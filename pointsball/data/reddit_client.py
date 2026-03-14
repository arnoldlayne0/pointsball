import datetime
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import polars as pl
from dotenv import load_dotenv
from praw import Reddit
from praw.models import Subreddit

logger = logging.getLogger(__name__)

RMT_SCHEMA = [
    'submission_created_utc',
    'submission_id',
    'submission_title',
    'submission_selftext',
    'submission_score',
    'submission_upvote_ratio',
    'comment_created_utc',
    'comment_id',
    'comment_body',
    'comment_score',
    'comment_parent_id',
]
TIME_FILTER_THRESHOLDS = {"day": 1, "week": 7, "month": 30}


@dataclass
class RedditCredentials:
    client_id: str
    secret: str
    user_agent: str
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "RedditCredentials":
        load_dotenv()
        return cls(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            secret=os.environ["REDDIT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
            username=os.environ["REDDIT_USERNAME"],
            password=os.environ["REDDIT_PASSWORD"],
        )


def _generate_rate_my_team_submissions(
    subreddit: Subreddit, comment_limit: int = 32, time_filter: str = "year"
) -> Iterator:
    for s in subreddit.search(query="Rate My Team", time_filter=time_filter):
        logger.info(f"Fetching submission {s.id}.")
        s.comments.replace_more(limit=comment_limit)
        for c in s.comments.list():
            yield (
                s.created_utc,
                s.id,
                s.title,
                s.selftext,
                s.score,
                s.upvote_ratio,
                c.created_utc,
                c.id,
                c.body,
                c.score,
                c.parent_id,
            )


def fetch_rate_my_team(credentials: RedditCredentials, data_dir: Path = Path("data/raw")) -> None:
    today = datetime.date.today()
    try:
        cur_lazy_rmt_df = pl.scan_parquet(str(data_dir / "reddit_rmt_submissions/*/*.parquet"), hive_partitioning=True)
        last_date_partition = cur_lazy_rmt_df.select("date").max().collect().item()
        last_date_partition_date = datetime.datetime.strptime(last_date_partition, "%Y-%m-%d").date()
        days_since_last_partition = (today - last_date_partition_date).days
        for filter_name, threshold in TIME_FILTER_THRESHOLDS.items():
            if days_since_last_partition <= threshold:
                time_filter = filter_name
                break
        else:
            time_filter = "year"
    except FileNotFoundError:
        time_filter = "year"
        last_date_partition_date = today - datetime.timedelta(days=365)

    logger.info(f"Fetching submissions from the last {time_filter}.")
    reddit = Reddit(
        client_id=credentials.client_id,
        client_secret=credentials.secret,
        user_agent=credentials.user_agent,
        username=credentials.username,
        password=credentials.password,
    )
    fpl_subreddit = reddit.subreddit("FantasyPL")
    lazy_rmt_df = pl.LazyFrame(
        _generate_rate_my_team_submissions(fpl_subreddit, time_filter=time_filter), schema=RMT_SCHEMA
    )
    lazy_rmt_df = (
        lazy_rmt_df.with_columns(
            pl.from_epoch(pl.col("submission_created_utc")).dt.date().alias("submission_date"),
            pl.from_epoch(pl.col("comment_created_utc")).dt.date().alias("comment_date"),
            pl.col("comment_parent_id").str.starts_with("t3").alias("top_level_comment"),
        )
        .filter(pl.col("submission_date") == pl.col("comment_date"))
        .filter(pl.col("submission_date") > last_date_partition_date)
        .drop(["comment_date"])
    )
    output_path = str(data_dir / "reddit_rmt_submissions")
    lazy_rmt_df.collect().rename({"submission_date": "date"}).write_parquet(
        file=output_path, use_pyarrow=True, pyarrow_options={"partition_cols": ["date"]}
    )
