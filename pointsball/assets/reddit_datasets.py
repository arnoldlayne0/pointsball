import datetime
from typing import Iterator

import polars as pl
from dagster import AssetExecutionContext, asset
from praw import Reddit
from praw.models import Subreddit

from pointsball.resources.reddit_resources import RedditResource

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


def generate_rate_my_team_submissions(
    subreddit: Subreddit, context: AssetExecutionContext, comment_limit: int = 32, time_filter: str = "year"
) -> Iterator:
    for s in subreddit.search(query="Rate My Team", time_filter=time_filter):
        context.log.info(f"Fetching submission {s.id}.")
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


@asset()
def reddit_rate_my_team(context: AssetExecutionContext, reddit_credentials: RedditResource):
    today = datetime.date.today()
    try:
        cur_lazy_rmt_df = pl.scan_parquet("data/raw/reddit_rmt_submissions/*/*.parquet")
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
    context.log.info(f"Fetching submissions from the last {time_filter}.")
    reddit = Reddit(
        client_id=reddit_credentials.client_id,
        client_secret=reddit_credentials.secret,
        user_agent=reddit_credentials.user_agent,
        username=reddit_credentials.username,
        password=reddit_credentials.password,
    )
    fpl_subreddit = reddit.subreddit("FantasyPL")
    lazy_rmt_df = pl.LazyFrame(
        generate_rate_my_team_submissions(fpl_subreddit, context=context, time_filter=time_filter), schema=RMT_SCHEMA
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
    lazy_rmt_df.collect().rename({"submission_date": "date"}).write_parquet(
        file="data/raw/reddit_rmt_submissions", use_pyarrow=True, pyarrow_options={"partition_cols": ["date"]}
    )
