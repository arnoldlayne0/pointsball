from typing import Iterator

import polars as pl
from dagster import asset
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


def generate_rate_my_team_submissions(subreddit: Subreddit, comment_limit: int = 32) -> Iterator:
    for s in subreddit.search(query="Rate My Team", time_filter="year"):
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


@asset
def reddit_rate_my_team(reddit_credentials: RedditResource):
    reddit = Reddit(
        client_id=reddit_credentials.client_id,
        client_secret=reddit_credentials.secret,
        user_agent=reddit_credentials.user_agent,
    )
    fpl_subreddit = reddit.subreddit("FantasyPL")
    lazy_rmt_df = pl.LazyFrame(generate_rate_my_team_submissions(fpl_subreddit), schema=RMT_SCHEMA)
    lazy_rmt_df = lazy_rmt_df.with_columns(
        pl.from_epoch(pl.col("submission_created_utc")).dt.date().alias("submission_date"),
        pl.from_epoch(pl.col("comment_created_utc")).dt.date().alias("comment_date"),
        pl.col("comment_parent_id").str.starts_with("t3").alias("top_level_comment"),
    ).filter(pl.col("submission_date") == pl.col("comment_date"))
    lazy_rmt_df.sink_parquet("data/raw/rate_my_team.parquet")
