import logging
from pathlib import Path

import aiohttp
import polars as pl
from fpl import FPL

from pointsball.schemas.fpl_schemas import FPL_DATASET_COLUMNS

logger = logging.getLogger(__name__)


async def fetch_fpl_elements() -> pl.DataFrame:
    async with aiohttp.ClientSession() as session:
        fpl_session = FPL(session)
        players_json = await fpl_session.get_players(return_json=True)
    return pl.DataFrame(players_json).rename({"id": "player_id", "web_name": "player_name"})


async def fetch_fpl_player_histories(player_ids: list[int]) -> pl.DataFrame:
    async with aiohttp.ClientSession() as session:
        fpl_session = FPL(session)
        player_summaries = await fpl_session.get_player_summaries(player_ids=player_ids, return_json=True)
    histories = [hist for ps in player_summaries for hist in ps["history"]]
    df = pl.DataFrame(histories).rename(
        {
            "element": "player_id",
            "round": "gameweek",
            "fixture": "fixture_id",
        }
    )
    return df.with_columns(
        pl.col("expected_goals").cast(pl.Float64),
        pl.col("expected_assists").cast(pl.Float64),
        pl.col("expected_goal_involvements").cast(pl.Float64),
        pl.col("expected_goals_conceded").cast(pl.Float64),
    )


async def fetch_fpl_fixtures() -> pl.DataFrame:
    async with aiohttp.ClientSession() as session:
        fpl_session = FPL(session)
        all_fixtures = await fpl_session.get_fixtures(return_json=True)
    return pl.DataFrame(all_fixtures).rename({"event": "gameweek", "id": "fixture_id"})


async def fetch_fpl_teams() -> pl.DataFrame:
    async with aiohttp.ClientSession() as session:
        fpl_session = FPL(session)
        all_teams = await fpl_session.get_teams(return_json=True)
    return pl.DataFrame(all_teams).rename(
        {
            "id": "team_id",
            "code": "team_code",
            "name": "team_name",
            "short_name": "team_name_short",
        }
    )


def build_fpl_dataset(
    elements: pl.DataFrame,
    player_histories: pl.DataFrame,
    teams: pl.DataFrame,
    fixtures: pl.DataFrame,
) -> pl.DataFrame:
    elements_to_join = elements[FPL_DATASET_COLUMNS["players"]]
    histories_to_join = player_histories[FPL_DATASET_COLUMNS["player_histories"]]
    teams_to_join = teams[FPL_DATASET_COLUMNS["teams"]]
    fixtures_to_join = fixtures.filter(pl.col("finished")).select(FPL_DATASET_COLUMNS["fixtures"])
    return (
        elements_to_join.join(histories_to_join, on="player_id")
        .join(teams_to_join, on="team_code")
        .join(fixtures_to_join, on="fixture_id")
    )


async def run_fpl_pipeline(data_dir: Path = Path("data/raw")) -> pl.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching FPL elements...")
    elements = await fetch_fpl_elements()
    elements.write_parquet(data_dir / "fpl_elements.parquet")

    logger.info("Fetching FPL player histories...")
    player_ids = elements["player_id"].unique().to_list()
    player_histories = await fetch_fpl_player_histories(player_ids)
    player_histories.write_parquet(data_dir / "fpl_player_histories.parquet")

    logger.info("Fetching FPL fixtures...")
    fixtures = await fetch_fpl_fixtures()
    fixtures.write_parquet(data_dir / "fpl_fixtures.parquet")

    logger.info("Fetching FPL teams...")
    teams = await fetch_fpl_teams()
    teams.write_parquet(data_dir / "fpl_teams.parquet")

    logger.info("Building FPL dataset...")
    dataset = build_fpl_dataset(elements, player_histories, teams, fixtures)
    dataset.write_parquet(data_dir / "fpl_dataset.parquet")

    logger.info(f"FPL dataset written to {data_dir / 'fpl_dataset.parquet'} ({dataset.shape[0]} rows).")
    return dataset
