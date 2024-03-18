import aiohttp
import polars as pl
from dagster import AssetIn, Output, asset
from fpl import FPL

from pointsball.schemas.fpl_schemas import FPL_DATASET_COLUMNS


@asset(
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "fpl_elements"],
)
async def fpl_elements() -> Output[pl.DataFrame]:
    async with aiohttp.ClientSession() as session:
        fpl_session = FPL(session)
        players_json = await fpl_session.get_players(return_json=True)
    players_df = pl.DataFrame(players_json).rename({"id": "player_id", "web_name": "player_name"})
    return Output(players_df, metadata={"num_rows": players_df.shape[0]})


@asset(
    ins={"fpl_elements": AssetIn("fpl_elements")},
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "fpl_player_histories"],
)
async def fpl_player_histories(fpl_elements: pl.DataFrame) -> Output[pl.DataFrame]:
    player_ids = fpl_elements["player_id"].unique().to_list()
    async with aiohttp.ClientSession() as session:
        fpl_session = FPL(session)
        player_summaries = await fpl_session.get_player_summaries(player_ids=player_ids, return_json=True)
    histories = [hist for ps in player_summaries for hist in ps["history"]]
    player_histories_df = pl.DataFrame(histories).rename({"element": "player_id", "round": "gameweek"})
    return Output(
        player_histories_df,
        metadata={
            "num_rows": player_histories_df.shape[0],
            "min_round": int(player_histories_df["gameweek"].min()),
            "max_round": int(player_histories_df["gameweek"].max()),
        },
    )


@asset(
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "fpl_fixtures"],
)
async def fpl_fixtures() -> Output[pl.DataFrame]:
    async with aiohttp.ClientSession() as session:
        fpl_session = FPL(session)
        all_fixtures = await fpl_session.get_fixtures(return_json=True)
    fixtures_df = pl.DataFrame(all_fixtures)
    return Output(
        fixtures_df,
        metadata={
            "num_rows": fixtures_df.shape[0],
        },
    )


@asset(
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "fpl_teams"],
)
async def fpl_teams() -> Output[pl.DataFrame]:
    async with aiohttp.ClientSession() as session:
        fpl_session = FPL(session)
        all_teams = await fpl_session.get_teams(return_json=True)
    teams_df = pl.DataFrame(all_teams).rename(
        {
            "id": "team_id",
            "code": "team_code",
            "name": "team_name",
            "short_name": "team_name_short",
        }
    )
    return Output(teams_df, metadata={"num_rows": teams_df.shape[0]})


@asset(
    ins={
        "fpl_elements": AssetIn("fpl_elements"),
        "fpl_player_histories": AssetIn("fpl_player_histories"),
        "fpl_teams": AssetIn("fpl_teams"),
    },
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "fpl_dataset"],
)
def fpl_dataset(fpl_elements: pl.DataFrame, fpl_player_histories: pl.DataFrame, fpl_teams: pl.DataFrame):
    elements_to_join = fpl_elements[FPL_DATASET_COLUMNS["players"]]
    player_histories_to_join = fpl_player_histories[FPL_DATASET_COLUMNS["player_histories"]]
    teams_to_join = fpl_teams[FPL_DATASET_COLUMNS["teams"]]
    fpl_dataset_df = elements_to_join.join(player_histories_to_join, on="player_id").join(teams_to_join, on="team_code")
    return Output(fpl_dataset_df)
