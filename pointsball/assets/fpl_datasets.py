import aiohttp
import polars as pl
from dagster import AssetIn, Output, asset
from fpl import FPL


@asset(
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "fpl_elements"],
)
async def fpl_elements() -> Output[pl.DataFrame]:
    async with aiohttp.ClientSession() as session:
        fpl_session = FPL(session)
        players_json = await fpl_session.get_players(return_json=True)
    players_pldf = pl.DataFrame(players_json).rename({"id": "player_id", "web_name": "player_name"})
    return Output(players_pldf, metadata={"num_rows": players_pldf.shape[0]})


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
    player_histories_pldf = pl.DataFrame(histories).rename({"element": "player_id", "round": "gameweek"})
    return Output(
        player_histories_pldf,
        metadata={
            "num_rows": player_histories_pldf.shape[0],
            "min_round": int(player_histories_pldf["gameweek"].min()),
            "max_round": int(player_histories_pldf["gameweek"].max()),
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
    fixtures_pldf = pl.DataFrame(all_fixtures)
    return Output(
        fixtures_pldf,
        metadata={
            "num_rows": fixtures_pldf.shape[0],
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
    teams_pldf = pl.DataFrame(all_teams).rename(
        {
            "id": "team_id",
            "code": "team_code",
            "name": "team_name",
            "short_name": "team_name_short",
        }
    )
    return Output(teams_pldf, metadata={"num_rows": teams_pldf.shape[0]})
