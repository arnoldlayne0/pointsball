import polars as pl
import pytest


@pytest.fixture
def fpl_elements_fixture():
    return pl.read_csv("tests/data/fpl_elements.csv")


@pytest.fixture
def fpl_player_histories_fixture():
    return pl.read_csv("tests/data/fpl_player_histories.csv")


@pytest.fixture
def fpl_teams_fixture():
    return pl.read_csv("tests/data/fpl_teams.csv")


@pytest.fixture
def fpl_dataset_fixture():
    return pl.read_csv("tests/data/fpl_dataset.csv")


@pytest.fixture
def rolling_points_bps_fixture():
    return pl.read_csv("tests/data/rolling_points_bps.csv")
