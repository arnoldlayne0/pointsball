from polars.testing import assert_frame_equal

from pointsball.data.fpl_client import build_fpl_dataset


def test_fpl_elements(fpl_elements_fixture):
    assert fpl_elements_fixture.shape == (3, 88)


def test_fpl_player_histories(fpl_player_histories_fixture):
    assert fpl_player_histories_fixture.shape == (15, 36)


def test_fpl_teams(fpl_teams_fixture):
    assert fpl_teams_fixture.shape == (3, 21)


def test_fpl_dataset(
    fpl_elements_fixture, fpl_player_histories_fixture, fpl_teams_fixture, fpl_fixtures_fixture, fpl_dataset_fixture
):
    result = build_fpl_dataset(
        fpl_elements_fixture, fpl_player_histories_fixture, fpl_teams_fixture, fpl_fixtures_fixture
    )
    assert_frame_equal(result, fpl_dataset_fixture)
