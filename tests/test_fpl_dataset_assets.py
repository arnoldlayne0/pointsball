from polars.testing import assert_frame_equal

from pointsball.assets.fpl_datasets import fpl_dataset


def test_fpl_elements(fpl_elements_fixture):
    assert fpl_elements_fixture.shape == (3, 88)


def test_fpl_player_histories(fpl_player_histories_fixture):
    assert fpl_player_histories_fixture.shape == (15, 36)


def test_fpl_teams(fpl_teams_fixture):
    assert fpl_teams_fixture.shape == (3, 21)


def test_fpl_dataset(fpl_elements_fixture, fpl_player_histories_fixture, fpl_teams_fixture, fpl_dataset_fixture):
    result = fpl_dataset(fpl_elements_fixture, fpl_player_histories_fixture, fpl_teams_fixture)
    assert_frame_equal(result.value, fpl_dataset_fixture)
