def test_fpl_elements(fpl_elements_fixture):
    assert fpl_elements_fixture.shape == (3, 88)


def test_fpl_player_histories(fpl_player_histories_fixture):
    assert fpl_player_histories_fixture.shape == (15, 36)


def test_fpl_teams(fpl_teams_fixture):
    assert fpl_teams_fixture.shape == (3, 21)
