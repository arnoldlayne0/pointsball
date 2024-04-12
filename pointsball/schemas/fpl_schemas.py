FPL_DATASET_PLAYERS_COLUMNS = [
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "element_type",
    "first_name",
    "second_name",
    "player_id",
    "now_cost",
    "selected_by_percent",
    "team_code",
    "player_name",
]
FPL_DATASET_PLAYER_HISTORIES_COLUMNS = [
    "player_id",
    "opponent_team",
    "total_points",
    "was_home",
    "kickoff_time",
    "team_h_score",
    "team_a_score",
    "gameweek",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",  # read as string currently
    "expected_assists",  # read as string currently
    "expected_goal_involvements",  # read as string currently
    "expected_goals_conceded",  # read as string currently
    "value",
    "transfers_balance",
    "selected",
    "transfers_in",
    "transfers_out",
    "fixture_id",
]
FPL_DATASET_TEAMS_COLUMNS = [
    "team_code",
    "team_id",
    "team_name",
    "team_name_short",
]
FPL_DATASET_FIXTURE_COLUMNS = [
    "team_h",
    "team_a",
    "team_h_difficulty",
    "team_a_difficulty",
    "fixture_id",
]
FPL_DATASET_COLUMNS = {
    "players": FPL_DATASET_PLAYERS_COLUMNS,
    "player_histories": FPL_DATASET_PLAYER_HISTORIES_COLUMNS,
    "teams": FPL_DATASET_TEAMS_COLUMNS,
    "fixtures": FPL_DATASET_FIXTURE_COLUMNS,
}
