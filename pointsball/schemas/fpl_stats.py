from enum import Enum


class PlayerFormStatsEnum(Enum):
    total_points = "total_points"
    bps = "bps"
    influence = "influence"
    creativity = "creativity"
    threat = "threat"
    expected_goal_involvements = "expected_goal_involvements"
    expected_goals_conceded = "expected_goals_conceded"
    goals_scored = "goals_scored"
    assists = "assists"
    clean_sheets = "clean_sheets"
    goals_conceded = "goals_conceded"
    own_goals = "own_goals"
    penalties_saved = "penalties_saved"
    penalties_missed = "penalties_missed"
    yellow_cards = "yellow_cards"
    red_cards = "red_cards"


class PlayerSelectionStatsEnum(Enum):
    selected_by_percent = "selected_by_percent"
    value = "value"
    transfers_balance = "transfers_balance"


class PlayerIdEnum(Enum):
    player_id = "player_id"
    gameweek = "gameweek"


class TeamIdEnum(Enum):
    team_code = "team_code"
    gameweek = "gameweek"


class StatelessPlayerFeaturesEnum(Enum):
    element_type = "element_type"
    chance_of_playing_next_round = "chance_of_playing_next_round"


class FixtureFeaturesEnum(Enum):
    opponent_team = "opponent_team"
    was_home = "was_home"
    team_h_score = "team_h_score"
    team_a_score = "team_a_score"
