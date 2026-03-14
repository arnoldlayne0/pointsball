from typing import List

import polars as pl

from pointsball.schemas.fpl_stats import PlayerFormStatsEnum, TeamFormStatsEnum


class GroupRollingAvgFeatures:
    def __init__(self, groups: List[str], period: str, statistics: List[str], suffix: str = ""):
        self.feature_names = [f"{statistic}_rolling_previous_{period}_{suffix}" for statistic in statistics]
        self.groups = groups
        self.period = period
        self.statistic = statistics

    def transform(self, df: pl.DataFrame):
        df = df.sort("gameweek")
        features = df.rolling(
            index_column="gameweek",
            period=self.period,
            by=self.groups,
        ).agg(
            [
                pl.mean(statistic).alias(feature_name)
                for statistic, feature_name in zip(self.statistic, self.feature_names, strict=True)
            ]
        )
        return features


### Ugly massive function
# does not yet handle double gameweeks properly - they get duplicated in the joins
def generate_features_dataset(fpl_dataset_df: pl.DataFrame):
    # Player features
    player_form_features_overall = GroupRollingAvgFeatures(
        groups=["player_id"], period="5i", statistics=[pfs.name for pfs in PlayerFormStatsEnum]
    ).transform(fpl_dataset_df)
    player_form_features_home_away = GroupRollingAvgFeatures(
        groups=["player_id", "was_home"],
        period="5i",
        statistics=[pfs.name for pfs in PlayerFormStatsEnum],
        suffix="home_away",
    ).transform(fpl_dataset_df)
    # Team features
    teams_df = fpl_dataset_df.group_by(["team_id", "gameweek", "was_home"]).agg(
        pl.sum("goals_scored", "expected_goals"),
        pl.max("goals_conceded", "expected_goals_conceded"),
        pl.first("team_code"),
    )
    teams_elements_df = fpl_dataset_df.group_by(["team_id", "gameweek", "was_home", "element_type"]).agg(
        pl.sum("goals_scored", "expected_goals"),
        pl.max("goals_conceded", "expected_goals_conceded"),
        pl.first("team_code"),
    )
    team_form_features = GroupRollingAvgFeatures(
        groups=["team_id"], period="5i", statistics=[pfs.name for pfs in TeamFormStatsEnum], suffix="team_overall"
    ).transform(teams_df)
    team_form_features_home_away = GroupRollingAvgFeatures(
        groups=["team_id", "was_home"],
        period="5i",
        statistics=[pfs.name for pfs in TeamFormStatsEnum],
        suffix="team_home_away",
    ).transform(teams_df)
    team_form_features_position = GroupRollingAvgFeatures(
        groups=["team_id", "element_type"],
        period="5i",
        statistics=[pfs.name for pfs in TeamFormStatsEnum],
        suffix="team_position",
    ).transform(teams_elements_df)
    base_df = fpl_dataset_df.select(["player_id", "team_id", "gameweek", "was_home", "element_type", "opponent_team"])

    features_df = (
        base_df.join(player_form_features_overall, on=["player_id", "gameweek"], how="left")
        .join(player_form_features_home_away, on=["player_id", "gameweek", "was_home"], how="left")
        .join(team_form_features, on=["team_id", "gameweek"], how="left")
        .join(team_form_features_home_away, on=["team_id", "gameweek", "was_home"], how="left")
        # Opponent team stats — Polars 1.x coalesces right-side join keys, so no manual drop needed
        .join(
            team_form_features.rename({c: f"{c}_opponent" for c in team_form_features.columns}),
            left_on=["opponent_team", "gameweek"],
            right_on=["team_id_opponent", "gameweek_opponent"],
            how="left",
        )
        .join(
            team_form_features_home_away.with_columns(~pl.col("was_home")).rename(
                {c: f"{c}_opponent" for c in team_form_features_home_away.columns}
            ),
            left_on=["opponent_team", "gameweek", "was_home"],
            right_on=["team_id_opponent", "gameweek_opponent", "was_home_opponent"],
            how="left",
        )
        .join(
            team_form_features_position.rename({c: f"{c}_opponent" for c in team_form_features_position.columns}),
            left_on=["opponent_team", "gameweek", "element_type"],
            right_on=["team_id_opponent", "gameweek_opponent", "element_type_opponent"],
            how="left",
        )
    )
    return features_df
