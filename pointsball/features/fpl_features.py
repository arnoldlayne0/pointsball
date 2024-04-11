from typing import List

import polars as pl


class GroupRollingAvgFeatures:
    def __init__(self, groups: List[str], period: str, statistics: List[str], suffix: str = ""):
        self.feature_names = [f"{statistic}_rolling_previous_{period}_{suffix}" for statistic in statistics]
        self.groups = groups
        self.period = period
        self.statistic = statistics

    def transform(self, df: pl.DataFrame):
        features = df.rolling(
            index_column="gameweek",
            period=self.period,
            by=self.groups,
        ).agg(
            [
                pl.mean(statistic).alias(feature_name)
                for statistic, feature_name in zip(self.statistic, self.feature_names)
            ]
        )
        return features


# def generate_features_dataset(df: pl.DataFrame):
#     # Stateless player features
#     player_form_features_overall = GroupRollingAvgFeatures(
#         groups=["player_id"], period="5i", statistics=[pfs.name for pfs in PlayerFormStatsEnum]
#     ).transform(df)
#     player_form_features_home_away = GroupRollingAvgFeatures(
#         groups=["player_id", "was_home"],
#         period="5i",
#         statistics=[pfs.name for pfs in PlayerFormStatsEnum],
#         suffix="home_away",
#     ).transform(df)
