from typing import List

import polars as pl


class BaseFeature:
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def transform(self, df: pl.DataFrame):
        raise NotImplementedError


class StatelessFeature(BaseFeature):
    def __init__(self, feature_name: str):
        super().__init__(feature_name=feature_name)

    def transform(self, df: pl.DataFrame):
        return df


class GroupCumulFeature(BaseFeature):
    def __init__(
        self, feature_name: str, groups: List[str], cumul_window: str, aggregation_function: pl.expr, statistic: str
    ):
        super().__init__(feature_name=feature_name)
        self.groups = groups
        self.cumul_window = cumul_window
        self.aggregation_function = aggregation_function
        self.statistic = statistic

    def transform(self, df: pl.DataFrame):
        feature = df.rolling(
            index_column="gameweek",
            period=self.cumul_window,
            by=self.groups,
        ).agg(self.aggregation_function(self.statistic).alias(self.feature_name))
        return feature
