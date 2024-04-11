from polars.testing import assert_frame_equal

from pointsball.features.fpl_features import GroupRollingAvgFeatures


def test_group_rolling_avg_features(rolling_points_bps_fixture, fpl_dataset_fixture):
    features_df = GroupRollingAvgFeatures(
        groups=["player_id"], period="2i", statistics=["total_points", "bps"]
    ).transform(fpl_dataset_fixture)
    assert_frame_equal(features_df, rolling_points_bps_fixture)
