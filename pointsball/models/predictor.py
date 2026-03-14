import logging
from pathlib import Path

import joblib
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from pointsball.features.fpl_features import generate_features_dataset

logger = logging.getLogger(__name__)

TARGET_COL = "total_points"

# Columns that identify a row but carry no predictive signal
_IDENTIFIER_COLS = {"player_id", "team_id", "gameweek", "opponent_team"}

# Columns from the raw dataset to include as extra features alongside rolling averages
_EXTRA_FEATURE_COLS = [
    "chance_of_playing_next_round",
    "value",
    "selected_by_percent",
]


def _fixture_difficulty(df: pl.DataFrame) -> pl.DataFrame:
    """Replace team_h/a_difficulty with a single fixture_difficulty column based on was_home."""
    return df.with_columns(
        pl.when(pl.col("was_home"))
        .then(pl.col("team_h_difficulty"))
        .otherwise(pl.col("team_a_difficulty"))
        .alias("fixture_difficulty")
    ).drop(["team_h_difficulty", "team_a_difficulty"])


def _feature_cols(df: pl.DataFrame) -> list[str]:
    skip = _IDENTIFIER_COLS | {TARGET_COL}
    return [c for c in df.columns if c not in skip]


def build_training_data(dataset_df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Generate features and join with target + extra columns. Returns (df, feature_cols)."""
    features_df = generate_features_dataset(dataset_df)
    extras = dataset_df.select(
        ["player_id", "gameweek", "was_home", TARGET_COL, "team_h_difficulty", "team_a_difficulty"]
        + _EXTRA_FEATURE_COLS
    ).pipe(_fixture_difficulty)

    df = features_df.join(extras, on=["player_id", "gameweek", "was_home"])

    feature_cols = _feature_cols(df)

    # Drop rows where ALL rolling features are null (early GWs with no rolling history yet)
    rolling_cols = [c for c in feature_cols if "rolling" in c]
    if rolling_cols:
        df = df.filter(pl.any_horizontal(pl.col(rolling_cols).is_not_null()))

    return df, feature_cols


def train(dataset_df: pl.DataFrame) -> tuple[HistGradientBoostingRegressor, list[str]]:
    """Train a HistGradientBoostingRegressor on historical FPL data. Returns (model, feature_cols)."""
    df, feature_cols = build_training_data(dataset_df)
    logger.info("Training on %d rows with %d features.", len(df), len(feature_cols))

    X = df.select(feature_cols).to_numpy()
    y = df[TARGET_COL].to_numpy()

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    logger.info("Training complete.")
    return model, feature_cols


def predict_upcoming(
    model: HistGradientBoostingRegressor,
    feature_cols: list[str],
    dataset_df: pl.DataFrame,
    fixtures_df: pl.DataFrame,
    elements_df: pl.DataFrame,
) -> pl.DataFrame:
    """Predict total_points for each player in upcoming (unfinished) fixtures.

    Returns a DataFrame with columns: player_id, player_name, gameweek, fixture_id,
    was_home, opponent_team, predicted_points.
    """
    features_df = generate_features_dataset(dataset_df)

    # Latest rolling-average snapshot per player per home/away context
    latest_features = features_df.filter(
        pl.col("gameweek") == pl.col("gameweek").max().over(["player_id", "was_home"])
    ).drop(["gameweek", "opponent_team", "team_id"])

    # Upcoming fixtures (event/gameweek must be assigned, i.e. not blank gameweeks)
    upcoming = fixtures_df.filter(~pl.col("finished") & pl.col("gameweek").is_not_null()).select(
        ["fixture_id", "gameweek", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    )

    if upcoming.is_empty():
        logger.warning("No upcoming fixtures found — returning empty predictions.")
        return pl.DataFrame()

    home_rows = upcoming.select(
        "fixture_id",
        "gameweek",
        pl.col("team_h").alias("team_id"),
        pl.col("team_a").alias("opponent_team"),
        pl.lit(True).alias("was_home"),
        pl.col("team_h_difficulty").alias("fixture_difficulty"),
    )
    away_rows = upcoming.select(
        "fixture_id",
        "gameweek",
        pl.col("team_a").alias("team_id"),
        pl.col("team_h").alias("opponent_team"),
        pl.lit(False).alias("was_home"),
        pl.col("team_a_difficulty").alias("fixture_difficulty"),
    )
    fixture_rows = pl.concat([home_rows, away_rows])

    # Current player info (elements_df uses raw API column names: "team" and "now_cost")
    player_info = elements_df.select(
        [
            "player_id",
            pl.col("team").alias("team_id"),
            "player_name",
            "chance_of_playing_next_round",
            pl.col("now_cost").alias("value"),
            "selected_by_percent",
        ]
    )

    # Build prediction rows: player × upcoming fixture
    pred_base = player_info.join(fixture_rows, on="team_id")

    # Join rolling features matched on player + home/away context
    pred_df = pred_base.join(
        latest_features.drop(["element_type"]),
        on=["player_id", "was_home"],
        how="left",
    )

    # Ensure all training feature columns are present (fill missing with null)
    for col in feature_cols:
        if col not in pred_df.columns:
            pred_df = pred_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    X = pred_df.select(feature_cols).to_numpy()
    predictions = model.predict(X)

    return pred_df.select(
        ["player_id", "player_name", "gameweek", "fixture_id", "was_home", "opponent_team"]
    ).with_columns(pl.Series("predicted_points", predictions))


def save_model(model: HistGradientBoostingRegressor, feature_cols: list[str], path: Path) -> None:
    """Persist model and feature column list to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, path)
    logger.info("Model saved to %s.", path)


def load_model(path: Path) -> tuple[HistGradientBoostingRegressor, list[str]]:
    """Load model and feature column list from disk."""
    artifact = joblib.load(path)
    return artifact["model"], artifact["feature_cols"]
