import asyncio
import logging
from pathlib import Path

import typer

app = typer.Typer(
    name="pointsball",
    help="Automated Fantasy Premier League team management.",
    no_args_is_help=True,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@app.command()
def fetch(
    data_dir: Path = typer.Option(Path("data/raw"), "--data-dir", "-d", help="Directory to write parquet files."),
) -> None:
    """Fetch all FPL data (players, histories, fixtures, teams) and write parquets."""
    from pointsball.data.fpl_client import run_fpl_pipeline

    asyncio.run(run_fpl_pipeline(data_dir))


@app.command()
def train(
    data_dir: Path = typer.Option(
        Path("data/raw"), "--data-dir", "-d", help="Directory containing fpl_dataset.parquet."
    ),
    model_dir: Path = typer.Option(
        Path("data/models"), "--model-dir", "-m", help="Directory to write the trained model."
    ),
) -> None:
    """Train the points-prediction model on historical FPL data."""
    import polars as pl

    from pointsball.models.predictor import save_model
    from pointsball.models.predictor import train as train_model

    dataset_df = pl.read_parquet(data_dir / "fpl_dataset.parquet")
    model, feature_cols = train_model(dataset_df)
    save_model(model, feature_cols, model_dir / "model.joblib")


@app.command()
def predict(
    data_dir: Path = typer.Option(Path("data/raw"), "--data-dir", "-d", help="Directory containing FPL parquets."),
    model_dir: Path = typer.Option(
        Path("data/models"), "--model-dir", "-m", help="Directory containing the trained model."
    ),
    output_dir: Path = typer.Option(
        Path("data/predictions"), "--output-dir", "-o", help="Directory to write predictions."
    ),
) -> None:
    """Predict player points for upcoming gameweeks."""
    import polars as pl

    from pointsball.models.predictor import load_model, predict_upcoming

    dataset_df = pl.read_parquet(data_dir / "fpl_dataset.parquet")
    fixtures_df = pl.read_parquet(data_dir / "fpl_fixtures.parquet")
    elements_df = pl.read_parquet(data_dir / "fpl_elements.parquet")

    model, feature_cols = load_model(model_dir / "model.joblib")
    predictions_df = predict_upcoming(model, feature_cols, dataset_df, fixtures_df, elements_df)

    if predictions_df.is_empty():
        typer.echo("No upcoming fixtures found — nothing to predict.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predictions.parquet"
    predictions_df.write_parquet(out_path)
    typer.echo(f"Predictions written to {out_path} ({len(predictions_df)} rows).")


@app.command()
def run(
    data_dir: Path = typer.Option(Path("data/raw"), "--data-dir", "-d", help="Directory for raw FPL parquets."),
    model_dir: Path = typer.Option(Path("data/models"), "--model-dir", "-m", help="Directory for the trained model."),
    predictions_dir: Path = typer.Option(
        Path("data/predictions"), "--predictions-dir", "-p", help="Directory for predictions."
    ),
    use_my_team: bool = typer.Option(False, "--use-my-team", help="Optimise transfers against your current FPL squad."),
    horizon: int = typer.Option(5, "--horizon", "-n", help="Number of upcoming gameweeks to optimise over."),
    discount: float = typer.Option(
        0.9, "--discount", help="Discount factor per GW (e.g. 0.9 means GW+1 worth 90%% of GW+0)."
    ),
) -> None:
    """Run the full pipeline: fetch → train → predict → optimize.

    The recommended weekly workflow. Use --use-my-team to get transfer
    recommendations for your actual squad instead of a fresh squad selection.
    Points in future GWs are discounted: GW+k is weighted by discount^k.
    """
    import polars as pl

    from pointsball.data.fpl_client import run_fpl_pipeline
    from pointsball.models.predictor import predict_upcoming, save_model
    from pointsball.models.predictor import train as train_model
    from pointsball.optimizer.squad import format_squad, optimize_transfers, prepare_player_pool, select_squad

    typer.echo("── Step 1/4: Fetching FPL data ──────────────────────────")
    dataset_df = asyncio.run(run_fpl_pipeline(data_dir))

    typer.echo("── Step 2/4: Training model ─────────────────────────────")
    model, feature_cols = train_model(dataset_df)
    save_model(model, feature_cols, model_dir / "model.joblib")

    typer.echo("── Step 3/4: Predicting upcoming gameweeks ──────────────")
    fixtures_df = pl.read_parquet(data_dir / "fpl_fixtures.parquet")
    elements_df = pl.read_parquet(data_dir / "fpl_elements.parquet")
    predictions_df = predict_upcoming(model, feature_cols, dataset_df, fixtures_df, elements_df)

    if predictions_df.is_empty():
        typer.echo("No upcoming fixtures found — skipping optimize step.")
        return

    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.write_parquet(predictions_dir / "predictions.parquet")

    typer.echo(f"── Step 4/4: Optimizing squad (horizon={horizon}, discount={discount}) ──")
    player_pool = prepare_player_pool(predictions_df, elements_df, horizon=horizon, discount=discount)

    if use_my_team:
        from pointsball.data.fpl_account import fetch_my_team

        my_team = fetch_my_team()
        result = optimize_transfers(
            player_pool, my_team.squad_ids, free_transfers=my_team.free_transfers, bank=my_team.bank
        )
    else:
        result = select_squad(player_pool)

    typer.echo(format_squad(result, player_pool))


@app.command()
def show_team(
    data_dir: Path = typer.Option(
        Path("data/raw"), "--data-dir", "-d", help="Directory containing fpl_elements.parquet."
    ),
) -> None:
    """Show current FPL squad, bank, and free transfers (requires FPL credentials in .env)."""
    import polars as pl

    from pointsball.data.fpl_account import fetch_my_team, format_my_team

    my_team = fetch_my_team()
    elements_df = pl.read_parquet(data_dir / "fpl_elements.parquet")
    typer.echo(format_my_team(my_team, elements_df))


@app.command()
def optimize(
    predictions_dir: Path = typer.Option(
        Path("data/predictions"), "--predictions-dir", "-p", help="Directory containing predictions.parquet."
    ),
    data_dir: Path = typer.Option(
        Path("data/raw"), "--data-dir", "-d", help="Directory containing fpl_elements.parquet."
    ),
    squad: str = typer.Option("", "--squad", "-s", help="Space-separated player_ids of current squad (15 ids)."),
    use_my_team: bool = typer.Option(
        False, "--use-my-team", help="Fetch current squad from FPL API (requires credentials in .env)."
    ),
    free_transfers: int = typer.Option(
        1,
        "--free-transfers",
        "-f",
        help="Free transfers available. Ignored when --use-my-team is set (value read from API).",
    ),
    bank: int = typer.Option(
        0, "--bank", "-b", help="Extra funds in bank in tenths of millions. Ignored when --use-my-team is set."
    ),
    horizon: int = typer.Option(5, "--horizon", "-n", help="Number of upcoming gameweeks to optimise over."),
    discount: float = typer.Option(
        0.9, "--discount", help="Discount factor per GW (e.g. 0.9 means GW+1 worth 90%% of GW+0)."
    ),
) -> None:
    """Select an optimal squad or optimize transfers using predicted points.

    Points in future GWs are discounted: GW+k is weighted by discount^k.
    Use --horizon to control how many gameweeks ahead to consider.
    """
    import polars as pl

    from pointsball.optimizer.squad import (
        format_squad,
        optimize_transfers,
        prepare_player_pool,
        select_squad,
    )

    predictions_path = predictions_dir / "predictions.parquet"
    if not predictions_path.exists():
        typer.echo(
            f"Error: {predictions_path} not found.\n"
            "Run the full pipeline first:  uv run pointsball run [--use-my-team]",
            err=True,
        )
        raise typer.Exit(1)

    predictions_df = pl.read_parquet(predictions_path)
    elements_df = pl.read_parquet(data_dir / "fpl_elements.parquet")
    player_pool = prepare_player_pool(predictions_df, elements_df, horizon=horizon, discount=discount)

    if use_my_team:
        from pointsball.data.fpl_account import fetch_my_team

        my_team = fetch_my_team()
        result = optimize_transfers(
            player_pool,
            my_team.squad_ids,
            free_transfers=my_team.free_transfers,
            bank=my_team.bank,
        )
    elif squad:
        current_ids = [int(x) for x in squad.split()]
        if len(current_ids) != 15:
            typer.echo(f"Error: --squad must contain exactly 15 player_ids (got {len(current_ids)}).", err=True)
            raise typer.Exit(1)
        result = optimize_transfers(player_pool, current_ids, free_transfers=free_transfers, bank=bank)
    else:
        result = select_squad(player_pool)

    typer.echo(format_squad(result, player_pool))


if __name__ == "__main__":
    app()
