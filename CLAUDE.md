# Pointsball

## Project Overview
Pointsball is an automated framework to manage a Fantasy Premier League (FPL) team. It integrates data fetching, machine learning for performance prediction, and operations research (integer programming) to determine the mathematically optimal team setup and transfers.

## Core Components
1. **FPL API Client:** Interfaces with the official FPL API to fetch historical/live data and connect to the user's FPL account to read/update the team.
2. **ML Prediction Engine:** Uses historical data and fixture schedules to predict player points for the upcoming N gameweeks.
3. **Optimizer:** Uses Integer Programming to find the optimal squad and transfers, strictly enforcing FPL rules (budget constraints, max players per team, 1 free transfer/week, hit penalties, etc.).
4. **CLI Entrypoint:** A command-line interface to orchestrate the pipeline and execute commands.

## Tech Stack & Tooling
- **Language:** Python
- **Dependency Management:** `uv` (Note: We are migrating from Poetry. Use `uv` strictly for all package management and run commands. Ignore `poetry.lock`).
- **Data Processing:** `polars` (used for all tabular data transformations).
- **Data Storage:** Persist datasets and predictions locally as `.parquet` files using Polars.
- **Machine Learning:** `scikit-learn` (specifically using `HistGradientBoostingRegressor`).
- **Optimization:** Google OR-Tools (`ortools`).
- **CLI Framework:** `typer`.
- **Orchestration:** Designed to be decoupled. Currently triggered via CLI, but architected so that an orchestrator like Dagster could be added later.

## Development Guidelines & Conventions
- **Strict Typing:** All functions, classes, and methods must include standard Python type hints.
- **Data Schemas:** All datasets must define explicit schemas (using Polars schemas or Pydantic) to ensure data integrity between the API, ML model, and Optimizer.
- **Secrets Management:** FPL credentials and session cookies must be loaded from a local `.env` file using `python-dotenv` or `os.environ`. Never hardcode secrets.
- **Testing:** The project maintains a test suite in the `/tests` directory. Always write or update tests when adding new features or modifying logic.
- **Pre-commit:** The repository uses `pre-commit` hooks. Ensure code passes all formatting and linting checks before committing.

## Common Commands
- **Install Dependencies:** `uv sync`
- **Add a Package:** `uv add <package>`
- **Run Tests:** `uv run pytest tests/`
- **Run CLI:** `uv run python -m pointsball` (or specific Typer command)
