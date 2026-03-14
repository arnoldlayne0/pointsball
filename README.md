# Pointsball

Automated Fantasy Premier League team management. Pointsball fetches live FPL data, trains a machine learning model to predict player points, and uses integer programming to select the mathematically optimal squad and transfers.

---

## How it works

### 1. Data fetching

Pointsball pulls data from the public FPL API:

- **Player elements** — name, position, club, current price, ownership, chance of playing
- **Player histories** — per-gameweek stats for every player across previous seasons
- **Fixtures** — home/away, difficulty ratings, gameweek assignment
- **Teams** — team IDs and names

All data is stored locally as `.parquet` files under `data/raw/`.

### 2. Feature engineering

Historical player stats are transformed into rolling window features:

| Feature | Description |
|---|---|
| `total_points_roll3` / `roll5` | Rolling average points over last 3 and 5 GWs |
| `goals_scored_roll3` / `roll5` | Rolling goal average |
| `assists_roll3` / `roll5` | Rolling assist average |
| `minutes_roll3` / `roll5` | Rolling minutes played |
| `bps_roll3` / `roll5` | Rolling bonus points average |
| `fixture_difficulty` | Opponent difficulty rating (1–5), adjusted for home/away |

Rolling windows are computed per `(player_id, was_home)` grouping so home and away performances are tracked separately.

### 3. ML model

The predictor uses `HistGradientBoostingRegressor` (scikit-learn), which handles missing values natively — no imputation needed for early-season rows where rolling windows are partially null.

**Training**: The model is trained on all historical gameweeks where at least some rolling features are present. The target is `total_points` per fixture.

**Prediction**: For each player with an upcoming fixture, the model takes their most recent rolling feature snapshot plus the incoming fixture difficulty. Players without upcoming fixtures are excluded.

### 4. Optimizer

Squad selection and transfer optimization are solved as a Mixed-Integer Program using Google OR-Tools (CBC solver).

**Decision variables:**
- `squad[i]` — binary, player `i` is in the 15-player squad
- `xi[i]` — binary, player `i` is in the starting XI
- `captain[i]` — binary, player `i` is captain (doubles their points)

**Constraints:**
- Exactly 15 players in squad (2 GK, 5 DEF, 5 MID, 3 FWD)
- Exactly 11 starters (1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD)
- Total squad cost ≤ £100M
- Maximum 3 players from any single club
- Captain must be in starting XI; vice-captain selected greedily post-solve

**Transfer penalty:** Extra transfers beyond the free allowance cost 4 points each. The penalty is linearized with an auxiliary variable `e ≥ transfers_out − free_transfers`, added to the objective as `−4e`.

**Objective:** Maximise total predicted points from the starting XI, plus captain's predicted points (so captain contributes 2× their prediction).

---

## Setup

### Prerequisites

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency management

### Install

```bash
git clone <repo>
cd pointsball
uv sync
```

### Configure

Create a `.env` file in the project root:

```env
# Required: your FPL team ID (visible in the URL on the FPL website)
FPL_TEAM_ID=123456
```

Your FPL team ID is in the URL when you view your team: `https://fantasy.premierleague.com/entry/123456/`

---

## CLI commands

### Full pipeline (recommended weekly workflow)

```bash
uv run pointsball run
```

Runs all four steps in sequence: fetch → train → predict → optimize. Prints the optimal squad at the end.

To optimize transfers against your **current squad** instead of selecting from scratch:

```bash
uv run pointsball run --use-my-team
```

---

### Individual commands

#### Fetch FPL data

```bash
uv run pointsball fetch
```

Downloads all FPL data and writes parquet files to `data/raw/`. Run this first, or use `run` which calls it automatically.

#### Train the model

```bash
uv run pointsball train
```

Trains the points-prediction model on `data/raw/fpl_dataset.parquet` and saves it to `data/models/model.joblib`.

#### Generate predictions

```bash
uv run pointsball predict
```

Loads the trained model, scores all players with upcoming fixtures, and writes `data/predictions/predictions.parquet`.

#### Optimize squad / transfers

```bash
# Select the best possible squad from scratch
uv run pointsball optimize

# Optimize transfers against your current squad (reads from FPL API)
uv run pointsball optimize --use-my-team

# Optimize transfers against a manually specified squad
uv run pointsball optimize --squad "123 456 789 ..." --free-transfers 2 --bank 15
```

`--bank` is in tenths of millions (e.g. `15` = £1.5M extra budget). `--squad` expects exactly 15 space-separated player IDs.

#### View your current squad

```bash
uv run pointsball show-team
```

Prints your current FPL squad, bank balance, and estimated free transfers for next week. Reads from the public FPL API — no authentication required.

---

## Data flow

```
FPL API
  └─► fpl_client.py ──► data/raw/*.parquet
                              │
                     fpl_features.py
                              │
                     predictor.py (train)
                              │
                     data/models/model.joblib
                              │
                     predictor.py (predict)
                              │
                     data/predictions/predictions.parquet
                              │
                     optimizer/squad.py
                              │
                     stdout: squad / transfer recommendations
```

---

## Development

```bash
# Run tests
uv run pytest tests/

# Add a dependency
uv add <package>
```

The project uses `pre-commit` hooks for formatting and linting. Install them with:

```bash
uv run pre-commit install
```

---

## Known limitations

- **Double gameweek handling** — players with two fixtures in one GW are represented as duplicate rows. Rolling windows are computed on gameweek number, so double GWs slightly inflate feature values.
- **Apply transfers** — submitting transfers via the FPL API requires an authenticated session. The login endpoint is behind Datadome bot-detection, which blocks all programmatic HTTP clients. Transfer submission is not currently supported.
