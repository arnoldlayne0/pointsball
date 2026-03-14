"""
FPL squad optimizer using Integer Programming (OR-Tools CBC).

Squad rules enforced:
  - 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD
  - Starting XI of 11: exactly 1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD
  - Budget: £100M (1000 in FPL tenths-of-millions units)
  - Max 3 players from any single team
  - 1 captain (2× points), 1 vice-captain

Transfer rules enforced:
  - Each transfer beyond free_transfers costs 4 points (a "hit")
"""

import logging
from dataclasses import dataclass

import polars as pl
from ortools.linear_solver import pywraplp

logger = logging.getLogger(__name__)

BUDGET = 1000  # £100M in tenths of millions
SQUAD_SIZE = 15
XI_SIZE = 11
MAX_PER_TEAM = 3
HIT_PENALTY = 4  # points per extra transfer

# Required counts in the 15-man squad by position (1=GK, 2=DEF, 3=MID, 4=FWD)
POSITION_SQUAD = {1: 2, 2: 5, 3: 5, 4: 3}
# Minimum starters in XI by position
POSITION_XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}


@dataclass
class OptimizedSquad:
    squad_ids: list[int]
    xi_ids: list[int]
    captain_id: int
    vice_captain_id: int
    total_predicted_points: float
    transfers_made: int
    hit_cost: int


def prepare_player_pool(
    predictions_df: pl.DataFrame,
    elements_df: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate predicted points across GWs and join with cost/position data.

    Returns a DataFrame with columns:
        player_id, player_name, predicted_points, cost, element_type, team_id
    """
    points_by_player = predictions_df.group_by("player_id").agg(pl.col("predicted_points").sum())
    player_info = elements_df.select(
        ["player_id", "player_name", "now_cost", "element_type", pl.col("team").alias("team_id")]
    ).rename({"now_cost": "cost"})

    return (
        points_by_player.join(player_info, on="player_id")
        .filter(pl.col("predicted_points").is_not_null())
        .sort("predicted_points", descending=True)
    )


def _solve(
    player_pool: pl.DataFrame,
    budget: int,
    current_squad_ids: set[int] | None = None,
    free_transfers: int = 0,
) -> OptimizedSquad:
    rows = player_pool.to_dicts()
    n = len(rows)
    player_ids = [r["player_id"] for r in rows]
    costs = [r["cost"] for r in rows]
    points = [r["predicted_points"] for r in rows]
    positions = [r["element_type"] for r in rows]
    teams = [r["team_id"] for r in rows]

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("Could not create CBC solver.")

    squad = [solver.IntVar(0, 1, f"s_{i}") for i in range(n)]
    xi = [solver.IntVar(0, 1, f"x_{i}") for i in range(n)]
    captain = [solver.IntVar(0, 1, f"c_{i}") for i in range(n)]

    # Objective: points for each starter + bonus points for the captain
    obj = solver.Objective()
    for i in range(n):
        obj.SetCoefficient(xi[i], points[i])
        obj.SetCoefficient(captain[i], points[i])  # captain earns an extra copy of their points
    obj.SetMaximization()

    # Squad / XI / captain sizes
    solver.Add(sum(squad) == SQUAD_SIZE)
    solver.Add(sum(xi) == XI_SIZE)
    solver.Add(sum(captain) == 1)

    for i in range(n):
        solver.Add(xi[i] <= squad[i])  # can only start if in squad
        solver.Add(captain[i] <= xi[i])  # captain must start

    # Budget
    solver.Add(sum(costs[i] * squad[i] for i in range(n)) <= budget)

    # Squad composition by position
    for pos, count in POSITION_SQUAD.items():
        idx = [i for i, p in enumerate(positions) if p == pos]
        solver.Add(sum(squad[i] for i in idx) == count)

    # XI composition by position (min starters + exactly 1 GK)
    for pos, min_xi in POSITION_XI_MIN.items():
        idx = [i for i, p in enumerate(positions) if p == pos]
        solver.Add(sum(xi[i] for i in idx) >= min_xi)
    gk_idx = [i for i, p in enumerate(positions) if p == 1]
    solver.Add(sum(xi[i] for i in gk_idx) == 1)

    # Max 3 players per club
    for team in set(teams):
        idx = [i for i, t in enumerate(teams) if t == team]
        solver.Add(sum(squad[i] for i in idx) <= MAX_PER_TEAM)

    # Transfer penalty (only applies when optimizing transfers)
    extra_var = None
    if current_squad_ids is not None:
        keep_vars = [squad[i] for i, pid in enumerate(player_ids) if pid in current_squad_ids]
        extra_var = solver.IntVar(0, SQUAD_SIZE, "extra_transfers")
        # extra_var >= transfers - free_transfers
        # transfers = SQUAD_SIZE - sum(keep_vars)
        # => sum(keep_vars) + extra_var >= SQUAD_SIZE - free_transfers
        solver.Add(sum(keep_vars) + extra_var >= SQUAD_SIZE - free_transfers)
        obj.SetCoefficient(extra_var, -float(HIT_PENALTY))

    logger.info("Solving with %d players...", n)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"Optimizer found no feasible solution (status={status}).")

    squad_ids = [player_ids[i] for i in range(n) if squad[i].solution_value() > 0.5]
    xi_ids = [player_ids[i] for i in range(n) if xi[i].solution_value() > 0.5]
    captain_id = next(player_ids[i] for i in range(n) if captain[i].solution_value() > 0.5)

    # Vice-captain: highest predicted points among starters, excluding the captain
    vc_id = (
        player_pool.filter(pl.col("player_id").is_in(xi_ids))
        .filter(pl.col("player_id") != captain_id)
        .sort("predicted_points", descending=True)["player_id"][0]
    )

    transfers_made = 0
    hit_cost = 0
    if current_squad_ids is not None:
        transfers_made = sum(1 for pid in squad_ids if pid not in current_squad_ids)
        hit_cost = HIT_PENALTY * max(0, transfers_made - free_transfers)

    return OptimizedSquad(
        squad_ids=squad_ids,
        xi_ids=xi_ids,
        captain_id=captain_id,
        vice_captain_id=vc_id,
        total_predicted_points=solver.Objective().Value(),
        transfers_made=transfers_made,
        hit_cost=hit_cost,
    )


def select_squad(player_pool: pl.DataFrame, budget: int = BUDGET) -> OptimizedSquad:
    """Select an optimal fresh 15-man squad from the player pool."""
    logger.info("Running fresh squad selection.")
    return _solve(player_pool, budget)


def optimize_transfers(
    player_pool: pl.DataFrame,
    current_squad_ids: list[int],
    free_transfers: int = 1,
    bank: int = 0,
) -> OptimizedSquad:
    """Optimize transfers given the current squad.

    Args:
        player_pool: output of prepare_player_pool().
        current_squad_ids: list of 15 player_ids currently in the squad.
        free_transfers: number of free transfers available (usually 1, max 2).
        bank: extra funds in the bank in tenths of millions (e.g. 5 = £0.5M).
    """
    current_set = set(current_squad_ids)
    current_cost = player_pool.filter(pl.col("player_id").is_in(current_set))["cost"].sum()
    available_budget = current_cost + bank
    logger.info(
        "Running transfer optimization. Budget=£%.1fM, free_transfers=%d.",
        available_budget / 10,
        free_transfers,
    )
    return _solve(player_pool, available_budget, current_set, free_transfers)


def format_squad(squad: OptimizedSquad, player_pool: pl.DataFrame) -> str:
    """Return a human-readable squad summary."""
    lookup = {r["player_id"]: r for r in player_pool.to_dicts()}
    lines = []

    pos_labels = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    def player_line(pid: str, in_xi: bool) -> str:
        r = lookup[pid]
        tags = []
        if pid == squad.captain_id:
            tags.append("(C)")
        elif pid == squad.vice_captain_id:
            tags.append("(V)")
        if not in_xi:
            tags.append("[bench]")
        tag_str = " ".join(tags)
        return f"  {pos_labels[r['element_type']]:3s}  {r['player_name']:<25s} £{r['cost']/10:.1f}M  {r['predicted_points']:.1f}pts  {tag_str}"

    lines.append("=== Starting XI ===")
    for pid in sorted(squad.xi_ids, key=lambda p: lookup[p]["element_type"]):
        lines.append(player_line(pid, in_xi=True))

    lines.append("\n=== Bench ===")
    bench_ids = [pid for pid in squad.squad_ids if pid not in squad.xi_ids]
    for pid in sorted(bench_ids, key=lambda p: lookup[p]["element_type"]):
        lines.append(player_line(pid, in_xi=False))

    lines.append(f"\nPredicted points (XI + captain bonus): {squad.total_predicted_points:.1f}")
    if squad.transfers_made:
        lines.append(f"Transfers made: {squad.transfers_made}  |  Hit cost: -{squad.hit_cost}pts")

    return "\n".join(lines)
