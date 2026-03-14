import polars as pl

from pointsball.optimizer.squad import (
    BUDGET,
    MAX_PER_TEAM,
    POSITION_SQUAD,
    POSITION_XI_MIN,
    SQUAD_SIZE,
    XI_SIZE,
    optimize_transfers,
    select_squad,
)


def _make_pool(n_per_position: dict[int, int], base_cost: int = 60, team_spread: int = 20) -> pl.DataFrame:
    """Build a minimal synthetic player pool with varied predicted points and costs."""
    rows = []
    pid = 1
    for pos, count in n_per_position.items():
        for j in range(count):
            rows.append(
                {
                    "player_id": pid,
                    "player_name": f"P{pid}",
                    "predicted_points": float(pos * 10 + j),  # varied by position + index
                    "cost": base_cost + j * 2,
                    "element_type": pos,
                    "team_id": (pid % team_spread) + 1,
                }
            )
            pid += 1
    return pl.DataFrame(rows)


# Pool large enough for a valid squad: more than required per position
POOL = _make_pool({1: 6, 2: 12, 3: 12, 4: 8})


def test_select_squad_size():
    result = select_squad(POOL)
    assert len(result.squad_ids) == SQUAD_SIZE
    assert len(result.xi_ids) == XI_SIZE


def test_select_squad_xi_subset_of_squad():
    result = select_squad(POOL)
    assert set(result.xi_ids).issubset(set(result.squad_ids))


def test_select_squad_captain_in_xi():
    result = select_squad(POOL)
    assert result.captain_id in result.xi_ids


def test_select_squad_vc_in_xi_and_not_captain():
    result = select_squad(POOL)
    assert result.vice_captain_id in result.xi_ids
    assert result.vice_captain_id != result.captain_id


def test_select_squad_position_composition():
    result = select_squad(POOL)
    pool_lookup = {r["player_id"]: r["element_type"] for r in POOL.to_dicts()}

    squad_by_pos: dict[int, int] = {}
    for pid in result.squad_ids:
        pos = pool_lookup[pid]
        squad_by_pos[pos] = squad_by_pos.get(pos, 0) + 1

    for pos, required in POSITION_SQUAD.items():
        assert (
            squad_by_pos.get(pos, 0) == required
        ), f"Position {pos}: expected {required}, got {squad_by_pos.get(pos, 0)}"


def test_select_squad_xi_position_minimums():
    result = select_squad(POOL)
    pool_lookup = {r["player_id"]: r["element_type"] for r in POOL.to_dicts()}

    xi_by_pos: dict[int, int] = {}
    for pid in result.xi_ids:
        pos = pool_lookup[pid]
        xi_by_pos[pos] = xi_by_pos.get(pos, 0) + 1

    for pos, min_count in POSITION_XI_MIN.items():
        assert (
            xi_by_pos.get(pos, 0) >= min_count
        ), f"XI position {pos}: expected >={min_count}, got {xi_by_pos.get(pos, 0)}"
    # Exactly 1 GK
    assert xi_by_pos.get(1, 0) == 1


def test_select_squad_budget():
    pool_lookup = {r["player_id"]: r["cost"] for r in POOL.to_dicts()}
    result = select_squad(POOL, budget=BUDGET)
    total_cost = sum(pool_lookup[pid] for pid in result.squad_ids)
    assert total_cost <= BUDGET


def test_select_squad_max_per_team():
    pool_lookup = {r["player_id"]: r["team_id"] for r in POOL.to_dicts()}
    result = select_squad(POOL)
    from collections import Counter

    team_counts = Counter(pool_lookup[pid] for pid in result.squad_ids)
    for team, count in team_counts.items():
        assert count <= MAX_PER_TEAM, f"Team {team} has {count} players (max {MAX_PER_TEAM})"


def test_optimize_transfers_respects_free_transfer():
    """With 1 free transfer, making >1 transfer should incur a hit."""
    initial = select_squad(POOL)
    result = optimize_transfers(POOL, initial.squad_ids, free_transfers=1)
    if result.transfers_made > 1:
        assert result.hit_cost == 4 * (result.transfers_made - 1)
    else:
        assert result.hit_cost == 0


def test_optimize_transfers_valid_squad():
    """Transfer result must still satisfy all squad constraints."""
    initial = select_squad(POOL)
    result = optimize_transfers(POOL, initial.squad_ids, free_transfers=1)
    assert len(result.squad_ids) == SQUAD_SIZE
    assert len(result.xi_ids) == XI_SIZE
    assert result.captain_id in result.xi_ids
