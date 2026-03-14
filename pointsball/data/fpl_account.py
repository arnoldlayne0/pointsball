import logging
import os
from dataclasses import dataclass

import polars as pl
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
_ENTRY_URL = "https://fantasy.premierleague.com/api/entry/{}/"
_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{}/event/{}/picks/"


@dataclass
class MyTeam:
    squad_ids: list[int]  # 15 player IDs
    captain_id: int
    vice_captain_id: int
    bank: int  # tenths of millions (e.g. 15 = £1.5M)
    free_transfers: int  # estimated transfers available for next GW


def _team_id() -> int:
    load_dotenv()
    raw = os.environ.get("FPL_TEAM_ID")
    if not raw:
        raise ValueError("FPL_TEAM_ID is not set in .env")
    return int(raw)


def _current_gameweek() -> int:
    bootstrap = requests.get(_BOOTSTRAP_URL, timeout=15).json()
    return next(e["id"] for e in bootstrap["events"] if e["is_current"])


def _estimate_free_transfers(event_transfers: int, event_transfers_cost: int) -> int:
    """Estimate free transfers available for the *next* gameweek.

    Logic: if 0 transfers were made this GW the rollover gives 2 next week;
    otherwise 1 (standard weekly allowance). Taking a hit doesn't affect
    next week's allowance.
    """
    if event_transfers == 0:
        return 2
    return 1


def fetch_my_team(team_id: int | None = None) -> MyTeam:
    """Fetch current squad and transfer info from the public FPL API.

    No authentication required — FPL exposes all team data publicly by team ID.
    Reads FPL_TEAM_ID from .env if team_id is not supplied.
    """
    if team_id is None:
        team_id = _team_id()

    gw = _current_gameweek()
    logger.info("Fetching team %d picks for GW%d.", team_id, gw)

    picks_resp = requests.get(_PICKS_URL.format(team_id, gw), timeout=15)
    picks_resp.raise_for_status()
    data = picks_resp.json()

    picks = data["picks"]
    history = data["entry_history"]

    free_transfers = _estimate_free_transfers(
        history["event_transfers"],
        history["event_transfers_cost"],
    )

    return MyTeam(
        squad_ids=[p["element"] for p in picks],
        captain_id=next(p["element"] for p in picks if p["is_captain"]),
        vice_captain_id=next(p["element"] for p in picks if p["is_vice_captain"]),
        bank=history["bank"],
        free_transfers=free_transfers,
    )


def format_my_team(my_team: MyTeam, elements_df: pl.DataFrame) -> str:
    """Return a human-readable summary of the current FPL squad."""
    pos_labels = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    lookup = {r["player_id"]: r for r in elements_df.to_dicts()}

    lines = ["=== Current Squad ==="]
    for pid in sorted(my_team.squad_ids, key=lambda p: lookup.get(p, {}).get("element_type", 9)):
        info = lookup.get(pid, {})
        pos = pos_labels.get(info.get("element_type", 0), "???")
        name = info.get("player_name", f"ID:{pid}")
        cost = info.get("now_cost", 0) / 10
        tags = []
        if pid == my_team.captain_id:
            tags.append("(C)")
        if pid == my_team.vice_captain_id:
            tags.append("(V)")
        lines.append(f"  {pos:3s}  {name:<25s} £{cost:.1f}M  {'  '.join(tags)}")

    lines.append(f"\nBank:           £{my_team.bank / 10:.1f}M")
    lines.append(f"Free transfers: {my_team.free_transfers} (estimated)")
    lines.append("\nTo optimize transfers run:\n  uv run pointsball optimize --use-my-team")
    return "\n".join(lines)
