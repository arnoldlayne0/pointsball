# Pointsball — Progress

## Component status

| Component | Status | File |
|---|---|---|
| FPL API fetching | ✅ Done | `pointsball/data/fpl_client.py` |
| Dataset building | ✅ Done | `pointsball/data/fpl_client.py` |
| Feature engineering | 🟡 Partial | `pointsball/features/fpl_features.py` — rolling windows only; double-GW bug acknowledged |
| ML model | ✅ Done | `pointsball/models/predictor.py` |
| Optimizer | ✅ Done | `pointsball/optimizer/squad.py` |
| FPL account (read) | ✅ Done | `pointsball/data/fpl_account.py` |
| Reddit RMT client | ✅ Done | `pointsball/data/reddit_client.py` |
| CLI | ✅ Done | `pointsball/__main__.py` |
| Apply transfers | ❌ Not started | Would require FPL auth (blocked by Datadome) |

---

## Known issues / future work

- **Double gameweek handling** — `generate_features_dataset()` duplicates rows for players with two fixtures in one GW. The rolling windows are computed on gameweek number, so double GWs inflate features.
- **Feature engineering improvements** — fixture difficulty (FDR) could be added as an explicit rolling or static feature.
- **Reddit signal** — mention count + sentiment per player per week could add a useful qualitative signal. Requires player name matching and a sentiment library.
- **Transfer writing** — submitting transfers via the FPL API requires authenticated session. The login endpoint (`users.premierleague.com`) is behind Datadome bot-detection, blocking all programmatic access.

---

## Bugs fixed

| Date | Bug | Fix |
|---|---|---|
| 2026-03-14 | `fpl_features.py` dropped `team_id_opponent`/`gameweek_opponent` after joins, but Polars 1.x auto-coalesces right-side join keys | Removed the three redundant `.drop()` calls |
| 2026-03-14 | `reddit_client.py` scanned Hive-partitioned parquet without `hive_partitioning=True`, causing `ColumnNotFoundError: "date"` | Added `hive_partitioning=True` to `pl.scan_parquet()` |

---

## Architecture decisions

- **No Dagster** — removed in favour of plain Python + CLI. Can be re-added as an orchestration layer later without changing the core modules.
- **Public FPL API for team reads** — `fantasy.premierleague.com/api/entry/{id}/event/{gw}/picks/` is public; no authentication needed to read squad, bank, or transfer history.
- **Datadome blocks programmatic login** — `users.premierleague.com/accounts/login/` is protected by Datadome TLS fingerprinting. Headless browsers, stealth patches, and all Python HTTP clients are blocked. Write operations (apply transfers) are deferred until a workaround is found.
