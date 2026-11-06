"""Per-GW rows for finished gameweeks the community archive does not have yet (current season), from the
public FPL API: event/{gw}/live + fixtures + bootstrap -> data/archive/live_gws.parquet (archive schema).

Usage: uv run python -I scripts/load_live_gws.py [--season 2026-27] [--archive data/archive] [--cache DIR]

Why: the blind replay (fpl_rocket.replay) scores and trains on per-GW rows; the community archive
(scripts/load_archive.py) lags the live season (2026-27 stops at GW6 on 2026-11-06). Only GWs that are
finished AND data_checked and absent from all_gws.parquet are written.

One row per (player, fixture) of every club with a fixture in the GW. A club with two fixtures in the GW
(double GW): counting stats come from each fixture's `explain` entry and the expected-goals family (not in
`explain`) is split across the fixtures in proportion to minutes.
Limitations, documented in the replay report:
`team` is the club in the current bootstrap (no mid-season move is detectable from the live endpoint), `value`
is the CURRENT now_cost and `selected` is null (neither is used for scoring or by the replay's predictors).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket.api import FPLClient  # noqa: E402
from fpl_rocket.archive import (  # noqa: E402
    ARCHIVE_COLUMNS,
    DEFCON_STATS,
    FLOAT_STATS,
    INT_STATS,
    NULLABLE_INT,
)

SPLIT_BY_MINUTES = (
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
)


def live_rows(season: str, gw: int, live: dict, fixtures: list[dict], bootstrap: dict) -> pd.DataFrame:
    """Archive-schema rows for one finished GW (pure: API json in, frame out)."""
    teams = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    players = {int(p["id"]): p for p in bootstrap["elements"]}
    fx = [f for f in fixtures if f.get("event") == gw]
    by_team: dict[int, list[dict]] = {}
    for f in fx:
        by_team.setdefault(int(f["team_h"]), []).append(f)
        by_team.setdefault(int(f["team_a"]), []).append(f)
    rows = []
    for el in live["elements"]:
        e = int(el["id"])
        p = players.get(e)
        if p is None:
            continue
        team = int(p["team"])
        mine = sorted(by_team.get(team, []), key=lambda f: (f["kickoff_time"] or "", f["id"]))
        if not mine:
            continue
        stats = el["stats"]
        explain = {
            int(x["fixture"]): {s["identifier"]: s["value"] for s in x["stats"]} for x in el["explain"]
        }
        total_min = float(stats.get("minutes") or 0)
        for f in mine:
            home = int(f["team_h"]) == team
            if len(mine) == 1:
                s = dict(stats)
            else:
                ex = explain.get(int(f["id"]), {})
                s = {k: ex.get(k, 0) for k in (*INT_STATS, *DEFCON_STATS, "total_points")}
                share = float(ex.get("minutes", 0)) / total_min if total_min > 0 else 0.0
                for k in SPLIT_BY_MINUTES:
                    s[k] = float(stats.get(k) or 0) * share
                s["total_points"] = sum(
                    x["points"] for xx in el["explain"] if xx["fixture"] == f["id"] for x in xx["stats"]
                )
            opp = int(f["team_a"]) if home else int(f["team_h"])
            row = {
                "season": season,
                "gw": gw,
                "element": e,
                "code": int(p["code"]),
                "player_name": str(p.get("web_name", "")),
                "element_type": int(p["element_type"]),
                "team": teams[team],
                "team_id": team,
                "opponent": teams[opp],
                "opponent_id": opp,
                "was_home": home,
                "kickoff_time": pd.Timestamp(f["kickoff_time"]),
                "fixture": int(f["id"]),
                "value": int(p["now_cost"]),
                "selected": np.nan,
                "transfers_in": np.nan,
                "transfers_out": np.nan,
                "total_points": s.get("total_points", 0),
            }
            for c in (*INT_STATS, *FLOAT_STATS, *DEFCON_STATS):
                row[c] = s.get(c, np.nan)
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in (
        *INT_STATS,
        *DEFCON_STATS,
        "value",
        "selected",
        "transfers_in",
        "transfers_out",
        "total_points",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype(NULLABLE_INT)
    for c in FLOAT_STATS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")
    for c in ("gw", "element", "code", "element_type", "team_id", "opponent_id", "fixture"):
        df[c] = df[c].astype(NULLABLE_INT)
    df["was_home"] = df["was_home"].astype("boolean")
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    for c in ("player_name", "team", "opponent"):
        df[c] = df[c].astype("string")
    return df[list(ARCHIVE_COLUMNS)].sort_values(["gw", "element", "fixture"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--archive", default="data/archive")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)
    archive = Path(args.archive)
    have = pd.read_parquet(archive / "all_gws.parquet", columns=["season", "gw"])
    done = set(have.loc[have["season"] == args.season, "gw"].astype(int))
    client = FPLClient(args.cache)
    bootstrap = client.bootstrap()
    fixtures = client.fixtures()
    gws = [
        int(e["id"])
        for e in bootstrap["events"]
        if e["finished"] and e["data_checked"] and int(e["id"]) not in done
    ]
    frames = [live_rows(args.season, g, client.event_live(g), fixtures, bootstrap) for g in gws]
    out = archive / "live_gws.parquet"
    if not frames:
        print("nothing to add")
        return 0
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"{out}: {len(df)} rows for {args.season} GW {gws}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
