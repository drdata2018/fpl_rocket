"""Builds the recorded replay fixtures (run once from the repository root; the outputs are committed):

    uv run python tests/fixtures/replay/build_fixture.py

asknown_mini/   a slice of data/asknown for 2021-22 GW5 (bootstrap players / teams / events, fixtures as
                known, lagged crowd tables, deadlines of 2021-22, English clubs' European matches of 2021-22)
history_mini.parquet  points-history rows (data/replay_inputs/points_history.parquet) of 2021-22 GW1-6
                and of every 2021-22 GW17 fixture, including the REAL rearranged fixture 166 (Leicester v
                Spurs: in GW17 as known at the GW17 deadline, postponed, played 2022-01-19 in GW22)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "data" / "asknown"
SEASON, GW = "2021-22", 5
KEEP_COLS = [
    "season",
    "gw",
    "element",
    "code",
    "player_name",
    "element_type",
    "team",
    "team_id",
    "opponent",
    "opponent_id",
    "was_home",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "defensive_contribution",
    "total_points",
    "value",
    "selected",
    "kickoff_time",
    "fixture",
    "known_at",
]


def main() -> None:
    out = HERE / "asknown_mini"
    if out.exists():
        shutil.rmtree(out)
    for kind in ("players", "teams", "events"):
        d = out / "bootstrap" / kind / SEASON
        d.mkdir(parents=True)
        shutil.copy2(SRC / "bootstrap" / kind / SEASON / f"gw{GW}.parquet", d / f"gw{GW}.parquet")
    (out / "fixtures" / SEASON).mkdir(parents=True)
    shutil.copy2(SRC / "fixtures" / SEASON / f"gw{GW}.parquet", out / "fixtures" / SEASON / f"gw{GW}.parquet")
    dl = pd.read_parquet(SRC / "deadlines.parquet")
    dl[dl["season"] == SEASON].to_parquet(out / "deadlines.parquet", index=False)
    (out / "crowd").mkdir()
    for kind in ("players", "tiers"):
        t = pd.read_parquet(SRC / "crowd" / f"lagged_{kind}.parquet")
        t[(t["season"] == SEASON) & (t["gw"] == GW)].to_parquet(
            out / "crowd" / f"lagged_{kind}.parquet", index=False
        )
    (out / "european").mkdir()
    eu = pd.read_parquet(SRC / "european" / "english_club_matches.parquet")
    eu[eu["season"] == SEASON].to_parquet(out / "european" / "english_club_matches.parquet", index=False)
    h = pd.read_parquet(ROOT / "data" / "replay_inputs" / "points_history.parquet")
    s = h[h["season"] == SEASON]
    gw17 = pd.read_parquet(SRC / "fixtures" / SEASON / "gw17.parquet")
    gw17_ids = set(gw17.loc[gw17["event"] == 17, "fixture"].astype(int))
    keep = s[(s["gw"] <= 6) | s["fixture"].isin(gw17_ids)]
    keep[list(KEEP_COLS)].to_parquet(HERE / "history_mini.parquet", index=False)
    total = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    print(f"fixtures: {total / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
