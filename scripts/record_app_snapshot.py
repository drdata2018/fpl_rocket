"""Record a trimmed, de-personalised API snapshot for the offline app tests (tests/test_app.py).

Usage: uv run python scripts/record_app_snapshot.py --entries <id> [--clone-as 1000001] [--out DIR]

--clone-as copies the first entry under a fake ID (renamed "Test Team 2") so that two-session tests need no
second real manager's data in the repo.

The snapshot is in FPLClient's raw-cache layout, so the app replays it with
FPL_ROCKET_CACHE_DIR=<dir> FPL_ROCKET_OFFLINE=1. Personal data is never written: team and manager names,
regions, badges and league names are replaced; league standings keep only entry ids, ranks and totals.
Player fields are trimmed to what the app reads; live stats drop the per-fixture `explain` lists.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket.api import FPLClient  # noqa: E402
from fpl_rocket.views import LIVE_STATS  # noqa: E402

PLAYER_FIELDS = (
    "id",
    "web_name",
    "team",
    "element_type",
    "now_cost",
    "selected_by_percent",
    "total_points",
    "points_per_game",
    "form",
    "minutes",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "bonus",
    "bps",
    "defensive_contribution",
    "clean_sheets",
    "saves",
    "status",
    "news",
    "news_added",
    "chance_of_playing_next_round",
    "cost_change_event",
    "cost_change_start",
    "transfers_in_event",
    "transfers_out_event",
)
EVENT_FIELDS = (
    "id",
    "name",
    "deadline_time",
    "finished",
    "data_checked",
    "is_previous",
    "is_current",
    "is_next",
    "average_entry_score",
    "highest_score",
    "most_captained",
    "most_vice_captained",
    "top_element_info",
)


def _trim_bootstrap(b: dict[str, Any]) -> dict[str, Any]:
    out = {k: b[k] for k in ("chips", "game_settings", "element_types", "teams", "total_players")}
    out["events"] = [{k: e.get(k) for k in EVENT_FIELDS} for e in b["events"]]
    out["elements"] = [{k: p.get(k) for k in PLAYER_FIELDS} for p in b["elements"]]
    return out


def _trim_live(live: dict[str, Any]) -> dict[str, Any]:
    return {
        "elements": [
            {"id": el["id"], "stats": {k: el["stats"].get(k, 0) for k in LIVE_STATS}}
            for el in live["elements"]
        ]
    }


def _scrub_entry(e: dict[str, Any], n: int) -> dict[str, Any]:
    keep = (
        "id",
        "started_event",
        "current_event",
        "entered_events",
        "summary_overall_points",
        "summary_overall_rank",
        "summary_event_points",
        "summary_event_rank",
        "last_deadline_bank",
        "last_deadline_value",
        "last_deadline_total_transfers",
        "joined_time",
    )
    out = {k: e.get(k) for k in keep}
    out["name"] = f"Test Team {n}"
    out["player_first_name"] = f"Manager{n}"
    classic = []
    for i, lg in enumerate(e.get("leagues", {}).get("classic", [])[:3], 1):
        classic.append(
            {
                "id": lg["id"],
                "name": f"League {i}",
                "league_type": lg.get("league_type"),
                "entry_rank": lg.get("entry_rank"),
                "entry_last_rank": lg.get("entry_last_rank"),
                "rank_count": lg.get("rank_count"),
                "active_phases": [
                    {"phase": p.get("phase"), "total": p.get("total"), "rank": p.get("rank")}
                    for p in lg.get("active_phases") or []
                ],
            }
        )
    out["leagues"] = {"classic": classic, "h2h": [], "cup": {}, "cup_matches": []}
    return out


def _scrub_standings(s: dict[str, Any]) -> dict[str, Any]:
    # other managers' entry ids are replaced by small synthetic ids
    rows = [
        {"entry": i, **{k: r.get(k) for k in ("rank", "rank_sort", "last_rank", "total", "event_total")}}
        for i, r in enumerate(s["standings"]["results"], 1)
    ]
    return {"league": {"id": s["league"]["id"]}, "standings": {"has_next": False, "page": 1, "results": rows}}


def _clone(out: Path, src: int, dst: int, n: int) -> None:
    """Copy every entry_<src>* file to entry_<dst>*, swapping the ID and the scrubbed names."""
    for f in out.glob(f"entry_{src}*.json"):
        body = json.loads(f.read_text(encoding="utf-8"))
        text = json.dumps(body, ensure_ascii=False).replace(str(src), str(dst))
        body = json.loads(text)
        if isinstance(body, dict) and "player_first_name" in body:
            body["name"], body["player_first_name"] = f"Test Team {n}", f"Manager{n}"
        target = out / f.name.replace(f"entry_{src}", f"entry_{dst}", 1)
        target.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entries", type=int, nargs="+", required=True)
    ap.add_argument("--clone-as", type=int, default=1000001)
    ap.add_argument("--out", default="tests/fixtures/app_snapshot")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    live_client = FPLClient(backoff=0.5)
    snap = FPLClient(out)  # used only for its cache layout (_cache_write)

    boot = live_client.bootstrap()
    snap._cache_write("bootstrap-static/", None, _trim_bootstrap(boot))
    snap._cache_write(
        "fixtures/", None, [{k: v for k, v in f.items() if k != "stats"} for f in live_client.fixtures()]
    )
    finished = [int(e["id"]) for e in boot["events"] if e["finished"]]
    for gw in finished:
        snap._cache_write(f"event/{gw}/live/", None, _trim_live(live_client.event_live(gw)))

    summaries: set[int] = set()
    meta: dict[str, Any] = {"entries": args.entries, "finished_gws": finished, "leagues": {}}
    snap._cache_write("entry/999999999/", None, {"__404__": True, "__body__": None})  # unknown-ID case
    for n, eid in enumerate(args.entries, 1):
        entry = live_client.entry(eid)
        hist = live_client.entry_history(eid)
        snap._cache_write(f"entry/{eid}/", None, _scrub_entry(entry, n))
        snap._cache_write(
            f"entry/{eid}/history/", None, {"current": hist["current"], "chips": hist["chips"], "past": []}
        )
        snap._cache_write(f"entry/{eid}/transfers/", None, live_client.entry_transfers(eid))
        for h in hist["current"]:
            gw = int(h["event"])
            pk = live_client.entry_picks(eid, gw)
            snap._cache_write(f"entry/{eid}/event/{gw}/picks/", None, pk)
        started = int(entry.get("started_event") or 1)
        first = live_client.entry_picks(eid, started)
        summaries |= {int(p["element"]) for p in (first or {}).get("picks", [])}
        meta["leagues"][str(eid)] = [lg["id"] for lg in _scrub_entry(entry, n)["leagues"]["classic"]]
        for lg in _scrub_entry(entry, n)["leagues"]["classic"]:
            st = live_client.league_classic_standings(lg["id"], 1)
            snap._cache_write(
                f"leagues-classic/{lg['id']}/standings/", {"page_standings": 1}, _scrub_standings(st)
            )
    for pid in sorted(summaries):
        es = live_client.element_summary(pid)
        rows = [{k: r.get(k) for k in ("round", "value", "total_points", "minutes")} for r in es["history"]]
        snap._cache_write(f"element-summary/{pid}/", None, {"history": rows, "fixtures": []})
    if args.clone_as:
        _clone(out, args.entries[0], args.clone_as, len(args.entries) + 1)
        meta["entries"] = [*args.entries, args.clone_as]
        meta["leagues"][str(args.clone_as)] = meta["leagues"][str(args.entries[0])]
    (out / "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"wrote {len(list(out.iterdir()))} files, {size / 1e6:.2f} MB to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
