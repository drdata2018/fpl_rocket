"""Record trimmed API JSON into tests/fixtures/ for offline tests.

Usage: uv run python scripts/record_fixtures.py --entry <entry_id> --gw <finished_gw> [--out tests/fixtures]

Writes:
    bootstrap_trim.json         events, chips, game_settings, element_types, teams, and every player with
                                the fields the tests need (id, element_type, team, now_cost, web_name)
    event_<gw>_live.json        event/<gw>/live, all players (compact, ~0.5 MB); --keep-live N trims
                                to the first N players plus the entry's squad
    entry_<id>/{entry,history,transfers,picks_<gw>,picks_<started>,initial_values}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fpl_rocket.api import FPLClient
from fpl_rocket.state import fetch_inputs

PLAYER_FIELDS = ("id", "element_type", "team", "now_cost", "web_name", "status")
# personal data (names, region, badge, league names) is never written into the repo
ENTRY_FIELDS = (
    "id",
    "started_event",
    "current_event",
    "entered_events",
    "last_deadline_bank",
    "last_deadline_value",
    "last_deadline_total_transfers",
    "summary_overall_points",
    "summary_event_points",
    "joined_time",
)


def _dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry", type=int, required=True)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--out", default="tests/fixtures")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--keep-live", type=int, default=0, help="0 = keep all players")
    args = ap.parse_args(argv)
    out = Path(args.out)
    client = FPLClient(args.cache)

    inputs = fetch_inputs(client, args.entry)
    b = inputs["bootstrap"]
    trimmed = {k: b[k] for k in ("events", "chips", "game_settings", "element_types", "teams")}
    trimmed["elements"] = [{k: p.get(k) for k in PLAYER_FIELDS} for p in b["elements"]]
    _dump(out / "bootstrap_trim.json", trimmed)

    last_pub = max(inputs["picks"])
    squad = {p["element"] for p in (inputs["picks"][last_pub] or {}).get("picks", [])}
    live = client.event_live(args.gw)
    n = args.keep_live
    keep = [el for i, el in enumerate(live["elements"]) if not n or i < n or el["id"] in squad]
    _dump(out / f"event_{args.gw}_live.json", {"elements": keep})

    d = out / f"entry_{args.entry}"
    _dump(d / "entry.json", {k: inputs["entry"].get(k) for k in ENTRY_FIELDS})
    _dump(d / "history.json", inputs["history"])
    _dump(d / "transfers.json", inputs["transfers"])
    for gw, p in inputs["picks"].items():
        _dump(d / f"picks_{gw}.json", p)
    _dump(d / "initial_values.json", {str(k): v for k, v in inputs["initial_values"].items()})
    for f in sorted(out.rglob("*.json")):
        print(f"{f.stat().st_size / 1024:8.1f} KiB  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
