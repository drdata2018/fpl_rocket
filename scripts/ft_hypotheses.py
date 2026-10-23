"""Test free-transfer chip-week policies against real managers' observed hit costs.

Samples entries from a classic league's standings (default: overall league 314, several pages), replays
each manager's free-transfer state machine under every chip_week_policy and counts managers whose
observed event_transfers_cost contradicts the replay. The policy with zero contradictions is the rule.

Usage: uv run python scripts/ft_hypotheses.py [--league 314] [--pages 1 20 200 2000 20000] [--cache DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

from fpl_rocket.api import FPLAPIError, FPLClient
from fpl_rocket.rules.season_2026_27 import TRANSFERS
from fpl_rocket.state import replay_free_transfers

POLICIES = ("carry", "carry_accrue", "reset_one")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", type=int, default=314)
    ap.add_argument("--pages", type=int, nargs="+", default=[1, 20, 200, 2000, 20000, 100000])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args(argv)
    client = FPLClient(args.cache)
    upcoming = next(e["id"] for e in client.bootstrap()["events"] if e["is_next"])

    entries: list[int] = []
    for page in args.pages:
        try:
            st = client.league_classic_standings(args.league, page)
        except FPLAPIError as exc:
            print(f"page {page}: {exc}")
            continue
        entries += [r["entry"] for r in st["standings"]["results"]]

    bad = {p: [] for p in POLICIES}
    chip_users = 0
    informative = 0  # managers whose chip week was followed by a GW with hits (discriminates policies)
    for eid in entries:
        try:
            hist = client.entry_history(eid)
            entry = client.entry(eid)
        except FPLAPIError:
            continue
        time.sleep(args.sleep if not args.cache else 0)
        chips = [c for c in hist["chips"] if c["name"] in ("wildcard", "freehit")]
        chip_users += bool(chips)
        started = int(entry.get("started_event") or 1)
        results = {}
        for policy in POLICIES:
            _, warns = replay_free_transfers(
                hist["current"], hist["chips"], started, upcoming, replace(TRANSFERS, chip_week_policy=policy)
            )
            results[policy] = warns
            if warns:
                bad[policy].append((eid, warns[:2]))
        informative += len({bool(w) for w in results.values()}) > 1
    print(
        f"sampled {len(entries)} entries; {chip_users} used WC/FH; "
        f"{informative} discriminate between policies"
    )
    for policy in POLICIES:
        print(f"  policy {policy:<13} contradicted by {len(bad[policy])} managers")
        for eid, w in bad[policy][:3]:
            print(f"      {eid}: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
