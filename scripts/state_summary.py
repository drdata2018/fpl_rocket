"""Reconstruct and print the planning state for one or more entry IDs.

Usage: uv run python scripts/state_summary.py <entry_id> [<entry_id> ...] [--cache DIR] [--verbose]
"""

from __future__ import annotations

import argparse
import sys

from fpl_rocket.api import FPLAPIError, FPLClient
from fpl_rocket.state import PlanningStateError, build_planning_state, fetch_inputs, validate_squad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("entries", type=int, nargs="+")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    client = FPLClient(args.cache)
    bad = 0
    for eid in args.entries:
        try:
            st = build_planning_state(**fetch_inputs(client, eid))
        except (FPLAPIError, PlanningStateError) as exc:
            print(f"entry {eid}: ERROR {exc}")
            bad += 1
            continue
        illegal = validate_squad(st.squad, st.bank)
        bad += bool(illegal)
        print(st.summary() + (f" ILLEGAL {illegal}" if illegal else " legal"))
        if args.verbose:
            for w in st.warnings:
                print("   warn:", w)
            for p in st.squad:
                print(
                    f"   {p.element:>4} pos{p.element_type} team{p.team:>2} buy {p.purchase_price} now "
                    f"{p.current_price} sell {p.selling_price} ({p.purchase_source})"
                )
            print(
                "   chips available:",
                [(c["name"], c["start_event"], c["stop_event"]) for c in st.chips_available],
            )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
