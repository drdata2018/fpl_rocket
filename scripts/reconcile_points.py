"""Recompute every player's GW points from event/{gw}/live stats and compare with the API total.

Usage: uv run python scripts/reconcile_points.py <gw> [--cache DIR]
Exit code 1 when any mismatch is found.
"""

from __future__ import annotations

import argparse
import sys

from fpl_rocket.api import FPLClient
from fpl_rocket.reconcile import reconcile_gw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gw", type=int)
    ap.add_argument("--cache", default=None, help="optional raw JSON cache directory")
    args = ap.parse_args(argv)

    client = FPLClient(args.cache)
    bootstrap = client.bootstrap()
    live = client.event_live(args.gw)
    checked, mismatches = reconcile_gw(live, bootstrap)
    print(f"GW{args.gw}: checked {checked} players, mismatches {len(mismatches)}")
    for m in mismatches[:20]:
        api_parts = [
            (s["identifier"], s["value"], s["points"]) for fx in m.api_explain for s in fx.get("stats", [])
        ]
        print(
            f"  element {m.element} type {m.element_type}: api {m.api_total} vs computed {m.computed_total}"
            f"  breakdown={m.breakdown}  api_explain={api_parts}"
        )
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
