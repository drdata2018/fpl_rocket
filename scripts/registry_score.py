"""Score the prospective registry once a GW is finished (fpl_rocket.registry).

Usage: uv run python scripts/registry_score.py 9 [10 ...] --csv registry_scores.csv
       uv run python scripts/registry_score.py --auto --csv ../spike-log/registry/registry_scores.csv

For each GW: downloads the deadline-gw<N> registry assets (--source: the releases URL, or a local directory
holding deadline-gw<N>/ folders), scores them against the FPL API (event live points, the roster managers'
actual picks) and appends rows to --csv. --auto scores every finished (and data-checked) GW from
REGISTRY_START_GW on that has no row yet. A week without registry assets gets a status=missing row: weeks are
never skipped.

--trajectories-csv also scores the four virtual managers (fpl_rocket.trajectories) of every finished GW not
yet in that CSV: realised points with automatic substitutions, hits, cumulative, vs the official GW average; a
week without its deadline-gw<N> trajectories is scored on the rolled-forward squad (status=missing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket import registry, trajectories  # noqa: E402
from fpl_rocket.api import FPLClient  # noqa: E402
from fpl_rocket.config import RELEASES_URL  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gws", nargs="*", type=int)
    ap.add_argument("--auto", action="store_true", help="score every finished GW not yet in --csv")
    ap.add_argument("--csv", default="registry_scores.csv")
    ap.add_argument("--trajectories-csv", default=None, help="also score the virtual managers into this CSV")
    ap.add_argument("--source", default=RELEASES_URL)
    ap.add_argument("--start", type=int, default=registry.REGISTRY_START_GW)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)
    client = FPLClient(args.cache, offline=args.offline)
    failed = False
    if args.trajectories_csv:
        tgws = list(args.gws)
        if args.auto:
            tgws += registry.pending_gws(
                client.bootstrap(), trajectories.scored_gws(args.trajectories_csv), args.start
            )
        for gw in sorted(set(tgws)):
            try:
                rows = trajectories.score_and_append(gw, args.trajectories_csv, client, args.source)
            except Exception as exc:  # noqa: BLE001 - the registry is still scored; retried next run
                failed = True
                print(f"GW{gw} virtual managers: FAILED ({type(exc).__name__}: {exc})")
                break
            line = ", ".join(f"{r['policy']} {r.get('net', '-')} ({r['status']})" for r in rows)
            print(f"GW{gw} virtual managers: {line}")
    gws = list(args.gws)
    if args.auto:
        gws += registry.pending_gws(client.bootstrap(), registry.scored_gws(args.csv), args.start)
    if not gws:
        print("nothing to score")
        return int(failed)
    for gw in sorted(set(gws)):
        rows = registry.score_and_append(gw, args.csv, client, args.source)
        head = rows[0]
        detail = head.get("reason") or (
            f"RMSE model {head.get('rmse_model', float('nan')):.3f} / baseline "
            f"{head.get('rmse_baseline', float('nan')):.3f}"
        )
        print(f"GW{gw}: {head['status']} - {detail}; {len(rows) - 1} entry rows")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
