"""Write the four virtual managers' deadline step for a pre-deadline bundle (fpl_rocket.trajectories).

Usage: uv run python scripts/trajectories_write.py --bundle data/bundles/<id> --out data/registry
                                                  [--archive data/archive/all_gws.parquet]
                                                  [--previous RELEASES_URL | DIR | none] [--time-limit 60]

Reads the bundle (sha256-checked), computes the frozen baseline's per-fixture xP (form per 90 x availability
on the bundle's current season + the archived previous season, as registry_write.py does), loads the previous
state from the latest EARLIER deadline-gw<k> release (--previous: the releases URL, a local directory of
deadline-gw<k>/ folders, or `none` = start afresh), takes the GW N step for every trajectory and writes
trajectories_state.json and trajectories_plans.json to --out. The bundle workflow uploads them into
deadline-gw<N> before the deadline. Exits 0 without writing for a GW before REGISTRY_START_GW. A read error
of an earlier release (other than "not found") aborts: the trajectories are never restarted by accident.
Needs the `model` dependency group (bundle_history_rows).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket import registry, trajectories  # noqa: E402
from fpl_rocket.config import RELEASES_URL  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default="data/registry")
    ap.add_argument("--archive", default="data/archive/all_gws.parquet")
    ap.add_argument("--previous", default=RELEASES_URL, help="releases URL, local deadline dir, or 'none'")
    ap.add_argument("--gw", type=int, default=None, help="override the bundle's next GW (dry runs only)")
    ap.add_argument("--time-limit", type=float, default=None, help="per-solve limit in seconds (default 60)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    inp = trajectories.inputs_from_bundle(args.bundle, args.archive, gw=args.gw, time_limit=args.time_limit)
    if inp.gw < registry.REGISTRY_START_GW:
        print(f"GW{inp.gw} is before REGISTRY_START_GW={registry.REGISTRY_START_GW}: nothing to write")
        return 0
    previous = None
    if args.previous != "none":
        found = trajectories.latest_state(args.previous, inp.gw)
        if found is not None:
            k, previous = found
            print(f"previous state: deadline-gw{k} (bundle {previous.get('bundle_id')})")
    if previous is None:
        print(f"no earlier state: creating the start at GW{inp.gw} ({trajectories.START_RULE})")
    state, plans = trajectories.step(previous, inp, log=print)
    paths = trajectories.write_files(args.out, state, plans)
    names = {int(p["id"]): p.get("web_name", str(p["id"])) for p in inp.bootstrap["elements"]}
    if state["created_gw"] == inp.gw:
        squad = ", ".join(names.get(e, str(e)) for e in state["start_squad"])
        print(
            f"start squad (cost {state['start_cost'] / 10:.1f}, bank {state['start_bank'] / 10:.1f}): {squad}"
        )
    for name, rec in plans["trajectories"].items():
        print(
            f"  {name}: {rec['status']} captain {names.get(rec['captain'])} vice {names.get(rec['vice'])};"
            f" in {[names.get(e) for e in rec['transfers_in']]}"
            f" out {[names.get(e) for e in rec['transfers_out']]}; hits {rec['hits']};"
            f" xP {rec.get('expected_points')}"
        )
    for p in paths:
        print(f"  {p} ({p.stat().st_size:,} bytes)")
    print(f"done in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
