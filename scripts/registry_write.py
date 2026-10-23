"""Write the prospective-registry assets for a pre-deadline bundle (fpl_rocket.registry).

Usage: uv run python scripts/registry_write.py --bundle data/bundles/<id> --out data/registry
                                              [--archive data/archive/all_gws.parquet]

Reads the bundle's files (sha256-checked against its manifest), computes the frozen baseline's xP for the next
GW (form per 90 x availability on the bundle's current season + the archived previous season), rebuilds the
public planning state of every roster entry from the FPL API (bootstrap and round prices from the bundle),
solves the shadow planner's objectives (a), (b) and (c) for each (fpl_rocket.registry.PLAN_KEYS) and writes
registry_xp.parquet (the xp table AS ISSUED; the simulations are not needed), registry_plans.json and
registry_meta.json (model_version, model_n_sims) to --out. The bundle workflow uploads
them into the deadline-gw<N> release. Needs the `model` dependency group (bundle_history_rows/bundle_targets).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket import registry, views  # noqa: E402
from fpl_rocket.api import FPLClient  # noqa: E402
from fpl_rocket.backtest import previous_season  # noqa: E402
from fpl_rocket.bundle import APP_TABLES, app_shared, read_manifest, verified_file  # noqa: E402
from fpl_rocket.rules import CURRENT_SEASON  # noqa: E402
from fpl_rocket.state import build_planning_state, fetch_inputs  # noqa: E402


class _BundleBootstrapClient:
    """FPLClient whose bootstrap is the bundle's (prices and events consistent with the issued xP)."""

    def __init__(self, client: FPLClient, bootstrap: dict[str, Any]) -> None:
        self._c, self._b = client, bootstrap

    def bootstrap(self) -> dict[str, Any]:
        return self._b

    def __getattr__(self, name: str) -> Any:
        return getattr(self._c, name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default="data/registry")
    ap.add_argument("--archive", default="data/archive/all_gws.parquet")
    ap.add_argument("--cache", default=None, help="optional raw JSON cache directory (tests)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="per-solve limit in seconds (default 60, the app uses 10)",
    )
    args = ap.parse_args(argv)

    bdir = Path(args.bundle)
    manifest = read_manifest(bdir)
    if "xp.parquet" not in {f["name"] for f in manifest.get("files", [])}:
        raise SystemExit(f"bundle {manifest['bundle_id']} has no xp table ({manifest.get('model_error')})")
    raw: dict[str, bytes] = {}
    for name in [f"{t}.parquet" for t in APP_TABLES] + ["element_history.parquet", "xp.parquet"]:
        raw[name] = (bdir / name).read_bytes()
        verified_file(manifest, name, raw[name])
    shared = app_shared(manifest, raw, views.LIVE_STATS)
    tables = {
        t: pd.read_parquet(bdir / f"{t}.parquet")
        for t in ("players", "teams", "events", "fixtures", "element_history")
    }
    xp = pd.read_parquet(bdir / "xp.parquet")

    from fpl_rocket.model.pipeline import bundle_history_rows, bundle_targets

    history = bundle_history_rows(tables, CURRENT_SEASON)
    if Path(args.archive).exists():
        arch = pd.read_parquet(args.archive)
        history = pd.concat(
            [arch[arch["season"] == previous_season(CURRENT_SEASON)], history], ignore_index=True
        )
    targets, _ = bundle_targets(tables, n_gws=1)
    gw = int(manifest.get("next_event") or targets["gw"].min())
    baseline = registry.baseline_xp(history, targets[targets["gw"] == gw])

    client = _BundleBootstrapClient(FPLClient(args.cache, offline=args.offline), shared.bootstrap)
    states, errors = {}, {}
    for entry_id in registry.REGISTRY_ENTRY_IDS:
        try:
            inputs = fetch_inputs(client, entry_id, round_prices=shared.round_prices)
            states[entry_id] = build_planning_state(**inputs, prices_as_of=f"bundle {manifest['bundle_id']}")
        except Exception as exc:  # noqa: BLE001 - recorded per entry
            errors[entry_id] = f"{type(exc).__name__}: {exc}"
    reg_xp, plans, meta = registry.build_registry(
        manifest=manifest,
        bootstrap=shared.bootstrap,
        xp=xp,
        baseline=baseline,
        states=states,
        errors=errors,
        time_limit=args.time_limit,
    )
    paths = registry.write_files(args.out, reg_xp, plans, meta)
    ok = sum(1 for e in plans["entries"] if e["status"] == "ok")
    print(
        f"registry GW{gw} bundle {manifest['bundle_id']} model {meta['model_version']} "
        f"(N={meta['model_n_sims']}): {ok}/{len(plans['entries'])} entries planned"
    )
    for e in plans["entries"]:
        st = {k: (e.get(k) or {}).get("status", "-") for k in registry.PLAN_KEYS}
        print(f"  entry {e['entry_id']}: {e['status']} {st if e['status'] == 'ok' else e.get('error')}")
    print(
        f"  time limit {meta['solver_time_limit_s']:g} s; hit it: {meta['plans_hit_time_limit'] or 'none'};"
        f" solver errors: {meta['plans_solver_error'] or 'none'}"
    )
    for p in paths:
        print(f"  {p} ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
