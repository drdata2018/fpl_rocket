"""Write a frozen baseline record (fpl_rocket.replay.baseline) for a finished replay run.

Usage:
    uv run python -I scripts/freeze_baseline.py --run data/replays/20261106T151927Z --name baseline_v0 \
        --commit 7a14ba1999a2b578199f00dd1587cd3f7949edf6 [--note "..."]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_rocket.model.config import DEFAULT_CONFIG, MODEL_BASE_VERSION  # noqa: E402
from fpl_rocket.replay import baseline, chips, forecast, policies, run  # noqa: E402
from fpl_rocket.replay import inputs as I  # noqa: E402
from fpl_rocket.rules import season_rules  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--note", action="append", default=[])
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    seasons = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    logs: dict = {}
    settings: dict = {}
    for s in seasons:
        logs[s] = {"rows_csv_sha256": baseline.sha256_file(run_dir / s / "rows.csv"), "policies": {}}
        for p in policies.POLICY_NAMES:
            path = run_dir / s / f"{p}.jsonl"
            logs[s]["policies"][p] = baseline.log_record(path)
            dec, _ = baseline.read_log(path)
            first = dec[min(dec)]
            settings.setdefault(p, first["policy"])
    ak = ROOT / "data" / "asknown"
    rec = {
        "name": args.name,
        "tag": args.tag or args.name.replace("_", "-"),
        "commit": args.commit,
        "run": run_dir.name,
        "notes": args.note,
        "policies": list(policies.POLICY_NAMES),
        "policy_settings": settings,
        "planner": asdict(policies.PLANNER),
        "chip_rule": chips.CHIP_RULE.describe(),
        "chip_priority": list(chips.PRIORITY),
        "wide_pool": [list(x) for x in policies.WIDE_POOL],
        "wide_cheapest": policies.WIDE_CHEAPEST,
        "lagged_tier": policies.LAGGED_TIER,
        "seeds": {
            "model_seed": DEFAULT_CONFIG.seed,
            "model_seed_per_gw": "model_seed + gw",
            "leak_seed": run.LEAK_SEED,
            "leak_k": run.LEAK_K,
        },
        "model": {
            "version": MODEL_BASE_VERSION,
            "n_sims": DEFAULT_CONFIG.n_sims,
            "forecast_version": forecast.FORECAST_VERSION,
            "forecast_horizon": forecast.FORECAST_HORIZON,
            "low_history_gws": forecast.LOW_HISTORY_GWS,
        },
        "rules_versions": {s: season_rules(s).version for s in seasons},
        "asknown": {
            "manifest_sha256": {
                p.name: baseline.sha256_file(p) for p in sorted((ak / "manifest").glob("*.json"))
            },
            "deadlines_sha256": baseline.sha256_file(ak / "deadlines.parquet"),
            "report_sha256": baseline.sha256_file(ak / "REPORT.txt"),
        },
        "points_history_sha256": baseline.sha256_file(I.INPUTS_DIR / I.HISTORY_FILE),
        "forecast_cache": baseline.forecast_cache_record(I.INPUTS_DIR / "forecast_cache", seasons),
        "logs": logs,
        "environment": baseline.environment(),
        "canonical_form": (
            "per GW json.dumps(sort_keys=True) of the decision record without decide_seconds and "
            "package.solver.seconds (packages / decisions) and of the score row (scores); sha256 over the "
            "lines in GW order (fpl_rocket.replay.baseline)"
        ),
    }
    out = baseline.BASELINES_DIR / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
