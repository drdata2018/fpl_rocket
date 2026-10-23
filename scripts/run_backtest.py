"""Retrospective backtest of a predictor on the normalised archive (scripts/load_archive.py first).

Usage: uv run python -I scripts/run_backtest.py --predictor baseline --seasons 2024-25 2025-26

Writes data/backtests/<timestamp>_<predictor>.csv (one row per season x GW), ..._calibration.csv and
..._summary.csv, and prints the per-season summary. Labelled RETROSPECTIVE: the archive is end-of-GW data.
The decision metric (best XI + captain by prediction vs the GW average) is an UNMATCHED comparison: no
squad, budget or transfers. GW averages are official (bootstrap events[].average_entry_score) for the
current season and a proxy (backtest.gw_average_proxy) for older seasons.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket.backtest import PREDICTORS, run_backtest, summarise  # noqa: E402
from fpl_rocket.rules import CURRENT_SEASON  # noqa: E402


def official_averages(seasons: list[str]) -> dict[tuple[str, int], float]:
    """Current-season GW averages from the live bootstrap (other seasons have none)."""
    if CURRENT_SEASON not in seasons:
        return {}
    try:
        from fpl_rocket.api import FPLClient

        events = FPLClient().bootstrap()["events"]
    except Exception as exc:  # noqa: BLE001 - fall back to the proxy, loudly
        print(f"warning: bootstrap unavailable ({exc!r}); {CURRENT_SEASON} uses the proxy average")
        return {}
    return {
        (CURRENT_SEASON, int(e["id"])): float(e["average_entry_score"])
        for e in events
        if e.get("finished") and e.get("average_entry_score") is not None
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictor", default="baseline", choices=sorted(PREDICTORS))
    ap.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    ap.add_argument("--archive", default="data/archive/all_gws.parquet")
    ap.add_argument("--out", default="data/backtests")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.archive)
    result = run_backtest(df, args.seasons, PREDICTORS[args.predictor](), official_averages(args.seasons))
    summary = summarise(result)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = out / f"{stamp}_{args.predictor}"
    result.folds.to_csv(f"{base}.csv", index=False)
    result.calibration.to_csv(f"{base}_calibration.csv", index=False)
    summary.to_csv(f"{base}_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"RETROSPECTIVE backtest, predictor={args.predictor}, seasons={args.seasons}")
    print(f"results: {base}.csv (+ _calibration.csv, _summary.csv)\n")
    print(summary.round(3).to_string(index=False))
    pooled = result.calibration[result.calibration["season"] == "all"]
    print("\ncalibration by predicted-points decile (all seasons):")
    print(pooled.drop(columns="season").round(2).to_string(index=False))
    print(
        "\nDecision metric = best legal XI + captain by prediction (no squad, budget or transfers) minus the"
        " GW average: an UNMATCHED comparison, not a policy result."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
