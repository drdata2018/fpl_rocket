"""Matched-policy backtest (fpl_rocket.matched): MODEL planner vs BASELINE planner vs HOLD, same start.

Usage: uv run python -I scripts/run_matched.py --seasons 2024-25 2025-26 [2026-27]

Needs the archive (scripts/load_archive.py) and the `model` dependency group. Writes
data/backtests/<timestamp>_matched.csv (one row per season x GW x policy), ..._matched_summary.csv,
..._matched_paired.csv and ..._matched_summary.md, and prints the tables. RETROSPECTIVE and MATCHED: see the
fpl_rocket.matched docstring for the rules and caveats.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket import matched  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _md(summary: pd.DataFrame, paired: pd.DataFrame, seasons: list[str], minutes: float) -> str:
    lines = [
        f"# Matched-policy backtest ({datetime.now(UTC):%Y-%m-%d %H:%M} UTC)",
        "",
        f"**{matched.LABEL}.** Start: the xP-optimal legal 15 under 100.0 by the baseline's GW1 "
        f"prediction; 1 FT entering GW2; no chips. MODEL / BASELINE = planner (horizon {matched.HORIZON}, "
        f"decay {matched.DECAY}, <= {matched.MAX_TRANSFERS} transfers per GW, hits allowed) on the model / "
        "baseline predictions (refit per GW); HOLD = never transfer, best XI by the baseline. Later GWs of "
        "the horizon use the target GW's per-fixture xP x fixture count. Prices = archive value per GW, "
        "selling price by the rules engine. Points = actual, with automatic substitutions, minus hits. GW "
        "average = proxy (fpl_rocket.backtest.gw_average_proxy).",
        "",
        f"Seasons: {', '.join(seasons)}. Runtime {minutes:.1f} min.",
        "",
        "| season | policy | net points | transfers | hits | net - GW avg | max drawdown vs avg "
        "| fallbacks |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {r.season} | {r.policy} | {r.net_points} | {r.transfers} | {r.hits} | "
            f"{r.net_minus_average:+.1f} | {r.max_drawdown_vs_average:.1f} | {r.fallback_plans} |"
        )
    lines += [
        "",
        f"Paired per-GW difference MODEL minus BASELINE (net points), moving-block bootstrap (block "
        f"{matched.BLOCK}, {matched.RESAMPLES} resamples), 95% CI:",
        "",
        "| season | GWs | mean per GW | 95% CI | season total | total 95% CI | GWs model ahead / behind |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in paired.itertuples(index=False):
        lines.append(
            f"| {r.season} | {r.gws} | {r.mean_diff_per_gw:+.2f} | [{r.ci_low:+.2f}, {r.ci_high:+.2f}] | "
            f"{r.total_diff:+.0f} | [{r.total_ci_low:+.0f}, {r.total_ci_high:+.0f}] | "
            f"{r.gws_model_ahead} / {r.gws_baseline_ahead} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    ap.add_argument("--archive", default=str(ROOT / "data" / "archive" / "all_gws.parquet"))
    ap.add_argument("--out", default=str(ROOT / "data" / "backtests"))
    ap.add_argument("--time-limit", type=float, default=10.0)
    args = ap.parse_args()
    df = pd.read_parquet(args.archive)
    t0 = time.perf_counter()
    rows, summary, paired = matched.run(
        df, args.seasons, log=lambda m: print(m, flush=True), time_limit=args.time_limit
    )
    minutes = (time.perf_counter() - t0) / 60
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_matched"
    rows.to_csv(f"{base}.csv", index=False)
    summary.to_csv(f"{base}_summary.csv", index=False)
    paired.to_csv(f"{base}_paired.csv", index=False)
    md = _md(summary, paired, args.seasons, minutes)
    Path(f"{base}_summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"written: {base}*.csv / _summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
