"""Matched-policy backtest (fpl_rocket.matched): planner policies vs HOLD from one start squad.

Usage: uv run python -I scripts/run_matched.py --seasons 2024-25 2025-26 [2026-27]
           [--policies aggressive|restrained] [--start baseline_xp|most_owned]
           [--fixtures-as-known data/archive/fixtures_as_known] [--pred-cache DIR --workers N]
           [--block 4 --resamples 2000] [--time-limit 60]

--policies aggressive (default): MODEL vs BASELINE planner (<= 2 transfers, hits allowed) vs HOLD.
--policies restrained: MODEL_RESTRAINED vs FORM_RESTRAINED (<= 1 transfer, never a hit) vs HOLD, the rule of
the prospective trajectories (fpl_rocket.trajectories); with --start most_owned they also start from the
trajectories' published start rule (the most-owned legal 15 at GW1).
--fixtures-as-known DIR: the planner sees the schedule as known at each deadline (scripts/
build_fixtures_as_known.py: <DIR>/<season>/gw<N>.parquet); scoring still uses the actual results.
--pred-cache DIR: per-season predictions are read from / written to DIR/<season>.pkl (replays that differ
only in policies or fixture knowledge reuse the same refits); --workers N refits the GWs in a process pool.

Needs the archive (scripts/load_archive.py) and the `model` dependency group. Writes
data/backtests/<timestamp>_<tag>.csv (one row per season x GW x policy), _summary.csv, _paired.csv and
_summary.md, and prints the tables. RETROSPECTIVE and MATCHED: see the fpl_rocket.matched docstring.
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
from fpl_rocket.planner import BATCH_TIME_LIMIT_S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_fixtures_as_known(root: str | Path, season: str) -> dict[int, pd.DataFrame]:
    """{gw: schedule as known at the GW deadline} from <root>/<season>/gw<N>.parquet."""
    folder = Path(root) / season
    out = {int(p.stem[2:]): pd.read_parquet(p) for p in sorted(folder.glob("gw*.parquet"))}
    if not out:
        raise SystemExit(f"no fixtures-as-known snapshots in {folder} (scripts/build_fixtures_as_known.py)")
    return out


def describe(policies: str, start: str, known: bool) -> str:
    start_txt = (
        "the xP-optimal legal 15 under 100.0 by the baseline's GW1 prediction"
        if start == "baseline_xp"
        else "the most-owned legal 15 under 100.0 at GW1 (the prospective trajectories' published rule, "
        "by the archive's GW1 ownership)"
    )
    pol = (
        f"MODEL / BASELINE = planner (horizon {matched.HORIZON}, decay {matched.DECAY}, <= "
        f"{matched.MAX_TRANSFERS} transfers per GW, hits allowed) on the model / baseline predictions"
        if policies == "aggressive"
        else f"MODEL_RESTRAINED / FORM_RESTRAINED = planner (horizon {matched.HORIZON}, decay "
        f"{matched.DECAY}, <= 1 transfer per GW, never a hit, free transfers banked under the rules) on "
        "the model / baseline predictions; hits are zero by construction, so gross = net"
    )
    fx = (
        "The planner sees the schedule AS KNOWN at each deadline (fixtures-as-known snapshots); points are "
        "actual."
        if known
        else "Fixture counts are AS PLAYED (the known fixture leak, see the audit)."
    )
    return (
        f"**{matched.LABEL}.** Start: {start_txt}; 1 FT entering GW2; no chips. {pol} (refit per GW); HOLD = "
        "never transfer, best XI by the baseline. Later GWs of the horizon use the target GW's per-fixture "
        f"xP x fixture count. {fx} Prices = archive value per GW, selling price by the rules engine. "
        "Points = actual, with automatic substitutions, minus hits. GW average = proxy "
        "(fpl_rocket.backtest.gw_average_proxy)."
    )


def md_tables(summary: pd.DataFrame, paired: pd.DataFrame, block: int, resamples: int) -> list[str]:
    lines = [
        "| season | policy | net points | gross | transfers | hits | net - GW avg | max drawdown vs avg "
        "| fallbacks | time-limited |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {r.season} | {r.policy} | {r.net_points} | {r.gross_points} | {r.transfers} | {r.hits} | "
            f"{r.net_minus_average:+.1f} | {r.max_drawdown_vs_average:.1f} | {r.fallback_plans} | "
            f"{r.time_limited_plans} |"
        )
    lines += [
        "",
        "Paired per-GW differences A minus B (net points), moving-block bootstrap (circular blocks of "
        f"{block} within seasons, {resamples} resamples; pooled = GW-weighted mean of the per-season "
        "resamples), 95% CI; sign test = two-sided exact binomial on the GW differences, ties dropped:",
        "",
        "| A - B | season | GWs | mean per GW | 95% CI | total | total 95% CI | GWs A ahead / B ahead "
        "| sign test p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in paired.itertuples(index=False):
        lines.append(
            f"| {r.a} - {r.b} | {r.season} | {r.gws} | {r.mean_diff_per_gw:+.2f} | "
            f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}] | {r.total_diff:+.0f} | "
            f"[{r.total_ci_low:+.0f}, {r.total_ci_high:+.0f}] | {r.gws_a_ahead} / {r.gws_b_ahead} | "
            f"{r.sign_test_p:.3f} |"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    ap.add_argument("--archive", default=str(ROOT / "data" / "archive" / "all_gws.parquet"))
    ap.add_argument("--out", default=str(ROOT / "data" / "backtests"))
    ap.add_argument("--policies", choices=sorted(matched.POLICY_SETS), default="aggressive")
    ap.add_argument("--start", choices=matched.STARTS, default=None, help="default: baseline_xp")
    ap.add_argument("--fixtures-as-known", default=None, help="folder of <season>/gw<N>.parquet snapshots")
    ap.add_argument("--pred-cache", default=None, help="per-season prediction cache folder")
    ap.add_argument("--workers", type=int, default=1, help="processes for the per-GW refits")
    ap.add_argument("--block", type=int, default=matched.BLOCK)
    ap.add_argument("--resamples", type=int, default=matched.RESAMPLES)
    ap.add_argument("--tag", default=None, help="file-name tag (default: matched / restrained [+ _known])")
    ap.add_argument(
        "--time-limit",
        type=float,
        default=BATCH_TIME_LIMIT_S,
        help="per-solve time limit in seconds (default 60; 10 s under load moved a season ~90 points)",
    )
    args = ap.parse_args()
    start = args.start or "baseline_xp"
    df = pd.read_parquet(args.archive)
    log = lambda m: print(m, flush=True)  # noqa: E731
    t0 = time.perf_counter()
    inputs = {
        s: matched.season_inputs(
            df, s, cache_dir=args.pred_cache, archive=args.archive, workers=args.workers, log=log
        )
        for s in args.seasons
    }
    known = (
        {s: load_fixtures_as_known(args.fixtures_as_known, s) for s in args.seasons}
        if args.fixtures_as_known
        else None
    )
    rows, summary, paired = matched.run(
        df,
        args.seasons,
        log=log,
        inputs=inputs,
        fixtures_as_known=known,
        block=args.block,
        resamples=args.resamples,
        policies=matched.POLICY_SETS[args.policies],
        start=start,
        time_limit=args.time_limit,
    )
    minutes = (time.perf_counter() - t0) / 60
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("matched" if args.policies == "aggressive" else args.policies) + (
        "_known" if known else ""
    )
    base = out / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{tag}"
    rows.to_csv(f"{base}.csv", index=False)
    summary.to_csv(f"{base}_summary.csv", index=False)
    paired.to_csv(f"{base}_paired.csv", index=False)
    md = "\n".join(
        [
            f"# Matched-policy backtest: {args.policies} ({datetime.now(UTC):%Y-%m-%d %H:%M} UTC)",
            "",
            describe(args.policies, start, known is not None),
            "",
            f"Seasons: {', '.join(args.seasons)}. Solver limit {args.time_limit:g} s. "
            f"Runtime {minutes:.1f} min.",
            "",
            *md_tables(summary, paired, args.block, args.resamples),
        ]
    )
    Path(f"{base}_summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"written: {base}*.csv / _summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
