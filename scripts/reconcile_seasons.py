"""Validate each season's scoring table against the archive: recompute every per-fixture row's points from its
stats with that season's rules module and compare with the recorded total_points.

Usage: uv run python -I scripts/reconcile_seasons.py [--seasons 2021-22 ... 2026-27] [--out FILE.csv]

Rows: data/archive/all_gws.parquet (+ live_gws.parquet for the current season). Prints per season the rows
checked, the mismatching rows and the GWs in which they occur; --out writes every mismatching row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket.reconcile import reconcile_archive  # noqa: E402
from fpl_rocket.rules import SEASONS  # noqa: E402


def load_rows(archive: Path) -> pd.DataFrame:
    df = pd.read_parquet(archive / "all_gws.parquet")
    live = archive / "live_gws.parquet"
    if live.exists():
        df = pd.concat([df, pd.read_parquet(live)], ignore_index=True)
    return df


def summary(df: pd.DataFrame, seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, bad = [], []
    for s in seasons:
        r = reconcile_archive(df, s)
        m = r[r["diff"] != 0]
        bad.append(m)
        rows.append(
            {
                "season": s,
                "gws": int(r["gw"].nunique()),
                "rows_checked": len(r),
                "mismatching_rows": len(m),
                "mismatch_rate": len(m) / max(len(r), 1),
                "gws_with_mismatch": ",".join(str(int(g)) for g in sorted(m["gw"].unique())) or "-",
                "net_point_diff": int(m["diff"].sum()),
            }
        )
    return pd.DataFrame(rows), pd.concat(bad, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=list(SEASONS))
    ap.add_argument("--archive", default="data/archive")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    table, bad = summary(load_rows(Path(args.archive)), args.seasons)
    print(table.to_string(index=False))
    if args.out:
        bad.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
