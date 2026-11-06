"""Robustness statistics of a finished replay run (fpl_rocket.replay.robustness): no new replays.

Usage:
    uv run python -I scripts/replay_robustness.py --run data/replays/20261106T151927Z \
        [--out <run>/ROBUSTNESS.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fpl_rocket.replay import robustness as R  # noqa: E402

PAIRS = (("minimal_model", "form_restrained"), ("minimal_model", "lagged_crowd"))


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rows = R.load_rows(args.run)
    complete = R.UNTOUCHED + R.EXPOSED
    md = [f"# Robustness ({Path(args.run).name}); {R.RESAMPLES} resamples, blocks within seasons", ""]
    for a, b in PAIRS:
        d = R.diffs(rows, a, b)
        boot = []
        for group in (R.UNTOUCHED, R.EXPOSED, ("2024-25",), ("2025-26",), complete):
            for blk in R.BLOCKS:
                boot.append(R.pooled_bootstrap(d, group, blk))
        md += [f"## {a} minus {b}", "", "Pooled moving-block bootstrap (95% CI of the mean per GW)", ""]
        md += [_md(pd.DataFrame(boot)), ""]
        md += ["Leave-one-season-out (five complete seasons)", "", _md(pd.DataFrame(R.loso(d, complete))), ""]
        md += ["Sign test per season", "", _md(pd.DataFrame(R.sign_rows(d, complete))), ""]
        shares = [R.top_share(d, g) for g in (R.UNTOUCHED, R.EXPOSED, complete)]
        md += [
            "Share of the pooled total from the 10 largest GW differences",
            "",
            _md(pd.DataFrame(shares)),
            "",
        ]
    text = "\n".join(md) + "\n"
    out = Path(args.out) if args.out else Path(args.run) / "ROBUSTNESS.md"
    out.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
