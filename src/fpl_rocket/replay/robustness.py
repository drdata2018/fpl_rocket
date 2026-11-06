"""Robustness statistics of paired per-GW differences from a finished replay run (no new replays).

For a pair (a, b) and a group of seasons: per-GW differences d = net_a - net_b; moving-block bootstrap of the
mean (circular blocks of `block` GWs WITHIN each season, `resamples` resamples, seed = the season's index in
SEASON_ORDER, the same generator as the replay report); the pooled mean = the GW-weighted mean of the
per-season resample means. Also: leave-one-season-out pooled means, the exact sign test per season, and
the share of the pooled total that comes from the group's `top` largest single-GW differences.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..matched import block_bootstrap_mean, sign_test

SEASON_ORDER = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
UNTOUCHED = ("2021-22", "2022-23", "2023-24")
EXPOSED = ("2024-25", "2025-26")
BLOCKS = (4, 8, 12)
RESAMPLES = 5000


def load_rows(run_dir: str | Path, seasons=SEASON_ORDER) -> pd.DataFrame:
    parts = [
        pd.read_csv(Path(run_dir) / s / "rows.csv")
        for s in seasons
        if (Path(run_dir) / s / "rows.csv").exists()
    ]
    return pd.concat(parts, ignore_index=True)


def diffs(rows: pd.DataFrame, a: str, b: str) -> dict[str, np.ndarray]:
    out = {}
    for s, g in rows.groupby("season", sort=True):
        w = g.pivot(index="gw", columns="policy", values="net").sort_index()
        if a in w and b in w:
            out[str(s)] = (w[a] - w[b]).dropna().to_numpy(float)
    return out


def pooled_bootstrap(d: dict[str, np.ndarray], seasons, block: int, resamples: int = RESAMPLES) -> dict:
    boots, xs = [], []
    for s in seasons:
        x = d[s]
        boots.append((len(x), block_bootstrap_mean(x, block, resamples, seed=SEASON_ORDER.index(s))))
        xs.append(x)
    n = sum(k for k, _ in boots)
    pooled = sum(k * bs for k, bs in boots) / n
    allx = np.concatenate(xs)
    lo, hi = np.nanpercentile(pooled, [2.5, 97.5])
    return {
        "seasons": "+".join(seasons),
        "block": block,
        "gws": int(n),
        "mean_per_gw": float(allx.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "total": float(allx.sum()),
        "total_ci_low": float(lo * n),
        "total_ci_high": float(hi * n),
        "p_boot_le_0": float((pooled <= 0).mean()),
    }


def loso(d: dict[str, np.ndarray], seasons) -> list[dict]:
    out = []
    for s in seasons:
        rest = [x for t, x in d.items() if t in seasons and t != s]
        allx = np.concatenate(rest)
        out.append(
            {"left_out": s, "gws": len(allx), "mean_per_gw": float(allx.mean()), "total": float(allx.sum())}
        )
    return out


def sign_rows(d: dict[str, np.ndarray], seasons) -> list[dict]:
    return [
        {
            "season": s,
            "gws": len(d[s]),
            "a_ahead": int((d[s] > 0).sum()),
            "b_ahead": int((d[s] < 0).sum()),
            "ties": int((d[s] == 0).sum()),
            "total": float(d[s].sum()),
            "sign_test_p": sign_test(d[s]),
        }
        for s in seasons
    ]


def top_share(d: dict[str, np.ndarray], seasons, top: int = 10) -> dict:
    allx = np.concatenate([d[s] for s in seasons])
    best = np.sort(allx)[::-1][:top]
    total = float(allx.sum())
    return {
        "seasons": "+".join(seasons),
        "total": total,
        "top_sum": float(best.sum()),
        "share": float(best.sum() / total) if total else float("nan"),
        "rest": total - float(best.sum()),
    }
