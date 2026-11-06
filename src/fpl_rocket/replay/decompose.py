"""Where the points difference between two replay policies comes from (step 4e), from a replay's decision logs
and the actual results; no new replays. DIAGNOSTICS ONLY.

Convention (one per GW and policy; every player counted once):
    lineup    the points of the XI that scored: the XI after FPL's automatic substitutions (bench boost: the
              11 picked starters; the bench is the chip's)
    captain   the armband's additional multiple, ONCE: the points of the armband holder (the captain, or the
              vice when the captain did not play; 0 when neither played)
    chip      bench boost: the 4 bench players' points; triple captain: the third multiple (the armband
              holder's points once more); free hit / wildcard: 0 (their points stay in lineup / captain and
              the week is flagged, since a one-week number understates a squad change)
    hits      - hit cost x hits (0 for every frozen policy: they never take a hit)
    lineup + captain + chip + hits = the engine's net points (checked per GW: `check` column).
A void GW (2022-23 GW7) scores nothing; a chip played there is reinstated and counts as no chip.
The difference of two policies is taken component by component per GW.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..planner.lineup import realised_points
from .baseline import read_log

COMPONENTS = ("lineup", "captain", "chip", "hits")


def gw_components(
    pkg: Mapping[str, Any],
    points: Mapping[int, int],
    minutes: Mapping[int, int],
    element_type: Mapping[int, int],
    limits: Mapping[int, tuple[int, int]],
    hits: int = 0,
    hit_cost: int = 4,
    void: bool = False,
) -> dict[str, Any]:
    """Components of one package's GW score (module docstring). `pkg` = a decision-log package."""
    xi = [int(e) for e in pkg["xi"]]
    bench = [int(e) for e in pkg["bench"]]
    cap, vice = int(pkg["captain"]), int(pkg["vice"])
    chip = pkg.get("chip") or None
    if void:
        points, minutes = {}, {}
        eff = None
    else:
        eff = chip
    res = realised_points(xi, bench, cap, vice, points, minutes, element_type, chip=eff, limits=limits)
    armband = res["armband"]
    arm_pts = int(points.get(armband, 0)) if armband is not None else 0
    if eff == "bboost":
        lineup = sum(int(points.get(e, 0)) for e in xi)
        chip_pts = sum(int(points.get(e, 0)) for e in bench)
    else:
        lineup = sum(int(points.get(e, 0)) for e in res["scorers"])
        chip_pts = arm_pts if eff == "3xc" else 0
    xi_pts = [int(points.get(e, 0)) for e in xi]
    out = {
        "lineup": lineup,
        "captain": arm_pts,
        "chip": chip_pts,
        "hits": -hit_cost * int(hits),
        "chip_played": chip or "",
        "chip_effective": eff or "",
        "flag_squad_chip": eff in ("freehit", "wildcard"),
        "captain_el": cap,
        "vice_el": vice,
        "armband": armband if armband is not None else -1,
        "captain_points": int(points.get(cap, 0)),
        "vice_points": int(points.get(vice, 0)),
        "captain_played": int(minutes.get(cap, 0)) > 0,
        "vice_took_armband": armband is not None and armband == vice,
        "captain_top_of_xi": bool(xi_pts) and int(points.get(cap, 0)) >= max(xi_pts),
        "auto_subs": len(res["subs"]),
        "void": void,
    }
    out["total"] = out["lineup"] + out["captain"] + out["chip"] + out["hits"]
    return out


def element_types(history: pd.DataFrame, season: str) -> dict[int, int]:
    """{element: element_type} of a season from the points history (one type per player and season)."""
    s = history[history["season"] == season]
    return {int(e): int(t) for e, t in s.groupby("element")["element_type"].last().items()}


def policy_components(
    log_path: str | Path,
    season: str,
    results: Callable[[int], pd.DataFrame],
    element_type: Mapping[int, int],
    limits: Mapping[int, tuple[int, int]],
    void_gws: Sequence[int] = (),
    hit_cost: int = 4,
) -> pd.DataFrame:
    """One row per GW of a decision log: components, the package's predicted captain / vice scores and the
    engine's net (`check` = total - net, must be 0). results(gw) = per element points / minutes."""
    decisions, scores = read_log(log_path)
    rows = []
    for gw in sorted(scores):
        pkg = decisions[gw]["package"]
        row = scores[gw]
        r = results(gw)
        pts = {int(e): int(v) for e, v in r["points"].items()}
        mins = {int(e): int(v) for e, v in r["minutes"].items()}
        c = gw_components(
            pkg, pts, mins, element_type, limits, int(row["hits"]), hit_cost, void=gw in set(void_gws)
        )
        sc = {int(k): float(v) for k, v in (pkg.get("scores") or {}).items()}
        xi = [int(e) for e in pkg["xi"]]
        ranked = sorted(xi, key=lambda e: (-sc.get(e, 0.0), e))
        second = next((e for e in ranked if e != c["captain_el"]), -1)
        rows.append(
            {
                "season": season,
                "gw": gw,
                **c,
                "net": int(row["net"]),
                "check": c["total"] - int(row["net"]),
                "pred_captain": sc.get(c["captain_el"], np.nan),
                "pred_vice": sc.get(c["vice_el"], np.nan),
                "pred_second_el": second,
                "pred_second": sc.get(second, np.nan),
            }
        )
    return pd.DataFrame(rows)


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Per GW A minus B of every component and of the net (same season, GW keys)."""
    cols = [*COMPONENTS, "net"]
    m = a[["season", "gw", *cols, "flag_squad_chip"]].merge(
        b[["season", "gw", *cols, "flag_squad_chip"]], on=["season", "gw"], suffixes=("_a", "_b")
    )
    out = m[["season", "gw"]].copy()
    for c in cols:
        out[c] = m[f"{c}_a"] - m[f"{c}_b"]
    out["squad_chip_week"] = m["flag_squad_chip_a"] | m["flag_squad_chip_b"]
    return out


def leave_largest_out(d: Sequence[float], k: int) -> dict[str, float]:
    """Drop the k GWs with the largest differences (most favourable to A); remaining total and mean per GW."""
    x = np.sort(np.asarray(d, float))
    kept = x[: max(0, len(x) - int(k))]
    return {
        "k": int(k),
        "gws": len(kept),
        "total": float(kept.sum()),
        "mean": float(kept.mean()) if len(kept) else float("nan"),
    }


def margin_bins(df: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """Predicted captain margin (captain xP - vice xP) in equal-count bins against the realised points
    difference of the same two players. df: policy_components rows (non-void)."""
    d = df[~df["void"]].copy()
    d["pred_margin"] = d["pred_captain"] - d["pred_vice"]
    d["real_margin"] = d["captain_points"] - d["vice_points"]
    d = d.dropna(subset=["pred_margin"])
    edges = np.quantile(d["pred_margin"], np.linspace(0, 1, n_bins + 1))
    d["bin"] = np.clip(np.searchsorted(edges[1:-1], d["pred_margin"], side="right"), 0, n_bins - 1)
    out = d.groupby("bin").agg(
        gws=("gw", "size"),
        pred_lo=("pred_margin", "min"),
        pred_hi=("pred_margin", "max"),
        pred_mean=("pred_margin", "mean"),
        real_mean=("real_margin", "mean"),
        real_median=("real_margin", "median"),
        captain_ahead=("real_margin", lambda x: int((x > 0).sum())),
        vice_ahead=("real_margin", lambda x: int((x < 0).sum())),
        tied=("real_margin", lambda x: int((x == 0).sum())),
    )
    return out.reset_index()


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    a = pd.Series(np.asarray(x, float)).rank().to_numpy()
    b = pd.Series(np.asarray(y, float)).rank().to_numpy()
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def crowd_captains(after: pd.DataFrame, lagged: pd.DataFrame, tier: str = "10000") -> pd.DataFrame:
    """The sample's most-captained player per (season, GW of the picks), AFTER THE FACT: from the after-the-
    fact table (gw = the picks' GW) and, for GWs it lacks (GW1), the lagged table (source_gw = the picks'
    GW). Ties: the lower element id; `tied` counts them."""
    a = after[after["tier"].astype(str) == tier][["season", "gw", "element", "captains", "managers"]]
    a = a.assign(source="after_the_fact")
    lg = lagged[lagged["tier"].astype(str) == tier]
    lg = lg.assign(gw=lg["source_gw"])[["season", "gw", "element", "captains", "managers"]]
    have = set(zip(a["season"], a["gw"], strict=True))
    lg = lg[[(s, g) not in have for s, g in zip(lg["season"], lg["gw"], strict=True)]].assign(source="lagged")
    d = pd.concat([a, lg], ignore_index=True)
    d = d.sort_values(["season", "gw", "captains", "element"], ascending=[True, True, False, True])
    top = d.groupby(["season", "gw"], as_index=False).first()
    mx = d.groupby(["season", "gw"])["captains"].transform("max")
    tied = d[d["captains"] == mx].groupby(["season", "gw"]).size().rename("tied").reset_index()
    top = top.merge(tied, on=["season", "gw"])
    top["tied"] = top["tied"] - 1
    return top.rename(columns={"element": "crowd_captain", "captains": "crowd_captains"})
