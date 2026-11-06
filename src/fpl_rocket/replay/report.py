"""REPORT.md of a replay run: season tables, chips, solver fallbacks, leakage test, KNOWN_UNCERTAIN rules,
paired block-bootstrap comparisons and the within-sample comparison with real managers.

Paired comparisons: per-GW net-point differences MINIMAL_MODEL minus each benchmark, moving-block bootstrap of
the mean (circular blocks of 4 GWs, 5000 resamples, blocks within seasons, seed = season index), season by
season and pooled (the GW-weighted mean of the per-season resamples), with an exact sign test.

Real managers (data/asknown/crowd after_the_fact_*: the sampled-manager archive, earliest committed version
of each GW file; NOT known at T - used only as an after-the-fact comparator): for the random "Overall" tier
and the top-10k tier ("10000"), each sampled manager's GW points are recomputed from the picks (positions
1-11 XI, 12-15 bench in order), captain / vice flags, the chip recorded for that GW (bench boost: all 15;
triple captain: x3; assistant manager: the manager pick, position 16, added) and actual results with the
same automatic-substitution rule as the replay; transfer costs (hits) are the recorded event_transfers_cost.
Agreement with the recorded gw_points is reported. A policy's weekly percentile within a tier = 100 x (share
of the tier's managers with lower net points + half the share with equal), void GWs excluded. Everything is
"within this sample"; no overall percentile is claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..asknown.reader import DEFAULT_ROOT
from ..matched import _paired_row, block_bootstrap_mean
from ..planner.lineup import realised_points
from ..reconcile import reconcile_archive
from ..rules import get_rules, season_rules
from .chips import CHIP_RULE
from .inputs import HISTORY_FILE, INPUTS_DIR, HistoryStore
from .policies import POLICY_NAMES

BLOCK = 4
RESAMPLES = 5000
TIERS = ("Overall", "10000")
TIER_LABEL = {"Overall": "random 'Overall' sample", "10000": "top-10k sample"}
REFERENCE = "minimal_model"
ENGINE_UNCERTAIN = (
    "Club limit after a player changes club: a squad that made transfers may not end with any club above "
    "max(3, its count before the transfers) (FPL's exact enforcement not verified).",
    "2021-22 and the 2020-21 training season have no FPL expected goals / assists / starts: the model uses "
    "goals, assists and minutes >= 60 as proxies (replay.inputs).",
    "2026-27 GW7-9 points rows come from the live API (scripts/load_live_gws.py): team = current club, "
    "value = current price, no ownership (none of these is used for scoring).",
    "Bench boost and triple captain are played on the plan made without the chip (transfers not re-planned "
    "for the chip).",
    "HOLD never plays a wildcard (it would be a permanent transfer); its wildcards are unused.",
)


# ======================================================================================= loading
def load_rows(out: Path, seasons: list[str]) -> pd.DataFrame:
    parts = [pd.read_csv(out / s / "rows.csv") for s in seasons if (out / s / "rows.csv").exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_leak(out: Path, seasons: list[str]) -> pd.DataFrame:
    rows = []
    for s in seasons:
        p = out / s / "leak_test.jsonl"
        if p.exists():
            rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)


def load_events(out: Path, seasons: list[str], policies: list[str]) -> pd.DataFrame:
    """Solver / chip events from the decision logs."""
    rows = []
    for s in seasons:
        for p in policies:
            f = out / s / f"{p}.jsonl"
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec["event"] != "decision":
                    continue
                pkg = rec["package"]
                for n in pkg.get("notes", []):
                    if isinstance(n, dict) and n.get("event"):
                        rows.append({"season": s, "policy": p, "gw": rec["gw"], **n})
                if pkg["solver"].get("time_limited"):
                    rows.append({"season": s, "policy": p, "gw": rec["gw"], "event": "time_limited_plan"})
    return pd.DataFrame(rows)


# ======================================================================================= tables
def _gain(x, reinstated) -> str:
    if reinstated:
        return "reinstated"
    return f"gain {int(x)}" if pd.notna(x) else "gain n/a"


def season_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (season, policy), g in rows.groupby(["season", "policy"], sort=True):
        chips = g[g["chip"].notna() & (g["chip"].astype(str) != "")]
        out.append(
            {
                "season": season,
                "policy": policy,
                "gws": len(g),
                "points": int(g["net"].sum()),
                "hits": int(g["hits"].sum()),
                "transfers": int(g["transfers"].sum()),
                "chips": "; ".join(
                    f"{c} GW{int(w)} ({_gain(x, r)})"
                    for c, w, x, r in zip(
                        chips["chip"], chips["gw"], chips["chip_gain"], chips["chip_reinstated"], strict=True
                    )
                )
                or "-",
                "solver_fallbacks": int(g["fallback"].sum()),
            }
        )
    return pd.DataFrame(out)


def paired(
    rows: pd.DataFrame, ref: str = REFERENCE, block: int = BLOCK, resamples: int = RESAMPLES
) -> pd.DataFrame:
    out = []
    others = [p for p in POLICY_NAMES if p != ref and p in set(rows["policy"])]
    for b in others:
        boots, diffs = [], []
        for i, (season, g) in enumerate(rows.groupby("season", sort=True)):
            w = g.pivot(index="gw", columns="policy", values="net").sort_index()
            if ref not in w or b not in w:
                continue
            d = (w[ref] - w[b]).dropna().to_numpy(float)
            bs = block_bootstrap_mean(d, block, resamples, seed=i)
            boots.append((len(d), bs))
            diffs.append(d)
            out.append(_paired_row(season, ref, b, d, bs))
        if len(boots) > 1:
            n = sum(k for k, _ in boots)
            pooled = sum(k * bs for k, bs in boots) / n
            out.append(_paired_row("pooled", ref, b, np.concatenate(diffs), pooled))
    return pd.DataFrame(out)


# ======================================================================================= real managers
def manager_points(season: str, store: HistoryStore, root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Per (tier, entry, gw): recomputed gross points, recorded points, hits, net (module docstring)."""
    crowd = root / "crowd"
    filt = [("season", "=", season), ("tier", "in", list(TIERS))]
    picks = pq.read_table(
        crowd / "after_the_fact_picks.parquet",
        columns=["season", "gw", "tier", "entry", "element", "position", "is_captain", "is_vice_captain"],
        filters=filt,
    ).to_pandas()
    mgr = pq.read_table(
        crowd / "after_the_fact_managers.parquet",
        columns=["season", "gw", "tier", "entry", "chip", "gw_points", "event_transfers_cost"],
        filters=filt,
    ).to_pandas()
    if picks.empty:
        return pd.DataFrame()
    rules = season_rules(season)
    d = store.df[store.df["season"] == season]
    et = dict(zip(d["element"].astype(int), d["element_type"].astype(int), strict=True))
    res = {int(g): store.results(season, int(g)) for g in sorted(picks["gw"].unique())}
    chip = {
        (t, int(e), int(g)): c
        for t, e, g, c in zip(mgr["tier"], mgr["entry"], mgr["gw"], mgr["chip"], strict=True)
    }
    rows = []
    picks = picks.sort_values(["tier", "entry", "gw", "position"], kind="stable")
    for (tier, entry, gw), p in picks.groupby(["tier", "entry", "gw"], sort=False):
        r = res[int(gw)]
        pts = {int(e): int(v) for e, v in r["points"].items()}
        mins = {int(e): int(v) for e, v in r["minutes"].items()}
        pos = p["position"].astype(int).to_numpy()
        el = p["element"].astype(int).to_numpy()
        xi = [int(e) for e, q in zip(el, pos, strict=True) if q <= 11]
        bench = [int(e) for e, q in zip(el, pos, strict=True) if 12 <= q <= 15]
        extra = [int(e) for e, q in zip(el, pos, strict=True) if q > 15]
        cap = next((int(e) for e, c in zip(el, p["is_captain"], strict=True) if c), xi[0])
        vice = next((int(e) for e, c in zip(el, p["is_vice_captain"], strict=True) if c), xi[1])
        c = chip.get((tier, int(entry), int(gw)))
        c = None if c is None or (isinstance(c, float) and np.isnan(c)) or c == "" else str(c)
        ets = {e: et.get(e, 3) for e in xi + bench}
        try:
            got = realised_points(xi, bench, cap, vice, pts, mins, ets, chip=c, limits=rules.formation)[
                "points"
            ]
        except (KeyError, IndexError):
            got = np.nan
        got = got + sum(pts.get(e, 0) for e in extra)
        rows.append(
            {"season": season, "tier": tier, "entry": int(entry), "gw": int(gw), "computed": got, "chip": c}
        )
    out = pd.DataFrame(rows).merge(
        mgr[["tier", "entry", "gw", "gw_points", "event_transfers_cost"]].astype(
            {"entry": "int64", "gw": "int64"}
        ),
        on=["tier", "entry", "gw"],
        how="left",
    )
    out["hits_cost"] = out["event_transfers_cost"].fillna(0).astype(int)
    out["net"] = out["computed"] - out["hits_cost"]
    return out


def official_average_check(mgr: pd.DataFrame, root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Sanity check of the reconstruction: the random 'Overall' tier's mean recomputed gross points per GW vs
    FPL's official average_entry_score (bootstrap events of the season's last as-known snapshot)."""
    out = []
    for season, g in mgr[mgr["tier"] == "Overall"].groupby("season"):
        d = root / "bootstrap" / "events" / season
        files = sorted(d.glob("gw*.parquet"), key=lambda q: int(q.stem[2:])) if d.exists() else []
        if not files:
            continue
        ev = pd.read_parquet(files[-1])
        ev = ev[ev["finished"].fillna(False).astype(bool)].set_index("event")["average_entry_score"]
        m = g.groupby("gw")["computed"].mean()
        both = pd.concat([m.rename("sample_mean"), ev.rename("official")], axis=1, join="inner").dropna()
        both = both[both["official"] > 0]
        if both.empty:
            continue
        diff = both["sample_mean"] - both["official"].astype(float)
        out.append(
            {
                "season": season,
                "gws": len(both),
                "mean_sample_minus_official": float(diff.mean()),
                "mean_abs_diff": float(diff.abs().mean()),
                "correlation": float(np.corrcoef(both["sample_mean"], both["official"].astype(float))[0, 1]),
            }
        )
    return pd.DataFrame(out)


def percentiles(rows: pd.DataFrame, mgr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(weekly percentiles, distribution per season x tier x policy, season sums vs tier mean)."""
    weekly = []
    for (season, tier, gw), g in mgr.groupby(["season", "tier", "gw"], sort=True):
        vals = g["net"].dropna().to_numpy(float)
        if not len(vals):
            continue
        pol = rows[(rows["season"] == season) & (rows["gw"] == gw)]
        for p, net, void in zip(pol["policy"], pol["net"], pol["void_gw"], strict=True):
            if bool(void):
                continue
            pct = 100.0 * (np.mean(vals < net) + 0.5 * np.mean(vals == net))
            weekly.append(
                {
                    "season": season,
                    "tier": tier,
                    "gw": int(gw),
                    "policy": p,
                    "net": int(net),
                    "pct": pct,
                    "n": len(vals),
                    "tier_mean": float(vals.mean()),
                }
            )
    w = pd.DataFrame(weekly)
    dist = (
        w.groupby(["season", "tier", "policy"])["pct"]
        .agg(gws="size", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    tier_mean = mgr.groupby(["season", "tier", "gw"])["net"].mean().reset_index()
    sums = (
        tier_mean.groupby(["season", "tier"])
        .agg(tier_mean_season=("net", "sum"), gws=("gw", "nunique"))
        .reset_index()
    )
    # policy totals over the SAME GWs as the tier (the rank tiers have no GW1 sample)
    covered = tier_mean.groupby(["season", "tier"])["gw"].apply(set).to_dict()
    for p in sorted(rows["policy"].unique()):
        r = rows[rows["policy"] == p]
        sums[p] = [
            int(r[(r["season"] == s) & r["gw"].isin(covered[(s, t)])]["net"].sum())
            for s, t in zip(sums["season"], sums["tier"], strict=True)
        ]
    return w, dist, sums


# ======================================================================================= markdown
def _md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    if df is None or df.empty:
        return "(none)\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def known_uncertain(seasons: list[str]) -> list[str]:
    out = []
    for s in seasons:
        out += list(get_rules(s).KNOWN_UNCERTAIN)
    return out + list(ENGINE_UNCERTAIN)


def write_report(out: str | Path, seasons: list[str], policies: list[str] | None = None) -> Path:
    out = Path(out)
    policies = list(policies or POLICY_NAMES)
    rows = load_rows(out, seasons)
    store = HistoryStore(INPUTS_DIR / HISTORY_FILE)
    parts: list[str] = [f"# Blind replay report ({out.name})\n"]
    parts.append(
        "Contract: issue time T = deadline - 2 h; every package decided from asknown.load(season, gw) + "
        "points rows of matches with kickoff + 3 h <= T only; validated against that season's rules; "
        "scored with actual "
        "results. Hits: never (every policy). Chip rule (shared, prespecified, not tuned): "
        f"`{json.dumps(CHIP_RULE.describe())}` + last eligible GW. All comparisons with managers are "
        "WITHIN THIS SAMPLE; no overall percentile is claimed.\n"
    )
    parts.append("## Season totals per policy (net points; hits = 0 by construction)\n")
    st = season_table(rows)
    parts.append(_md(st))
    wide = st.pivot(index="season", columns="policy", values="points")
    parts.append("\nPoints by season (wide):\n\n" + _md(wide.reset_index(), "{:.0f}"))
    parts.append("\n## Chips played (GW, realised chip gain)\n")
    parts.append(
        "Gain: bench boost = bench points; triple captain = the armband player's points once more; "
        "free hit / wildcard = GW points minus the points the squad held before the chip would have scored "
        "with the policy's own line-up rule (one GW only). Reinstated = played in a void GW "
        "(2022-23 GW7).\n\n"
    )
    ch = rows[rows["chip"].notna() & (rows["chip"].astype(str) != "")][
        ["season", "policy", "gw", "chip", "chip_gain", "chip_reinstated", "net"]
    ]
    parts.append(_md(ch.sort_values(["season", "policy", "gw"]), "{:.0f}"))
    parts.append("\n## Solver fallbacks and solver events\n")
    ev = load_events(out, seasons, policies)
    if len(ev):
        parts.append(
            _md(ev.groupby(["season", "policy", "event"]).size().rename("count").reset_index(), "{:.0f}")
        )
        parts.append("\n" + _md(ev.head(60)))
    else:
        parts.append("No solver fallback, time-limited plan or skipped chip.\n")
    parts.append("\n## Leakage test (K deadlines per season, everything after T corrupted)\n")
    lk = load_leak(out, seasons)
    if len(lk):
        agg = lk.groupby("season").agg(
            deadlines=("gw", lambda x: ",".join(str(int(v)) for v in sorted(set(x)))),
            packages=("passed", "size"),
            passed=("passed", "sum"),
            inputs_identical=("inputs_identical", "sum"),
            rows_corrupted=("rows_corrupted", "max"),
        )
        parts.append(_md(agg.reset_index(), "{:.0f}"))
        bad = lk[~lk["passed"]]
        if len(bad):
            parts.append("\nFAILURES:\n\n" + _md(bad))
    else:
        parts.append("(not run)\n")
    parts.append("\n## KNOWN_UNCERTAIN rules and limitations\n")
    parts += [f"- {x}\n" for x in known_uncertain(seasons)]
    parts.append("\n## Rules reconciliation (archive points recomputed with each season's scoring table)\n")
    rec = []
    for s in seasons:
        r = reconcile_archive(store.df, s)
        m = r[r["diff"] != 0]
        rec.append({"season": s, "rows": len(r), "mismatches": len(m), "gws": int(r["gw"].nunique())})
    parts.append(_md(pd.DataFrame(rec), "{:.0f}"))
    parts.append(
        f"\n## Paired per-GW differences: {REFERENCE} minus each benchmark (moving-block bootstrap, "
        f"block {BLOCK}, {RESAMPLES} resamples, blocks within seasons; 95% CI)\n"
    )
    pr = paired(rows)
    parts.append(_md(pr))
    parts.append("\n## Real managers (sampled-manager archive) - within this sample\n")
    mgr_parts = []
    for s in seasons:
        m = manager_points(s, store)
        if len(m):
            mgr_parts.append(m)
    if mgr_parts:
        mgr = pd.concat(mgr_parts, ignore_index=True)
        rec = mgr["gw_points"].fillna(0) > 0
        agree = (
            mgr.assign(
                recorded_zero=~rec,
                match=(mgr["computed"] == mgr["gw_points"]).where(rec),
            )
            .groupby(["season", "tier"])
            .agg(
                manager_gws=("entry", "size"),
                managers=("entry", "nunique"),
                recorded_points_zero=("recorded_zero", "mean"),
                checkable_gws=("match", "count"),
                recomputed_equals_recorded=("match", "mean"),
            )
            .reset_index()
        )
        parts.append(
            "Reconstruction: gross points recomputed from picks + actual results (auto-subs, captain / vice, "
            "chip; assistant-manager pick added). Hits = the recorded event_transfers_cost (transfers are "
            "made before the deadline, so this is known when the picks are fetched; it is not "
            "reconstructable from picks alone). The recorded entry_history points are 0 whenever the "
            "sample file was fetched before the GW's matches (recorded_points_zero = that share); "
            "agreement is measured on the other manager-GWs only. The recomputed points are used for "
            "every manager-GW:\n\n" + _md(agree, "{:.4f}")
        )
        parts.append(
            "\nSanity check: random-tier mean of the recomputed gross points vs the official GW average "
            "(average_entry_score):\n\n" + _md(official_average_check(mgr))
        )
        w, dist, sums = percentiles(rows, mgr)
        parts.append(
            "\nWeekly percentile of each policy within the tier (void GWs excluded): median and IQR over "
            "GWs\n\n"
        )
        parts.append(_md(dist, "{:.1f}"))
        parts.append(
            "\nSeason sum of the tier's mean weekly net points vs each policy's total over the same GWs "
            "(the rank tiers have no GW1 sample; the random tier is redrawn each GW)\n\n"
        )
        parts.append(_md(sums, "{:.0f}"))
        w.to_csv(out / "manager_percentiles_weekly.csv", index=False)
        mgr.to_csv(out / "manager_points.csv", index=False)
    else:
        parts.append("(no sampled managers for these seasons)\n")
    st.to_csv(out / "season_table.csv", index=False)
    pr.to_csv(out / "paired.csv", index=False)
    path = out / "REPORT.md"
    path.write_text("".join(p if p.endswith("\n") else p + "\n" for p in parts), encoding="utf-8")
    return path
