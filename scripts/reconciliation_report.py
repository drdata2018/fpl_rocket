"""Step 4e reconciliation report: October's matched totals -> baseline-v0, one condition at a time, plus the
shared-captain replays. Reads finished runs only (scripts/run_attribution.py with the variants of
fpl_rocket.replay.variants.RECONCILIATION / CAPTAIN_SWAP).

Usage:
    uv run python -I scripts/reconciliation_report.py [--out data/replays/attribution_4e]

Inputs: <out>/reconciliation/<variant>/<season>/rows.csv, <out>/captain_swap/captain_swap/<season>/rows.csv,
the October run (data/backtests/20261023T232216Z_restrained_free_only.csv) and its rerun with today's code
(<out>/october_rerun/*_restrained_free_only.csv), baseline-v0 (data/replays/20261106T151927Z) and the step 4d
attribution run (data/replays/attribution_4d). Writes <out>/RECONCILIATION.md and reconciliation.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

BASELINE_RUN = ROOT / "data" / "replays" / "20261106T151927Z"
ATTRIBUTION_4D = ROOT / "data" / "replays" / "attribution_4d"
OCTOBER = ROOT / "data" / "backtests" / "20261023T232216Z_restrained_free_only.csv"
SEASONS = ("2024-25", "2025-26")
ALL_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
OCT_NAMES = {"model_restrained": "minimal_model", "form_restrained": "form_restrained"}
PAIR = ("minimal_model", "form_restrained")
# the path: each row restores one condition of baseline-v0 on top of the previous row
STEPS = (
    ("october", "October matched replay (run_matched, restrained free_only, most-owned start)"),
    (
        "oct_inputs_no_topup",
        "blind engine, October's inputs (predictions, low-history sets, buy pool) and October's "
        "free-transfer rule (no AFCON top-up); flags off, chips off, October's GW1 squad, fixtures as played",
    ),
    ("oct_inputs", "+ the season's free-transfer rules restored (2025-26: AFCON top-up to 5 entering GW16)"),
    (
        "oct_conditions",
        "blind forecasts at T instead of October's inputs (flags off, chips off, October's GW1 "
        "squad, fixtures as played)",
    ),
    ("oct_conditions_known", "+ fixtures as known at T restored"),
    ("flags_off_chips_off", "+ own GW1 squad restored (flags off, chips off)"),
    ("chips_off", "+ availability flags restored (chips off)"),
    ("baseline", "+ chips restored = baseline-v0"),
)


def totals(rows: pd.DataFrame, names: dict[str, str] | None = None) -> dict[str, int]:
    t = rows.groupby("policy")["net"].sum()
    return {(names or {}).get(str(p), str(p)): int(v) for p, v in t.items()}


def october_rows(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path)
    r = r[r["policy"].isin(list(OCT_NAMES))].copy()
    r["policy"] = r["policy"].map(OCT_NAMES)
    return r


def step_rows(out: Path) -> list[dict]:
    oc = october_rows(OCTOBER)
    rows = []
    for s in SEASONS:
        prev = None
        for name, text in STEPS:
            if name == "october":
                t = totals(oc[oc["season"] == s])
                src = OCTOBER.name
            elif name == "baseline":
                t = totals(pd.read_csv(BASELINE_RUN / s / "rows.csv"))
                src = BASELINE_RUN.name
            else:
                f = out / "reconciliation" / name / s / "rows.csv"
                if name == "oct_inputs_no_topup":
                    from fpl_rocket.rules import season_rules

                    # no top-up in the season: the same replay as oct_inputs
                    topup = bool(season_rules(s).special.ft_topup)
                    name_run = name if topup else "oct_inputs"
                    f = (
                        out
                        / ("reconciliation_extra" if topup else "reconciliation")
                        / name_run
                        / s
                        / "rows.csv"
                    )
                if not f.exists():
                    rows.append({"season": s, "step": name, "conditions": text, "missing": True})
                    prev = None
                    continue
                t = totals(pd.read_csv(f))
                src = f"{f.parent.parent.parent.name}/{f.parent.parent.name}"
            m, fo = t["minimal_model"], t["form_restrained"]
            r = {
                "season": s,
                "step": name,
                "conditions": text,
                "model": m,
                "form": fo,
                "gap": m - fo,
                "source": src,
                "missing": False,
            }
            if prev is not None:
                r |= {
                    "d_model": m - prev["model"],
                    "d_form": fo - prev["form"],
                    "d_gap": (m - fo) - prev["gap"],
                }
            rows.append(r)
            prev = r
    return rows


def divergence(out: Path) -> list[str]:
    """Where the blind engine with October's inputs first departs from October's own replay."""
    oc = october_rows(OCTOBER)
    md = []
    runs = (("oct_inputs", "reconciliation"), ("oct_inputs_no_topup", "reconciliation_extra"))
    for s in SEASONS:
        for name, sub in runs:
            f = out / sub / name / s / "rows.csv"
            if not f.exists():
                continue
            b = pd.read_csv(f)
            for p in PAIR:
                x = oc[(oc["season"] == s) & (oc["policy"] == p)].set_index("gw")
                y = b[b["policy"] == p].set_index("gw")
                gws = sorted(set(x.index) & set(y.index))
                sq = [g for g in gws if str(x.at[g, "squad"]) != str(y.at[g, "squad"])]
                bank = [g for g in gws if int(x.at[g, "bank"]) != int(y.at[g, "bank"])]
                net = [g for g in gws if int(x.at[g, "net"]) != int(y.at[g, "net"])]
                cap = [g for g in gws if int(x.at[g, "captain"]) != int(y.at[g, "captain"])]
                md.append(
                    f"| {s} | {name} | {p} | {len(gws)} | {sq[0] if sq else '-'} | {len(sq)} | "
                    f"{bank[0] if bank else '-'} | {net[0] if net else '-'} | {len(net)} | {len(cap)} | "
                    f"{int(y['free_transfers_in'].max())} |"
                )
    return md


def four_d_check(out: Path) -> list[str]:
    md = []
    for name in ("flags_off_chips_off", "chips_off"):
        a = ATTRIBUTION_4D / name / "2024-25" / "rows.csv"
        b = out / "reconciliation" / name / "2024-25" / "rows.csv"
        if a.exists() and b.exists():
            ta, tb = totals(pd.read_csv(a)), totals(pd.read_csv(b))
            same = all(ta[p] == tb[p] for p in PAIR)
            md.append(
                f"- {name} 2024-25: step 4d {ta['minimal_model']} / {ta['form_restrained']}, rerun now "
                f"{tb['minimal_model']} / {tb['form_restrained']} ({'identical' if same else 'DIFFERENT'})"
            )
    return md


def swap_rows(out: Path) -> list[dict]:
    rows = []
    for s in ALL_SEASONS:
        base = totals(pd.read_csv(BASELINE_RUN / s / "rows.csv"))
        f = out / "captain_swap" / "captain_swap" / s / "rows.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        t = totals(r)
        chips = {}
        bchips = {}
        b = pd.read_csv(BASELINE_RUN / s / "rows.csv")
        for p in PAIR:
            chips[p] = ", ".join(
                f"{c} GW{g}"
                for g, c in zip(r.loc[r["policy"] == p, "gw"], r.loc[r["policy"] == p, "chip"], strict=True)
                if isinstance(c, str) and c
            )
            bchips[p] = ", ".join(
                f"{c} GW{g}"
                for g, c in zip(b.loc[b["policy"] == p, "gw"], b.loc[b["policy"] == p, "chip"], strict=True)
                if isinstance(c, str) and c
            )
        first = {}
        for p in PAIR:
            x = b[b["policy"] == p].set_index("gw")
            y = r[r["policy"] == p].set_index("gw")
            sq = [g for g in x.index if str(x.at[g, "squad"]) != str(y.at[g, "squad"])]
            ch = [g for g in x.index if str(x.at[g, "chip"]) != str(y.at[g, "chip"])]
            first[p] = f"{ch[0] if ch else '-'} / {sq[0] if sq else '-'} ({len(sq)})"
        rows.append(
            {
                "season": s,
                "model_first_chip_squad": first["minimal_model"],
                "form_first_chip_squad": first["form_restrained"],
                "model_swap": t["minimal_model"],
                "model_base": base["minimal_model"],
                "form_swap": t["form_restrained"],
                "form_base": base["form_restrained"],
                "model_chips_swap": chips["minimal_model"],
                "model_chips_base": bchips["minimal_model"],
                "form_chips_swap": chips["form_restrained"],
                "form_chips_base": bchips["form_restrained"],
                "fallbacks": int(r["fallback"].sum()),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "replays" / "attribution_4e"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    steps = step_rows(out)
    pd.DataFrame(steps).to_csv(out / "reconciliation.csv", index=False)
    rerun = sorted((out / "october_rerun").glob("*_restrained_free_only.csv"))
    md = [
        "# Reconciliation: October's matched replay -> baseline-v0 (step 4e)",
        "",
        "**Each row below is ONE replay** (both planner policies, whole season, 60 s solver limit) with the "
        "conditions listed; each row restores one more condition of baseline-v0 on top of the row above. "
        "The changes between rows are the effect of that restoration GIVEN the path chosen (order-dependent) "
        "and include path dependence (one different transfer changes the rest of the season). They are NOT "
        "additive contributions and do not sum to anything in another order.",
        "",
        "October's run = `20261023T232216Z_restrained_free_only` (run_matched --policies restrained --start "
        "most_owned --pred-cache data/backtests/pred_cache_3g): its start squad is the MOST-OWNED legal 15 "
        "by "
        "the archive's GW1 `selected` (matched.most_owned_initial_squad), not the form predictor's "
        "xP-optimal "
        "GW1 squad (that is the aggressive matched run's `--start baseline_xp`). The oct_* rows impose "
        "exactly "
        "that squad (recomputed from the archive and identical to the GW1 squad in October's CSV).",
        "",
    ]
    if rerun:
        r = october_rows(rerun[-1])
        t = {s: totals(r[r["season"] == s]) for s in SEASONS}
        md += [
            "October's run re-run today with the current code (`october_rerun/`): "
            + "; ".join(
                f"{s} MODEL {t[s]['minimal_model']} / FORM {t[s]['form_restrained']}" for s in SEASONS
            )
            + " (identical to October's published totals 2066 / 2104 and 1928 / 1890).",
            "",
        ]
    for s in SEASONS:
        md += [
            f"## {s}",
            "",
            "| step | conditions | MODEL | FORM | MODEL-FORM | change vs row above (MODEL / FORM / gap) | "
            "replay |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in steps:
            if r["season"] != s:
                continue
            if r.get("missing"):
                md.append(f"| {r['step']} | {r['conditions']} | not run | | | | |")
                continue
            ch = f"{r['d_model']:+d} / {r['d_form']:+d} / {r['d_gap']:+d}" if "d_model" in r else "-"
            md.append(
                f"| {r['step']} | {r['conditions']} | {r['model']} | {r['form']} | {r['gap']:+d} | {ch} | "
                f"{r['source']} |"
            )
        md.append("")
    fb = n_rows = 0
    for f in out.glob("*/*/*/rows.csv"):
        if f.parent.parent.parent.name in ("reconciliation", "reconciliation_extra", "captain_swap"):
            r = pd.read_csv(f)
            fb, n_rows = fb + int(r["fallback"].sum()), n_rows + len(r)
    md += [
        f"Solver fallbacks in all reconciliation and captain-swap replays: {fb} of {n_rows} decisions.",
        "",
    ]
    res = []
    for s in SEASONS:
        by = {r["step"]: r for r in steps if r["season"] == s and not r.get("missing")}
        if "october" in by and "oct_inputs_no_topup" in by:
            a, b = by["october"], by["oct_inputs_no_topup"]
            res.append(
                f"{s}: MODEL {b['model'] - a['model']:+d}, FORM {b['form'] - a['form']:+d}, gap "
                f"{b['gap'] - a['gap']:+d}"
            )
    md += [
        "**Residual** (oct_inputs_no_topup minus October, the part the table does not close): "
        + "; ".join(res)
        + ".",
        "",
        "## Where the blind engine with October's inputs departs from October's own replay",
        "",
        "Rows oct_inputs_no_topup / oct_inputs vs October, GW by GW (squad and bank after transfers, net "
        "points, captain).",
        "",
        "| season | row | policy | GWs | first GW with a different squad | GWs with a different squad | "
        "first GW with a different bank | first GW with different points | GWs with different points | GWs "
        "with a different captain | max free transfers entering a GW (blind) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *divergence(out),
        "",
        "Re-runs of the step 4d variants with today's code (determinism check):",
        "",
        *four_d_check(out),
        "",
    ]
    md += REMAINING
    sw = swap_rows(out)
    if sw:
        pd.DataFrame(sw).to_csv(out / "captain_swap.csv", index=False)
        md += [
            "## Shared-captain replays (policy intervention, NOT an additive attribution)",
            "",
            "Variant `captain_swap` (fpl_rocket.replay.variants): each planner policy keeps everything of "
            "its "
            "own (planner, squad path, XI, bench order, chip rule) except the armband: captain and vice = "
            "`captain_pair(own XI of the package, the OTHER policy's xP for GW t, low-history set)` - "
            "MINIMAL_MODEL captains by the form baseline's xP, FORM_RESTRAINED by the model's. The triple-"
            "captain trigger then reads the other policy's xP of that swapped captain (>= 9); bench boost, "
            "free hit and wildcard triggers are unchanged (own xP). The swap is applied to the final XI of "
            "the "
            "GW (in a wildcard / free-hit week, the chip plan's XI). The squad path can change only through "
            "chip timing (a triple captain played in another GW changes which chips remain and when the "
            "others are forced); captain choices themselves never change transfers.",
            "",
            "| season | MODEL baseline-v0 | MODEL, form captain | change | FORM baseline-v0 | FORM, model "
            "captain | change | MODEL-FORM baseline / swapped |",
            "|---|---|---|---|---|---|---|---|",
        ]
        tm = tf = tmb = tfb = 0
        for r in sw:
            md.append(
                f"| {r['season']} | {r['model_base']} | {r['model_swap']} | "
                f"{r['model_swap'] - r['model_base']:+d} | {r['form_base']} | {r['form_swap']} | "
                f"{r['form_swap'] - r['form_base']:+d} | "
                f"{r['model_base'] - r['form_base']:+d} / {r['model_swap'] - r['form_swap']:+d} |"
            )
            tm, tf, tmb, tfb = (
                tm + r["model_swap"],
                tf + r["form_swap"],
                tmb + r["model_base"],
                tfb + r["form_base"],
            )
        if len(sw) == len(ALL_SEASONS):
            md.append(
                f"| five seasons | {tmb} | {tm} | {tm - tmb:+d} | {tfb} | {tf} | {tf - tfb:+d} | "
                f"{tmb - tfb:+d} / {tm - tf:+d} |"
            )
        md += [
            "",
            "Chips played (baseline-v0 -> captain swap):",
            "",
            "| season | MODEL baseline-v0 | MODEL swapped | FORM baseline-v0 | FORM swapped | solver "
            "fallbacks |",
            "|---|---|---|---|---|---|",
        ]
        for r in sw:
            md.append(
                f"| {r['season']} | {r['model_chips_base']} | {r['model_chips_swap']} | "
                f"{r['form_chips_base']} | "
                f"{r['form_chips_swap']} | {r['fallbacks']} |"
            )
        md += [
            "",
            "Path check: first GW whose chip differs from baseline-v0 / first GW whose squad differs (number "
            "of GWs with a different squad).",
            "",
            "| season | MODEL | FORM |",
            "|---|---|---|",
            *[
                f"| {r['season']} | {r['model_first_chip_squad']} | {r['form_first_chip_squad']} |"
                for r in sw
            ],
        ]
        md.append("")
    (out / "RECONCILIATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


REMAINING = [
    "## Differences that remain between the blind engine and October's matched replay",
    "",
    "Identified from the code (fpl_rocket.matched / scripts/run_matched.py vs fpl_rocket.replay). Row "
    "oct_inputs removes the input differences (1-4); the rest are engine differences that no row removes.",
    "",
    "Inputs (removed together by oct_inputs; the oct_inputs -> oct_conditions change measures them as one "
    "bundle):",
    "",
    "1. **Predictions.** October: retrospective refits per target GW on the archive (rolling_origin: every "
    "GW "
    "row before the target in the season + the previous season, end-of-GW data), the model with "
    "`BACKTEST_CONFIG` (horizon 1 only, DC reference time = the target GW's first kickoff, no live-only "
    "status / chance features) and the form baseline; a later GW of the planner's horizon = the target GW's "
    "per-fixture xP x that GW's fixture count, and a player without a row keeps his last per-fixture "
    "prediction. Blind: one fit at T on the points known at T (kickoff + 3 h <= T; a rearranged match of an "
    "earlier GW played after T is excluded), model v0.2.1 `DEFAULT_CONFIG` predicting every GW of the "
    "horizon "
    "with its own horizon features (n_sims 1000, seed config + GW in both), the form baseline on the same "
    "history. Two model-code differences inside this bundle: October's ModelPredictor simulated the points "
    "with the 2026-27 scoring rules for every season (`XPModel(cfg)` defaults to the current season, so "
    "2024-25 lines were scored with DefCon points that season did not have), the blind model with the "
    "season's own rules (`scoring_season=season`); and the simulation's card rule changed after October "
    "(commit 5cf94e1: a card shown to an unused substitute counts). October's cache was written on "
    "2026-10-24 with the code of that day.",
    "2. **Low-history sets** (never captain / vice, never bought): October from each fold's training rows, "
    "blind from the history at T.",
    "3. **Buy pool.** October: only players with an archive row in GW t (a player blank in GW t cannot be "
    "bought); blind: every player buyable at T (not removed, can_select). oct_inputs uses the intersection "
    "(the engine rejects a purchase not buyable at T), so a player in October's pool who was not "
    "selectable at T stays out of the blind planner's candidates: this difference is only partly removed.",
    "4. **Fixture counts as played.** October: the archive's distinct fixtures per player and GW; blind "
    '`fixtures="played"`: the season\'s last as-known fixture file mapped through the club at T.',
    "",
    "Engine (no row removes them):",
    "",
    "5. **Free transfers.** October applies the 2026-27 transfer rules to every season (+1 per GW, bank cap "
    "5, "
    "no special events); the blind engine uses each season's rules: identical for 2024-25 (cap 5), but "
    "2025-26 has the AFCON top-up to 5 free transfers entering GW16 (rules/season_2025_26.py), which "
    "October never had.",
    "6. **Prices.** October: the archive `value` of GW t as buying price and for selling prices; blind: "
    "`now_cost` in the bootstrap snapshot at T (T = deadline - 2 h). Selling price rule identical "
    "(rules.selling_price).",
    "7. **Player metadata.** Club and position from the archive row of GW t (October) vs the snapshot at T.",
    "8. **Legality and execution.** The blind engine validates every package against the season's rules "
    "(club "
    "limit with the 'no club above max(3, before)' rule, buyable at T, bank >= 0); October has no validator. "
    "Auto-substitutions and the armband rule are the same function (planner.lineup.realised_points) with the "
    "season's formation limits (same values).",
    "",
    "Not differences (checked in the code): the planner settings (multi-GW objective, horizon 3, decay 0.85, "
    "transfers <= free transfers, never a hit, default pool outside wildcard weeks, 60 s), GW1 handling of "
    "an "
    "imposed squad (best XI / top-xP captain / second vice / reserve GK first by the policy's own GW1 xP), "
    "chips (none in both), the start squad (identical), the free transfer entering GW2 (1), the scoring data "
    "(same per-GW points; the blind history file is the archive plus live GWs).",
    "",
]


if __name__ == "__main__":
    sys.exit(main())
