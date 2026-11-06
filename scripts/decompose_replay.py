"""Where the points difference comes from (fpl_rocket.replay.decompose): MODEL minus FORM and MODEL minus
LAGGED_CROWD per GW, split into line-up / captain / chip components, from an existing replay run (no replays).

Usage:
    uv run python -I scripts/decompose_replay.py [--run data/replays/20261106T151927Z]

Writes <run>/DECOMPOSITION.md, decomposition.csv (per season, GW and policy: components, captain facts,
predicted captain / vice xP) and decomposition_paired.csv (per GW differences of each pair).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fpl_rocket.replay import decompose as D  # noqa: E402
from fpl_rocket.replay import inputs as I  # noqa: E402
from fpl_rocket.rules import season_rules  # noqa: E402

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
UNTOUCHED = ("2021-22", "2022-23", "2023-24")
EXPOSED = ("2024-25", "2025-26")
POLICIES = ("minimal_model", "form_restrained", "hold", "lagged_crowd")
SHORT = {"minimal_model": "MODEL", "form_restrained": "FORM", "hold": "HOLD", "lagged_crowd": "LAGGED"}
PAIRS = (("minimal_model", "form_restrained"), ("minimal_model", "lagged_crowd"))
GROUPS = {
    **{s: (s,) for s in SEASONS},
    "untouched 2021-22..2023-24": UNTOUCHED,
    "exposed 2024-25..2025-26": EXPOSED,
    "all five": SEASONS,
}
KS = (0, 1, 3, 5, 10)
CROWD = ROOT / "data" / "asknown" / "crowd"


def components(run_dir: Path, store: I.HistoryStore) -> pd.DataFrame:
    parts = []
    for s in SEASONS:
        rules = season_rules(s)
        et = D.element_types(store.df, s)
        for p in POLICIES:
            df = D.policy_components(
                run_dir / s / f"{p}.jsonl",
                s,
                lambda gw, s=s: store.results(s, gw),
                et,
                rules.formation,
                sorted(rules.special.void_gws),
                rules.transfers.hit_cost,
            )
            parts.append(df.assign(policy=p))
    return pd.concat(parts, ignore_index=True)


def _signs(x: pd.Series) -> str:
    return f"{int((x > 0).sum())} / {int((x < 0).sum())}"


def section_split(comp: pd.DataFrame, pairs_df: dict) -> list[str]:
    md = [
        "## 1. Total difference = line-up + captain + chip components",
        "",
        "Per pair, season and pooled group: the net difference and its components (hits are 0 for every "
        "policy: never a hit), with the number of GWs each component was positive / negative for MODEL. "
        "'squad-chip weeks' = GWs where either policy played a free hit or wildcard (their points stay in "
        "the line-up / captain rows; the GW count and their net difference are shown).",
        "",
    ]
    for (a, b), d in pairs_df.items():
        md += [
            f"### {SHORT[a]} minus {SHORT[b]}",
            "",
            "| group | GWs | net | line-up | captain | chip | net +/- GWs | line-up +/- | captain +/- | "
            "chip +/- | squad-chip weeks (net diff) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, seasons in GROUPS.items():
            g = d[d["season"].isin(seasons)]
            sq = g[g["squad_chip_week"]]
            assert int(g["hits"].sum()) == 0
            md.append(
                f"| {name} | {len(g)} | {int(g['net'].sum()):+d} | {int(g['lineup'].sum()):+d} | "
                f"{int(g['captain'].sum()):+d} | {int(g['chip'].sum()):+d} | {_signs(g['net'])} | "
                f"{_signs(g['lineup'])} | {_signs(g['captain'])} | {_signs(g['chip'])} | "
                f"{len(sq)} ({int(sq['net'].sum()):+d}) |"
            )
        md.append("")
    md += [
        "### Each policy's own components (season totals)",
        "",
        "| season | policy | net | line-up | captain | chip | bench boost | triple captain | FH / WC weeks |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in SEASONS:
        for p in POLICIES:
            g = comp[(comp["season"] == s) & (comp["policy"] == p)]
            bb = int(g.loc[g["chip_effective"] == "bboost", "chip"].sum())
            tc = int(g.loc[g["chip_effective"] == "3xc", "chip"].sum())
            sq = g[g["flag_squad_chip"]]
            fw = ", ".join(f"{c} GW{int(gw)}" for c, gw in zip(sq["chip_effective"], sq["gw"], strict=True))
            md.append(
                f"| {s} | {SHORT[p]} | {int(g['net'].sum())} | {int(g['lineup'].sum())} | "
                f"{int(g['captain'].sum())} | {int(g['chip'].sum())} | {bb} | {tc} | {fw or '-'} |"
            )
    md.append("")
    return md


def section_leave_out(pairs_df: dict) -> list[str]:
    md = [
        "## 2. Leave-largest-k-gameweeks-out sensitivity",
        "",
        "The k GWs with the LARGEST differences in MODEL's favour are dropped (within the group; pooled "
        "groups drop their k largest across the pooled GWs); remaining total and mean per remaining GW. "
        "k = 0 is the principal result, which keeps every week; this table only shows how much of it rests "
        "on a few "
        "weeks.",
        "",
    ]
    for (a, b), d in pairs_df.items():
        md += [
            f"### {SHORT[a]} minus {SHORT[b]}",
            "",
            "| group | " + " | ".join(f"k={k}: total (mean/GW)" for k in KS) + " |",
            "|---|" + "---|" * len(KS),
        ]
        for name, seasons in GROUPS.items():
            x = d.loc[d["season"].isin(seasons), "net"].to_numpy(float)
            cells = []
            for k in KS:
                r = D.leave_largest_out(x, k)
                cells.append(f"{r['total']:+.0f} ({r['mean']:+.2f})")
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")
    return md


def section_captains(comp: pd.DataFrame) -> list[str]:
    md = [
        "## 3. Captain decisions",
        "",
        "Per policy and season (void GW excluded): captain's actual points (single, whether or not he "
        "played), "
        "GWs the captain played 0 minutes, of those the vice took the armband / neither played, GWs the "
        "captain was the top scorer of the policy's own picked XI (ties count), total captain component (the "
        "armband's extra multiple once) and triple-captain extra (the third multiple).",
        "",
        "| season | policy | GWs | captain pts (single) | per GW | captain 0 min | vice took armband | "
        "neither played | captain top of own XI | captain component | TC extra |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in SEASONS:
        for p in POLICIES:
            g = comp[(comp["season"] == s) & (comp["policy"] == p) & ~comp["void"]]
            md.append(
                f"| {s} | {SHORT[p]} | {len(g)} | {int(g['captain_points'].sum())} | "
                f"{g['captain_points'].mean():.2f} | {int((~g['captain_played']).sum())} | "
                f"{int(g['vice_took_armband'].sum())} | {int((g['armband'] < 0).sum())} | "
                f"{int(g['captain_top_of_xi'].sum())} | {int(g['captain'].sum())} | "
                f"{int(g.loc[g['chip_effective'] == '3xc', 'chip'].sum())} |"
            )
    m = comp[(comp["policy"] == "minimal_model") & ~comp["void"]].copy()
    not_second = int((m["vice_el"] != m["pred_second_el"]).sum())
    bins = D.margin_bins(m, 3)
    m["pred_margin"] = m["pred_captain"] - m["pred_vice"]
    m["real_margin"] = m["captain_points"] - m["vice_points"]
    md += [
        "",
        "### MINIMAL_MODEL: predicted captain margin at T vs realised margin",
        "",
        "Predicted margin = the model's xP for GW t (package scores at T) of the captain minus that of the "
        "vice (the second-best captaincy candidate: captain_pair's second, outside the low-history set); "
        f"the vice was not the XI's second-highest xP in {not_second} of {len(m)} GWs (low-history "
        "exclusions or the planner's own captain choice). Realised margin = the captain's actual points "
        "minus "
        "the vice's (single points, both counted whether or not they played). Bins = equal-count thirds of "
        "the pooled five seasons' predicted margins.",
        "",
        "| bin | GWs | predicted margin range | mean predicted | mean realised | median realised | "
        "captain ahead / vice ahead / tied |",
        "|---|---|---|---|---|---|---|",
    ]
    names = ("small", "middle", "large")
    for r in bins.itertuples(index=False):
        md.append(
            f"| {names[int(r.bin)]} | {r.gws} | {r.pred_lo:.2f} to {r.pred_hi:.2f} | {r.pred_mean:.2f} | "
            f"{r.real_mean:+.2f} | {r.real_median:+.1f} | {r.captain_ahead} / {r.vice_ahead} / {r.tied} |"
        )
    md += [
        "",
        "| season | GWs | Spearman(predicted, realised) | mean predicted | mean realised |",
        "|---|---|---|---|---|",
    ]
    for s in (*SEASONS, "pooled"):
        g = m if s == "pooled" else m[m["season"] == s]
        md.append(
            f"| {s} | {len(g)} | {D.spearman(g['pred_margin'], g['real_margin']):+.3f} | "
            f"{g['pred_margin'].mean():.2f} | {g['real_margin'].mean():+.2f} |"
        )
    md.append("")
    return md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(ROOT / "data" / "replays" / "20261106T151927Z"))
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    store = I.HistoryStore(I.INPUTS_DIR / I.HISTORY_FILE)
    comp = components(run_dir, store)
    bad = comp[comp["check"] != 0]
    if len(bad):
        raise SystemExit(f"components do not add up to the engine's net in {len(bad)} rows:\n{bad.head()}")
    comp.to_csv(run_dir / "decomposition.csv", index=False)
    pairs_df = {}
    for a, b in PAIRS:
        d = D.paired(comp[comp["policy"] == a], comp[comp["policy"] == b])
        pairs_df[(a, b)] = d
    pd.concat([d.assign(pair=f"{SHORT[a]}-{SHORT[b]}") for (a, b), d in pairs_df.items()]).to_csv(
        run_dir / "decomposition_paired.csv", index=False
    )
    md = [
        f"# Where the points difference comes from ({run_dir.name}, baseline-v0)",
        "",
        "From the frozen decision logs and the actual results; no new replays. Seasons 2021-22..2025-26 "
        "(complete seasons only).",
        "",
        "**Convention** (one per GW and policy, every player counted once; fpl_rocket.replay.decompose):",
        "",
        "- **line-up** = points of the XI that scored: each policy's XI after FPL's automatic substitutions "
        "(in a bench-boost week the 11 picked starters; the bench belongs to the chip);",
        "- **captain** = the armband's additional multiple, ONCE: the armband holder's points (the captain, "
        "or the vice when the captain played 0 minutes; 0 when neither played);",
        "- **chip** = bench boost: the bench players' points; triple captain: the third multiple (the "
        "armband holder's points once more); free hit / wildcard: 0 here - their weeks are flagged and their "
        "points stay in line-up / captain, because a one-week gain understates a squad change;",
        "- hits = 0 for every policy (never a hit).",
        "",
        f"line-up + captain + chip = the engine's net points in every one of the {len(comp)} (season, GW, "
        "policy) rows (checked). A difference is taken component by component per GW and summed; negative "
        "contributions are kept. The void GW 2022-23 GW7 scores 0 for everyone.",
        "",
    ]
    md += section_split(comp, pairs_df)
    md += section_leave_out(pairs_df)
    md += section_captains(comp)
    md += section_crowd(comp, store)
    md += [
        "Files: `decomposition.csv` (per season, GW, policy: components, chip, captain / vice / armband, "
        "captain facts, predicted captain / vice / second xP), `decomposition_paired.csv` (per GW A - B of "
        "each component and of net, squad-chip-week flag).",
    ]
    (run_dir / "DECOMPOSITION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


def section_crowd(comp: pd.DataFrame, store: I.HistoryStore) -> list[str]:
    """The top-10k sample's most-captained player per GW (after the fact) with his actual points."""
    after = pd.read_parquet(CROWD / "after_the_fact_players.parquet")
    lagged = pd.read_parquet(CROWD / "lagged_players.parquet")
    top = D.crowd_captains(after, lagged)
    top = top[top["season"].isin(SEASONS)].copy()
    pts = []
    for s, gw, e in zip(top["season"], top["gw"], top["crowd_captain"], strict=True):
        r = store.results(s, int(gw))
        pts.append(int(r["points"].get(int(e), 0)))
    top["crowd_points"] = pts
    return _crowd_table(comp, top, after, lagged)


def _crowd_table(comp, top, after, lagged) -> list[str]:
    a = after[after["tier"].astype(str) == "10000"].set_index(["season", "gw", "element"])["captains"]
    lg = lagged[lagged["tier"].astype(str) == "10000"]
    lg = lg.set_index(["season", "source_gw", "element"])["captains"]
    lg.index = lg.index.set_names(["season", "gw", "element"])
    common = a.index.intersection(lg.index)
    agree = int((a.loc[common] == lg.loc[common]).sum())
    md = [
        "## 4. After-the-fact comparator: the top-10k sample's most-captained player",
        "",
        "**NOT KNOWABLE BEFORE THE DEADLINE**: the sample's captaincy of GW t is published after the GW t "
        "deadline. Source: the archived tier-10000 manager sample used by LAGGED_CROWD (about 370 managers "
        "per GW), picks of GW t (after-the-fact table, which has no GW1; GW1 from the lagged table's "
        "source GW 1 where that tier exists - not in 2021-22 and 2022-23 - and the void GW 2022-23 GW7 "
        "excluded). The two "
        f"tables agree on {agree} of {len(common)} common (season, GW, player) captain counts. "
        "Most-captained = most captain picks (ties: the lower element id). Points are single (no "
        "multiplier).",
        "",
        "| season | GWs with a sample | crowd captain pts | MODEL captain pts | MODEL - crowd | same player "
        "| "
        "MODEL ahead / crowd ahead | GWs with a tied crowd top | FORM captain pts | LAGGED captain pts |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    tot = np.zeros(4, int)
    for s in SEASONS:
        m = comp[(comp["season"] == s) & ~comp["void"]]
        mm = m[m["policy"] == "minimal_model"].set_index("gw")
        ff = m[m["policy"] == "form_restrained"].set_index("gw")
        ll = m[m["policy"] == "lagged_crowd"].set_index("gw")
        t = top[(top["season"] == s) & top["gw"].isin(mm.index)]
        gws = t["gw"].to_numpy(int)
        crowd = t["crowd_points"].to_numpy(int)
        model = mm.loc[gws, "captain_points"].to_numpy(int)
        same = int((mm.loc[gws, "captain_el"].to_numpy(int) == t["crowd_captain"].to_numpy(int)).sum())
        d = model - crowd
        tot += [crowd.sum(), model.sum(), (d > 0).sum(), (d < 0).sum()]
        md.append(
            f"| {s} | {len(t)} | {int(crowd.sum())} | {int(model.sum())} | {int(d.sum()):+d} | {same} | "
            f"{int((d > 0).sum())} / {int((d < 0).sum())} | {int((t['tied'] > 0).sum())} | "
            f"{int(ff.loc[gws, 'captain_points'].sum())} | {int(ll.loc[gws, 'captain_points'].sum())} |"
        )
    md.append(
        f"| five seasons | - | {int(tot[0])} | {int(tot[1])} | {int(tot[1] - tot[0]):+d} | - | "
        f"{int(tot[2])} / {int(tot[3])} | - | - | - |"
    )
    md.append("")
    return md


if __name__ == "__main__":
    sys.exit(main())
