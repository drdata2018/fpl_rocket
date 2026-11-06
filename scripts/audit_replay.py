"""Independent ledger audit of a replay run (fpl_rocket.ledger_audit) + ledger excerpts.

Usage:
    uv run python -I scripts/audit_replay.py --run data/replays/20261106T151927Z \
        [--focus 2024-25 minimal_model form_restrained] [--top 8] [--out <run>/AUDIT.md]

Audits every season x policy of the run (disagreements with the engine and legality issues, row by row), then
prints the ledger of the focus policy for its chip weeks and the --top GWs with the largest focus minus
comparison difference, with the comparison policy's ledger and the players that produced the gap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_rocket import ledger_audit as LA  # noqa: E402

POLICIES = ("minimal_model", "form_restrained", "hold", "lagged_crowd")


def _packages(path: Path) -> dict[int, dict]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["event"] == "decision":
                out[int(rec["gw"])] = rec["package"]
    return out


def _p(data: LA.SeasonData, e: int, gw: int) -> str:
    return f"{data.name(e)} {data.points.get(gw, {}).get(e, 0)}"


def ledger_text(row: dict, pkg: dict, data: LA.SeasonData, label: str) -> list[str]:
    gw = row["gw"]
    pts = data.points.get(gw, {})
    mins = data.minutes.get(gw, {})
    cost = {}
    pl = data.players(gw)
    for e in set(row["t_in"]) | set(row["t_out"]):
        cost[e] = int(pl.loc[e, "now_cost"]) if e in pl.index else None
    lines = [
        f"- **{label}** GW{gw}: chip {row['chip'] or '-'}; free transfers in {row['free_transfers_in']}, "
        f"transfers {row['transfers']}, hits {row['hits']}; bank {row['bank_before'] / 10:.1f} -> "
        f"{row['bank'] / 10:.1f}"
    ]
    if row["t_out"] or row["t_in"]:
        outs = ", ".join(f"{data.name(e)} ({(cost[e] or 0) / 10:.1f})" for e in row["t_out"])
        ins = ", ".join(f"{data.name(e)} ({(cost[e] or 0) / 10:.1f})" for e in row["t_in"])
        lines.append(f"  - out: {outs or '-'}; in: {ins}")
    xi = [
        f"{data.name(e)}{'(C)' if e == pkg['captain'] else '(V)' if e == pkg['vice'] else ''} "
        f"{pts.get(e, 0)}{'' if mins.get(e, 0) else '*'}"
        for e in pkg["xi"]
    ]
    lines.append("  - XI: " + ", ".join(xi))
    lines.append("  - bench: " + ", ".join(f"{data.name(e)} {pts.get(e, 0)}" for e in pkg["bench"]))
    if row["subs"]:
        lines.append(
            "  - auto-subs: " + ", ".join(f"{data.name(a)} -> {data.name(b)}" for a, b in row["subs"])
        )
    arm = row["armband"]
    lines.append(
        f"  - armband: {data.name(arm) if arm else 'none'} x{row['mult']} "
        f"({pts.get(arm, 0) if arm else 0} pts); GW points {row['points']}, net {row['net']}"
        + (f"; chip gain {row['chip_gain']}" if row["chip_gain"] is not None else "")
    )
    return lines


def contributions(row: dict, data: LA.SeasonData) -> Counter:
    pts = data.points.get(row["gw"], {})
    c: Counter = Counter()
    for e in row["scorers"]:
        c[e] += pts.get(e, 0)
    if row["armband"] is not None:
        c[row["armband"]] += (row["mult"] - 1) * pts.get(row["armband"], 0)
    return c


def gap_text(a: dict, b: dict, data: LA.SeasonData, la: str, lb: str) -> list[str]:
    ca, cb = contributions(a, data), contributions(b, data)
    diff = {e: ca.get(e, 0) - cb.get(e, 0) for e in set(ca) | set(cb)}
    plus = sorted(((e, d) for e, d in diff.items() if d > 0), key=lambda x: -x[1])
    minus = sorted(((e, d) for e, d in diff.items() if d < 0), key=lambda x: x[1])
    hits = 4 * (a["hits"] - b["hits"])
    return [
        f"  - gap {la} - {lb} = {a['net'] - b['net']}: "
        + ", ".join(f"{data.name(e)} +{d}" for e, d in plus)
        + ("; " if minus else "")
        + ", ".join(f"{data.name(e)} {d}" for e, d in minus)
        + (f"; hits {-hits}" if hits else "")
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seasons", nargs="+", default=None)
    ap.add_argument("--focus", nargs=3, default=["2024-25", "minimal_model", "form_restrained"])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # player names (python -I ignores PYTHONUTF8)
    run = Path(args.run)
    seasons = args.seasons or sorted(p.name for p in run.iterdir() if p.is_dir() and p.name[:2] == "20")
    res = LA.audit_run(run, seasons, list(POLICIES))
    md = ["# Independent ledger audit", "", f"Run {run.name}; auditor fpl_rocket.ledger_audit.", ""]
    md += [
        "| season | policy | gws | ledger total | disagreements | legality issues |",
        "|---|---|---|---|---|---|",
    ]
    n_dis = n_iss = 0
    for (s, p), r in sorted(res.items()):
        tot = sum(x["net"] for x in r["rows"])
        md.append(
            f"| {s} | {p} | {len(r['rows'])} | {tot} | {len(r['disagreements'])} | {len(r['issues'])} |"
        )
        n_dis += len(r["disagreements"])
        n_iss += len(r["issues"])
    md += ["", f"TOTAL disagreements {n_dis}; legality issues {n_iss}", ""]
    for (_s, _p), r in sorted(res.items()):
        for d in r["disagreements"] + r["issues"]:
            md.append(f"- {_p} {json.dumps(d, default=str)}")
    s, fa, fb = args.focus
    if (s, fa) in res and (s, fb) in res:
        data = LA.SeasonData(s)
        ra = {r["gw"]: r for r in res[(s, fa)]["rows"]}
        rb = {r["gw"]: r for r in res[(s, fb)]["rows"]}
        pa, pb = _packages(run / s / f"{fa}.jsonl"), _packages(run / s / f"{fb}.jsonl")
        chip_gws = [g for g, r in sorted(ra.items()) if r["chip"]]
        diffs = sorted(ra, key=lambda g: -(ra[g]["net"] - rb[g]["net"]))[: args.top]
        md += ["", f"## Ledger {s} {fa}: chip weeks {chip_gws}", ""]
        for g in chip_gws:
            md += ledger_text(ra[g], pa[g], data, fa)
            md += ledger_text(rb[g], pb[g], data, fb)
            md += gap_text(ra[g], rb[g], data, fa, fb)
        md += ["", f"## The {args.top} GWs with the largest {fa} minus {fb}: {sorted(diffs)}", ""]
        for g in sorted(diffs):
            md += ledger_text(ra[g], pa[g], data, fa)
            md += ledger_text(rb[g], pb[g], data, fb)
            md += gap_text(ra[g], rb[g], data, fa, fb)
        top_sum = sum(ra[g]["net"] - rb[g]["net"] for g in diffs)
        all_sum = sum(ra[g]["net"] - rb[g]["net"] for g in ra)
        md += ["", f"Season gap {all_sum}; the {args.top} largest GWs sum to {top_sum}."]
    text = "\n".join(md) + "\n"
    out = Path(args.out) if args.out else run / "AUDIT.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written {out}")
    return 1 if n_dis or n_iss else 0


if __name__ == "__main__":
    sys.exit(main())
