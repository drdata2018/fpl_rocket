"""Real final-rank mapping (fpl_rocket.rankmap): fetch real managers' past seasons, then place replay totals.

Usage:
    uv run python -I scripts/rank_map.py fetch [--extra-random 2000] [--min-interval 1.0]
    uv run python -I scripts/rank_map.py report --replay data/replays/20261106T151927Z

fetch   1) the members of the frozen cohorts (registry/cohorts/cohort_<freeze>.json on the spike-log branch,
           read with `git show`), 2) --extra-random ids drawn with the seeded scheme of fpl_rocket.rankmap
           from
           1..total_players at the freeze (data/raw/cohorts/seed_<seed>/state.json). One entry/{id}/history/
           request per id, >= --min-interval seconds apart, cached in data/raw/rankmap/histories.jsonl (entry
           id + past seasons' points and final ranks only; no names); resumable.
report  per season 2021-22..2025-26: pairs, points range, monotonic violations, anchors near 2,100-2,450,
        and every replay policy's season total placed on the curve (rank, band, percentile).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_rocket import cohorts, rankmap  # noqa: E402

CACHE_DIR = ROOT / "data" / "raw" / "rankmap"
FREEZE = "20261106"


def _log(path: Path):
    def log(msg: str) -> None:
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return log


def cohort_members(freeze: str = FREEZE) -> dict[str, list[int]]:
    body = subprocess.run(
        ["git", "show", f"origin/spike-log:registry/cohorts/cohort_{freeze}.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    data = json.loads(body.decode("utf-8"))
    return {name: [int(m["entry_id"]) for m in members] for name, members in data["cohorts"].items()}


def cmd_fetch(args) -> int:
    from fpl_rocket.api import FPLClient

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log = _log(CACHE_DIR / "fetch.log")
    cache = cohorts.JsonlCache(CACHE_DIR / rankmap.CACHE_NAME)
    session = cohorts.PoliteSession(min_interval=args.min_interval)
    client = FPLClient(session=session, backoff=2.0)
    members = cohort_members(args.freeze)
    total = {}
    for name in ("random", "top10k"):
        total[name] = rankmap.fetch(client, members[name], cache, f"cohort_{name}", log=log)
        log(f"cohort {name}: {total[name]}")
    if args.extra_random:
        state = json.loads(
            (ROOT / "data" / "raw" / "cohorts" / f"seed_{args.seed}" / "state.json").read_text(
                encoding="utf-8"
            )
        )
        exclude = {e for v in members.values() for e in v}
        ids = rankmap.random_ids(args.seed, int(state["total_players"]), args.extra_random, exclude)
        total["random_extra"] = rankmap.fetch(client, ids, cache, "random_extra", log=log)
        log(f"random extra: {total['random_extra']}")
    log(f"requests {session.n_requests} by status {dict(session.by_status)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--freeze", default=FREEZE)
    f.add_argument("--seed", type=int, default=int(FREEZE))
    f.add_argument("--extra-random", type=int, default=0)
    f.add_argument("--min-interval", type=float, default=1.0)
    r = sub.add_parser("report")
    r.add_argument("--replay", required=True)
    r.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        return cmd_fetch(args)
    return cmd_report(args)


def cmd_report(args) -> int:
    import pandas as pd

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = rankmap.load_cache(CACHE_DIR / rankmap.CACHE_NAME)
    pairs = rankmap.pairs_from_cache(records)
    run_dir = Path(args.replay)
    lo, hi = rankmap.INTEREST
    md = [
        "# Real final-rank mapping",
        "",
        f"Entries fetched: {len(records)} ({sum(1 for r in records.values() if r['status'] == 'ok')} found); "
        f"pairs (season, points, final rank) in 2021-22..2025-26: "
        f"{int(pairs['season'].isin(rankmap.SEASONS).sum())}.",
        "",
        "The replay policies are blind robots that never missed a week; real managers include abandoned "
        "teams. A placement is where that season total WOULD HAVE ranked in that season, not a claim about "
        "the future.",
        "",
        "| season | pairs | entries | points range | monotonic violations | totals with tied ranks | "
        "anchors in 2100-2450 | largest rank in sample | entrants (source) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    curves = {}
    for s in rankmap.SEASONS:
        d = pairs[pairs["season"] == s]
        if d.empty:
            md.append(f"| {s} | 0 | 0 | - | - | - | - | - |")
            continue
        c = rankmap.fit_curve(pairs, s)
        curves[s] = c
        ent, src = rankmap.ENTRANTS[s]
        md.append(
            f"| {s} | {len(d)} | {d['entry'].nunique()} | {d['points'].min()}-{d['points'].max()} | "
            f"{c.violations} | {rankmap.tie_groups(d)} | {int(d['points'].between(lo, hi).sum())} | "
            f"{d['rank'].max():,} | {ent:,} ({src}) |"
        )
    md += [
        "",
        "Placement: rank_est = isotonic log10(rank) curve interpolated at the total; band = ranks of the "
        "nearest observed anchors with at least / at most the total; pct = rank / entrants x 100 (top x%).",
        "",
        "| season | policy | total | rank_est | band (best-worst) | anchors (points above / below) | "
        "top % est | top % band | pairs within +-25 pts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    out_rows = []
    for s in rankmap.SEASONS:
        f = run_dir / s / "rows.csv"
        if s not in curves or not f.exists():
            continue
        tot = pd.read_csv(f).groupby("policy")["net"].sum()
        ent = rankmap.ENTRANTS[s][0]
        for p, t in tot.items():
            pl = curves[s].place(int(t), ent)
            out_rows.append({"policy": p, **pl})

            def fmt(v):
                return "-" if v is None else f"{v:,}"

            def pct(v):
                return "-" if v is None else f"{v:.2f}"

            band = f"{fmt(pl['rank_best'])}-{fmt(pl['rank_worst'])}"
            md.append(
                f"| {s} | {p} | {int(t)} | {fmt(pl['rank_est'])} | {band} | "
                f"{pl['anchor_above_points']} / {pl['anchor_below_points']} | {pct(pl.get('pct_est'))} | "
                f"{pct(pl.get('pct_best'))}-{pct(pl.get('pct_worst'))} | {pl['anchors_within_25']} |"
            )
    text = "\n".join(md) + "\n"
    out = Path(args.out) if args.out else run_dir / "RANKMAP.md"
    out.write_text(text, encoding="utf-8")
    pd.DataFrame(out_rows).to_csv(out.with_suffix(".csv"), index=False)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
