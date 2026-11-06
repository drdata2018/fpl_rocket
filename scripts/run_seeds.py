"""Seed diagnostic (fpl_rocket.replay.seeds): MINIMAL_MODEL replayed with other Monte Carlo seeds of its
forecast, everything else frozen. DIAGNOSTICS ONLY - computational variation, not football uncertainty.

Usage:
    uv run python -I scripts/run_seeds.py [--seasons 2021-22 ... 2025-26] [--seeds 1 2 ... 8]
        [--phases tables runs report] [--workers 8] [--out data/replays/seed_4e]
        [--baseline-run data/replays/20261106T151927Z]

tables  one refit at T per (season, GW) + one prediction per seed index (0 = baseline-v0, must equal the
        frozen forecast cache: checked here), cached in data/replay_inputs/seed_cache
runs    one process per (seed index, season): MINIMAL_MODEL (the frozen policy) on the seed's model table;
        <out>/seed_<k>/<season>/minimal_model.jsonl + rows.csv
report  <out>/SEEDS.md, seed_rows.csv (per seed and GW), seed_totals.csv (per seed and season)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fpl_rocket import rankmap  # noqa: E402
from fpl_rocket.replay import baseline, run, seeds  # noqa: E402
from fpl_rocket.replay import inputs as I  # noqa: E402
from fpl_rocket.replay.policies import make_policies  # noqa: E402

CACHE = I.INPUTS_DIR / "forecast_cache"
BASELINE_RUN = ROOT / "data" / "replays" / "20261106T151927Z"
RANKMAP_CACHE = ROOT / "data" / "raw" / "rankmap" / rankmap.CACHE_NAME
SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
POLICY = "minimal_model"
_STORE: dict[str, I.HistoryStore] = {}


def _store() -> I.HistoryStore:
    key = str(I.INPUTS_DIR / I.HISTORY_FILE)
    if key not in _STORE:
        _STORE[key] = I.HistoryStore(key)
    return _STORE[key]


def _logger(path: Path):
    def log(msg: str) -> None:
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return log


def _tables_task(season: str, gw: int, ks: list[int]) -> tuple[str, int, float, float]:
    t0 = time.perf_counter()
    inp = I.load_inputs(season, gw, _store())
    tabs = seeds.load_or_build(inp, run.forecast_digest(inp), [0, *ks])
    base = run.forecasts_for(inp, CACHE, require_cache=True).xp("model")
    mine = tabs[0]
    if set(base) != set(mine):
        raise RuntimeError(f"{season} GW{gw}: seed refit keys differ from the frozen forecasts")
    diff = max((abs(base[k] - mine[k]) for k in base), default=0.0)
    return season, gw, diff, time.perf_counter() - t0


def _run_task(k: int, season: str, out: str, log_path: str) -> tuple[int, str]:
    log = _logger(Path(log_path))

    def hook(fc, inp):
        table = seeds.load_or_build(inp, run.forecast_digest(inp), [k], build=False)[k]
        return seeds.with_model_table(fc, table, k)

    run.run_season(
        season,
        Path(out) / f"seed_{k}",
        _store(),
        policy_names=(POLICY,),
        cache_dir=CACHE,
        log=lambda m: log(f"[seed {k}] {m}"),
        require_cache=True,
        policies={POLICY: make_policies()[POLICY]},
        forecast_hook=hook,
        model_seed=lambda gw: seeds.mc_seed(k, gw),
    )
    return k, season


# ================================================================================================ report
def _log_path(out: Path, baseline_run: Path, k: int, season: str, policy: str = POLICY) -> Path:
    return (
        baseline_run / season / f"{policy}.jsonl"
        if k == 0
        else out / f"seed_{k}" / season / f"{policy}.jsonl"
    )


def collect(out: Path, baseline_run: Path, seasons: list[str], ks: list[int]) -> pd.DataFrame:
    """One row per (season, seed index, GW): net points, captain, vice, chip, transfers, changes vs k = 0."""
    rows = []
    for s in seasons:
        base_dec, _ = baseline.read_log(_log_path(out, baseline_run, 0, s))
        for k in [0, *ks]:
            p = _log_path(out, baseline_run, k, s)
            if not p.exists():
                continue
            dec, sc = baseline.read_log(p)
            for gw in sorted(sc):
                pk = dec[gw]["package"]
                ch = seeds.package_changes(base_dec[gw]["package"], pk)
                rows.append(
                    {
                        "season": s,
                        "seed_index": k,
                        "mc_seed": seeds.mc_seed(k, gw),
                        "gw": gw,
                        "net": int(sc[gw]["net"]),
                        "captain": int(pk["captain"]),
                        "vice": int(pk["vice"]),
                        "chip": pk.get("chip") or "",
                        "transfers": int(sc[gw]["transfers"]),
                        "transfers_in": " ".join(str(e) for e in sorted(pk["transfers_in"])),
                        "transfers_out": " ".join(str(e) for e in sorted(pk["transfers_out"])),
                        "fallback": bool(sc[gw].get("fallback", False)),
                        **{f"diff_{c}": bool(v) for c, v in ch.items()},
                    }
                )
    return pd.DataFrame(rows)


def other_policy_rows(baseline_run: Path, seasons: list[str]) -> pd.DataFrame:
    parts = []
    for s in seasons:
        r = pd.read_csv(baseline_run / s / "rows.csv")
        parts.append(
            r[r["policy"].isin(["form_restrained", "lagged_crowd"])][["season", "gw", "policy", "net"]]
        )
    return pd.concat(parts, ignore_index=True)


def _fmt_pct(v) -> str:
    return "-" if v is None else f"{v:.2f}"


def report(out: Path, baseline_run: Path, seasons: list[str], ks: list[int]) -> Path:
    rows = collect(out, baseline_run, seasons, ks)
    rows.to_csv(out / "seed_rows.csv", index=False)
    others = other_policy_rows(baseline_run, seasons)
    records = rankmap.load_cache(RANKMAP_CACHE)
    pairs = rankmap.pairs_from_cache(records)
    curves = {s: rankmap.fit_curve(pairs, s) for s in seasons if (pairs["season"] == s).any()}
    tot = rows.groupby(["season", "seed_index"]).agg(
        total=("net", "sum"),
        gws=("gw", "size"),
        fallbacks=("fallback", "sum"),
        **{f"gws_{c}": (f"diff_{c}", "sum") for c in ("any", "transfers", "captain", "xi_bench", "chip")},
    )
    tot = tot.reset_index()
    for p in ("form_restrained", "lagged_crowd"):
        o = others[others["policy"] == p].set_index(["season", "gw"])["net"]
        diffs = []
        for r in tot.itertuples(index=False):
            m = rows[(rows["season"] == r.season) & (rows["seed_index"] == r.seed_index)].set_index("gw")[
                "net"
            ]
            d = m - o.loc[r.season].reindex(m.index)
            diffs.append((int(d.sum()), float(d.mean())))
        tot[f"minus_{p}"] = [a for a, _ in diffs]
        tot[f"minus_{p}_per_gw"] = [b for _, b in diffs]
    pct = []
    for r in tot.itertuples(index=False):
        c = curves.get(r.season)
        pl = c.place(int(r.total), rankmap.ENTRANTS[r.season][0]) if c is not None else {}
        pct.append((pl.get("rank_est"), pl.get("pct_est")))
    tot["rank_est"] = [a for a, _ in pct]
    tot["top_pct_est"] = [b for _, b in pct]
    tot.to_csv(out / "seed_totals.csv", index=False)
    n = len(ks)
    md = [
        "# Seed diagnostic: MINIMAL_MODEL with other Monte Carlo seeds (step 4e)",
        "",
        "**These seeds measure COMPUTATIONAL variation only** (the random draws inside the model's "
        "simulation). "
        "They are not extra seasons and say nothing about uncertainty in football outcomes: every run "
        "below plays the same five real seasons with the same results.",
        "",
        "What the seed controls: `XPModel.predict(seed=...)` seeds the numpy generator of the Monte Carlo "
        "simulation (`model.simulate.simulate_gw`, n_sims 1000 joint draws per player-fixture behind every "
        "expected-points value). The fit does not read the seed: Dixon-Coles, the rates, the newcomer prior "
        "and the bonus link are deterministic, and the boosted models (minutes classifier / regressor, bonus "
        "regressor) use a fixed random_state=0 with early stopping off. So the fit is the same in every run "
        "and only the simulation draws change (checked: a second fit at T gives identical predictions; the "
        "`lines` before simulation are identical across seeds). A 'both vary' run is therefore not "
        "applicable. Baseline-v0 = seed index 0: Monte Carlo seed 20261023 + GW. Seed index k: "
        f"20261023 + {seeds.SEED_STRIDE:,} x k + GW (k = 1..{n}). n_sims 1000 throughout.",
        "",
        "Everything else is frozen: the same inputs at T, the same fit, the same form baseline table (chip "
        "triggers of FORM / LAGGED are unaffected), the frozen MINIMAL_MODEL policy (planner, chip rule, "
        "60 s "
        "solver limit). FORM_RESTRAINED and LAGGED_CROWD do not use the model, so their baseline-v0 totals "
        "are the comparators. Seed-0 tables re-predicted from one refit equal the frozen forecast cache "
        "exactly (self-check in run.log).",
        "",
    ]
    fb = int(tot.loc[tot["seed_index"] > 0, "fallbacks"].sum())
    md += [f"Solver fallbacks in the seed runs: {fb}.", ""]
    md += [
        f"## Season totals (seed index 0 = baseline-v0, then 1..{n})",
        "",
        "| season | totals k=0..8 | mean | sd | min | max | FORM | LAGGED |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in seasons:
        t = tot[tot["season"] == s].sort_values("seed_index")
        if t.empty:
            continue
        f_tot = int(others[(others["season"] == s) & (others["policy"] == "form_restrained")]["net"].sum())
        l_tot = int(others[(others["season"] == s) & (others["policy"] == "lagged_crowd")]["net"].sum())
        v = t["total"].to_numpy(float)
        md.append(
            f"| {s} | {' / '.join(str(int(x)) for x in v)} | {v.mean():.1f} | {v.std(ddof=1):.1f} | "
            f"{int(v.min())} | {int(v.max())} | {f_tot} | {l_tot} |"
        )
    five = tot.groupby("seed_index")["total"].sum()
    if len(five) > 1:
        v = five.to_numpy(float)
        md.append(
            f"| five seasons | {' / '.join(str(int(x)) for x in v)} | {v.mean():.1f} | {v.std(ddof=1):.1f} | "
            f"{int(v.min())} | {int(v.max())} | - | - |"
        )
    md += [
        "",
        "## Approximate final-rank percentile of each total (top x %, rank map of RANKMAP.md)",
        "",
        "| season | k=0 | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---|" * (len(ks) + 1),
    ]
    for s in seasons:
        t = tot[tot["season"] == s].sort_values("seed_index")
        cells = [f"{int(r.total)}: {_fmt_pct(r.top_pct_est)}" for r in t.itertuples(index=False)]
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md += [
        "",
        "## Gameweeks whose package differs from baseline-v0's (per seed)",
        "",
        "Counts of GWs (out of the season's GWs) where the seed run's package differs: any part; transfers "
        "(in / out sets); captain (captain or vice); XI / bench (XI set or bench order); chip (chip played "
        "that GW). A GW can count in several columns. Once one transfer differs, the squads differ and later "
        "differences are path-dependent, not separate seed effects. GW1 'transfers' = the initial "
        "squad.",
        "",
        "| season | k | total | any | transfers | captain | XI / bench | chip | first differing GW | "
        "first GW with different transfers |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in seasons:
        for r in tot[(tot["season"] == s) & (tot["seed_index"] > 0)].sort_values("seed_index").itertuples():
            g = rows[(rows["season"] == s) & (rows["seed_index"] == r.seed_index)]
            first = int(g.loc[g["diff_any"], "gw"].min()) if g["diff_any"].any() else "-"
            first_t = int(g.loc[g["diff_transfers"], "gw"].min()) if g["diff_transfers"].any() else "-"
            md.append(
                f"| {s} | {r.seed_index} | {int(r.total)} | {int(r.gws_any)}/{int(r.gws)} | "
                f"{int(r.gws_transfers)} | {int(r.gws_captain)} | {int(r.gws_xi_bench)} | "
                f"{int(r.gws_chip)} | {first} | {first_t} |"
            )
    md += [
        "",
        "## Paired differences vs FORM_RESTRAINED and LAGGED_CROWD (their baseline-v0 rows), per seed",
        "",
        "| season | k | MODEL | MODEL-FORM total | per GW | MODEL-LAGGED total | per GW |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in seasons:
        for r in tot[tot["season"] == s].sort_values("seed_index").itertuples():
            md.append(
                f"| {s} | {r.seed_index} | {int(r.total)} | {r.minus_form_restrained:+d} | "
                f"{r.minus_form_restrained_per_gw:+.2f} | {r.minus_lagged_crowd:+d} | "
                f"{r.minus_lagged_crowd_per_gw:+.2f} |"
            )
    md += [
        "",
        "| season | MODEL-FORM over seeds: mean (sd) [min, max] | "
        "MODEL-LAGGED over seeds: mean (sd) [min, max] |",
        "|---|---|---|",
    ]
    for s in [*seasons, "five seasons"]:
        t = tot if s == "five seasons" else tot[tot["season"] == s]
        if s == "five seasons":
            t = t.groupby("seed_index")[["minus_form_restrained", "minus_lagged_crowd"]].sum().reset_index()
        a, b = t["minus_form_restrained"].to_numpy(float), t["minus_lagged_crowd"].to_numpy(float)
        md.append(
            f"| {s} | {a.mean():+.1f} ({a.std(ddof=1):.1f}) [{a.min():+.0f}, {a.max():+.0f}] | "
            f"{b.mean():+.1f} ({b.std(ddof=1):.1f}) [{b.min():+.0f}, {b.max():+.0f}] |"
        )
    md += [
        "",
        "## Standard error of a seed-averaged season total by the number of seeds",
        "",
        f"From the {n} additional seeds (k = 1..{n}; the baseline seed is left out so the groups are "
        "disjoint and equally sized): se(m) = sd / sqrt(m) with sd the sample sd of the per-seed totals; "
        "empirical = sd of the means of disjoint groups of m consecutive seeds (needs >= 2 groups).",
        "",
        "| season | sd (m=1) | se m=2 | se m=4 | se m=8 | empirical m=1 / 2 / 4 |",
        "|---|---|---|---|---|---|",
    ]
    for s in [*seasons, "five seasons"]:
        t = tot[tot["seed_index"] > 0]
        v = (
            t.groupby("seed_index")["total"].sum()
            if s == "five seasons"
            else t[t["season"] == s].set_index("seed_index")["total"]
        ).sort_index()
        se = {r["m"]: r for r in seeds.se_by_count(list(v.to_numpy(float)))}
        emp = " / ".join(
            "-" if np.isnan(se[m]["empirical_sd"]) else f"{se[m]['empirical_sd']:.1f}" for m in (1, 2, 4)
        )
        md.append(
            f"| {s} | {se[1]['se']:.1f} | {se[2]['se']:.1f} | {se[4]['se']:.1f} | {se[8]['se']:.1f} | {emp} |"
        )
    md += [
        "",
        "Files: `seed_rows.csv` (season, seed_index, mc_seed, gw, net, captain, vice, chip, transfers, "
        "transfers_in / out, fallback, diff_* vs k = 0; k = 0 rows are baseline-v0's), `seed_totals.csv`. "
        "Decision logs: `seed_<k>/<season>/minimal_model.jsonl`. Seed tables: "
        "`data/replay_inputs/seed_cache/<season>/gwNN.pkl` (indices 0..8).",
    ]
    path = out / "SEEDS.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=list(SEASONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 9)))
    ap.add_argument("--phases", nargs="+", default=["tables", "runs", "report"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data" / "replays" / "seed_4e"))
    ap.add_argument("--baseline-run", default=str(BASELINE_RUN))
    args = ap.parse_args(argv)
    if 0 in args.seeds:
        raise SystemExit("seed index 0 is baseline-v0 (not replayed here)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    log = _logger(log_path)
    log(f"out {out}; seasons {args.seasons}; seeds {args.seeds}; phases {args.phases}")
    failed = []
    if "tables" in args.phases:
        tasks = [(s, g) for s in args.seasons for g in run.season_gws(s)]
        worst = 0.0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_tables_task, s, g, args.seeds): (s, g) for s, g in tasks}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    s, g, d, sec = f.result()
                except Exception as exc:  # noqa: BLE001 - logged, the others go on
                    failed.append(f"tables {futs[f]}: {exc!r}")
                    log(f"FAILED tables {futs[f]}: {exc!r}")
                    continue
                worst = max(worst, d)
                if i % 10 == 0 or i == len(futs):
                    log(
                        f"tables {i}/{len(futs)} ({s} GW{g} {sec:.0f} s); max |seed 0 - frozen| = {worst:.3g}"
                    )
        if worst != 0.0:
            log(f"WARNING: seed index 0 differs from the frozen forecasts by up to {worst}")
    if "runs" in args.phases and not failed:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_run_task, k, s, str(out), str(log_path)): (k, s)
                for k in args.seeds
                for s in args.seasons
            }
            for f in as_completed(futs):
                try:
                    k, s = f.result()
                    log(f"done seed {k} {s}")
                except Exception as exc:  # noqa: BLE001 - logged, the others go on
                    failed.append(f"{futs[f]}: {exc!r}")
                    log(f"FAILED {futs[f]}: {exc!r}")
    if "report" in args.phases:
        log(f"report {report(out, Path(args.baseline_run), args.seasons, args.seeds)}")
    (out / "seeds.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "stride": seeds.SEED_STRIDE,
                "scheme": "20261023 + stride x k + gw",
                "seasons": args.seasons,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    if failed:
        log(f"{len(failed)} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
