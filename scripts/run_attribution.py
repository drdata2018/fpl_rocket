"""Controlled replays against the frozen baseline-v0 (fpl_rocket.replay.variants): one factor at a time, the
main pairs, and tie-break perturbations. DIAGNOSTICS ONLY - no default changes.

Usage:
    uv run python -I scripts/run_attribution.py --seasons 2024-25 2023-24 \
        [--variants model_flags_off ... tiebreak_1 ...] [--phases tables runs report] [--workers 5] \
        [--out data/replays/attribution_<ts>] [--baseline-run data/replays/20261106T151927Z]

tables  the variant model tables (one refit at T per (season, GW), cached in data/replay_inputs/variant_cache;
        the as-known capped prediction must equal the frozen forecast cache: checked here)
runs    one process per (variant, season); a variant replays only the policies it changes (model_flags_off and
        imposed_start: MINIMAL_MODEL; form_flags_off: FORM_RESTRAINED; the rest: both); the unchanged policy
        keeps its baseline-v0 total (identical by construction: same inputs, same code)
report  <out>/ATTRIBUTION.md + attribution.csv (season totals, MODEL minus FORM, changes vs baseline-v0)
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

import pandas as pd  # noqa: E402

from fpl_rocket.replay import inputs as I  # noqa: E402
from fpl_rocket.replay import run, variants  # noqa: E402

CACHE = I.INPUTS_DIR / "forecast_cache"
BASELINE_RUN = ROOT / "data" / "replays" / "20261106T151927Z"
PAIR = ("minimal_model", "form_restrained")
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


def policies_for(v: variants.Variant) -> tuple[str, ...]:
    if v.name == "model_flags_off" or v.name == "imposed_start":
        return ("minimal_model",)
    if v.name == "form_flags_off":
        return ("form_restrained",)
    return PAIR


def imposed_squad(baseline_run: Path, season: str) -> list[int]:
    rec = run.recorded_packages(baseline_run, season, "form_restrained")
    first = rec[min(rec)]
    return sorted(int(e) for e in first["package"]["transfers_in"])


def _tables_task(season: str, gw: int) -> tuple[str, int, float, float]:
    t0 = time.perf_counter()
    inp = I.load_inputs(season, gw, _store())
    digest = run.forecast_digest(inp)
    tabs = variants.load_or_build_tables(inp, digest)
    fc = run.forecasts_for(inp, CACHE, require_cache=True)
    base = fc.xp("model")
    mine = tabs["model_known_capped"]
    if set(base) != set(mine):
        raise RuntimeError(f"{season} GW{gw}: variant refit keys differ from the frozen forecasts")
    diff = max((abs(base[k] - mine[k]) for k in base), default=0.0)
    return season, gw, diff, time.perf_counter() - t0


def _run_task(vname: str, season: str, out: str, baseline_run: str, log_path: str) -> tuple[str, str]:
    log = _logger(Path(log_path))
    v = variants.by_name(vname)
    imposed = imposed_squad(Path(baseline_run), season) if v.imposed_start else None
    pols = variants.make_variant_policies(v, imposed)

    def hook(fc, inp):
        tabs = None
        if v.needs_model_tables():
            tabs = variants.load_or_build_tables(inp, run.forecast_digest(inp), build=False)
        return variants.apply(v, fc, inp, tabs)

    def vlog(msg: str) -> None:
        log(f"[{vname}] {msg}")

    run.run_season(
        season,
        Path(out) / vname,
        _store(),
        policy_names=policies_for(v),
        cache_dir=CACHE,
        log=vlog,
        require_cache=True,
        policies=pols,
        forecast_hook=hook,
    )
    return vname, season


def baseline_totals(baseline_run: Path, seasons: list[str]) -> dict[tuple[str, str], int]:
    out = {}
    for s in seasons:
        rows = pd.read_csv(baseline_run / s / "rows.csv")
        for p, t in rows.groupby("policy")["net"].sum().items():
            out[(s, str(p))] = int(t)
    return out


def report(out: Path, seasons: list[str], names: list[str], baseline_run: Path) -> Path:
    base = baseline_totals(baseline_run, seasons)
    rows = []
    for s in seasons:
        for n in ["baseline", *names]:
            v = variants.by_name(n)
            tot = {p: base[(s, p)] for p in PAIR}
            ran = []
            if n != "baseline":
                f = out / n / s / "rows.csv"
                if not f.exists():
                    continue
                r = pd.read_csv(f)
                for p, t in r.groupby("policy")["net"].sum().items():
                    tot[str(p)] = int(t)
                    ran.append(str(p))
            m, f_ = tot["minimal_model"], tot["form_restrained"]
            rows.append(
                {
                    "season": s,
                    "variant": n,
                    "replayed": ",".join(ran) if ran else "-",
                    "model": m,
                    "form": f_,
                    "model_minus_form": m - f_,
                    "d_model": m - base[(s, "minimal_model")],
                    "d_form": f_ - base[(s, "form_restrained")],
                    "d_gap": (m - f_) - (base[(s, "minimal_model")] - base[(s, "form_restrained")]),
                    "note": v.note,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "attribution.csv", index=False)
    lines = [
        "# Attribution replays vs baseline-v0",
        "",
        "Totals = net season points. d_* = change vs baseline-v0. A policy not replayed keeps its "
        "baseline total.",
        "",
        "| season | variant | replayed | MODEL | FORM | MODEL-FORM | dMODEL | dFORM | dGAP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['season']} | {r['variant']} | {r['replayed']} | {r['model']} | {r['form']} | "
            f"{r['model_minus_form']} | {r['d_model']:+d} | {r['d_form']:+d} | {r['d_gap']:+d} |"
        )
    tb = df[df["variant"].str.startswith("tiebreak_") | (df["variant"] == "baseline")]
    if (tb["variant"] != "baseline").any():
        lines += [
            "",
            "Tie-break sensitivity (baseline + the noise seeds): spread of season totals",
            "",
            "| season | runs | MODEL min-max (sd) | FORM min-max (sd) | MODEL-FORM min-max |",
            "|---|---|---|---|---|",
        ]
        for s, g in tb.groupby("season", sort=False):
            lines.append(
                f"| {s} | {len(g)} | {g['model'].min()}-{g['model'].max()} ({g['model'].std():.1f}) | "
                f"{g['form'].min()}-{g['form'].max()} ({g['form'].std():.1f}) | "
                f"{g['model_minus_form'].min()}-{g['model_minus_form'].max()} |"
            )
    path = out / "ATTRIBUTION.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=["2024-25", "2023-24"])
    default_v = [v.name for v in variants.ATTRIBUTION[1:]] + [
        f"tiebreak_{s}" for s in variants.TIEBREAK_SEEDS
    ]
    ap.add_argument("--variants", nargs="+", default=default_v)
    ap.add_argument("--phases", nargs="+", default=["tables", "runs", "report"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline-run", default=str(BASELINE_RUN))
    args = ap.parse_args(argv)
    out = Path(args.out or ROOT / "data" / "replays" / f"attribution_{datetime.now(UTC):%Y%m%dT%H%M%SZ}")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    log = _logger(log_path)
    log(f"out {out}; seasons {args.seasons}; variants {args.variants}; phases {args.phases}")
    vs = [variants.by_name(n) for n in args.variants]
    failed = []
    if "tables" in args.phases and any(v.needs_model_tables() for v in vs):
        tasks = [(s, g) for s in args.seasons for g in run.season_gws(s)]
        worst = 0.0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_tables_task, s, g) for s, g in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                s, g, d, sec = f.result()
                worst = max(worst, d)
                if i % 10 == 0 or i == len(futs):
                    log(f"tables {i}/{len(futs)} ({s} GW{g} {sec:.0f} s); max |refit - frozen| = {worst:.3g}")
        if worst != 0.0:
            log(f"WARNING: the refit differs from the frozen forecasts by up to {worst}")
    if "runs" in args.phases:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_run_task, v.name, s, str(out), args.baseline_run, str(log_path)): (v.name, s)
                for v in vs
                for s in args.seasons
            }
            for f in as_completed(futs):
                try:
                    n, s = f.result()
                    log(f"done {n} {s}")
                except Exception as exc:  # noqa: BLE001 - logged, the others go on
                    failed.append(f"{futs[f]}: {exc!r}")
                    log(f"FAILED {futs[f]}: {exc!r}")
    if "report" in args.phases:
        log(f"report {report(out, args.seasons, args.variants, Path(args.baseline_run))}")
    (out / "variants.json").write_text(json.dumps([v.describe() for v in vs], indent=1), encoding="utf-8")
    if failed:
        log(f"{len(failed)} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
