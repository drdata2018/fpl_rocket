"""Blind replay of whole seasons under the PLAN_v3 contract (fpl_rocket.replay).

Usage:
    uv run python -I scripts/run_replay.py --seasons 2021-22 2022-23 --policies minimal_model hold \
        [--out data/replays/<ts>] [--gws 1 2 3] [--workers 6] [--phases forecasts seasons leak report]

Phases (all by default, in order):
    forecasts  predictions at T for every (season, GW), built in a process pool and cached by input digest in
               data/replay_inputs/forecast_cache (identical when rebuilt; the leakage test rebuilds them)
    seasons    the season loop per season (one process per season), resumable per season / policy
               (<out>/<season>/<policy>.done); decisions in <out>/<season>/<policy>.jsonl
    leak       the built-in leakage test: K = 3 deadlines per season (fixed seed), rows after T corrupted
    report     <out>/REPORT.md (fpl_rocket.replay.report)
The points history (data/replay_inputs/points_history.parquet) is rebuilt from data/archive first.

    uv run python -I scripts/run_replay.py --verify-baseline baseline_v0 --seasons 2024-25 2023-24 \
        --policies minimal_model
replays the chosen seasons / policies from the caches (history and forecast cache checked against the frozen
sha256 first; nothing rebuilt) and asserts byte-identical canonical packages, decisions and score rows
(fpl_rocket.replay.baseline; frozen record src/fpl_rocket/replay/baselines/<name>.json).
Solver limit 60 s per plan; a solver fallback is a logged event (decision log, report), never silent.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")  # sklearn's OpenMP threads per worker (set before imports)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket.replay import baseline, run  # noqa: E402
from fpl_rocket.replay import inputs as I  # noqa: E402
from fpl_rocket.replay.policies import POLICY_NAMES  # noqa: E402

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
PHASES = ("forecasts", "seasons", "leak", "report")
CACHE = I.INPUTS_DIR / "forecast_cache"

_STORE: dict[str, I.HistoryStore] = {}


def _store(path: str) -> I.HistoryStore:
    if path not in _STORE:
        _STORE[path] = I.HistoryStore(path)
    return _STORE[path]


def _forecast_task(season: str, gw: int, history: str, cache: str) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    inp = I.load_inputs(season, gw, _store(history))
    run.forecasts_for(inp, cache)
    return season, gw, time.perf_counter() - t0


def _logger(path: Path):
    def log(msg: str) -> None:
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return log


def _season_task(season, out, history, cache, policies, gws, log_path) -> str:
    log = _logger(Path(log_path))
    run.run_season(
        season, out, _store(history), policy_names=tuple(policies), cache_dir=cache, gws=gws, log=log
    )
    return season


def _leak_task(season, out, history, policies, gws, log_path) -> str:
    log = _logger(Path(log_path))
    run.leak_test(season, out, history, policy_names=tuple(policies), gws=gws, log=log)
    return season


def _verify_task(season, out, history, cache, policies, log_path) -> str:
    log = _logger(Path(log_path))
    run.run_season(
        season,
        out,
        _store(history),
        policy_names=tuple(policies),
        cache_dir=cache,
        log=log,
        require_cache=True,
    )
    return season


def verify_baseline(name: str, seasons: list[str], policies: list[str], out: Path, workers: int) -> int:
    """Replay the chosen seasons / policies from the caches (never rebuilt: a cache miss stops the run) and
    assert that the canonical packages, decisions and score rows are byte-identical to the frozen baseline
    (fpl_rocket.replay.baseline). Returns 0 when every pair is identical."""
    rec = baseline.load(name)
    log = _logger(out / "verify.log")
    history = I.INPUTS_DIR / I.HISTORY_FILE
    log(
        f"verify {name} (commit {rec['commit'][:7]}, run {rec['run']}): seasons {seasons}; "
        f"policies {policies}"
    )
    if baseline.sha256_file(history) != rec["points_history_sha256"]:
        log("FAIL: the points history differs from the baseline's (sha256)")
        return 1
    fcr = baseline.forecast_cache_record(CACHE, seasons)
    for s in seasons:
        if fcr["per_season"][s] != rec["forecast_cache"]["per_season"][s]:
            log(f"FAIL: the forecast cache of {s} differs from the baseline's")
            return 1
    log("inputs: points history and forecast cache identical to the baseline's")
    with ProcessPoolExecutor(max_workers=min(workers, len(seasons))) as pool:
        futs = [
            pool.submit(
                _verify_task, s, str(out), str(history), str(CACHE), policies, str(out / "verify.log")
            )
            for s in seasons
        ]
        for f in as_completed(futs):
            log(f"replayed {f.result()}")
    ref_dir = I.INPUTS_DIR.parent / "replays" / rec["run"]
    failures = 0
    for s in seasons:
        for p in policies:
            got = baseline.log_record(out / s / f"{p}.jsonl")
            want = rec["logs"][s]["policies"][p]
            keys = ("packages", "decisions", "scores", "gws", "total_net")
            same = all(got[k] == want[k] for k in keys)
            detail = ""
            if not same:
                failures += 1
                ref = ref_dir / s / f"{p}.jsonl"
                if ref.exists():
                    cmp = baseline.compare_logs(ref, out / s / f"{p}.jsonl")
                    detail = "; ".join(f"{k} differ in GWs {cmp[k]['different']}" for k in keys[:3])
            log(
                f"{s} {p}: {'IDENTICAL' if same else 'DIFFERENT'} "
                f"({got['gws']} GWs, total {got['total_net']}, "
                f"packages {got['packages'][:12]} vs {want['packages'][:12]}) {detail}"
            )
    log(f"verify {name}: {'PASS' if not failures else f'FAIL ({failures} pair(s))'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=list(SEASONS))
    ap.add_argument("--policies", nargs="+", default=list(POLICY_NAMES), choices=POLICY_NAMES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gws", nargs="+", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--phases", nargs="+", default=list(PHASES), choices=PHASES)
    ap.add_argument("--no-rebuild-history", action="store_true")
    ap.add_argument(
        "--verify-baseline",
        default=None,
        metavar="NAME",
        help="replay --seasons x --policies from the caches and assert byte-identical canonical packages "
        "against replay/baselines/NAME.json (nothing is rebuilt)",
    )
    args = ap.parse_args(argv)
    if args.verify_baseline:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = Path(args.out or f"data/replays/verify_{args.verify_baseline}_{stamp}")
        out.mkdir(parents=True, exist_ok=True)
        return verify_baseline(args.verify_baseline, args.seasons, args.policies, out, args.workers)
    out = Path(args.out or f"data/replays/{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    log = _logger(log_path)
    history = I.INPUTS_DIR / I.HISTORY_FILE
    if not args.no_rebuild_history or not history.exists():
        history = I.build_points_history()
    log(f"out {out}; seasons {args.seasons}; policies {args.policies}; phases {args.phases}")
    if "forecasts" in args.phases:
        tasks = [(s, g) for s in args.seasons for g in run.season_gws(s, gws=args.gws)]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_forecast_task, s, g, str(history), str(CACHE)) for s, g in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                s, g, sec = f.result()
                if i % 10 == 0 or i == len(futs):
                    log(
                        f"forecasts {i}/{len(futs)} ({s} GW{g} {sec:.0f} s; {time.perf_counter() - t0:.0f} s)"
                    )
    failed: list[str] = []
    if "seasons" in args.phases:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.seasons))) as pool:
            futs = [
                pool.submit(
                    _season_task,
                    s,
                    str(out),
                    str(history),
                    str(CACHE),
                    args.policies,
                    args.gws,
                    str(log_path),
                )
                for s in args.seasons
            ]
            for f in as_completed(futs):
                try:
                    log(f"season {f.result()} done")
                except Exception as exc:  # noqa: BLE001 - one season's failure is logged loudly, the rest go on
                    failed.append(repr(exc))
                    log(f"SEASON FAILED: {exc!r}")
    if "leak" in args.phases:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.seasons))) as pool:
            futs = [
                pool.submit(_leak_task, s, str(out), str(history), args.policies, args.gws, str(log_path))
                for s in args.seasons
            ]
            for f in as_completed(futs):
                try:
                    log(f"leak test {f.result()} done")
                except Exception as exc:  # noqa: BLE001
                    failed.append(repr(exc))
                    log(f"LEAK TEST FAILED: {exc!r}")
    if "report" in args.phases:
        from fpl_rocket.replay import report

        path = report.write_report(out, args.seasons, args.policies)
        log(f"report {path}")
    if failed:
        log(f"{len(failed)} failure(s): {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
