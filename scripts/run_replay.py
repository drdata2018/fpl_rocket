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

from fpl_rocket.replay import inputs as I  # noqa: E402
from fpl_rocket.replay import run  # noqa: E402
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=list(SEASONS))
    ap.add_argument("--policies", nargs="+", default=list(POLICY_NAMES), choices=POLICY_NAMES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gws", nargs="+", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--phases", nargs="+", default=list(PHASES), choices=PHASES)
    ap.add_argument("--no-rebuild-history", action="store_true")
    args = ap.parse_args(argv)
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
