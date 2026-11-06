"""Freeze the prospective cohorts of real managers ONCE (fpl_rocket.cohorts; Plan v3 Phase 1e).

Usage: uv run python scripts/cohort_freeze.py --seed 20261106 --out ../spike-log [--cache data/raw/cohorts]

Draws the random cohort (active managers, uniform over entry ids) and the top-10k cohort (uniform positions in
the overall league), records each member's state after the last finished GW, and writes
<out>/registry/cohorts/cohort_<YYYYMMDD>.json + cohort_<YYYYMMDD>_manifest.json (sha256 of the cohort file).
The top-10k standings are fetched first (they move once the next GW is live). Polite: browser User-Agent,
>= 1 s between requests, retries with backoff on 429/403/5xx, a 30 s cool-down after a 429/403. Resumable: the
bootstrap facts and every answer are cached under --cache/seed_<seed>/ (gitignored; ids and numbers only, no
names), so a rerun continues where it stopped and reproduces the same cohorts. Commit the two files to the
data-only `spike-log` branch afterwards; they are never rewritten.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket import cohorts  # noqa: E402
from fpl_rocket.api import FPLClient  # noqa: E402


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _say(msg: str) -> None:
    print(f"{datetime.now(UTC):%H:%M:%S} {msg}", flush=True)


def _dist(values: list[int]) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {}
    q = statistics.quantiles(v, n=4, method="inclusive")
    return {"n": len(v), "min": v[0], "p25": q[0], "median": q[1], "p75": q[2], "max": v[-1]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="root of the spike-log checkout")
    ap.add_argument("--cache", default="data/raw/cohorts")
    ap.add_argument("--size", type=int, default=cohorts.COHORT_SIZE)
    ap.add_argument("--min-interval", type=float, default=1.0)
    ap.add_argument("--max-candidates", type=int, default=cohorts.MAX_RANDOM_CANDIDATES)
    args = ap.parse_args(argv)

    cache = Path(args.cache) / f"seed_{args.seed}"
    cache.mkdir(parents=True, exist_ok=True)
    session = cohorts.PoliteSession(min_interval=args.min_interval)
    client = FPLClient(session=session, backoff=2.0)
    run_start, t0 = _now(), time.monotonic()

    state_path = cache / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _say(f"resuming freeze {state['freeze_id']} (started {state['started_at']})")
    else:
        boot = client.bootstrap()
        gw = cohorts.last_finished_gw(boot)
        ev = {int(e["id"]): e for e in boot["events"]}
        nxt = ev.get(gw + 1, {})
        state = {
            "freeze_id": datetime.now(UTC).strftime("%Y%m%d"),
            "seed": args.seed,
            "started_at": run_start,
            "total_players": int(boot["total_players"]),
            "last_finished_gw": gw,
            "last_finished_data_checked": bool(ev[gw].get("data_checked")),
            "next_gw": gw + 1 if nxt else None,
            "next_deadline": nxt.get("deadline_time"),
            "runs": [],
        }
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    if state["seed"] != args.seed:
        raise SystemExit("cache seed differs from --seed")
    gw = int(state["last_finished_gw"])
    _say(
        f"freeze GW{gw}, next GW{state['next_gw']} deadline {state['next_deadline']}, "
        f"total_players {state['total_players']}"
    )

    def _log_run(status: str) -> None:
        state["runs"].append(
            {
                "started_at": run_start,
                "ended_at": _now(),
                "seconds": round(time.monotonic() - t0, 1),
                "requests": session.n_requests,
                "by_status": dict(session.by_status),
                "status": status,
            }
        )
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")

    try:
        top, top_stats = cohorts.freeze_top(
            client,
            args.seed,
            gw,
            [
                cohorts.JsonlCache(cache / f"top10k_pages_s{s}.jsonl")
                for s in range(1, cohorts.TOP_SNAPSHOTS + 1)
            ],
            cohorts.JsonlCache(cache / "top10k_histories.jsonl"),
            size=args.size,
            progress=_say,
        )
        rnd, rnd_stats = cohorts.freeze_random(
            client,
            args.seed,
            state["total_players"],
            gw,
            cohorts.JsonlCache(cache / "random_candidates.jsonl"),
            size=args.size,
            max_candidates=args.max_candidates,
            progress=_say,
        )
    except BaseException as exc:
        _log_run(f"interrupted: {type(exc).__name__}: {exc}")
        _say(f"stopped ({type(exc).__name__}: {exc}); rerun to resume")
        raise
    _log_run("complete")

    runs = state["runs"]
    by_status: dict[str, int] = {}
    for r in runs:
        for k, v in r["by_status"].items():
            by_status[k] = by_status.get(k, 0) + v
    doc = {
        "freeze_id": state["freeze_id"],
        "freeze_gw": gw,
        "seed": args.seed,
        "cohorts": {cohorts.RANDOM: rnd, cohorts.TOP10K: top},
    }
    manifest = {
        "seed": args.seed,
        "seed_scheme": cohorts.SEED_SCHEME,
        "freeze_started_utc": state["started_at"],
        "freeze_time_utc": _now(),
        "total_players": state["total_players"],
        "last_finished_gw": gw,
        "last_finished_data_checked": state["last_finished_data_checked"],
        "next_gw": state["next_gw"],
        "next_deadline": state["next_deadline"],
        "rules": cohorts.RULES,
        "units": "value and bank in tenths of a million (FPL API units)",
        "counts": {cohorts.RANDOM: rnd_stats, cohorts.TOP10K: top_stats},
        "at_freeze": {
            name: {
                "overall_rank": _dist([m["overall_rank"] for m in ms]),
                "total_points": _dist([m["total_points"] for m in ms]),
            }
            for name, ms in ((cohorts.RANDOM, rnd), (cohorts.TOP10K, top))
        },
        "requests": {
            "total": sum(r["requests"] for r in runs),
            "by_status": by_status,
            "seconds": round(sum(r["seconds"] for r in runs), 1),
            "runs": runs,
        },
    }
    cpath, mpath, sha = cohorts.write_freeze(args.out, state["freeze_id"], doc, manifest)
    _say(f"wrote {cpath} (sha256 {sha}) and {mpath}")
    print(json.dumps({k: manifest[k] for k in ("counts", "at_freeze", "requests")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
