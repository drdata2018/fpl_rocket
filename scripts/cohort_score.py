"""Score the frozen prospective cohorts (fpl_rocket.cohorts) for every finished GW after the freeze.

Usage: uv run python scripts/cohort_score.py --dir ../spike-log/registry/cohorts
           [--stage data/raw/cohorts/stage]

For each cohort_<id>_manifest.json in --dir: verifies the cohort file's sha256 (refuses to run on a mismatch),
lists the finished and data-checked GWs after the freeze GW that have no row yet, fetches entry/{id}/history/
ONCE per member (it covers every pending GW) into --stage (resumable), and only when every member was fetched
appends the rows to cohort_scores.csv and rewrites cohort_summary.csv (atomic replace). A partial fetch
publishes nothing (exit 1; the next run resumes). Idempotent: existing (freeze_id, gw, cohort, entry_id) rows
are never duplicated or rewritten.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket import cohorts  # noqa: E402
from fpl_rocket.api import FPLClient  # noqa: E402


def _say(msg: str) -> None:
    print(f"{datetime.now(UTC):%H:%M:%S} {msg}", flush=True)


def run(cohorts_dir: Path, stage_root: Path, client, retry_pause: float = 60.0) -> int:
    try:
        frozen = cohorts.load_frozen(cohorts_dir)
    except cohorts.CohortTampered as exc:
        _say(f"REFUSED: {exc}")
        return 2
    if not frozen:
        _say(f"no cohort manifest in {cohorts_dir}")
        return 0
    boot = client.bootstrap()
    scores = cohorts.read_scores(cohorts_dir / cohorts.SCORES_NAME)
    failed_any = False
    for manifest, doc in frozen:
        fid, fgw = str(manifest["freeze_id"]), int(manifest["last_finished_gw"])
        gws = cohorts.pending_gws(boot, fgw, cohorts.scored_gws(scores, fid))
        if not gws:
            _say(f"freeze {fid}: nothing to score")
            continue
        members = doc["cohorts"]
        ids = {int(m["entry_id"]) for ms in members.values() for m in ms}
        stage = stage_root / f"{fid}_gw{max(gws)}"
        _say(f"freeze {fid}: GWs {gws}, {len(ids)} members, staging in {stage}")
        hist, failed = cohorts.stage_histories(client, ids, stage, progress=_say)
        if failed:  # one more pass over the failures after a pause
            time.sleep(retry_pause)
            more, failed = cohorts.stage_histories(client, failed, stage)
            hist.update(more)
        if failed:
            failed_any = True
            _say(f"freeze {fid}: {len(failed)} members not fetched (e.g. {failed[:5]}); nothing published")
            continue
        rows = cohorts.score_rows(fid, members, hist, gws)
        added, scores = cohorts.publish(cohorts_dir, rows)
        _say(f"freeze {fid}: published {added} rows for GWs {gws}")
    return int(failed_any)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="registry/cohorts directory of the spike-log checkout")
    ap.add_argument("--stage", default="data/raw/cohorts/stage")
    ap.add_argument("--min-interval", type=float, default=1.0)
    args = ap.parse_args(argv)
    session = cohorts.PoliteSession(min_interval=args.min_interval)
    client = FPLClient(session=session, backoff=2.0)
    t0 = time.monotonic()
    code = run(Path(args.dir), Path(args.stage), client)
    _say(
        f"{session.n_requests} requests in {time.monotonic() - t0:.0f} s, by status {dict(session.by_status)}"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
