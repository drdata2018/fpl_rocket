"""Deadline-known fixture snapshots from the community archive's git history (fpl_rocket.fixtures_known).

Usage:
    git clone --filter=blob:none --no-checkout \
        https://github.com/vaastav/Fantasy-Premier-League data/vaastav_meta
    uv run python -I scripts/build_fixtures_as_known.py --repo data/vaastav_meta --seasons 2024-25 2025-26

For each season: `git log --format=%H,%cI -- data/<season>/fixtures.csv` lists the commits; for every GW
deadline (first as-played kickoff - 90 min, from data/archive/<season>/fixtures.csv) the last commit before it
is read with `git show <sha>:data/<season>/fixtures.csv` and written to
data/archive/fixtures_as_known/<season>/gw<N>.parquet (fixture, event, team_h, team_a, kickoff, snapshot_sha,
snapshot_at). Prints and writes the coverage report (coverage.csv / coverage_summary.csv / coverage.txt in the
output folder). The cloned repository is untrusted data: only git's read commands (log, show) are run against
it, with hooks disabled, and the CSVs are parsed by pandas.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpl_rocket import fixtures_known as fk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "--no-pager"]


def git(repo: Path, *args: str) -> bytes:
    res = subprocess.run([*GIT, "-C", str(repo), *args], capture_output=True, timeout=300, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {res.stderr.decode(errors='replace').strip()}")
    return res.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(ROOT / "data" / "vaastav_meta"))
    ap.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    ap.add_argument("--archive", default=str(ROOT / "data" / "archive"))
    ap.add_argument("--out", default=str(ROOT / "data" / "archive" / "fixtures_as_known"))
    args = ap.parse_args()
    repo, out = Path(args.repo), Path(args.out)
    cov = []
    for season in args.seasons:
        path = f"data/{season}/fixtures.csv"
        commits = fk.parse_log(git(repo, "log", "--format=%H,%cI", "--", path).decode())
        final = fk.parse_fixtures((Path(args.archive) / season / "fixtures.csv").read_bytes())
        dl = fk.deadlines(final)
        cache: dict[str, pd.DataFrame] = {}
        snapshots = {}
        folder = out / season
        folder.mkdir(parents=True, exist_ok=True)
        for g, deadline in sorted(dl.items()):
            commit = fk.snapshot_for(commits, deadline)
            if commit is None:
                snapshots[g] = (None, None)
                continue
            if commit.sha not in cache:
                cache[commit.sha] = fk.parse_fixtures(git(repo, "show", f"{commit.sha}:{path}"))
            snap = cache[commit.sha]
            snapshots[g] = (commit, snap)
            snap.assign(snapshot_sha=commit.sha, snapshot_at=commit.at).to_parquet(
                folder / f"gw{g}.parquet", index=False
            )
        rows = fk.coverage_rows(season, commits, final, snapshots)
        cov.extend(rows)
        print(f"{season}: {len(commits)} commits, {len(cache)} snapshots used, {len(dl)} deadlines")
    table = pd.DataFrame(cov)
    summary = fk.summarise_coverage(table)
    table.to_csv(out / "coverage.csv", index=False)
    summary.to_csv(out / "coverage_summary.csv", index=False)
    (out / "coverage.txt").write_text(summary.to_string(index=False) + "\n", encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(
        table[
            [
                "season",
                "gw",
                "deadline",
                "snapshot_at",
                "age_days",
                "current",
                "differ_gw_t",
                "differ_horizon",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
