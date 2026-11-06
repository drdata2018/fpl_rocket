"""Fixture schedule AS KNOWN at the issue time T, and the as-played (final) schedule.

As known: the community archive (github.com/vaastav/Fantasy-Premier-League) commits data/<season>/fixtures.csv
repeatedly through each season; for every GW the LAST commit with commit time <= T is read (the step-3g method
of fpl_rocket.fixtures_known, re-anchored from "deadline" to the issue time T = deadline - 2 h). Caveat: a git
commit time is when the commit was made, not when it was pushed; the archive's scraper pushes as it commits,
but this is not provable from git alone. The archive is STALE for the current season (last push 2026-10-22):
the snapshot age at each T is recorded so a stale schedule is visible, never silently trusted.

As played (retrospective, used only for first_kickoff_utc and for coverage statistics): the archive's latest
fixtures.csv for finished seasons, the live FPL fixtures endpoint for the current season.
"""

from __future__ import annotations

import json

import pandas as pd

from fpl_rocket import fixtures_known as fk

VAASTAV_URL = "https://github.com/vaastav/Fantasy-Premier-League"


def parse_csv_fixtures(raw: bytes | str) -> pd.DataFrame:
    return fk.parse_fixtures(raw)


def parse_api_fixtures(raw: bytes | str) -> pd.DataFrame:
    """FPL /api/fixtures/ JSON -> (fixture, event, team_h, team_a, kickoff)."""
    data = json.loads(raw)
    df = pd.DataFrame(
        {
            "fixture": [f.get("id") for f in data],
            "event": [f.get("event") for f in data],
            "team_h": [f.get("team_h") for f in data],
            "team_a": [f.get("team_a") for f in data],
            "kickoff": [f.get("kickoff_time") for f in data],
        }
    )
    for c in ("fixture", "event", "team_h", "team_a"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True, errors="coerce")
    return df.dropna(subset=["fixture", "team_h", "team_a"]).reset_index(drop=True)


def first_kickoffs(final: pd.DataFrame) -> dict[int, pd.Timestamp]:
    f = final.dropna(subset=["event", "kickoff"])
    return {int(g): k for g, k in f.groupby(f["event"].astype(int))["kickoff"].min().items()}


def commit_at_or_before(commits: list[fk.Commit], t: pd.Timestamp) -> fk.Commit | None:
    before = [c for c in commits if c.at <= t]
    return before[-1] if before else None


def as_known_rows(
    season: str,
    gw: int,
    issue_time: pd.Timestamp,
    commit: fk.Commit | None,
    snap: pd.DataFrame | None,
    final: pd.DataFrame | None,
) -> dict:
    """Coverage row for one GW: snapshot age at T and how many fixtures of GW t differ from as played."""
    row = {
        "season": season,
        "gw": gw,
        "issue_time_utc": issue_time,
        "snapshot_sha": None if commit is None else commit.sha,
        "snapshot_at": None if commit is None else commit.at,
        "age_days": None if commit is None else (issue_time - commit.at).total_seconds() / 86400,
    }
    if snap is not None and final is not None:
        row["fixtures_gw_known"] = int((snap["event"] == gw).sum())
        row["differ_gw_t"] = len(fk.differing(snap, final, [gw]))
    return row
