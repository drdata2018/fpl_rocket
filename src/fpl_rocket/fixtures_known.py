"""Deadline-known fixtures: the schedule as it was known before each GW deadline (removes the fixture leak of
the retrospective replays, data/backtests/*_matched_audit.md item (c)).

Source: the community archive repository (github.com/vaastav/Fantasy-Premier-League) commits
data/<season>/fixtures.csv repeatedly through each season, so its git history holds the schedule as known on
each date. scripts/build_fixtures_as_known.py lists those commits (`git log --format=%H,%cI -- <path>`),
takes for every GW deadline the LAST commit strictly before it and reads that version (`git show sha:path`).
The repository is untrusted data: only git's own read commands are run against it, and the CSV is parsed by
pandas (nothing from it is executed).

Deadline of GW t = the first kickoff of GW t in the archive's final (as-played) fixtures minus
DEADLINE_LEAD (90 minutes, FPL's usual lead).

Coverage report per season: deadlines with any snapshot before them, deadlines with a CURRENT snapshot (one
committed after the previous GW's deadline, i.e. during the week before the decision) and the snapshot age;
fixtures of GWs t..t+H-1 (H = 3, the planner's horizon) whose GW in the snapshot differs from the GW they
were played in (moved, or unscheduled in the snapshot), for GW t alone and for the whole horizon.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

DEADLINE_LEAD = pd.Timedelta(minutes=90)
COLUMNS = ("fixture", "event", "team_h", "team_a", "kickoff")
HORIZON = 3


@dataclass(frozen=True)
class Commit:
    sha: str
    at: pd.Timestamp  # commit time, UTC


def parse_log(text: str) -> list[Commit]:
    """`git log --format=%H,%cI` output -> commits, oldest first."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        sha, at = line.split(",", 1)
        if len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha):
            raise ValueError(f"not a commit line: {line!r}")
        out.append(Commit(sha, pd.Timestamp(at).tz_convert("UTC")))
    return sorted(out, key=lambda c: c.at)


def parse_fixtures(raw: bytes | str) -> pd.DataFrame:
    """A fixtures.csv version -> (fixture, event, team_h, team_a, kickoff); event NaN = not scheduled."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    df = pd.read_csv(io.StringIO(text), usecols=["id", "event", "team_h", "team_a", "kickoff_time"])
    out = pd.DataFrame(
        {
            "fixture": pd.to_numeric(df["id"], errors="coerce").astype("Int64"),
            "event": pd.to_numeric(df["event"], errors="coerce").astype("Int64"),
            "team_h": pd.to_numeric(df["team_h"], errors="coerce").astype("Int64"),
            "team_a": pd.to_numeric(df["team_a"], errors="coerce").astype("Int64"),
            "kickoff": pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce"),
        }
    )
    return out.dropna(subset=["fixture", "team_h", "team_a"]).reset_index(drop=True)


def deadlines(final: pd.DataFrame, lead: pd.Timedelta = DEADLINE_LEAD) -> dict[int, pd.Timestamp]:
    """{gw: deadline} = first as-played kickoff of the GW minus `lead`."""
    f = final.dropna(subset=["event", "kickoff"])
    first = f.groupby(f["event"].astype(int))["kickoff"].min()
    return {int(g): k - lead for g, k in first.items()}


def snapshot_for(commits: list[Commit], deadline: pd.Timestamp) -> Commit | None:
    """The last commit strictly before the deadline (None when the history starts after it)."""
    before = [c for c in commits if c.at < deadline]
    return before[-1] if before else None


def differing(snapshot: pd.DataFrame, final: pd.DataFrame, gws: Iterable[int]) -> pd.DataFrame:
    """Fixtures played in, or scheduled (in the snapshot) for, one of `gws` whose GW differs between the
    snapshot and the final schedule (event NaN in the snapshot = unscheduled)."""
    gws = {int(g) for g in gws}
    m = final[["fixture", "event"]].merge(
        snapshot[["fixture", "event"]], on="fixture", how="outer", suffixes=("_played", "_known")
    )
    played = m["event_played"].astype("Float64")
    known = m["event_known"].astype("Float64")
    touches = played.isin(gws).fillna(False) | known.isin(gws).fillna(False)
    differs = (played != known).fillna(True)
    return m[touches & differs].reset_index(drop=True)


def coverage_rows(
    season: str,
    commits: list[Commit],
    final: pd.DataFrame,
    snapshots: dict[int, tuple[Commit | None, pd.DataFrame | None]],
    horizon: int = HORIZON,
) -> list[dict]:
    """One row per GW deadline: snapshot sha / time / age, current (after the previous deadline), and the
    number of fixtures whose GW differs from as played (GW t; GWs t..t+H-1)."""
    dl = deadlines(final)
    gws = sorted(dl)
    rows = []
    for i, g in enumerate(gws):
        commit, snap = snapshots.get(g, (None, None))
        prev_deadline = dl[gws[i - 1]] if i else None
        row = {
            "season": season,
            "gw": g,
            "deadline": dl[g],
            "snapshot_sha": commit.sha if commit else None,
            "snapshot_at": commit.at if commit else None,
            "age_days": (dl[g] - commit.at).total_seconds() / 86400 if commit else None,
            "current": bool(commit and (prev_deadline is None or commit.at > prev_deadline)),
        }
        if snap is not None:
            horizon_gws = [x for x in gws if x >= g][:horizon]
            row["differ_gw_t"] = len(differing(snap, final, [g]))
            row["differ_horizon"] = len(differing(snap, final, horizon_gws))
        rows.append(row)
    return rows


def summarise_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for season, g in rows.groupby("season", sort=True):
        with_snap = g["snapshot_sha"].notna()
        out.append(
            {
                "season": season,
                "deadlines": len(g),
                "with_snapshot": int(with_snap.sum()),
                "current_snapshot": int(g["current"].sum()),
                "median_age_days": float(g.loc[with_snap, "age_days"].median()) if with_snap.any() else None,
                "max_age_days": float(g.loc[with_snap, "age_days"].max()) if with_snap.any() else None,
                "deadlines_gw_t_differs": int((g.get("differ_gw_t", pd.Series(dtype=float)) > 0).sum()),
                "fixtures_differ_gw_t": int(g.get("differ_gw_t", pd.Series(dtype=float)).fillna(0).sum()),
                "deadlines_horizon_differs": int((g.get("differ_horizon", pd.Series(dtype=float)) > 0).sum()),
                "fixtures_differ_horizon": int(
                    g.get("differ_horizon", pd.Series(dtype=float)).fillna(0).sum()
                ),
            }
        )
    return pd.DataFrame(out)
