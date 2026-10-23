"""Deadline-known fixtures: git log parsing, deadlines, snapshot choice, schedule differences, coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_rocket import fixtures_known as fk

CSV = """code,event,finished,id,kickoff_time,team_a,team_h,stats
1,1,True,1,2024-08-16T19:00:00Z,2,1,[]
2,1,True,2,2024-08-17T14:00:00Z,4,3,[]
3,2,True,3,2024-08-24T14:00:00Z,1,2,[]
4,{e4},True,4,2024-08-24T16:30:00Z,3,4,[]
"""


def test_parse_log_and_snapshot_choice():
    log = f"{'b' * 40},2024-08-20T10:00:00+02:00\n{'a' * 40},2024-08-02T10:16:31+02:00\n\n"
    commits = fk.parse_log(log)
    assert [c.sha[0] for c in commits] == ["a", "b"]  # oldest first
    assert commits[1].at == pd.Timestamp("2024-08-20T08:00:00Z")
    assert fk.snapshot_for(commits, pd.Timestamp("2024-08-16T17:30:00Z")).sha == "a" * 40
    assert fk.snapshot_for(commits, pd.Timestamp("2024-08-20T08:00:00Z")).sha == "a" * 40  # strictly before
    assert fk.snapshot_for(commits, pd.Timestamp("2024-08-01T00:00:00Z")) is None
    with pytest.raises(ValueError):
        fk.parse_log("not a commit,2024-01-01T00:00:00Z")


def test_deadlines_differences_and_coverage():
    final = fk.parse_fixtures(CSV.format(e4=2))
    assert list(final.columns) == list(fk.COLUMNS)
    dl = fk.deadlines(final)
    assert dl == {
        1: pd.Timestamp("2024-08-16T17:30:00Z"),
        2: pd.Timestamp("2024-08-24T12:30:00Z"),
    }
    early = fk.parse_fixtures(CSV.format(e4=""))  # fixture 4 not yet scheduled
    d = fk.differing(early, final, [2])
    assert list(d["fixture"]) == [4]
    assert fk.differing(final, final, [1, 2]).empty
    commits = fk.parse_log(f"{'a' * 40},2024-08-10T00:00:00Z\n{'b' * 40},2024-08-21T00:00:00Z")
    snaps = {1: (commits[0], early), 2: (commits[1], final)}
    rows = pd.DataFrame(fk.coverage_rows("2024-25", commits, final, snaps))
    assert list(rows["current"]) == [True, True] and list(rows["differ_horizon"]) == [1, 0]
    assert list(rows["differ_gw_t"]) == [0, 0]
    s = fk.summarise_coverage(rows).iloc[0]
    assert s["with_snapshot"] == 2 and s["deadlines_horizon_differs"] == 1
