"""Step 3i: the scored snapshot is the LAST bundle published between 12 h and 30 min before the deadline.

Window boundaries, supersede (a second pre-deadline bundle for the same GW replaces the first; the GW is
recomputed from the EARLIER GW's state; the superseded ids are recorded), the trajectories start at the first
GW with a snapshot, not_started weeks never count, and the scorers never duplicate or rewrite existing rows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from fpl_rocket import registry
from fpl_rocket import trajectories as tr
from fpl_rocket.bundle import PRE_DEADLINE_WINDOW_MIN, deadline_info
from test_trajectories import GWS, _Client, _inp, _live
from test_trajectories import world as _world_fixture

world = _world_fixture  # the synthetic league fixture of test_trajectories
NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def _events(minutes_left: float) -> list[dict]:
    deadline = NOW + timedelta(minutes=minutes_left)
    return [
        {"id": 9, "is_next": False, "deadline_time": "2026-10-31T11:00:00Z"},
        {"id": 10, "is_next": True, "deadline_time": deadline.isoformat().replace("+00:00", "Z")},
    ]


# ------------------------------------------------------------------------------------------ window
@pytest.mark.parametrize(
    "minutes_left, expected",
    [
        (12 * 60 + 1, False),
        (12 * 60, True),  # 12 h: inclusive
        (12 * 60 - 1, True),
        (390, True),
        (31, True),
        (30, True),  # 30 min: inclusive
        (29.5, False),
        (10, False),
        (-5, False),
    ],
)
def test_pre_deadline_window_is_12_hours_to_30_minutes_inclusive(minutes_left, expected):
    info = deadline_info(_events(minutes_left), NOW)
    assert PRE_DEADLINE_WINDOW_MIN == (30, 720)
    assert info["pre_deadline"] is expected and info["next_event"] == 10
    assert info["lead_time_minutes"] == pytest.approx(minutes_left, abs=0.05)


def test_no_next_event_has_no_lead_time():
    info = deadline_info([{"id": 38, "is_next": False}], NOW)
    assert info["pre_deadline"] is False and info["lead_time_minutes"] is None


def test_lead_time_from_manifest_fields():
    assert registry.lead_time_minutes({"lead_time_minutes": 41.5}) == 41.5
    m = {"created_at": "2026-11-06T12:00:00+00:00", "next_deadline_time": "2026-11-06T18:30:00+00:00"}
    assert registry.lead_time_minutes(m) == 390.0
    assert registry.lead_time_minutes({}) is None


# ------------------------------------------------------------------------------------------ supersede
def _bundle_inp(world, gw: int, bundle_id: str, src, xp=None, lead: float = 300.0) -> tr.StepInputs:
    inp = _inp(world, gw, xp)
    inp.bundle_id = bundle_id
    inp.lead_time_minutes = lead
    inp.superseded_bundles = tuple(registry.superseded_bundles(gw, bundle_id, str(src)))
    return inp


def _publish(src, gw: int, state: dict, plans: dict) -> None:
    """What the workflow does for the trajectories assets: overwrite them in deadline-gw<gw> (--clobber)."""
    tr.write_files(src / registry.deadline_tag(gw), state, plans)


def test_second_pre_deadline_bundle_replaces_the_first_from_the_earlier_gws_state(world, tmp_path):
    src = tmp_path / "releases"
    s1, p1 = tr.step(None, _bundle_inp(world, 1, "20261031T080000Z", src))
    _publish(src, 1, s1, p1)
    # first pre-deadline bundle for GW2
    k, prev = tr.latest_state(str(src), 2, start=1)
    assert k == 1
    a_inp = _bundle_inp(world, 2, "20261107T060000Z", src, lead=420.0)
    assert a_inp.superseded_bundles == ()
    sa, pa = tr.step(prev, a_inp)
    _publish(src, 2, sa, pa)
    # second bundle for GW2 (fresher forecasts): the starting state is still GW1's, never GW2's own upload
    k, prev2 = tr.latest_state(str(src), 2, start=1)
    assert k == 1 and prev2 == prev
    _, xp_df, *_ = world
    fresher = xp_df.assign(xp=xp_df["xp"] * 1.1)
    b_inp = _bundle_inp(world, 2, "20261107T170000Z", src, xp=fresher, lead=40.0)
    assert b_inp.superseded_bundles == ("20261107T060000Z",)
    sb, pb = tr.step(prev2, b_inp)
    _publish(src, 2, sb, pb)
    final = json.loads((src / "deadline-gw2" / tr.STATE_FILE).read_text(encoding="utf-8"))
    plans = json.loads((src / "deadline-gw2" / tr.PLANS_FILE).read_text(encoding="utf-8"))
    assert final["bundle_id"] == plans["bundle_id"] == "20261107T170000Z"
    assert final["previous_gw"] == 1 and final["gw"] == 2
    assert final["lead_time_minutes"] == plans["lead_time_minutes"] == 40.0
    assert final["superseded_bundles"] == plans["superseded_bundles"] == ["20261107T060000Z"]
    # the GW2 decision is recomputed from GW1's state: same start squad, one GW2 line-up per policy
    assert prev == json.loads(json.dumps(s1, sort_keys=True, default=str))
    for name, t in final["trajectories"].items():
        assert set(t["lineups"]) == {"1", "2"}
        assert t["transfers_total"] == len(pb["trajectories"][name]["transfers_in"])
    # the first upload's GW2 choices were made from the same GW1 state (not from each other)
    assert sa["previous_gw"] == final["previous_gw"] == 1
    assert sa["trajectories"]["hold"]["squad"] == final["trajectories"]["hold"]["squad"]
    # a third bundle lists both earlier ones (sorted, the current one excluded)
    c_inp = _bundle_inp(world, 2, "20261107T175000Z", src, lead=30.0)
    assert c_inp.superseded_bundles == ("20261107T060000Z", "20261107T170000Z")


def test_registry_meta_records_lead_time_and_superseded(tmp_path):
    from test_registry import _world

    _, bootstrap, xp_df, state, baseline, manifest = _world()
    src = tmp_path / "releases"
    tag = src / registry.deadline_tag(1)
    tag.mkdir(parents=True)
    (tag / "manifest.json").write_text(json.dumps({"bundle_id": "20261030T080000Z"}), encoding="utf-8")
    (tag / registry.META_FILE).write_text(
        json.dumps({"bundle_id": "20261030T080000Z", "superseded_bundles": ["20261030T020000Z"]}),
        encoding="utf-8",
    )
    sup = registry.superseded_bundles(1, manifest["bundle_id"], str(src))
    assert sup == ["20261030T020000Z", "20261030T080000Z"]
    assert registry.superseded_bundles(5, "x", str(src)) == []  # no release yet
    _, _, meta = registry.build_registry(
        manifest=manifest,
        bootstrap=bootstrap,
        xp=xp_df,
        baseline=baseline,
        states={registry.REGISTRY_ENTRY_IDS[0]: state},
        superseded=sup,
    )
    assert meta["lead_time_minutes"] == 60.0  # created 10:00, deadline 11:00
    assert meta["superseded_bundles"] == sup and meta["bundle_id"] == manifest["bundle_id"]


def test_superseded_lookup_never_blocks_the_snapshot():
    def boom(source, gw, name):
        raise ConnectionError("offline")

    assert registry.superseded_bundles(10, "b", "https://example.invalid", fetch_json=boom) == []


# ------------------------------------------------------------------------------------------ start + scoring
def test_trajectories_start_at_the_first_gw_with_a_snapshot_and_not_started_never_counts(world, tmp_path):
    pool, _, boot, _, _ = world
    src = tmp_path / "releases"
    # no deadline-gw1 release (like GW9): the first pre-deadline run for GW2 finds no earlier state
    assert tr.latest_state(str(src), 2, start=1) is None
    s2, p2 = tr.step(None, _bundle_inp(world, 2, "20261106T120000Z", src, lead=390.0))
    assert s2["created_gw"] == 2 and s2["gw"] == 2 and s2["previous_gw"] is None
    _publish(src, 2, s2, p2)
    client = _Client(boot, {g: _live(pool, set()) for g in GWS})
    csv = tmp_path / "trajectories_scores.csv"
    rows1 = tr.score_and_append(1, csv, client, str(src), start=1)
    assert {r["status"] for r in rows1} == {"not_started"}
    assert tr.summary(tr.read_scores(csv)) is None  # only a not_started week: nothing to chart
    rows2 = tr.score_and_append(2, csv, client, str(src), start=1)
    assert {r["status"] for r in rows2} == {"ok"}
    assert all(r["lead_time_minutes"] == 390.0 and r["bundle_id"] == "20261106T120000Z" for r in rows2)
    for r in rows2:
        assert r["cumulative_net"] == r["net"] and r["transfers_total"] == 0
    s = tr.summary(pd.read_csv(csv))
    assert s["gw"] == 2 and list(s["cumulative"].index) == [2]
    # a later run never duplicates or rewrites rows (GW1 not_started stays as written)
    before = pd.read_csv(csv)
    tr.append_scores(csv, rows1 + rows2)
    tr.score_and_append(1, csv, client, str(src), start=1)
    after = pd.read_csv(csv)
    assert len(after) == len(before) == 2 * len(tr.TRAJECTORIES)
    pd.testing.assert_frame_equal(before, after)


def test_registry_scorer_never_duplicates_or_rewrites_rows(tmp_path):
    csv = tmp_path / "registry_scores.csv"
    gw9 = registry.missing_row(9, "no deadline release (no bundle published in the pre-deadline window)")
    registry.append_scores(csv, [gw9])
    before = pd.read_csv(csv)
    registry.append_scores(csv, [registry.missing_row(9, "something else")])
    registry.append_scores(
        csv,
        [
            {"gw": 10, "kind": "xp", "status": "ok", "bundle_id": "B", "lead_time_minutes": 42.0},
            {"gw": 10, "kind": "entry", "entry_id": 7630179, "status": "ok"},
        ],
    )
    registry.append_scores(csv, [{"gw": 10, "kind": "entry", "entry_id": 7630179, "status": "error"}])
    after = pd.read_csv(csv)
    assert len(after) == 3
    pd.testing.assert_frame_equal(
        after[after["gw"] == 9].reset_index(drop=True), before.reset_index(drop=True), check_dtype=False
    )
    assert after.loc[after["kind"] == "entry", "status"].tolist() == ["ok"]
    assert after.loc[(after["gw"] == 10) & (after["kind"] == "xp"), "lead_time_minutes"].tolist() == [42.0]
    assert registry.pending_gws({"events": [{"id": 9, "finished": True}]}, registry.scored_gws(csv)) == []
