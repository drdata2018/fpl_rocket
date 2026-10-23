"""Bundle v2: element_history / round_prices tables, lossless bootstrap, manifest flags, app payload."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from bundle_fixture import SNAP, build_snapshot_bundle, offline_client, snapshot_summary_ids
from fpl_rocket.bundle import (
    APP_TABLES,
    BundleError,
    app_shared,
    deadline_info,
    element_history_frame,
    fetch_element_summaries,
    read_bundle,
    read_manifest,
    records_table,
    round_prices_frame,
    round_prices_lookup,
    table_records,
)
from fpl_rocket.state import build_planning_state, fetch_inputs
from fpl_rocket.views import LIVE_STATS, live_frame

OWNER = 7630179


def _snap(name: str):
    return json.loads((SNAP / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return build_snapshot_bundle(tmp_path_factory.mktemp("bundles"))


def _files(path):
    return {p.name: p.read_bytes() for p in path.iterdir() if p.suffix == ".parquet"}


# ------------------------------------------------------------------------------------------ tables
def test_records_table_is_lossless():
    recs = [
        {"id": 1, "chance": None, "news": "", "nested": {"a": [1, 2]}, "mixed": 1, "f": 0.5},
        {"id": 2, "chance": 75, "news": "knock", "nested": None, "mixed": "x", "f": None},
    ]
    t = records_table(recs)
    assert table_records(t) == recs
    assert t.schema.field("chance").type == "int64"  # ints with nulls stay ints


def test_v2_tables_and_manifest(bundle):
    m = read_manifest(bundle)
    tables = read_bundle(bundle)
    assert m["format_version"] == 2 and m["rules_version"].startswith("2026-27")
    assert set(tables) >= {*APP_TABLES, "element_history", "gw_live_explain"}
    assert m["counts"] == {f["table"]: f["rows"] for f in m["files"]}
    assert all(len(f["sha256"]) == 64 for f in m["files"])
    assert m["element_summary_missing"] == [] and m["element_summary_requested"] == 15
    # fixtures carry finished results
    fx = tables["fixtures"]
    assert {"team_h_score", "team_a_score", "finished", "kickoff_time"} <= set(fx.columns)
    assert fx["finished"].sum() > 0 and fx.loc[fx["finished"], "team_h_score"].notna().all()
    # element_history: one row per (element, round, fixture) for each fetched player
    eh = tables["element_history"]
    assert set(eh["element"]) == set(snapshot_summary_ids())
    assert not eh.duplicated([c for c in ("element", "round", "fixture") if c in eh]).any()
    assert list(eh.columns[:2]) == ["element", "round"]


def test_element_history_keeps_all_stats_and_converts_numeric_strings():
    summary = {
        "history": [
            {"element": 9, "fixture": 3, "round": 2, "value": 55, "minutes": 90, "expected_goals": "0.42"},
            {"element": 9, "fixture": 1, "round": 1, "value": 55, "minutes": 0, "expected_goals": "0.00"},
        ]
    }
    eh = element_history_frame({9: summary})
    assert list(eh["round"]) == [1, 2]  # sorted
    assert eh["expected_goals"].dtype == float and eh["expected_goals"].iloc[1] == pytest.approx(0.42)


def test_round_prices_dgw_dedup_and_upcoming_round():
    eh = pd.DataFrame(
        {"element": [1, 1, 1, 2], "round": [1, 2, 2, 1], "fixture": [1, 2, 3, 1], "value": [50, 51, 51, 60]}
    )
    boot = {
        "events": [{"id": 3, "is_next": True}],
        "elements": [{"id": 1, "now_cost": 52}, {"id": 2, "now_cost": 59}, {"id": 3, "now_cost": 45}],
    }
    rp = round_prices_frame(eh, boot)
    assert len(rp) == 3 + 3  # (1,1) (1,2) (2,1) + round 3 for all three players
    lookup = round_prices_lookup(rp)
    assert lookup[1] == {1: 50, 2: 51, 3: 52} and lookup[3] == {3: 45}
    assert set(rp.loc[rp["round"] == 3, "source"]) == {"now_cost"}


def test_fetch_element_summaries_reports_missing():
    got, missing = fetch_element_summaries(offline_client, [4, 12, 99999], min_interval=0)
    assert set(got) == {4, 12} and missing == [99999]


def test_too_many_missing_summaries_fail_the_build(tmp_path):
    from fpl_rocket.bundle import write_bundle

    with pytest.raises(BundleError, match="element summaries failed"):
        write_bundle(tmp_path, offline_client(), element_ids=[4, 99998, 99999], min_interval=0)
    assert not any(tmp_path.iterdir())  # nothing written, not even a partial bundle


# ------------------------------------------------------------------------------------------ manifest flags
@pytest.mark.parametrize(
    "hours_left, expected", [(4.0, False), (2.9, True), (0.5, True), (-0.1, False), (3.0, False)]
)
def test_pre_deadline_flag(hours_left, expected):
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    deadline = now + timedelta(hours=hours_left)
    events = [
        {"id": 8, "is_next": False, "deadline_time": "2026-10-17T17:30:00Z"},
        {"id": 9, "is_next": True, "deadline_time": deadline.isoformat().replace("+00:00", "Z")},
    ]
    info = deadline_info(events, now)
    assert info["pre_deadline"] is expected and info["next_event"] == 9
    assert info["next_deadline_time"].startswith(deadline.isoformat()[:16])


def test_no_next_event_after_season():
    assert deadline_info([{"id": 38, "is_next": False}], datetime.now(UTC))["pre_deadline"] is False


def test_manifest_flags_from_snapshot(tmp_path):
    boot = _snap("bootstrap-static.json")
    nxt = next(e for e in boot["events"] if e["is_next"])
    deadline = datetime.fromisoformat(nxt["deadline_time"].replace("Z", "+00:00"))
    early = read_manifest(build_snapshot_bundle(tmp_path / "a", deadline - timedelta(hours=10)))
    late = read_manifest(build_snapshot_bundle(tmp_path / "b", deadline - timedelta(hours=1)))
    assert early["pre_deadline"] is False and late["pre_deadline"] is True
    assert early["next_event"] == late["next_event"] == nxt["id"]
    assert late["next_deadline_time"] == deadline.isoformat(timespec="seconds")
    assert late["cron_interval_hours"] == 3


# ------------------------------------------------------------------------------------------ app payload
def test_app_shared_reproduces_the_api_payloads(bundle):
    m = read_manifest(bundle)
    s = app_shared(m, _files(bundle), LIVE_STATS)
    assert s.bootstrap == _snap("bootstrap-static.json")
    assert s.fixtures == _snap("fixtures.json")
    lives = {g: _snap(f"event_{g}_live.json") for g in range(1, 8)}
    pd.testing.assert_frame_equal(s.live, live_frame(lives))
    assert s.bundle_id == m["bundle_id"]


def test_app_shared_rejects_tampered_file(bundle):
    files = _files(bundle)
    files["fixtures.parquet"] = files["fixtures.parquet"][:-10] + b"0123456789"
    with pytest.raises(BundleError, match="checksum"):
        app_shared(read_manifest(bundle), files, LIVE_STATS)


def test_app_shared_rejects_v1_and_incomplete(bundle):
    m = read_manifest(bundle)
    with pytest.raises(BundleError, match="format v1"):
        app_shared({**m, "format_version": 1}, _files(bundle), LIVE_STATS)
    files = _files(bundle)
    del files["round_prices.parquet"]
    with pytest.raises(BundleError, match="missing"):
        app_shared(m, files, LIVE_STATS)


# ------------------------------------------------------------------------------------------ round prices
class _CountingClient:
    def __init__(self):
        self.inner = offline_client()
        self.summary_calls: list[int] = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def element_summary(self, pid):
        self.summary_calls.append(pid)
        return self.inner.element_summary(pid)


def test_round_prices_match_live_element_summaries_for_recorded_owner(bundle):
    rp = app_shared(read_manifest(bundle), _files(bundle), LIVE_STATS).round_prices
    live_client, bundle_client = _CountingClient(), _CountingClient()
    live = fetch_inputs(live_client, OWNER)
    via_bundle = fetch_inputs(bundle_client, OWNER, round_prices=rp)
    assert len(live_client.summary_calls) == 15 and bundle_client.summary_calls == []
    assert via_bundle["initial_values"] == live["initial_values"] and len(live["initial_values"]) == 15
    a, b = build_planning_state(**live), build_planning_state(**via_bundle)
    assert [(p.element, p.purchase_price, p.selling_price) for p in a.squad] == [
        (p.element, p.purchase_price, p.selling_price) for p in b.squad
    ]
    assert a.warnings == b.warnings == []


def test_player_missing_from_round_prices_falls_back_to_one_live_call(bundle):
    rp = dict(app_shared(read_manifest(bundle), _files(bundle), LIVE_STATS).round_prices)
    dropped = 411
    rp.pop(dropped)
    c = _CountingClient()
    out = fetch_inputs(c, OWNER, round_prices=rp)
    assert c.summary_calls == [dropped]
    assert out["initial_values"] == fetch_inputs(offline_client(), OWNER)["initial_values"]
