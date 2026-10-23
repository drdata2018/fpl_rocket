"""Hourly bundles: the incremental element_history rule (reuse vs refetch), provenance in the manifest, and
the optional model hook (tables + manifest fields; a model failure never blocks the data bundle)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from bundle_fixture import SNAP, build_snapshot_bundle, offline_client, snapshot_summary_ids
from fpl_rocket import bundle as bundle_mod
from fpl_rocket.bundle import (
    element_history_decision,
    element_history_origin,
    fetch_previous_release,
    read_bundle,
    read_manifest,
    write_bundle,
)

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def _manifest(source="fetched", hours_ago=1.0, finished=(1, 2, 3), bundle_id="B0"):
    return {
        "bundle_id": bundle_id,
        "element_history_source": source,
        "element_history_fetched_at": (NOW - timedelta(hours=hours_ago)).isoformat(),
        "element_history_finished_gws": list(finished),
    }


# ------------------------------------------------------------------------------------------ the rule
def test_no_previous_release_fetches():
    assert element_history_decision(None, [1, 2, 3], NOW) == (True, "no previous release")


@pytest.mark.parametrize(
    "hours_ago, finished, refetch",
    [
        (1.0, (1, 2, 3), False),  # fresh, nothing finished since: reuse
        (5.9, (1, 2, 3), False),
        (6.1, (1, 2, 3), True),  # older than 6 h
        (1.0, (1, 2), True),  # GW3 finished since the fetch
    ],
)
def test_refetch_when_stale_or_a_gw_finished(hours_ago, finished, refetch):
    got, reason = element_history_decision(_manifest(hours_ago=hours_ago, finished=finished), [1, 2, 3], NOW)
    assert got is refetch and reason


def test_origin_follows_reuse_chains_and_old_manifests():
    assert element_history_origin(_manifest(source="reused:B-7"))[0] == "B-7"
    assert element_history_origin(_manifest(bundle_id="B9"))[0] == "B9"
    old = {
        "bundle_id": "OLD",
        "source_snapshots": {"element_summary": NOW.isoformat(), "finished_gws": "1,2"},
    }
    origin, fetched_at, finished = element_history_origin(old)
    assert (origin, fetched_at, finished) == ("OLD", NOW.isoformat(), [1, 2])


# ------------------------------------------------------------------------------------------ builds
def _snapshot_finished() -> list[int]:
    boot = json.loads((SNAP / "bootstrap-static.json").read_text(encoding="utf-8"))
    return [e["id"] for e in boot["events"] if e.get("finished")]


def test_reuse_skips_element_summaries_and_records_provenance(tmp_path, monkeypatch):
    first = build_snapshot_bundle(tmp_path / "a", NOW)
    m1 = read_manifest(first)
    assert m1["element_history_source"] == "fetched" and m1["element_summary_requested"] == 15
    assert m1["element_history_finished_gws"] == _snapshot_finished()

    previous, note = fetch_previous_release(str(first))  # a local bundle dir stands in for the release
    assert note == "ok" and previous is not None

    def boom(*_a, **_k):
        raise AssertionError("element summaries must not be fetched when reusing")

    monkeypatch.setattr(bundle_mod, "fetch_element_summaries", boom)
    later = NOW + timedelta(hours=1)
    second = write_bundle(
        tmp_path / "b", offline_client(), now=lambda: later, previous=previous, min_interval=0
    )
    m2 = read_manifest(second)
    assert m2["element_history_source"] == f"reused:{m1['bundle_id']}"
    assert m2["element_history_fetched_at"] == m1["element_history_fetched_at"]
    assert m2["element_summary_requested"] == 0 and m2["element_history_previous_bundle"] == m1["bundle_id"]
    t1, t2 = read_bundle(first), read_bundle(second)
    pd.testing.assert_frame_equal(t1["element_history"], t2["element_history"])
    pd.testing.assert_frame_equal(t1["round_prices"], t2["round_prices"])  # rebuilt from history + now_cost

    # a reuse of a reuse keeps pointing at the bundle that fetched
    prev2, _ = fetch_previous_release(str(second))
    third = write_bundle(
        tmp_path / "c",
        offline_client(),
        now=lambda: later + timedelta(hours=1),
        previous=prev2,
        min_interval=0,
    )
    assert read_manifest(third)["element_history_source"] == f"reused:{m1['bundle_id']}"


def test_stale_previous_release_is_refetched(tmp_path):
    first = build_snapshot_bundle(tmp_path / "a", NOW)
    previous, _ = fetch_previous_release(str(first))
    later = NOW + timedelta(hours=7)
    second = write_bundle(
        tmp_path / "b",
        offline_client(),
        now=lambda: later,
        previous=previous,
        element_ids=snapshot_summary_ids(),
        min_interval=0,
    )
    m = read_manifest(second)
    assert m["element_history_source"] == "fetched" and "h old" in m["element_history_reason"]
    assert m["element_summary_requested"] == 15


def test_unreadable_previous_release_means_fetch(tmp_path):
    prev, note = fetch_previous_release(str(tmp_path / "nothing-here"))
    assert prev is None and "unusable" in note
    assert fetch_previous_release(None) == (None, "no previous release source")


# ------------------------------------------------------------------------------------------ model hook
def test_model_hook_adds_tables_and_manifest_fields(tmp_path):
    def model(tables):
        assert "element_history" in tables
        xp = pd.DataFrame({"element": [1, 2], "gw": [8, 8], "xp": [3.5, 1.0]})
        return {"xp": xp}, {
            "model_version": "v0.1-test",
            "model_fitted_on": {"rows": 2},
            "model_features": {},
        }

    path = write_bundle(
        tmp_path,
        offline_client(),
        now=lambda: NOW,
        model=model,
        element_ids=snapshot_summary_ids(),
        min_interval=0,
    )
    m = read_manifest(path)
    assert m["model_version"] == "v0.1-test" and m["model_fitted_on"] == {"rows": 2}
    assert "xp" in read_bundle(path) and m["counts"]["xp"] == 2


def test_model_failure_is_recorded_and_the_data_bundle_still_publishes(tmp_path):
    def broken(_tables):
        raise RuntimeError("no archive")

    path = write_bundle(
        tmp_path,
        offline_client(),
        now=lambda: NOW,
        model=broken,
        element_ids=snapshot_summary_ids(),
        min_interval=0,
    )
    m = read_manifest(path)
    assert m["model_version"] is None and m["model_error"] == "RuntimeError: no archive"
    assert "players" in read_bundle(path) and "xp" not in m["counts"]
