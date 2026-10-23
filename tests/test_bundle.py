"""Bundle writer/reader: checksums, immutability, partial-bundle refusal."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_rocket.bundle import (
    BundleError,
    latest_bundle,
    live_tables,
    read_bundle,
    read_manifest,
    write_tables,
)

FIX = Path(__file__).parent / "fixtures"


def _tables():
    live = json.loads((FIX / "event_7_live.json").read_text(encoding="utf-8"))
    gw_live, explain = live_tables({7: live})
    return {
        "teams": pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}),
        "gw_live": gw_live,
        "gw_live_explain": explain,
    }


def test_roundtrip_and_manifest(tmp_path):
    path = write_tables(_tables(), tmp_path, {"bootstrap-static": "t0"}, bundle_id="B1")
    m = read_manifest(path)
    assert m["bundle_id"] == "B1" and m["rules_version"].startswith("2026-27")
    assert {f["table"] for f in m["files"]} == {"teams", "gw_live", "gw_live_explain"}
    tables = read_bundle(path)
    assert len(tables["gw_live"]) == len(
        json.loads((FIX / "event_7_live.json").read_text("utf-8"))["elements"]
    )
    assert set(tables["gw_live_explain"].columns) >= {
        "gw",
        "element",
        "fixture",
        "identifier",
        "value",
        "points",
    }
    assert latest_bundle(tmp_path) == path


def test_bundles_are_immutable(tmp_path):
    write_tables(_tables(), tmp_path, {}, bundle_id="B1")
    with pytest.raises(BundleError):
        write_tables(_tables(), tmp_path, {}, bundle_id="B1")


def test_tampered_file_refused(tmp_path):
    path = write_tables(_tables(), tmp_path, {}, bundle_id="B1")
    pd.DataFrame({"id": [9]}).to_parquet(path / "teams.parquet", index=False)
    with pytest.raises(BundleError, match="checksum"):
        read_bundle(path)


def test_partial_and_missing_manifest_refused(tmp_path):
    partial = tmp_path / "B2.partial"
    partial.mkdir()
    with pytest.raises(BundleError):
        read_bundle(partial)
    nomanifest = tmp_path / "B3"
    nomanifest.mkdir()
    with pytest.raises(BundleError):
        read_bundle(nomanifest)
    path = write_tables(_tables(), tmp_path, {}, bundle_id="B4")
    (path / "teams.parquet").unlink()
    with pytest.raises(BundleError, match="missing"):
        read_bundle(path)
    assert latest_bundle(tmp_path) == path  # B2.partial and B3 are never selected
