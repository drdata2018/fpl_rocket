"""Release-reading paths of data.py with a fake HTTP layer: URLs, no release yet, stale bundle, timeout."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import requests
import streamlit as st

from bundle_fixture import build_snapshot_bundle
from fpl_rocket import data
from fpl_rocket.config import RELEASES_URL


class _Resp:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status_code, self.content = status, body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("FPL_ROCKET_OFFLINE", raising=False)
    monkeypatch.delenv("FPL_ROCKET_BUNDLE_SOURCE", raising=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def _serve(monkeypatch, routes: dict[str, bytes]) -> list[str]:
    seen: list[str] = []

    def fake_get(url, timeout=None, headers=None):
        seen.append(url)
        return _Resp(200, routes[url]) if url in routes else _Resp(404)

    monkeypatch.setattr(data.requests, "get", fake_get)
    return seen


def test_default_source_is_the_releases_redirect_path(monkeypatch):
    assert data.bundle_source() == RELEASES_URL and "api.github.com" not in RELEASES_URL
    monkeypatch.setenv("FPL_ROCKET_OFFLINE", "1")
    assert data.bundle_source() is None  # offline replay: bundles off unless a source is set
    monkeypatch.setenv("FPL_ROCKET_BUNDLE_SOURCE", "off")
    assert data.bundle_source() is None


def test_no_release_yet_means_fallback_with_reason(monkeypatch):
    seen = _serve(monkeypatch, {})
    b, status = data._active()
    assert b is None and status == {"mode": "fallback", "reason": "no bundle release published yet"}
    assert seen == [f"{RELEASES_URL}/latest/download/manifest.json"]
    data._active()
    assert len(seen) == 1  # the failed manifest fetch is cached (300 s), not retried on every rerun


def test_assets_come_from_the_manifests_own_tag(monkeypatch, tmp_path):
    bundle = build_snapshot_bundle(tmp_path)
    bid = bundle.name
    routes = {f"{RELEASES_URL}/latest/download/manifest.json": (bundle / "manifest.json").read_bytes()}
    for p in bundle.glob("*.parquet"):
        routes[f"{RELEASES_URL}/download/bundle-{bid}/{p.name}"] = p.read_bytes()
    seen = _serve(monkeypatch, routes)
    b, status = data._active()
    assert status["mode"] == "bundle" and status["bundle_id"] == bid and b is not None
    assets = [u for u in seen if "manifest" not in u]
    assert len(assets) == 7 and all(f"/download/bundle-{bid}/" in u for u in assets)
    assert "element_history.parquet" not in " ".join(assets)  # the app downloads only what it needs


def test_stale_bundle_is_ignored(monkeypatch, tmp_path):
    old = build_snapshot_bundle(tmp_path, datetime.now(UTC) - timedelta(hours=13))
    url = f"{RELEASES_URL}/latest/download/manifest.json"
    _serve(monkeypatch, {url: (old / "manifest.json").read_bytes()})
    b, status = data._active()
    assert b is None and "older than 12 h" in status["reason"]


def test_timeout_means_fallback(monkeypatch):
    def boom(url, timeout=None, headers=None):
        raise requests.Timeout("slow")

    monkeypatch.setattr(data.requests, "get", boom)
    assert data._active()[1] == {"mode": "fallback", "reason": "bundle download timed out"}
