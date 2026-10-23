"""Page smoke tests with Streamlit's AppTest, replaying a recorded de-personalised snapshot (no network).

Snapshot: tests/fixtures/app_snapshot (scripts/record_app_snapshot.py). Covers: every page renders for no
ID and for each recorded ID; invalid IDs; ?id= read on first load and never written; two sessions with
different IDs do not leak; an API outage degrades with warnings instead of tracebacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "streamlit_app.py")
SNAP = Path(__file__).parent / "fixtures" / "app_snapshot"
META = json.loads((SNAP / "_meta.json").read_text(encoding="utf-8"))
ENTRIES = [str(e) for e in META["entries"]]
PAGES = (
    "app_pages/season_overview.py",
    "app_pages/my_team.py",
    "app_pages/compare_players.py",
    "app_pages/differentials.py",
    "app_pages/model_planner.py",
)
TIMEOUT = 60


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("FPL_ROCKET_CACHE_DIR", str(SNAP))
    monkeypatch.setenv("FPL_ROCKET_OFFLINE", "1")
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def _app(team_id: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    if team_id is not None:
        at.query_params["id"] = team_id
    return at


def _texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [h.value for h in at.subheader]
    return "\n".join(str(p) for p in parts)


def test_pages_render_without_team_id():
    at = _app()
    for page in PAGES:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.warning, (page, [w.value for w in at.warning])
    at.switch_page("app_pages/my_team.py")
    at.run()
    assert any("Enter a team ID" in i.value for i in at.info)


@pytest.mark.parametrize("team_id", ENTRIES)
def test_pages_render_for_recorded_ids(team_id):
    at = _app(team_id)
    for page in PAGES:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.warning, (page, [w.value for w in at.warning])
    assert at.session_state["team_id"] == int(team_id)
    # lazy sections: captain hindsight and the first league
    at.switch_page("app_pages/my_team.py")
    at.session_state["cap_open"] = True
    league_keys = [f"league_{lg}" for lg in META["leagues"][team_id][:1]]
    for k in league_keys:
        at.session_state[k] = True
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    text = _texts(at)
    assert "Total gap" in text  # captain hindsight table rendered
    assert "Planning state" in text and "Status: awaiting confirmation" in text
    assert any(m.label == "Gap to leader" for m in at.metric) or not league_keys
    ft = next(m for m in at.metric if m.label.startswith("Free transfers for GW"))
    assert ft.value  # "2" or "1-2"


def test_header_shows_first_name_only_and_percentile():
    tid = ENTRIES[0]
    at = _app(tid)
    at.switch_page("app_pages/my_team.py")
    at.run()
    assert any(c.value.startswith("Manager: Manager1 · team ID") for c in at.caption)
    pct = next(m for m in at.metric if m.label == "Percentile")
    assert pct.value.startswith("top ") and pct.value.endswith("%")


@pytest.mark.parametrize(
    "raw, message",
    [("abc", "digits only"), ("0", "not a valid team ID"), ("999999999", "No FPL team with ID 999999999")],
)
def test_invalid_ids_show_error_and_load_nothing(raw, message):
    at = _app()
    at.run()
    at.sidebar.text_input(key="team_id_input").input(raw)
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception
    assert any(message in e.value for e in at.sidebar.error), [e.value for e in at.sidebar.error]
    assert at.session_state["team_id"] is None


def test_query_param_read_once_and_never_written():
    a, b = ENTRIES[0], ENTRIES[1]
    at = _app(a)
    at.run()
    assert at.session_state["team_id"] == int(a)
    at.sidebar.text_input(key="team_id_input").input(b)
    at.sidebar.button[0].click()
    at.run()
    assert at.session_state["team_id"] == int(b)
    assert "id" not in at.query_params  # stale ?id= removed, new id not written


def test_no_url_param_is_written_on_load():
    at = _app()
    at.run()
    at.sidebar.text_input(key="team_id_input").input(ENTRIES[0])
    at.sidebar.button[0].click()
    at.run()
    assert at.session_state["team_id"] == int(ENTRIES[0])
    assert "id" not in at.query_params


def test_two_sessions_do_not_leak():
    a, b = ENTRIES[0], ENTRIES[1]
    at_a, at_b = _app(a), _app(b)
    for _ in range(2):
        for at in (at_a, at_b):
            at.switch_page("app_pages/my_team.py")
            at.run()
    cap_a = " ".join(c.value for c in at_a.caption)
    cap_b = " ".join(c.value for c in at_b.caption)
    assert f"team ID {a}" in cap_a and f"team ID {b}" not in cap_a
    assert f"team ID {b}" in cap_b and f"team ID {a}" not in cap_b
    assert "Test Team 1" in " ".join(c.value for c in at_a.sidebar.caption)
    assert "Test Team 2" in " ".join(c.value for c in at_b.sidebar.caption)


def test_api_outage_degrades_with_warnings(monkeypatch, tmp_path):
    monkeypatch.setenv("FPL_ROCKET_CACHE_DIR", str(tmp_path))  # empty offline cache = every call fails
    at = _app()
    for page in PAGES:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
    at.switch_page("app_pages/season_overview.py")
    at.run()
    assert any("unavailable" in w.value for w in at.warning)


def _snapshot_of(at: AppTest) -> list:
    out = [(m.label, m.value) for m in at.metric]
    out += [d.value.to_json() for d in at.dataframe]
    out += [h.value for h in at.html]
    return out


def test_deterministic_replay_from_snapshot():
    runs = []
    for _ in range(2):
        st.cache_data.clear()
        at = _app(ENTRIES[0])
        at.switch_page("app_pages/my_team.py")
        at.run()
        runs.append([x for x in _snapshot_of(at) if "built" not in str(x)])
    assert runs[0] == runs[1] and runs[0]


# ------------------------------------------------------------------------------------ shared-data bundles
def _sidebar_text(at: AppTest) -> str:
    return " ".join(c.value for c in at.sidebar.caption)


def _run_all_pages(at: AppTest) -> None:
    for page in PAGES:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])


@pytest.fixture(scope="module")
def fixture_bundle(tmp_path_factory):
    from bundle_fixture import build_snapshot_bundle

    return build_snapshot_bundle(tmp_path_factory.mktemp("release"))


def test_bundle_path_serves_shared_data(monkeypatch, tmp_path, fixture_bundle):
    from bundle_fixture import per_id_cache

    # only per-ID payloads are replayable: bootstrap, fixtures, live GWs and element summaries MUST come
    # from the bundle (round_prices replace the 15 element-summary calls of the planning state)
    monkeypatch.setenv("FPL_ROCKET_CACHE_DIR", str(per_id_cache(tmp_path / "cache")))
    monkeypatch.setenv("FPL_ROCKET_BUNDLE_SOURCE", str(fixture_bundle))
    at = _app(ENTRIES[0])
    _run_all_pages(at)
    for page in PAGES:
        at.switch_page(page)
        at.run()
        assert not at.warning, (page, [w.value for w in at.warning])
    assert f"bundle {fixture_bundle.name}" in _sidebar_text(at)
    at.switch_page("app_pages/my_team.py")
    at.run()
    assert "Planning state" in _texts(at) and "Status: awaiting confirmation" in _texts(at)
    ft = next(m for m in at.metric if m.label.startswith("Free transfers for GW"))
    assert ft.value


def test_missing_release_falls_back_to_live_api(monkeypatch, tmp_path):
    monkeypatch.setenv("FPL_ROCKET_BUNDLE_SOURCE", str(tmp_path / "no_release_yet"))
    at = _app(ENTRIES[0])
    _run_all_pages(at)
    warn = [w.value for w in at.sidebar.warning]
    assert any("live API fallback" in w for w in warn), warn
    assert "(live API)" in _sidebar_text(at)
    assert not [w for w in at.main.warning], [w.value for w in at.main.warning]  # pages unaffected


def test_checksum_mismatch_falls_back_to_live_api(monkeypatch, tmp_path, fixture_bundle):
    import shutil

    bad = tmp_path / fixture_bundle.name
    shutil.copytree(fixture_bundle, bad)
    raw = (bad / "players.parquet").read_bytes()
    (bad / "players.parquet").write_bytes(raw[:-8] + b"tampered")
    monkeypatch.setenv("FPL_ROCKET_BUNDLE_SOURCE", str(bad))
    at = _app(ENTRIES[0])
    _run_all_pages(at)
    warn = [w.value for w in at.sidebar.warning]
    assert any("live API fallback" in w and "checksum" in w for w in warn), warn
    assert "(live API)" in _sidebar_text(at) and "bundle " not in _sidebar_text(at)
