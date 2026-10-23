"""Offline AppTest of the shadow planner and the experimental xP columns, on a fixture bundle with a small
xp.parquet (tests/bundle_fixture.py, with_xp=True) and the recorded per-ID snapshot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from fpl_rocket import registry

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "streamlit_app.py")
SNAP = Path(__file__).parent / "fixtures" / "app_snapshot"
TID = "7630179"
PAGE = "app_pages/model_planner.py"


@pytest.fixture(scope="module")
def xp_bundle(tmp_path_factory):
    from bundle_fixture import build_snapshot_bundle

    return build_snapshot_bundle(tmp_path_factory.mktemp("xp_release"), with_xp=True)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path, xp_bundle):
    from bundle_fixture import per_id_cache

    monkeypatch.setenv("FPL_ROCKET_CACHE_DIR", str(per_id_cache(tmp_path / "cache")))
    monkeypatch.setenv("FPL_ROCKET_OFFLINE", "1")
    monkeypatch.setenv("FPL_ROCKET_BUNDLE_SOURCE", str(xp_bundle))
    monkeypatch.delenv("FPL_ROCKET_REGISTRY_SCORES", raising=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def _app(team_id: str | None = TID) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    if team_id is not None:
        at.query_params["id"] = team_id
    return at


def _texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [h.value for h in at.subheader]
    parts += [i.value for i in at.info] + [e.value for e in at.error]
    return "\n".join(str(p) for p in parts).replace("\\", "")  # drop markdown escapes


def _planner(at: AppTest) -> AppTest:
    at.switch_page(PAGE)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _confirm(at: AppTest) -> AppTest:
    next(b for b in at.button if b.label == "Confirm and plan").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_planner_needs_a_team_id_and_shows_registry_status():
    at = _planner(_app(None))
    text = _texts(at)
    assert "Enter a team ID" in text
    assert "Prospective registry" in text and "No scores yet" in text
    assert "Model v0.1-fixture" in text


def test_shadow_plan_after_confirming_the_state():
    at = _planner(_app())
    text = _texts(at)
    assert "Planning state" in text and "Confirm (or correct and confirm)" in text
    assert not at.html or not any("Bench" in h.value for h in at.main.html)  # no plan before confirmation
    _confirm(at)
    text = _texts(at)
    assert "GW8 plan: best XI from your squad (no transfers)" in text
    assert "Captain " in text and "Bench order" in text
    labels = {m.label for m in at.metric}
    assert {"Captain", "Vice-captain", "Expected points (XI + captain)"} <= labels
    pitch = " ".join(h.value for h in at.html)
    assert "Bench" in pitch and ">C<" in pitch and ">VC<" in pitch
    assert "bundle " in text and "v0.1-fixture" in text
    # the experimental section is collapsed: nothing multi-GW is computed or shown
    assert "(c)" not in text and not at.warning


def test_user_edits_are_stamped_in_the_explanation():
    at = _planner(_app())
    at.number_input(key=f"ps_bank_{TID}").set_value(9.9)
    _confirm(at)
    assert "User-edited bank: 9.9" in _texts(at)
    # an illegal squad correction is refused with the reason, no plan
    at.multiselect(key=f"ps_squad_{TID}").unselect(at.multiselect(key=f"ps_squad_{TID}").value[0])
    _confirm(at)
    assert any("must have 15 players" in e.value for e in at.error)
    assert "plan: best XI" not in _texts(at)


def test_experimental_section_shows_shadow_plans():
    at = _planner(_app())
    _confirm(at)
    at.session_state["planner_exp_open"] = True
    at.session_state["planner_h"] = 2
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = _texts(at)
    assert "NOT VALIDATED" in text and "recorded for scoring only" in text
    assert "(b) Next GW with up to 2 transfer(s)" in text
    assert "(c) 2-GW horizon, decay 0.85" in text
    at.session_state["planner_chips"] = True
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "(d) Chips: conditional gain inside the evaluated window" in _texts(at)
    assert "recommend" not in _texts(at).lower().replace("not a recommendation", "")


def test_registry_scores_are_read_and_summarised(tmp_path, monkeypatch):
    csv = tmp_path / "registry_scores.csv"
    rows = [
        {"gw": 8, "kind": "xp", "status": "missing", "reason": "no deadline release"},
        {
            "gw": 9,
            "kind": "xp",
            "status": "ok",
            "bundle_id": "B9",
            "rmse_model": 1.9,
            "rmse_baseline": 2.1,
            "spearman_played_model": 0.35,
            "spearman_played_baseline": 0.3,
        },
        {
            "gw": 9,
            "kind": "entry",
            "entry_id": 1,
            "status": "ok",
            "captain_model_pts": 8,
            "captain_baseline_pts": 2,
            "captain_best_pts": 13,
            "captain_manager_pts": 6,
            "xi_model_pts": 60,
            "xi_baseline_pts": 52,
            "manager_pts": 58,
        },
    ]
    registry.append_scores(csv, rows)
    monkeypatch.setenv("FPL_ROCKET_REGISTRY_SCORES", str(csv))
    at = _planner(_app(None))
    text = _texts(at)
    assert "Last scored: **GW9**" in text or "Last scored: GW9" in text
    assert {"xP RMSE (model)", "RMSE (baseline)"} <= {m.label for m in at.metric}


def test_differentials_and_compare_show_one_gw_xp_only():
    at = _app()
    at.switch_page("app_pages/differentials.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    df = at.dataframe[0].value
    assert {"xp_next", "p_play"} <= set(df.columns)
    assert (df["xp_next"] == "insufficient history").any()
    assert not (df["p_play"].eq("insufficient history") ^ df["xp_next"].eq("insufficient history")).any()
    assert "experimental model v0.1-fixture" in _texts(at)
    at.switch_page("app_pages/compare_players.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    assert "xP next GW (v0.1, experimental)" in table.index and "P(play)" in table.index
    assert not any("3" in str(i) and "GW" in str(i) and "sum" in str(i).lower() for i in table.index)


def test_two_sessions_keep_their_own_planning_state():
    a, b = _planner(_app()), _planner(_app("1000001"))
    a.number_input(key=f"ps_bank_{TID}").set_value(9.9)
    _confirm(a)
    _confirm(b)
    assert "User-edited bank" in _texts(a) and "User-edited bank" not in _texts(b)
    assert isinstance(pd.DataFrame(), pd.DataFrame)
