"""Offline AppTest of the shadow planner and the experimental xP columns, on a fixture bundle with a small
xp.parquet (tests/bundle_fixture.py, with_xp=True) and the recorded per-ID snapshot.

Public setting (config.SHOW_SHADOW_PLANS False): the Model & Planner page shows the planning state and the
registry, never a plan. The plan UI itself is still tested with the flag switched on (`plans_on`)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from fpl_rocket import config, data, registry, trajectories

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "streamlit_app.py")
SNAP = Path(__file__).parent / "fixtures" / "app_snapshot"
TID = "7630179"
PAGE = "app_pages/model_planner.py"
PAGES = (
    "app_pages/season_overview.py",
    "app_pages/my_team.py",
    "app_pages/compare_players.py",
    "app_pages/differentials.py",
    "app_pages/model_planner.py",
)
BANDS = {"nailed", "likely", "rotation risk", "doubtful", "insufficient history", "-"}


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
    monkeypatch.delenv("FPL_ROCKET_TRAJECTORY_SCORES", raising=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture()
def plans_on(monkeypatch):
    """The plan UI as it would render with config.SHOW_SHADOW_PLANS = True (not the public setting)."""
    monkeypatch.setattr(config, "SHOW_SHADOW_PLANS", True)


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
    assert f"first scored deadline is GW{registry.REGISTRY_START_GW}" in text
    assert "Model v0.1-fixture" in text and "bundle " in text


def test_public_page_never_renders_or_computes_a_plan(monkeypatch):
    assert config.SHOW_SHADOW_PLANS is False  # the deployed setting

    def boom(*a, **k):
        raise AssertionError("a plan was computed with SHOW_SHADOW_PLANS False")

    monkeypatch.setattr(data, "plan", boom)
    monkeypatch.setattr(data, "chip_plans", boom)
    at = _planner(_app())
    text = _texts(at)
    assert "Planning state" in text and "Confirm (or correct and confirm) the planning state." in text
    assert "to see the plan" not in text
    assert not any(b.label == "Confirm and plan" for b in at.button)
    next(b for b in at.button if b.label == "Confirm").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = _texts(at)
    assert any("confirmed for this session" in s.value for s in at.success)
    assert "How the planner is being tested" in text and "recorded as missing" in text
    assert "Prospective registry" in text
    for hidden in (
        "plan: best XI",
        "(b) Next GW",
        "-GW horizon, decay",
        "(d) Chips",
        "Bench order",
        "NOT VALIDATED",
    ):
        assert hidden not in text, hidden
    assert not at.html, "no pitch is rendered"
    assert not {"Captain", "Vice-captain", "Expected points (XI + captain)"} & {m.label for m in at.metric}
    # edits are confirmed (session only) without a plan
    at.number_input(key=f"ps_bank_{TID}").set_value(9.9)
    next(b for b in at.button if b.label == "Confirm").click()
    at.run()
    assert any("user-edited: bank 9.9" in s.value for s in at.success)


def test_shadow_plan_after_confirming_the_state(plans_on):
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
    assert "-GW horizon, decay" not in text and not at.warning


def test_user_edits_are_stamped_in_the_explanation(plans_on):
    at = _planner(_app())
    at.number_input(key=f"ps_bank_{TID}").set_value(9.9)
    _confirm(at)
    assert "User-edited bank: 9.9" in _texts(at)
    # an illegal squad correction is refused with the reason, no plan
    at.multiselect(key=f"ps_squad_{TID}").unselect(at.multiselect(key=f"ps_squad_{TID}").value[0])
    _confirm(at)
    assert any("must have 15 players" in e.value for e in at.error)
    assert "plan: best XI" not in _texts(at)


def test_experimental_section_shows_shadow_plans(plans_on):
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
    means = at.dataframe[0].value
    assert means.at["Line-up: planner (a), no transfers", "mean points"] == 60
    assert pd.isna(means.at["Line-up: planner (b), with transfers, net of hits", "mean points"])


def test_virtual_managers_section_without_scores():
    at = _planner(_app(None))
    text = _texts(at)
    assert f"Four virtual managers (prospective, from GW{registry.REGISTRY_START_GW})" in text
    assert "not a recommendation" in text and "chosen and published *before* the deadline" in text
    assert f"the first scored gameweek is GW{registry.REGISTRY_START_GW}" in text
    assert not [e for e in at.main if "chart" in e.type] and not at.dataframe


def test_virtual_managers_section_with_scores(tmp_path, monkeypatch):
    import json

    boot = json.loads((SNAP / "bootstrap-static.json").read_text(encoding="utf-8"))
    ids = [int(p["id"]) for p in boot["elements"][:15]]
    rows = []
    for gw, nets in ((9, (60, 58, 50, 47)), (10, (55, 61, 49, 40))):
        for p, net in zip(trajectories.TRAJECTORIES, nets, strict=True):
            prev = [r for r in rows if r["policy"] == p.name]
            rows.append(
                {
                    "gw": gw,
                    "policy": p.name,
                    "label": p.label,
                    "status": "missing" if (gw, p.name) == (10, "hold") else "ok",
                    "flagged": (gw, p.name) == (10, "hold"),
                    "points": net,
                    "hits": 0,
                    "net": net,
                    "transfers": 1 if gw == 10 and not p.hold else 0,
                    "cumulative_net": net + sum(r["net"] for r in prev),
                    "gw_average": 52.0,
                    "net_minus_average": net - 52.0,
                    "cumulative_vs_average": net - 52.0 + sum(r["net_minus_average"] for r in prev),
                    "transfers_total": 1 if gw == 10 and not p.hold else 0,
                    "hits_total": 0,
                    "captain": ids[12],
                    "vice": ids[13],
                    "xi": ",".join(map(str, ids[:11])),
                    "bench": ",".join(map(str, ids[11:])),
                    "transfers_in": str(ids[14]) if gw == 10 and not p.hold else "",
                    "transfers_out": str(ids[0]) if gw == 10 and not p.hold else "",
                }
            )
    csv = tmp_path / "trajectories_scores.csv"
    trajectories.append_scores(csv, rows)
    monkeypatch.setenv("FPL_ROCKET_TRAJECTORY_SCORES", str(csv))
    at = _planner(_app(None))
    text = _texts(at)
    assert "Last scored: **GW10**" in text and "Flagged in GW10: Hold (no transfers)" in text
    charts = [e for e in at.main if "chart" in e.type]
    assert charts, sorted({e.type for e in at.main})
    table = at.dataframe[0].value
    assert list(table["Policy"]) == [p.label for p in trajectories.TRAJECTORIES]
    assert list(table["Cumulative"]) == [115, 119, 99, 87]
    assert list(table["Points this GW"]) == [55, 61, 49, 40]
    names = {int(p["id"]): p["web_name"] for p in boot["elements"]}
    exp = next(e for e in at.expander if "Line-ups played in GW10" in e.label)
    lineup_text = " ".join(str(m.value) for m in exp.markdown).replace("\\", "")
    assert f"captain {names[ids[12]]}" in lineup_text and "Transfers: out" in lineup_text


def test_differentials_and_compare_show_next_gw_and_3_gw_xp_and_bands():
    at = _app()
    at.switch_page("app_pages/differentials.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    df = at.dataframe[0].value
    assert {"xp_next", "xp_next3", "p_play"} <= set(df.columns)
    assert not [c for c in df.columns if "6" in str(c)]
    assert (df["xp_next"] == "insufficient history").any()
    assert not (df["p_play"].eq("insufficient history") ^ df["xp_next"].eq("insufficient history")).any()
    assert set(df["p_play"]) <= BANDS and not df["p_play"].str.contains("%").any()
    ok = df[df["xp_next"].ne("insufficient history") & df["xp_next"].ne("-")]
    assert len(ok) and (ok["xp_next3"].astype(float) >= ok["xp_next"].astype(float)).all()  # 3 single GWs
    text = _texts(at)
    assert "experimental model v0.1-fixture" in text and "bundle " in text
    at.switch_page("app_pages/compare_players.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    assert "xP next GW (v0.1, experimental)" in table.index and "P(play)" in table.index
    assert "xP next 3 GWs (v0.1, experimental, sum over listed fixtures)" in table.index
    assert not any("6" in str(i) for i in table.index)
    assert set(table.loc["P(play)"]) <= BANDS
    text = _texts(at)
    assert "experimental model v0.1-fixture" in text and "bundle " in text
    assert "calibration is still being measured" in text


def test_app_never_downloads_the_simulations(xp_bundle, monkeypatch):
    names = {f.name for f in Path(xp_bundle).iterdir()}
    assert "xp_sims.parquet" in names  # the fixture release has them, like the real one
    assert "xp_sims.parquet" not in data.APP_ASSETS and "xp_fixtures.parquet" not in data.APP_ASSETS
    with pytest.raises(ValueError, match="not an app asset"):
        data._read(str(xp_bundle), "xp_sims.parquet", "x")
    seen: list[str] = []
    real = data._read

    def spy(source, name, bundle_id=None):
        seen.append(name)
        return real(source, name, bundle_id)

    monkeypatch.setattr(data, "_read", spy)
    at = _app()
    for page in PAGES:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
    assert "xp.parquet" in seen and "manifest.json" in seen
    assert set(seen) <= set(data.APP_ASSETS) and "xp_sims.parquet" not in seen


def test_pages_render_when_the_solver_cannot_be_imported(monkeypatch):
    import sys

    for mod in [m for m in sys.modules if m.startswith("fpl_rocket.planner")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setitem(sys.modules, "pulp", None)  # "import pulp" raises ImportError
    monkeypatch.setitem(sys.modules, "highspy", None)
    at = _app()
    for page in PAGES:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
    text = _texts(at)  # the Model & Planner page, last
    assert "Planning state" in text and "Prospective registry" in text
    assert "solver is not available on this host" in text


def test_two_sessions_keep_their_own_planning_state(plans_on):
    a, b = _planner(_app()), _planner(_app("1000001"))
    a.number_input(key=f"ps_bank_{TID}").set_value(9.9)
    _confirm(a)
    _confirm(b)
    assert "User-edited bank" in _texts(a) and "User-edited bank" not in _texts(b)
    assert isinstance(pd.DataFrame(), pd.DataFrame)
