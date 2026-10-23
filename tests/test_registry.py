"""Prospective registry: writer (pure builder + files), reader with checksums, scorer, missing weeks."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fpl_rocket import registry
from fpl_rocket.state import FreeTransfers, PlanningState, SquadPlayer
from planner_synth import cheap_squad, league

GW = 1


def _world():
    pool, xp = league(gws=(GW, GW + 1))
    bootstrap = {
        "elements": [
            {
                "id": c.element,
                "web_name": f"P{c.element}",
                "team": c.team,
                "element_type": c.element_type,
                "now_cost": c.buy,
                "selected_by_percent": "1.0",
            }
            for c in pool.values()
        ],
        "teams": [{"id": t, "short_name": f"T{t}", "name": f"Team {t}"} for t in range(1, 21)],
        "events": [{"id": GW, "finished": True, "data_checked": True}, {"id": GW + 1, "finished": False}],
    }
    xp_df = pd.DataFrame([{"element": e, "gw": g, "xp": v} for (e, g), v in xp.items()])
    xp_df["model_version"] = "v0.1-test"
    xp_df["low_history"] = xp_df["element"] == 1
    squad = cheap_squad(pool)
    state = PlanningState(
        entry_id=0,
        squad=[
            SquadPlayer(
                o.element, o.element_type, o.team, o.selling_price, o.selling_price, o.selling_price, "t"
            )
            for o in squad
        ],
        bank=10,
        team_value=1000,
        market_value=1000,
        selling_value=1000,
        free_transfers=FreeTransfers(1, 1, exact=True),
        chips_used=[],
        chips_available=[],
        last_published_gw=0,
        upcoming_gw=GW,
        transfers_since_publication=0,
        as_of={},
    )
    baseline = pd.Series({c.element: c.buy / 20 for c in pool.values()})
    manifest = {
        "bundle_id": "20261030T100000Z",
        "created_at": "2026-10-30T10:00:00+00:00",
        "next_event": GW,
        "next_deadline_time": "2026-10-30T11:00:00+00:00",
        "pre_deadline": True,
        "model_version": "v0.1-test",
        "rules_version": "2026-27.1",
    }
    return pool, bootstrap, xp_df, state, baseline, manifest


@pytest.fixture()
def built(tmp_path):
    pool, bootstrap, xp_df, state, baseline, manifest = _world()
    ids = registry.REGISTRY_ENTRY_IDS
    states = {ids[0]: state, ids[1]: state}
    reg_xp, plans, meta = registry.build_registry(
        manifest=manifest,
        bootstrap=bootstrap,
        xp=xp_df,
        baseline=baseline,
        states=states,
        errors={ids[2]: "EntryNotFound"},
    )
    registry.write_files(tmp_path / registry.deadline_tag(GW), reg_xp, plans, meta)
    return tmp_path, pool, bootstrap, reg_xp, plans, meta


def test_writer_records_plans_baseline_and_meta(built):
    src, _, _, reg_xp, plans, meta = built
    assert set(registry.FILES) <= {p.name for p in (src / registry.deadline_tag(GW)).iterdir()}
    assert {"bundle_id", "issued_gw", "baseline_xp", "model_version", "low_history"} <= set(reg_xp.columns)
    assert reg_xp.loc[reg_xp["gw"] == GW + 1, "baseline_xp"].isna().all()
    statuses = [e["status"] for e in plans["entries"]]
    assert statuses[:2] == ["ok", "ok"] and statuses[2:] == ["error"] * 3
    assert plans["entries"][2]["error"] == "EntryNotFound"
    ok = plans["entries"][0]
    assert len(ok["model"]["xi"]) == 11 and ok["model"]["captain"] in ok["model"]["xi"]
    assert ok["model"]["captain"] != 1  # low_history excluded from captaincy
    assert ok["baseline"]["captain"] in ok["baseline"]["xi"]
    assert (
        meta["gw"] == GW
        and meta["exclusions"]["low_history"] == 1
        and meta["exclusions"]["entries_failed"] == 3
    )
    assert meta["deadline"] == "2026-10-30T11:00:00+00:00" and meta["rules_version"] == "2026-27.1"
    assert meta["roster"][0] == 7630179


def test_reader_round_trip_and_checksum(built):
    src, *_ = built
    reg_xp, plans, meta = registry.read_registry(GW, str(src))
    assert len(reg_xp) and plans["gw"] == GW and meta["bundle_id"] == "20261030T100000Z"
    path = src / registry.deadline_tag(GW) / registry.PLANS_FILE
    path.write_text(path.read_text(encoding="utf-8").replace('"gw": 1', '"gw": 2'), encoding="utf-8")
    with pytest.raises(registry.RegistryMissing, match="checksum"):
        registry.read_registry(GW, str(src))


class _Client:
    def __init__(self, bootstrap, live, picks):
        self.b, self.live, self.picks = bootstrap, live, picks

    def bootstrap(self):
        return self.b

    def event_live(self, gw):
        return self.live

    def entry_picks(self, entry_id, gw):
        return self.picks.get(entry_id)


def _live(pool, star):
    return {
        "elements": [
            {
                "id": e,
                "stats": {"total_points": 15 if e == star else e % 4, "minutes": 0 if e % 7 == 0 else 90},
            }
            for e in pool
        ]
    }


def test_scorer_rows_and_missing_weeks(built, tmp_path):
    src, pool, bootstrap, reg_xp, plans, meta = built
    ok = plans["entries"][0]
    star = ok["model"]["squad"][5]
    live = _live(pool, star)
    manager = {
        "active_chip": None,
        "entry_history": {"points": 61},
        "picks": [
            {"element": e, "position": i + 1, "is_captain": i == 0, "is_vice_captain": i == 1}
            for i, e in enumerate(ok["model"]["squad"])
        ],
    }
    client = _Client(bootstrap, live, {registry.REGISTRY_ENTRY_IDS[0]: manager})
    csv = tmp_path / "scores" / "registry_scores.csv"
    rows = registry.score_and_append(GW, csv, client, str(src))
    xp_row = rows[0]
    assert xp_row["kind"] == "xp" and xp_row["status"] == "ok" and xp_row["n_players"] == len(pool)
    for k in ("rmse_model", "rmse_baseline", "spearman_model", "spearman_played_baseline"):
        assert xp_row[k] == xp_row[k]  # not NaN
    e0 = rows[1]
    assert e0["captain_best_pts"] == 15
    assert e0["manager_pts"] == 61 and e0["squad_overlap"] == 15
    assert e0["captain_manager"] == ok["model"]["squad"][0]
    assert isinstance(e0["xi_model_pts"], int) and isinstance(e0["xi_baseline_pts"], int)
    assert rows[2]["reason"] == "manager picks unavailable"
    assert [r["status"] for r in rows[3:]] == ["error"] * 3
    # a week with no deadline release, and one whose release lacks registry assets: rows, never skipped
    (src / registry.deadline_tag(3)).mkdir()
    (src / registry.deadline_tag(3) / "manifest.json").write_text("{}", encoding="utf-8")
    miss2 = registry.score_and_append(2, csv, client, str(src))
    miss3 = registry.score_and_append(3, csv, client, str(src))
    assert miss2[0]["status"] == "missing" and "no deadline release" in miss2[0]["reason"]
    assert miss3[0]["status"] == "missing" and "no registry assets" in miss3[0]["reason"]
    table = pd.read_csv(csv)
    assert list(table.columns) == list(registry.SCORE_COLUMNS)
    assert registry.scored_gws(csv) == {1, 2, 3}
    latest = registry.latest_scores(table)
    assert latest["gw"] == 3 and latest["status"] == "missing"


def test_pending_gws_start_and_data_checked():
    boot = {
        "events": [
            {"id": 8, "finished": True, "data_checked": True},
            {"id": 9, "finished": True, "data_checked": True},
            {"id": 10, "finished": True, "data_checked": False},
            {"id": 11, "finished": False},
        ]
    }
    assert registry.pending_gws(boot, scored=set(), start=9) == [9]
    assert registry.pending_gws(boot, scored={9}, start=9) == []
    assert json.loads(json.dumps(registry.missing_row(9, "x")))["status"] == "missing"
