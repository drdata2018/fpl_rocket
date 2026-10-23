"""Prospective policy trajectories (four virtual managers): start squad, carry-forward across deadline
releases, restrained rule (never a hit, free-transfer banking), missing weeks, scoring with automatic subs
and hits."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from fpl_rocket import trajectories as tr
from planner_synth import league

GWS = (1, 2, 3, 4, 5)
SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}


def _bootstrap(pool, unselectable: int | None = None) -> dict:
    elements = []
    for c in pool.values():
        own = round(c.buy / 10 + (c.element % 11) * 0.7, 1)  # ownership follows price, plus a spread
        elements.append(
            {
                "id": c.element,
                "web_name": f"P{c.element}",
                "team": c.team,
                "element_type": c.element_type,
                "now_cost": c.buy,
                "selected_by_percent": str(own),
                "can_select": c.element != unselectable,
                "removed": False,
            }
        )
    return {
        "elements": elements,
        "teams": [{"id": t, "short_name": f"T{t}", "name": f"Team {t}"} for t in range(1, 21)],
        "events": [
            {"id": g, "finished": False, "data_checked": False, "average_entry_score": 50 + g} for g in GWS
        ],
    }


def _xp_frame(xp: dict) -> pd.DataFrame:
    df = pd.DataFrame([{"element": e, "gw": g, "xp": v, "n_fixtures": 1} for (e, g), v in xp.items()])
    df["low_history"] = df["element"] == 1
    return df


@pytest.fixture(scope="module")
def world():
    pool, xp = league(gws=GWS)
    top = max(pool.values(), key=lambda c: c.buy).element  # the most-owned player: made unselectable
    boot = _bootstrap(pool, unselectable=top)
    baseline = {e: max(0.0, c.buy / 25 - (e % 5) * 0.3) for e, c in pool.items()}
    return pool, _xp_frame(xp), boot, baseline, top


def _inp(world, gw: int, xp: pd.DataFrame | None = None) -> tr.StepInputs:
    pool, xp_df, boot, baseline, _ = world
    return tr.StepInputs(
        gw=gw,
        bootstrap=boot,
        xp=xp_df if xp is None else xp,
        baseline_per_fixture=baseline,
        bundle_id=f"b{gw}",
        time_limit=20,
    )


def _legal(squad: list[dict], pool) -> None:
    ids = [m["element"] for m in squad]
    assert len(ids) == 15 == len(set(ids))
    assert Counter(pool[e].element_type for e in ids) == Counter(SHAPE)
    assert max(Counter(pool[e].team for e in ids).values()) <= 3


@pytest.fixture(scope="module")
def run(world, tmp_path_factory):
    """GW1 start, GW2 step from the GW1 release, GW3 missing, GW4 step from the GW2 release."""
    src = tmp_path_factory.mktemp("releases")
    s1, p1 = tr.step(None, _inp(world, 1))
    tr.write_files(src / "deadline-gw1", s1, p1)
    k, prev = tr.latest_state(str(src), 2, start=1)
    assert k == 1
    s2, p2 = tr.step(prev, _inp(world, 2))
    tr.write_files(src / "deadline-gw2", s2, p2)
    k, prev = tr.latest_state(str(src), 4, start=1)
    assert k == 2  # GW3 has no release: the latest EARLIER one is used
    s4, p4 = tr.step(prev, _inp(world, 4))
    tr.write_files(src / "deadline-gw4", s4, p4)
    return src, (s1, p1), (s2, p2), (s4, p4)


def test_most_owned_start_squad_is_legal_and_skips_unselectable(world):
    pool, _, boot, _, top = world
    players = tr.players_table(boot)
    squad = tr.most_owned_squad(players)
    _legal([{"element": e} for e in squad], pool)
    assert top not in squad
    assert sum(pool[e].buy for e in squad) <= 1000
    # greedy by ownership: the most-owned selectable player is the first pick; the budget reserve kept
    # the squad completable, so cheaper less-owned players fill the last slots
    sel = players[players["selectable"]].sort_values(["ownership", "price"], ascending=[False, True])
    assert squad[0] == int(sel.iloc[0]["element"])
    owned = sel.set_index("element")["ownership"]
    assert owned[squad[0]] >= owned[squad[-1]]
    assert tr.most_owned_squad(players) == squad  # deterministic


def test_start_state_same_squad_bank_and_no_transfers(world, run):
    pool, *_ = world
    _, (s1, p1), *_ = run
    assert s1["created_gw"] == 1 and s1["gw"] == 1 and s1["missing_gws"] == []
    cost = sum(pool[e].buy for e in s1["start_squad"])
    assert s1["start_cost"] == cost and s1["start_bank"] == 1000 - cost >= 0
    for name, t in s1["trajectories"].items():
        assert sorted(m["element"] for m in t["squad"]) == sorted(s1["start_squad"])
        assert t["bank"] == 1000 - cost and t["free_transfers_next"] == 1
        rec = p1["trajectories"][name]
        assert rec["status"] == "start" and rec["transfers_in"] == [] and rec["hits"] == 0
        assert len(rec["xi"]) == 11 and rec["captain"] in rec["xi"] and rec["captain"] != 1
    assert set(p1["trajectories"]) == {p.name for p in tr.TRAJECTORIES}


def test_carry_forward_missing_week_and_accounting(world, run):
    pool, *_ = world
    _, (s1, _), (s2, p2), (s4, p4) = run
    assert s2["previous_gw"] == 1 and s4["previous_gw"] == 2 and s4["missing_gws"] == [3]
    for policy in tr.TRAJECTORIES:
        t2, t4 = s2["trajectories"][policy.name], s4["trajectories"][policy.name]
        r2, r4 = p2["trajectories"][policy.name], p4["trajectories"][policy.name]
        _legal(t4["squad"], pool)
        assert r2["free_transfers"] == 1  # 1 free transfer entering the GW after the start
        n2 = len(r2["transfers_in"])
        ft_after_2 = min(5, 1 - min(n2, 1) + 1)
        assert t2["free_transfers_next"] == ft_after_2
        assert r4["free_transfers"] == min(5, ft_after_2 + 1)  # the missing GW3 banks one more
        # GW3: the GW2 squad and line-up carried, status missing
        lu3 = t4["lineups"]["3"]
        assert lu3["status"] == "missing" and lu3["xi"] == t2["lineups"]["2"]["xi"] and lu3["hits"] == 0
        # bank = start bank + sales - purchases over the transfer log
        moves = sum(o["sell"] for x in t4["transfer_log"] for o in x["out"]) - sum(
            i["buy"] for x in t4["transfer_log"] for i in x["in"]
        )
        assert t4["bank"] == s1["start_bank"] + moves >= 0
        for r in (r2, r4):
            assert len(r["transfers_in"]) == len(r["transfers_out"]) <= policy.max_transfers
            assert r["hits"] == max(0, len(r["transfers_in"]) - r["free_transfers"])
            if not policy.allow_hits:
                assert r["hits"] == 0
        if policy.hold:
            assert t4["transfers_total"] == 0 and sorted(m["element"] for m in t4["squad"]) == sorted(
                s1["start_squad"]
            )


def test_restrained_never_takes_a_hit_aggressive_may(world, run):
    _, (s1, _), *_ = run
    pool, xp_df, *_ = world
    owned = set(s1["start_squad"])
    # two cheap non-owned midfielders from different clubs become irresistible in GW2-4
    mids = sorted(
        (c for c in pool.values() if c.element_type == 3 and c.element not in owned and c.element != 1),
        key=lambda c: (c.buy, c.element),
    )
    stars = [mids[0].element, next(c.element for c in mids[1:] if c.team != mids[0].team)]
    boosted = xp_df.copy()
    boosted.loc[boosted["element"].isin(stars), "xp"] = 25.0
    _, plans = tr.step(s1, _inp(world, 2, boosted))
    agg, res = plans["trajectories"]["model_aggressive"], plans["trajectories"]["model_restrained"]
    assert set(agg["transfers_in"]) == set(stars) and agg["hits"] == 1 and agg["hit_cost"] == 4
    assert len(res["transfers_in"]) == 1 and res["hits"] == 0 and res["transfers_in"][0] in stars


def test_restrained_banks_free_transfers_when_nothing_is_worth_a_move(world, run):
    _, (s1, _), *_ = run
    _, xp_df, *_ = world
    flat = xp_df.assign(xp=0.0)  # no transfer can gain anything: the restrained policy keeps its FT
    state, plans = tr.step(s1, _inp(world, 2, flat))
    for name in ("model_restrained", "model_aggressive", "hold"):
        assert plans["trajectories"][name]["transfers_in"] == []
        assert state["trajectories"][name]["free_transfers_next"] == 2


def test_previous_state_must_be_earlier(world, run):
    _, _, (s2, _), _ = run
    with pytest.raises(ValueError, match="not before"):
        tr.step(s2, _inp(world, 2))


def _live(pool, zero_minutes: set[int], star: int | None = None) -> dict:
    return {
        "elements": [
            {
                "id": e,
                "stats": {
                    "total_points": 0 if e in zero_minutes else (12 if e == star else 2 + e % 3),
                    "minutes": 0 if e in zero_minutes else 90,
                },
            }
            for e in pool
        ]
    }


class _Client:
    def __init__(self, bootstrap, lives):
        self.b, self.lives = bootstrap, lives

    def bootstrap(self):
        return self.b

    def event_live(self, gw):
        return self.lives[gw]


def test_scoring_auto_subs_hits_cumulative_and_missing(world, run, tmp_path):
    from fpl_rocket.planner.lineup import realised_points

    pool, _, boot, _, _ = world
    src, (s1, p1), (s2, p2), _ = run
    lu1 = p1["trajectories"]["model_aggressive"]
    out_starter = next(e for e in lu1["xi"] if pool[e].element_type != 1 and e != lu1["captain"])
    lives = {g: _live(pool, {out_starter} if g == 1 else set(), star=lu1["captain"]) for g in GWS}
    client = _Client(boot, lives)
    csv = tmp_path / "trajectories_scores.csv"
    rows1 = tr.score_and_append(1, csv, client, str(src), start=1)
    assert [r["policy"] for r in rows1] == [p.name for p in tr.TRAJECTORIES]
    r = rows1[0]
    pts = {e["id"]: e["stats"]["total_points"] for e in lives[1]["elements"]}
    mins = {e["id"]: e["stats"]["minutes"] for e in lives[1]["elements"]}
    et = {e: pool[e].element_type for e in pool}
    expect = realised_points(lu1["xi"], lu1["bench"], lu1["captain"], lu1["vice"], pts, mins, et)
    assert r["auto_subs"] >= 1 and r["points"] == expect["points"] and r["net"] == r["points"]
    assert r["gw_average"] == 51 and r["net_minus_average"] == r["net"] - 51 and r["status"] == "ok"
    assert r["cumulative_net"] == r["net"] and r["flagged"] is False
    # a hit is deducted (scorer-level: a GW2 line-up with one hit)
    hit_lineups = {n: {**rec, "hits": 1, "hit_cost": 4} for n, rec in p2["trajectories"].items()}
    rows2 = tr.score_gw(2, hit_lineups, lives[2], boot, tr.read_scores(csv))
    for row, first in zip(rows2, rows1, strict=True):
        assert row["net"] == row["points"] - 4 and row["hits_total"] == 1
        assert row["cumulative_net"] == first["net"] + row["net"]
    tr.append_scores(csv, rows2)
    # GW3 has no deadline release: scored on the GW2 squad and line-up, flagged
    rows3 = tr.score_and_append(3, csv, client, str(src), start=1)
    for row in rows3:
        assert row["status"] == "missing" and row["flagged"] is True and row["transfers"] == 0
        assert row["xi"] == ",".join(str(e) for e in p2["trajectories"][row["policy"]]["xi"])
    assert tr.scored_gws(csv) == {1, 2, 3}
    s = tr.summary(pd.read_csv(csv))
    assert s["gw"] == 3 and list(s["cumulative"].index) == [1, 2, 3]
    assert list(s["table"]["Policy"]) == [p.label for p in tr.TRAJECTORIES]
    assert len(s["flagged"]) == 4


def test_scoring_before_any_trajectory_is_not_started(world, tmp_path):
    pool, _, boot, _, _ = world
    client = _Client(boot, {9: _live(pool, set())})
    rows = tr.score_and_append(9, tmp_path / "s.csv", client, str(tmp_path / "empty"))
    assert {r["status"] for r in rows} == {"not_started"} and all(r["flagged"] for r in rows)
    assert tr.summary(None) is None and tr.summary(tr.scores_frame([])) is None


def test_creation_from_fixture_bundle(tmp_path):
    from bundle_fixture import build_snapshot_bundle

    bdir = build_snapshot_bundle(tmp_path / "bundles", with_xp=True)
    # the recorded element summaries are trimmed (no fixture ids): the baseline is given, from season form
    players = pd.read_parquet(bdir / "players.parquet")
    form = dict(zip(players["id"].astype(int), pd.to_numeric(players["form"]).fillna(0.0), strict=True))
    inp = tr.inputs_from_bundle(bdir, archive=None, time_limit=20, baseline=form)
    state, plans = tr.step(None, inp)
    elements = {int(p["id"]): p for p in inp.bootstrap["elements"]}
    squad = state["start_squad"]
    assert len(squad) == 15
    assert Counter(int(elements[e]["element_type"]) for e in squad) == Counter(SHAPE)
    assert max(Counter(int(elements[e]["team"]) for e in squad).values()) <= 3
    cost = sum(int(elements[e]["now_cost"]) for e in squad)
    assert state["start_bank"] == 1000 - cost >= 0
    assert all(elements[e].get("can_select", True) for e in squad)
    for rec in plans["trajectories"].values():
        assert rec["status"] == "start" and rec["transfers_in"] == [] and rec["bank_after"] == 1000 - cost
    out = tr.write_files(tmp_path / "deadline", state, plans)
    assert [p.name for p in out] == list(tr.FILES)


def test_a_release_read_error_never_restarts_the_trajectories(monkeypatch):
    class _Resp:
        def __init__(self, code):
            self.status_code, self.content = code, b"{}"

    codes = {"deadline-gw11": 404, "deadline-gw10": 502}

    def get(url, **kw):
        return _Resp(next(c for tag, c in codes.items() if f"/{tag}/" in url))

    monkeypatch.setattr(tr.requests, "get", get)
    with pytest.raises(tr.SourceError, match="HTTP 502"):
        tr.latest_state("https://example.invalid/releases", 12)
    codes["deadline-gw10"] = 404
    codes["deadline-gw9"] = 404
    assert tr.latest_state("https://example.invalid/releases", 12) is None
