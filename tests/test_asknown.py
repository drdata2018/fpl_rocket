"""As-known archive (fpl_rocket.asknown) on small recorded fixtures (tests/fixtures/asknown): parsers, the
odds
inversion, and the reader contract (no row after T; corrupting everything after T changes nothing)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_rocket.asknown import bootstrap as bs
from fpl_rocket.asknown import crowd as cr
from fpl_rocket.asknown import minutes as mn
from fpl_rocket.asknown import odds as od
from fpl_rocket.asknown import uefa as ue
from fpl_rocket.asknown.common import ISSUE_LEAD, write_parquet
from fpl_rocket.asknown.reader import corrupt_after, load

FX = Path(__file__).parent / "fixtures" / "asknown"
SNAP_EARLY = FX / "fplcache" / "cache_2025_10_24_1254.json.xz"
SNAP_GW10 = FX / "fplcache" / "cache_2025_11_1_0633.json.xz"
T10 = pd.Timestamp("2025-11-01 11:30", tz="UTC")  # 2025-26 GW10: deadline 13:30 UTC - 2 h
T11 = pd.Timestamp("2025-11-08 09:00", tz="UTC")


def _snap(p: Path) -> dict:
    return bs.parse_snapshot(p.read_bytes())


# ===================================================================================== bootstrap
def test_snapshot_index_parses_paths_and_never_picks_a_later_snapshot():
    ls = pd.DataFrame(
        {
            "blob": ["a", "b", "c", "d"],
            "path": [
                "cache/2025/11/1/0633.json.xz",
                "cache/2025/11/1/1252.json.xz",
                "README.md",
                "cache/2025/10/31/1848.json.xz",
            ],
        }
    )
    idx = bs.snapshot_index(ls)
    assert list(idx["blob"]) == ["d", "a", "b"]
    assert idx["snapshot_ts"].iloc[1] == pd.Timestamp("2025-11-01 06:33", tz="UTC")
    assert bs.last_at_or_before(idx, T10)["blob"] == "a"  # 12:52 is after T: never used
    assert bs.last_at_or_before(idx, pd.Timestamp("2025-11-01 06:33", tz="UTC"))["blob"] == "a"  # <= T
    assert bs.last_at_or_before(idx, pd.Timestamp("2025-10-01", tz="UTC")) is None


def test_issue_snapshot_fixed_point_follows_a_moved_deadline():
    idx = pd.DataFrame(
        {
            "path": ["s1", "s2"],
            "blob": ["1", "2"],
            "snapshot_ts": pd.to_datetime(["2025-11-01 06:00", "2025-11-01 12:00"], utc=True),
        }
    )
    final = pd.Timestamp("2025-11-01 16:00", tz="UTC")
    earlier = pd.Timestamp("2025-11-01 13:00", tz="UTC")
    known = {"s1": earlier, "s2": earlier}  # both knew an earlier deadline than the one finally recorded
    snap, deadline, it = bs.resolve_issue_snapshot(idx, 10, final, lambda r, gw: known[r["path"]])
    assert snap["path"] == "s1" and deadline == earlier and it == 2  # T moved with the known deadline
    snap, deadline, it = bs.resolve_issue_snapshot(idx, 10, final, lambda r, gw: final)
    assert snap["path"] == "s2" and deadline == final and it == 1


def test_players_teams_events_frames_from_a_recorded_snapshot():
    d = _snap(SNAP_GW10)
    assert bs.snapshot_season(d) == "2025-26"
    dl = bs.event_deadlines(d)
    assert dl[10] - ISSUE_LEAD == T10
    ts = pd.Timestamp("2025-11-01 06:33", tz="UTC")
    p = bs.players_frame(d, "2025-26", 10, ts, T10)
    for c in (
        "element",
        "code",
        "now_cost",
        "status",
        "news_added",
        "penalties_order",
        "selected_by_percent",
        "transfers_in_event",
        "chance_of_playing_next_round",
        "ep_next",
        "snapshot_ts",
        "lag_minutes",
    ):
        assert c in p
    assert (p["lag_minutes"] == 297.0).all() and len(p) == len(d["elements"])
    assert str(p["news_added"].dt.tz) == "UTC"
    t = bs.teams_frame(d, "2025-26", 10, ts, T10)
    assert {"strength_attack_home", "strength_defence_away"} <= set(t.columns) and len(t) == 20
    e = bs.events_frame(d, "2025-26", 10, ts, T10)
    fin = e["finished"].astype(bool)
    assert e.loc[~fin, ["most_captained", "chip_plays"]].isna().all().all()  # crowd fields only once finished
    assert e.loc[fin, "most_captained"].notna().all()


def test_rules_record_and_diff():
    a = bs.rules_record(_snap(SNAP_EARLY))
    b = bs.rules_record(_snap(SNAP_GW10))
    assert {c["name"] for c in a["chips"]} >= {"wildcard", "freehit", "bboost", "3xc"}
    assert a["game_settings"]["squad_total_spend"] == 1000
    assert "featured_entries" not in a["game_settings"]
    flat = bs.flatten({"chips": a["chips"]})
    assert any(k.startswith("chips[wildcard@2-19]") for k in flat)
    changed = {k for k, _, _ in bs.diff_rules(a, b)}
    assert bs.diff_rules(a, a) == [] and isinstance(changed, set)
    mod = json.loads(json.dumps(b))
    mod["game_settings"]["max_extra_free_transfers"] = 9
    assert ("game_settings.max_extra_free_transfers", 4, 9) in bs.diff_rules(b, mod)


# ========================================================================================== odds
def test_margin_removal_proportional_and_shin():
    o = [1.5, 4.2, 7.0]
    p = od.proportional(o)
    assert p.sum() == pytest.approx(1.0) and p[0] / p[2] == pytest.approx(7.0 / 1.5)
    s, z = od.shin(o)
    assert s.sum() == pytest.approx(1.0) and 0 < z < 0.1
    assert s[2] < p[2] and s[0] > p[0]  # Shin shifts weight from the longshot to the favourite
    fair = 1.0 / np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(od.shin(fair)[0], [0.5, 0.3, 0.2])
    np.testing.assert_allclose(od.proportional(fair), [0.5, 0.3, 0.2])


@pytest.mark.parametrize("lh,la", [(1.6, 1.1), (2.4, 0.6), (0.9, 1.7), (1.2, 1.2)])
def test_poisson_inversion_recovers_the_rates(lh, la):
    ph, pd_, pa, pov = od.outcome_probs(lh, la)
    assert ph + pd_ + pa == pytest.approx(1.0)
    fh, fa, rmse, n = od.fit_lambdas(ph, pd_, pa, pov)
    assert (fh, fa) == (pytest.approx(lh, abs=1e-4), pytest.approx(la, abs=1e-4))
    assert rmse < 1e-7 and n == 4
    fh3, fa3, _, n3 = od.fit_lambdas(ph, pd_, pa)  # 1X2 only
    assert n3 == 3 and fh3 > fa3 if lh > la else True


def test_inconsistent_targets_leave_a_residual():
    _, _, rmse, _ = od.fit_lambdas(0.45, 0.10, 0.45, 0.2)  # a 10% draw cannot coexist with 20% over 2.5
    assert rmse > 0.01


def test_parse_season_files_and_columns_by_season():
    a = od.parse_csv((FX / "E0_2526.csv").read_bytes(), "2025-26")
    b = od.parse_csv((FX / "E0_2627.csv").read_bytes(), "2026-27")
    assert len(a) == 40 and len(b) == 5
    assert a["avg_h"].notna().all() and a["c_avg_h"].notna().all() and a["bfe_h"].notna().all()
    assert a["hxg"].isna().all() and b["hxg"].notna().all()  # xG only from 2026-27
    assert b["ps_h"].isna().all()  # no Pinnacle in 2026-27
    first = a.iloc[0]
    assert (first["home"], first["away"]) == ("Liverpool", "Bournemouth")
    assert first["kickoff_utc"] == pd.Timestamp("2025-08-15 19:00", tz="UTC")  # 20:00 BST


def test_team_mapping_uses_fpl_names():
    a = od.parse_csv((FX / "E0_2526.csv").read_bytes(), "2025-26")
    teams = bs.teams_frame(_snap(SNAP_GW10), "2025-26", 10, T10, T10)
    m = od.map_teams(a, teams)
    assert m["team_h"].notna().all() and m["team_a"].notna().all()
    utd = teams.loc[teams["name"] == "Man Utd", "team"].iloc[0]
    assert set(m.loc[m["home"] == "Man United", "team_h"]) == {utd}


def test_collection_time_rule():
    k = pd.Series(
        pd.to_datetime(
            [
                "2025-11-01 15:00",  # Saturday -> Friday 31 Oct 18:00 GMT
                "2025-11-03 20:00",  # Monday -> Friday 31 Oct
                "2025-10-31 20:00",  # Friday night -> same Friday
                "2025-10-29 19:45",  # Wednesday -> Tuesday 28 Oct
                "2025-08-16 14:00",  # Saturday in BST -> Friday 15 Aug 18:00 BST = 17:00 UTC
            ],
            utc=True,
        )
    )
    got = od.collection_time(k)
    exp = pd.to_datetime(
        ["2025-10-31 18:00", "2025-10-31 18:00", "2025-10-31 18:00", "2025-10-28 18:00", "2025-08-15 17:00"],
        utc=True,
    )
    assert list(got) == list(exp)
    lab = od.label_availability(pd.DataFrame({"kickoff_utc": k}), pd.Series([T10] * 5))
    assert lab["available_at_T"].tolist() == [True, True, True, True, True]
    early_t = pd.Series([pd.Timestamp("2025-10-31 16:30", tz="UTC")] * 5)  # a Friday-deadline GW
    lab = od.label_availability(pd.DataFrame({"kickoff_utc": k}), early_t)
    assert lab["available_at_T"].tolist()[:3] == [False, False, False]
    assert lab["available_at_T_noon"].tolist()[:3] == [True, True, True]
    assert not lab["closing_available_at_T"].any()


def test_live_next_round_function():
    raw = (FX / "fixtures_fd.csv").read_bytes()
    fetched = pd.Timestamp("2026-11-06 12:00", tz="UTC")
    out = od.fetch_next_round(raw=raw, fetched_at=fetched, issue_time=fetched + pd.Timedelta(hours=1))
    assert len(out) > 0 and (out["fetched_at"] == fetched).all() and out["available_at_T"].all()
    assert out["avg_lambda_h"].between(0.1, 5).all()
    late = od.fetch_next_round(raw=raw, fetched_at=fetched, issue_time=fetched - pd.Timedelta(hours=1))
    assert not late["available_at_T"].any()
    with pytest.raises(ValueError):
        od.fetch_next_round(raw=raw)


# ======================================================================================= minutes
def test_discover_layouts_by_season():
    paths = [
        "data/2024-2025/matches/GW1/matches.csv",
        "data/2024-2025/playermatchstats/GW1/playermatchstats.csv",
        "data/2024-2025/matches/matches.csv",
        "data/2025-2026/By Gameweek/GW10/matches.csv",
        "data/2025-2026/By Gameweek/GW10/playermatchstats.csv",
        "data/2025-2026/By Tournament/Champions League/GW10/matches.csv",
        "data/2026-2027/By Tournament/Friendlies/GW0/playermatchstats.csv",
        "data/2026-2027/By Gameweek/GW3/shots.csv",
    ]
    assert len(mn.discover(paths, "2024-25")) == 2
    d = mn.discover(paths, "2025-26")
    assert d["path"].str.contains("By Gameweek").all() and len(d) == 2
    assert mn.discover(paths, "2026-27")["gw"].tolist() == [0]


def _minutes():
    m = mn.parse_matches((FX / "ci_matches.csv").read_bytes(), 10)
    p = mn.parse_player_matches((FX / "ci_playermatchstats.csv").read_bytes(), 10)
    return mn.combine("2025-26", [m], [p])


def test_player_matches_have_kickoffs_and_fpl_ids():
    m, p = _minutes()
    assert p["kickoff_utc"].notna().all() and str(p["kickoff_utc"].dt.tz) == "UTC"
    assert set(m["tournament"]) == {"prem", "champions-league", "efl-cup"}
    assert (p["known_at"] == p["kickoff_utc"] + mn.FINISH_BUFFER).all()
    assert not p.duplicated(["player_id", "match_id"]).any()
    assert (p.loc[p["minutes_played"] == 0, "started"] == False).all()  # noqa: E712
    players = pd.read_csv(FX / "ci_players.csv")
    fpl = bs.players_frame(_snap(SNAP_GW10), "2025-26", 10, T10, T10)[["element", "code"]]
    j = mn.join_fpl(p, players, fpl)
    assert set(j["fpl_join"]) <= {"code", "code_other_id", "id_only", "none"}
    assert (j.loc[j["fpl_join"] == "code", "element"] == j.loc[j["fpl_join"] == "code", "player_id"]).all()
    npx = mn.team_npxg(m)
    assert len(npx) == 2 * len(m) and npx["npxg"].notna().any()


# ========================================================================================= crowd
def test_sample_parsing_drops_personal_fields_and_aggregates():
    raw = (FX / "fpl_sampled_small.json").read_bytes()
    at = pd.Timestamp("2024-11-05 00:17", tz="UTC")
    m, p = cr.parse_sample(raw, "2024-25", 10, at)
    assert len(m) == 16 and "name" not in m and "player_region_name" not in m
    assert set(m["tier_kind"]) == {"random", "rank", "league"}
    assert m["captain"].notna().all() and (m["bench_order"].map(lambda s: len(json.loads(s))) == 4).all()
    assert (p.groupby(["tier", "entry"]).size() == 15).all()
    pl, tiers = cr.aggregate(m, p)
    assert np.allclose(pl.groupby("tier")["captaincy_share"].sum(), 1.0)
    assert pl["ownership"].between(0, 1).all()
    assert (tiers["managers"] == 4).all()
    lagged = cr.lag(pl)
    assert (lagged["gw"] == 11).all() and (lagged["source_gw"] == 10).all()


# ========================================================================================== uefa
def test_uefa_page_and_english_mapping():
    df = ue.parse_page((FX / "uefa_page.json").read_bytes(), 1)
    assert len(df) == 10 and df["kickoff_utc"].notna().all()
    teams = {"2025-26": bs.teams_frame(_snap(SNAP_GW10), "2025-26", 10, T10, T10)}
    eng = ue.map_english(df, teams)
    assert len(eng) == 2 and set(eng["club"]) == {"Manchester City"}
    city = teams["2025-26"].loc[teams["2025-26"]["name"] == "Man City", "team"].iloc[0]
    assert (eng["fpl_team"] == city).all() and (eng["season"] == "2025-26").all()


# ======================================================================== reader + contract tests
def build_mini_store(root: Path) -> Path:
    """A tiny archive with the real layout: GW10 / GW11 of 2025-26, data before and after T10."""
    out = root / "asknown"
    early, gw10 = _snap(SNAP_EARLY), _snap(SNAP_GW10)
    ts10 = pd.Timestamp("2025-11-01 06:33", tz="UTC")
    ts11 = pd.Timestamp("2025-11-08 06:30", tz="UTC")  # a "later" snapshot (the recorded one, restamped)
    write_parquet(
        pd.DataFrame(
            {
                "season": ["2025-26", "2025-26"],
                "gw": [10, 11],
                "deadline_utc": [T10 + ISSUE_LEAD, T11 + ISSUE_LEAD],
                "issue_time_utc": [T10, T11],
                "status": ["ok", "ok"],
                "snapshot_ts": [ts10, ts11],
            }
        ),
        out / "deadlines.parquet",
    )
    for gw, d, ts, t in (
        (9, early, pd.Timestamp("2025-10-24 12:54", tz="UTC"), T10 - pd.Timedelta(days=7)),
        (10, gw10, ts10, T10),
        (11, gw10, ts11, T11),
    ):
        for kind, fn in (
            ("players", bs.players_frame),
            ("teams", bs.teams_frame),
            ("events", bs.events_frame),
        ):
            write_parquet(
                fn(d, "2025-26", gw, ts, t), out / "bootstrap" / kind / "2025-26" / f"gw{gw}.parquet"
            )
    fx = pd.DataFrame(
        {
            "fixture": [1, 2],
            "event": [10, 11],
            "team_h": [1, 12],
            "team_a": [13, 14],
            "kickoff": pd.to_datetime(["2025-11-01 15:00", "2025-11-08 15:00"], utc=True),
        }
    )
    for gw, at, t in (
        (10, pd.Timestamp("2025-10-30", tz="UTC"), T10),
        (11, pd.Timestamp("2025-11-07", tz="UTC"), T11),
    ):
        write_parquet(
            fx.assign(season="2025-26", gw=gw, snapshot_at=at, known_at=at, issue_time_utc=t),
            out / "fixtures" / "2025-26" / f"gw{gw}.parquet",
        )
    o = od.parse_csv((FX / "E0_2526.csv").read_bytes(), "2025-26")
    gw = np.where(
        o["kickoff_utc"] < T10 + ISSUE_LEAD, 9, np.where(o["kickoff_utc"] < T11 + ISSUE_LEAD, 10, 12)
    )
    t = pd.Series(
        np.where(gw == 10, T10, np.where(gw == 9, T10 - pd.Timedelta(days=7), T11 + pd.Timedelta(days=14)))
    )
    o = od.label_availability(o.assign(gw=gw), pd.to_datetime(t, utc=True))
    o["known_at"] = o["collected_at_assumed"]
    o["known_at_post"] = o["kickoff_utc"] + pd.Timedelta(hours=3)
    o = pd.concat([o, od.invert_frame(o, "avg"), od.invert_frame(o, "avg", closing=True)], axis=1)
    write_parquet(o, out / "odds" / "2025-26.parquet")
    m, p = _minutes()
    write_parquet(p, out / "minutes" / "2025-26" / "player_matches.parquet")
    write_parquet(mn.team_npxg(m), out / "minutes" / "2025-26" / "team_npxg.parquet")
    eu = ue.map_english(
        ue.parse_page((FX / "uefa_page.json").read_bytes(), 1),
        {"2025-26": bs.teams_frame(gw10, "2025-26", 10, T10, T10)},
    )
    write_parquet(eu, out / "european" / "english_club_matches.parquet")
    mm, pp = cr.parse_sample(
        (FX / "fpl_sampled_small.json").read_bytes(), "2025-26", 9, pd.Timestamp("2025-10-28", tz="UTC")
    )
    pl, tiers = cr.aggregate(mm, pp)
    for name, df in (("players", pl), ("tiers", tiers)):
        lag9 = cr.lag(df).assign(known_at=pd.Timestamp("2025-10-28", tz="UTC"))
        lag10 = cr.lag(df.assign(gw=10)).assign(known_at=pd.Timestamp("2025-11-04", tz="UTC"))
        write_parquet(pd.concat([lag9, lag10], ignore_index=True), out / "crowd" / f"lagged_{name}.parquet")
    return out


def test_reader_returns_nothing_after_T(tmp_path):
    store = build_mini_store(tmp_path)
    a = load("2025-26", 10, store)
    assert a.issue_time_utc == T10 and a.deadline_utc == T10 + ISSUE_LEAD
    assert a.violations() == {}
    assert (a.players_at_T["snapshot_ts"] <= T10).all() and len(a.players_at_T) > 0
    assert len(a.fixtures_as_known) == 2 and (a.fixtures_as_known["snapshot_at"] <= T10).all()
    assert (a.minutes_history["kickoff_utc"] + mn.FINISH_BUFFER <= T10).all()
    assert len(a.minutes_history) > 0
    assert (a.european_history["kickoff_utc"] < T10).all() and len(a.european_history) == 1
    assert (a.odds["collected_at_assumed"] <= T10).all()
    upcoming = a.odds[a.odds["kickoff_utc"] >= T10]
    assert len(upcoming) > 0 and upcoming["c_avg_h"].isna().all() and upcoming["result"].isna().all()
    past = a.odds[a.odds["kickoff_utc"] + pd.Timedelta(hours=3) <= T10]
    assert past["result"].notna().all()  # results of finished matches are allowed
    assert set(a.crowd_lagged_players["source_gw"]) == {9}
    assert a.crowd_lagged_tiers["known_at"].max() <= T10


@pytest.mark.parametrize("mode", ["mutate", "drop"])
def test_corrupting_everything_after_T_changes_nothing(tmp_path, mode):
    store = build_mini_store(tmp_path)
    before = load("2025-26", 10, store).digest()
    bad = tmp_path / f"corrupt_{mode}"
    n = corrupt_after(store, bad, T10, mode=mode)
    assert n > 0  # there was something after T to corrupt
    assert load("2025-26", 10, bad).digest() == before


def test_corruption_helper_has_teeth(tmp_path):
    """Corrupting from a cutoff BEFORE T must change the reader's output (the test can fail)."""
    store = build_mini_store(tmp_path)
    before = load("2025-26", 10, store).digest()
    corrupt_after(store, tmp_path / "early", T10 - pd.Timedelta(days=3))
    assert load("2025-26", 10, tmp_path / "early").digest() != before


def test_reader_refuses_a_gameweek_without_state(tmp_path):
    store = build_mini_store(tmp_path)
    with pytest.raises(KeyError):
        load("2025-26", 30, store)


def test_recorded_fixtures_stay_small():
    total = sum(f.stat().st_size for f in FX.rglob("*") if f.is_file())
    assert total < 400_000


BUILT = Path(__file__).resolve().parents[1] / "data" / "asknown"


@pytest.mark.skipif(not (BUILT / "deadlines.parquet").exists(), reason="as-known archive not built locally")
def test_contract_on_the_built_archive():
    dl = pd.read_parquet(BUILT / "deadlines.parquet")
    ok = dl[dl["status"] == "ok"]
    sample = ok.groupby("season").apply(lambda g: g.iloc[[0, len(g) // 2, -1]], include_groups=False)
    for (season, _), r in sample.iterrows():
        a = load(season, int(r["gw"]), BUILT)
        assert a.violations() == {}, (season, r["gw"])
        assert (a.players_at_T["snapshot_ts"] <= a.issue_time_utc).all()
