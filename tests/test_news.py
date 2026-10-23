"""Availability news (step 3d): status_history diffing, recency + staleness, name matching, the injury-table
parser (trimmed real page), the rule-based text parser on real FPL news strings, the combination rule, the
bundle hook (carried-forward history, failure isolation) and the live cap in the model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bundle_fixture import offline_client, snapshot_summary_ids
from fpl_rocket.bundle import fetch_previous_release, read_bundle, read_manifest, write_bundle
from fpl_rocket.model.config import DEFAULT_CONFIG, NEWS_TEXT_RULES, ModelConfig
from fpl_rocket.news import news_fn, run_news_stage
from fpl_rocket.news.availability import combine_news, fpl_cap, stale_relief
from fpl_rocket.news.history import recency_features, update_status_history
from fpl_rocket.news.items import NewsItem
from fpl_rocket.news.matching import Matcher, normalise
from fpl_rocket.news.premier_injuries import PremierInjuriesSource, parse_injury_table, parse_status
from fpl_rocket.news.rules import parse_item, parse_text

FIX = Path(__file__).parent / "fixtures" / "news" / "injury_table_trim.html"
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
CFG = DEFAULT_CONFIG


def _el(i, status="a", chance=None, news="", added=None, team=1, first="A", second="Player", web=None):
    return {
        "id": i,
        "team": team,
        "first_name": first,
        "second_name": second,
        "web_name": web or second,
        "status": status,
        "chance_of_playing_next_round": chance,
        "chance_of_playing_this_round": chance,
        "news": news,
        "news_added": added.isoformat().replace("+00:00", "Z") if added else None,
    }


# ------------------------------------------------------------------------------------------ status_history
def test_status_history_diffs_consecutive_bundles():
    t0, t1, t2 = NOW - timedelta(hours=2), NOW - timedelta(hours=1), NOW
    a0 = [_el(1), _el(2, "d", 75, "Knock - 75% chance of playing", t0 - timedelta(days=1))]
    h0 = update_status_history(None, a0, t0, None)
    assert list(h0["change"]) == ["initial", "initial"] and h0["prev_observed_at"].isna().all()

    h1 = update_status_history(h0, a0, t1, t0)  # nothing changed: no rows appended
    assert len(h1) == 2

    a2 = [
        _el(1, "i", 0, "Hamstring injury - Unknown return date", t2 - timedelta(minutes=5)),
        _el(2, "d", 75, "Knock - 75% chance of playing", t0 - timedelta(days=1)),
        _el(3),  # new player
    ]
    h2 = update_status_history(h1, a2, t2, t1)
    new = h2[h2["observed_at"] == pd.Timestamp(t2)]
    got = dict(zip(new["element"], new["change"], strict=True))
    assert got == {1: "status+chance+news+news_added", 3: "initial"}
    row1 = new[new["element"] == 1].iloc[0]
    assert row1["prev_observed_at"] == pd.Timestamp(t1) and row1["status"] == "i"
    assert len(h2) == 4 and not h2.duplicated(["element", "observed_at"]).any()


def test_chance_only_change_is_recorded():
    h0 = update_status_history(None, [_el(1, "d", 75, "x", NOW)], NOW - timedelta(hours=1), None)
    h1 = update_status_history(h0, [_el(1, "d", 50, "x", NOW)], NOW, NOW - timedelta(hours=1))
    assert list(h1["change"]) == ["initial", "chance"]


# ------------------------------------------------------------------------------------------ recency / stale
def test_recency_features_and_staleness_rule():
    old = NOW - timedelta(days=10)
    els = [
        _el(1, "d", 75, "Knock - 75% chance of playing", old),  # stale: 10 days, no change since
        _el(2, "d", 75, "Knock - 75% chance of playing", NOW - timedelta(days=3)),  # fresh
        _el(3, "i", 0, "Knee injury - Unknown return date", old),  # stale i
        _el(4),  # available: never stale
        _el(5, "d", 50, "Calf injury - 50% chance of playing", old),  # chance changed 2 days ago (history)
    ]
    first = NOW - timedelta(days=20)
    h = update_status_history(None, [_el(5, "d", 75, "Calf", old), *els[:4]], first, None)
    h = update_status_history(h, els, NOW - timedelta(days=2), first)
    r = recency_features(h, els, NOW, CFG).set_index("element")
    assert r.loc[1, "stale"] and r.loc[3, "stale"]
    assert not r.loc[2, "stale"] and not r.loc[4, "stale"] and not r.loc[5, "stale"]
    assert r.loc[1, "status_change_source"] == "news_added"  # flag predates the history
    assert r.loc[1, "days_since_status_change"] == pytest.approx(10.0)
    assert r.loc[5, "status_change_source"] == "history"
    assert r.loc[5, "days_since_status_change"] == pytest.approx(2.0)
    assert r.loc[4, "status_change_source"] == "first_seen" and np.isnan(r.loc[4, "days_since_news_added"])
    assert r.loc[5, "status_transitions_last_14d"] == 0  # chance changed, status did not
    h2 = update_status_history(h, [*els[:4], _el(5, "a")], NOW - timedelta(hours=1), NOW - timedelta(days=2))
    r2 = recency_features(h2, [*els[:4], _el(5, "a")], NOW, CFG).set_index("element")
    assert r2.loc[5, "status_transitions_last_14d"] == 1


def test_stale_relief_constants():
    status = pd.Series(["d", "d", "d", "d", "i", "d"])
    stale = pd.Series([False, True, True, True, True, True])
    days = pd.Series([30.0, 7.0, 21.0, 200.0, 30.0, 14.0])
    cap = fpl_cap(status, pd.Series([75, 75, 75, 75, 0, 25]))
    out = stale_relief(cap, status, stale, days, CFG)
    assert out[0] == pytest.approx(0.75)  # not stale: unchanged
    assert out[1] == pytest.approx(0.75)  # just stale: no relief yet
    assert out[2] == pytest.approx(0.875)  # one half-life: half the shortfall
    assert out[3] == pytest.approx(0.875)  # floor: never more than half the shortfall removed
    assert out[4] == 0.0  # 'i' flags are never relaxed by age alone
    assert out[5] == pytest.approx(1 - 0.75 * 0.5**0.5)


def test_fpl_cap_matches_the_v02_rule():
    st = pd.Series(["a", "d", "i", "s", "u", "n", "i"])
    ch = pd.Series([None, 50, 0, None, None, None, 25])
    assert list(fpl_cap(st, ch)) == [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.25]


# ------------------------------------------------------------------------------------------ matching
TEAMS = [
    {"id": 1, "name": "Arsenal", "short_name": "ARS"},
    {"id": 2, "name": "Spurs", "short_name": "TOT"},
    {"id": 3, "name": "Nott'm Forest", "short_name": "NFO"},
    {"id": 4, "name": "Coventry City", "short_name": "COV"},
    {"id": 5, "name": "Bournemouth", "short_name": "BOU"},
]
PLAYERS = [
    _el(10, team=1, first="Piero Martín", second="Hincapié Reyna", web="Hincapié"),
    _el(11, team=1, first="Martin", second="Ødegaard", web="Ødegaard"),
    _el(20, team=2, first="Pedro", second="Porro Sauceda", web="Pedro Porro"),
    _el(21, team=2, first="Dejan", second="Kulusevski", web="Kulusevski"),
    _el(22, team=2, first="Ben", second="Davies", web="B.Davies"),
    _el(23, team=2, first="Alfie", second="Davies", web="A.Davies"),
    _el(30, team=3, first="Callum", second="Hudson-Odoi", web="Hudson-Odoi"),
    _el(31, team=3, first="Neco", second="Williams", web="N.Williams"),
    _el(40, team=4, first="Haji", second="Wright", web="Wright"),
    _el(50, team=5, first="Julián", second="Araujo Zúñiga", web="J.Araujo"),
    _el(51, team=5, first="Adam", second="Smith", web="Smith"),
    _el(12, team=1, first="Ben", second="White", web="White"),
    _el(32, team=3, first="Jack", second="Smith", web="J.Smith"),
]


def test_normalise_strips_accents_case_and_punctuation():
    assert normalise("Hincapié  Reyna") == "hincapie reyna"
    assert normalise("Ødegaard") == "odegaard" and normalise("Nott'm Forest") == "nott m forest"
    assert normalise("Brighton & Hove Albion") == "brighton and hove albion"


@pytest.mark.parametrize(
    "name, team, element, method",
    [
        ("Piero Hincapie Reyna", "Arsenal", 10, "first_second"),  # accents, double surname
        ("MARTIN ODEGAARD", "Arsenal", 11, "full_name"),  # case + Ø
        ("Pedro Porro Sauceda", "Tottenham Hotspur", 20, "full_name"),  # club alias
        ("Dejan Kulusevski", "Spurs", 21, "full_name"),
        ("Callum Hudson-Odoi", "Nottingham Forest", 30, "full_name"),
        ("Haji Wright", "Coventry City", 40, "full_name"),  # FPL uses the long club name
        ("Julian Araujo", "AFC Bournemouth", 50, "first_surname"),  # double surname, initial web_name
        ("Adam Smith", "AFC Bournemouth", 51, "full_name"),  # same surname as 32, different club
        ("Jack Smith", "Nottingham Forest", 32, "full_name"),
    ],
)
def test_matching(name, team, element, method):
    m = Matcher(PLAYERS, TEAMS).match(name, team)
    assert (m.element, m.method) == (element, method)


def test_unmatched_items_are_never_guessed():
    mt = Matcher(PLAYERS, TEAMS)
    assert mt.match("Davies", "Spurs").reason == "ambiguous_player"  # two Davies at the same club
    assert mt.match("Ben Davies", "Spurs").element == 22  # the full name settles it
    assert mt.match("Some Player", "Hull City").reason == "no_team"
    assert mt.match("Unknown Person", "Arsenal").reason == "no_player"
    assert mt.match("x", None, element_hint=21).element == 21
    assert mt.match("x", None, element_hint=999).reason == "no_player"


# ------------------------------------------------------------------------------------------ injury table
def test_injury_table_parser_on_trimmed_page():
    assert FIX.stat().st_size < 100_000
    fetched = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    items, stamp = parse_injury_table(FIX.read_text(encoding="utf-8"), fetched)
    assert len(items) == 32
    assert stamp == "2026-10-13T09:46:00+00:00"  # "13 Oct 26 10:46 UK" = BST
    assert {i.team_hint for i in items} == {
        "AFC Bournemouth",
        "Brentford",
        "Coventry City",
        "Liverpool",
        "Tottenham Hotspur",
    }
    statuses = pd.Series([i.status_hint for i in items]).value_counts().to_dict()
    assert statuses == {"Ruled Out": 22, "50%": 6, "25%": 4}
    scott = next(i for i in items if i.player_name_raw == "Alex Scott")
    assert scott.team_hint == "AFC Bournemouth" and scott.chance_hint == 0.0
    assert scott.return_date_hint == "2026-11-21" and scott.published_at == "2026-10-08T12:00:00+00:00"
    assert scott.url.endswith("/newsroom/epl/players/alex-scott")
    assert scott.text.startswith("Thigh Injury. Currently Being Assessed. Status: Ruled Out.")
    assert "mob-title" not in scott.text and "TRACK" not in scott.text and "See Player Page" not in scott.text
    kroupi = next(i for i in items if i.player_name_raw == "Eli Kroupi")
    assert kroupi.published_at.startswith("2026-09-20")  # "Sept 20:" quote
    pct = [i for i in items if i.status_hint == "50%"]
    assert all(i.chance_hint == 50.0 for i in pct)
    assert all(i.source == "premier_injuries" and i.source_updated_at == stamp for i in items)


def test_parse_status_values():
    assert parse_status("Ruled Out") == 0.0 and parse_status("Suspended") == 0.0
    assert parse_status("75%") == 75.0 and parse_status("75% chance") == 75.0 and parse_status("") is None


class _Resp:
    def __init__(self, status, text):
        self.status_code, self.text = status, text


class _Session:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, 0

    def get(self, url, **kw):
        self.calls += 1
        assert kw["timeout"] == 10.0 and "Mozilla" in kw["headers"]["User-Agent"]
        if self.exc:
            raise self.exc
        return self.resp


@pytest.mark.parametrize(
    "session, needle",
    [
        (_Session(_Resp(403, "<title>Just a moment...</title>")), "Cloudflare"),
        (_Session(_Resp(200, "<html><body>maintenance</body></html>")), "no injury table"),
        (_Session(exc=ConnectionError("down")), "down"),
    ],
)
def test_injury_source_failure_returns_empty_and_logs(session, needle):
    src = PremierInjuriesSource(session=session)
    assert src.fetch() == [] and src.last_status == "error" and needle in src.last_error
    assert session.calls == 1  # one polite request, no retries


# ------------------------------------------------------------------------------------------ text rules
DEADLINE = datetime(2026, 10, 24, 10, 0, tzinfo=UTC)
# real FPL news strings (bootstrap-static, 2026-10-23)
REAL_FPL_NEWS = [
    ("Ankle injury - 75% chance of playing", "pct_chance", 0.75, "medium"),
    ("Leg injury - 50% chance of playing", "pct_chance", 0.5, "medium"),
    ("Knee injury - 25% chance of playing", "pct_chance", 0.25, "medium"),
    ("Knock - 75% chance of playing", "pct_chance", 0.75, "medium"),
    ("Unspecified injury - 75% chance of playing", "pct_chance", 0.75, "medium"),
    ("Back injury - Unknown return date", "unknown_return", 0.0, "high"),
    ("Head injury - Unknown return date", "unknown_return", 0.0, "high"),
    ("Hamstring injury - Expected back 1 Nov", "expected_back", 0.0, "high"),
    ("Hamstring injury - Expected back 2 Jan", "expected_back", 0.0, "high"),
    ("Thigh injury - Expected back 31 Oct", "expected_back", 0.0, "high"),
    ("Muscular injury - Expected back 7 Nov", "expected_back", 0.0, "high"),
    ("Suspended until 31 Oct", "suspended_until", 0.0, "high"),
    ("Suspended until 8 Nov", "suspended_until", 0.0, "high"),
    ("Has joined Al Hilal permanently", "left_permanent", 0.0, "high"),
    ("has departed the club as a free agent.", "left_permanent", 0.0, "high"),
    ("has returned to Getafe CF", "left_permanent", 0.0, "high"),
    ("Has joined Borussia Dortmund on loan for the rest of the season", "loan", 0.0, "high"),
    ("Has joined Oxford United on loan until January", "loan", 0.0, "high"),
    ("has joined  Kasimpasa on loan for the rest of the seaon.", "loan", 0.0, "high"),
    ("Has joined FC Köln on loan for the rest of the season", "loan", 0.0, "high"),
    ("not included in squad.", "not_in_squad", 0.0, "high"),
    ("Has joined West Ham United on loan for the rest of the season", "loan", 0.0, "high"),
    ("Has joined Paris Saint-Germain permanently", "left_permanent", 0.0, "high"),
]


@pytest.mark.parametrize("text, rule, p, conf", REAL_FPL_NEWS)
def test_rules_on_real_fpl_news(text, rule, p, conf):
    got = parse_text(text, next_deadline=DEADLINE, now=NOW, published_at=NOW - timedelta(days=5))
    assert (got.rule, got.p_play_next, got.confidence) == (rule, p, conf)


@pytest.mark.parametrize(
    "text, rule, p, conf",
    [
        ("Knock - doubtful for Saturday", "doubtful", 0.5, "medium"),
        ("Illness", "illness", 0.5, "low"),
        ("Lack of match fitness", "lack_of_fitness", 0.75, "low"),
        ("Personal reasons", "personal", None, "low"),
        ("Left the club", "left_permanent", 0.0, "high"),
        ("Ruled out for six weeks", "ruled_out", 0.0, "medium"),
        ("He trained with the group today", "unmatched", None, "low"),
        ("", "unmatched", None, "low"),
    ],
)
def test_other_patterns_and_unmatched_text(text, rule, p, conf):
    got = parse_text(text, next_deadline=DEADLINE, now=NOW)
    assert (got.rule, got.p_play_next, got.confidence) == (rule, p, conf)


def test_return_date_before_the_deadline_gives_no_estimate():
    got = parse_text(
        "Thigh injury - Expected back 24 Oct", next_deadline=DEADLINE + timedelta(days=1), now=NOW
    )
    assert got.rule == "expected_back" and got.p_play_next is None and got.confidence == "low"
    late = parse_text("Thigh injury - Expected back 2 Jan", next_deadline=DEADLINE, now=NOW)
    assert late.return_date == "2027-01-02" and late.p_play_next == 0.0


def test_rule_table_is_well_formed():
    names = [r[0] for r in NEWS_TEXT_RULES]
    assert len(names) == len(set(names))
    for _, _, p, p60, conf in NEWS_TEXT_RULES:
        assert conf in ("low", "medium", "high")
        assert p in ("pct", "date", None) or 0.0 <= float(p) <= 1.0
        assert p60 is None or 0.0 <= p60 <= 1.0


def _table_item(status, *, updated, ret="2026-11-21", published=None):
    return NewsItem(
        player_name_raw="X",
        team_hint="Arsenal",
        text=f"Thigh Injury. Not Available. Status: {status}. Potential return: x. Oct 08: 'quote'",
        status_hint=status,
        chance_hint=parse_status(status),
        return_date_hint=ret,
        source="premier_injuries",
        fetched_at=NOW.isoformat(),
        url="u",
        published_at=(published or updated).isoformat(),
        source_updated_at=updated.isoformat(),
    )


def test_injury_table_freshness_downgrades():
    fresh = parse_item(
        _table_item("Ruled Out", updated=NOW - timedelta(days=1)),
        next_deadline=DEADLINE,
        now=NOW,
        table_fresh_days=3,
    )
    assert (fresh.rule, fresh.p_play_next, fresh.confidence) == ("table_ruled_out", 0.0, "high")
    old = parse_item(
        _table_item("50%", updated=NOW - timedelta(days=10)),
        next_deadline=DEADLINE,
        now=NOW,
        table_fresh_days=3,
    )
    assert (old.rule, old.p_play_next, old.confidence) == ("table_pct", 0.5, "low")
    past = parse_item(
        _table_item("Ruled Out", updated=NOW, ret="2026-10-18"),
        next_deadline=DEADLINE,
        now=NOW,
        table_fresh_days=3,
    )
    assert past.confidence == "low"  # return date already passed: the row is probably out of date


# ------------------------------------------------------------------------------------------ combination
def _ev(source, p, conf="medium", published=NOW):
    return {"source": source, "p_play_next": p, "confidence": conf, "published_at": published.isoformat()}


def test_news_lowers_the_cap_with_enough_confidence():
    added = NOW - timedelta(days=1)
    cap, eff, src = combine_news(1.0, 1.0, False, added, [_ev("premier_injuries", 0.25)], CFG)
    assert (cap, eff, src) == (0.25, "lower", "premier_injuries")
    cap, eff, _ = combine_news(0.75, 0.75, False, added, [_ev("premier_injuries", 0.0, "high")], CFG)
    assert (cap, eff) == (0.0, "lower")
    assert combine_news(1.0, 1.0, False, added, [_ev("premier_injuries", 0.25, "low")], CFG)[1] == "none"
    # FPL's own text repeating its flag adds nothing (and does not undo the staleness relief)
    assert combine_news(0.75, 0.875, True, added, [_ev("fpl_news", 0.75)], CFG)[:2] == (0.875, "none")
    # FPL text saying less than its flag (e.g. 'Unknown return date' on a 'd' flag) lowers
    assert combine_news(0.75, 0.75, False, added, [_ev("fpl_news", 0.0, "high")], CFG)[:2] == (0.0, "lower")


def test_news_raises_only_a_stale_flag_and_only_when_newer():
    added = NOW - timedelta(days=12)
    newer = [_ev("premier_injuries", 0.75, published=NOW - timedelta(days=1))]
    cap, eff, src = combine_news(0.0, 0.0, True, added, newer, CFG)
    assert eff == "raise" and src == "premier_injuries" and cap == pytest.approx(0.375)  # half-way
    assert combine_news(0.0, 0.0, False, added, newer, CFG)[:2] == (0.0, "none")  # not stale: never raise
    older = [_ev("premier_injuries", 0.75, published=added - timedelta(days=1))]
    assert combine_news(0.0, 0.0, True, added, older, CFG)[:2] == (0.0, "none")  # older than news_added
    low = [_ev("premier_injuries", 0.75, "low", published=NOW)]
    assert combine_news(0.0, 0.0, True, added, low, CFG)[:2] == (0.0, "none")  # not confident enough
    fpl = [_ev("fpl_news", 1.0, "high", published=NOW)]
    assert combine_news(0.0, 0.0, True, added, fpl, CFG)[:2] == (0.0, "none")  # FPL text never raises
    # the newest external item wins over an older lowering one
    mixed = [_ev("premier_injuries", 0.0, published=NOW - timedelta(days=5)), *newer]
    assert combine_news(0.5, 0.5, True, added, mixed, CFG)[1] == "raise"


# ------------------------------------------------------------------------------------------ bundle hook
class _StubSource:
    name = "premier_injuries"

    def __init__(self, items, status="ok", error=None):
        self.items, self.last_status, self.last_error, self.source_updated_at = items, status, error, None

    def fetch(self):
        return list(self.items)


def _build(out, stamp, previous=None, sources=()):
    return write_bundle(
        out,
        offline_client(),
        now=lambda: stamp,
        element_ids=snapshot_summary_ids(),
        min_interval=0,
        previous=previous,
        news=news_fn(sources_factory=lambda now: list(sources)),
    )


def test_bundle_news_tables_and_carried_forward_history(tmp_path):
    t0 = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)
    failing = _StubSource([], status="error", error="RuntimeError: HTTP 403 (Cloudflare bot challenge)")
    first = _build(tmp_path / "a", t0, sources=[failing])
    m = read_manifest(first)
    tables = read_bundle(first)
    assert {"status_history", "availability", "news_items", "news_unmatched"} <= set(tables)
    n_players = len(tables["players"])
    assert len(tables["availability"]) == n_players and m["status_history_appended"] == n_players
    assert set(tables["status_history"]["change"]) == {"initial"}
    src = {s["source"]: s for s in m["news_sources_fetched"]}
    assert src["premier_injuries"]["status"] == "error" and "403" in src["premier_injuries"]["error"]
    assert src["fpl_news"]["items"] == m["news_items_matched"] > 0 and m["news_items_unmatched"] == 0
    assert m["news_fpl_text_rule_coverage"] is not None and "news_error" not in m
    av = tables["availability"]
    assert av["cap_with_news"].between(0, 1).all() and (av["cap_with_news"] == av["cap_model_only"]).all()
    assert av["p_play_with_news"].isna().all()  # no model in this build

    previous, note = fetch_previous_release(str(first))
    assert note == "ok" and len(previous.extra["status_history"]) == n_players
    second = _build(tmp_path / "b", t0 + timedelta(hours=1), previous=previous)
    m2 = read_manifest(second)
    assert m2["status_history_previous_rows"] == n_players and m2["status_history_appended"] == 0


def test_news_failure_never_blocks_the_bundle(tmp_path):
    def boom(tables, previous, now):
        raise RuntimeError("news broke")

    path = write_bundle(
        tmp_path, offline_client(), element_ids=snapshot_summary_ids(), min_interval=0, news=boom
    )
    m = read_manifest(path)
    assert m["news_error"] == "RuntimeError: news broke" and "availability" not in read_bundle(path)


def test_stage_matches_injury_table_items_against_the_snapshot(tmp_path):
    first = _build(tmp_path, datetime(2026, 10, 24, 9, 0, tzinfo=UTC))
    tables = read_bundle(first)
    p = tables["players"].iloc[0]
    team = tables["teams"].set_index("id").loc[int(p["team"]), "name"]
    stamp = datetime(2026, 10, 24, 8, 0, tzinfo=UTC)
    good = NewsItem(
        str(p["web_name"]),
        str(team),
        "Knee Injury. Not Available. Status: Ruled Out. Potential return: 21/11/2026.",
        "Ruled Out",
        0.0,
        "2026-11-21",
        "premier_injuries",
        stamp.isoformat(),
        "u",
        stamp.isoformat(),
        stamp.isoformat(),
    )
    bad = NewsItem(
        "Nobody Here",
        "Atlantis FC",
        "Status: 50%.",
        "50%",
        50.0,
        None,
        "premier_injuries",
        stamp.isoformat(),
        "u",
        stamp.isoformat(),
        stamp.isoformat(),
    )
    out, man = run_news_stage(
        tables, None, datetime(2026, 10, 24, 9, 0, tzinfo=UTC), sources=[_StubSource([good, bad])]
    )
    assert man["news_items_by_source"]["premier_injuries"] == {"matched": 1, "unmatched": 1}
    assert out["news_unmatched"].iloc[0]["unmatched_reason"] == "no_team"
    row = out["availability"].set_index("element").loc[int(p["id"])]
    assert row["ext_status"] == "Ruled Out" and row["cap_with_news"] == 0.0 and row["news_effect"] == "lower"


# ------------------------------------------------------------------------------------------ live model cap
def test_live_cap_uses_availability_and_records_both_p_play():
    pytest.importorskip("sklearn")
    from fpl_rocket.model.minutes import availability_factor
    from fpl_rocket.model.pipeline import run_bundle_model
    from model_synth import SEASON1, SEASON2, bundle_tables, world

    h = pd.Series([1, 2, 3])
    f = availability_factor(pd.Series(["d"] * 3), pd.Series([75] * 3), h, 0.5, pd.Series([0.875] * 3))
    assert np.allclose(f, [0.875, 1 - 0.125 * 0.5, 1 - 0.125 * 0.25])
    assert np.allclose(availability_factor(pd.Series(["u"] * 3), None, h, 0.5, pd.Series([1.0] * 3)), 0)

    df = world()
    tables = bundle_tables(df, next_gw=8)
    pl = tables["players"]
    target = int(pl.loc[pl["id"] % 100 == 5, "id"].iloc[0])  # a regular, no FPL flag
    av = pd.DataFrame({"element": pl["id"].astype(int)})
    av["cap_model_only"] = fpl_cap(pl["status"], pl["chance_of_playing_next_round"])
    av["cap_with_news"] = np.where(av["element"] == target, 0.0, av["cap_model_only"])
    for c in ("p_play_unconstrained", "p_play_model_only", "p_play_with_news"):
        av[c] = np.nan
    cfg = ModelConfig(n_sims=30, horizons=(1, 2), hgb_max_iter=20, bonus_max_iter=20)
    out, man = run_bundle_model(
        {**tables, "availability": av},
        df[df["season"] == SEASON1],
        season=SEASON2,
        now=datetime(2026, 10, 10, tzinfo=UTC),
        cfg=cfg,
        n_gws=2,
    )
    a = out["availability"].set_index("element")
    assert man["model_availability_cap"] == "cap_with_news"
    assert a.loc[target, "p_play_with_news"] == 0.0 and a.loc[target, "p_play_model_only"] > 0.3
    has = a["p_play_unconstrained"].notna()
    assert (a.loc[has, "p_play_model_only"] <= a.loc[has, "p_play_unconstrained"] + 1e-12).all()
    xp = out["xp"]
    assert xp.loc[(xp["element"] == target) & (xp["gw"] == 8), "p_play"].eq(0).all()


def test_flag_applies_in_full_to_the_gw_it_refers_to_while_a_gw_is_in_progress():
    """chance_of_playing_next_round is about the NEXT event: no decay for it even when its horizon is 2."""
    pytest.importorskip("sklearn")
    from fpl_rocket.model.minutes import availability_factor
    from fpl_rocket.model.pipeline import bundle_targets, run_bundle_model
    from model_synth import SEASON1, SEASON2, bundle_tables, world

    st, ch = pd.Series(["s", "s", "s"]), pd.Series([0.0, 0.0, 0.0])
    h = pd.Series([2, 3, 4])  # GW in progress: the next GW is horizon 2
    assert np.allclose(availability_factor(st, ch, h, 0.5), [0.5, 0.75, 0.875])  # v0.2: decayed too early
    assert np.allclose(availability_factor(st, ch, h, 0.5, steps=pd.Series([0, 1, 2])), [0.0, 0.5, 0.75])

    df = world()
    tables = bundle_tables(df, next_gw=8)
    in_progress = {**tables, "events": tables["events"].assign(finished=tables["events"]["id"] < 7)}
    targets, _ = bundle_targets(in_progress, n_gws=2, season=SEASON2)
    assert sorted(targets["horizon"].unique()) == [2, 3] and sorted(targets["flag_steps"].unique()) == [0, 1]
    cfg = ModelConfig(n_sims=30, horizons=(1, 2, 3), hgb_max_iter=20, bonus_max_iter=20)
    out, _ = run_bundle_model(
        in_progress,
        df[df["season"] == SEASON1],
        season=SEASON2,
        now=datetime(2026, 10, 10, tzinfo=UTC),
        cfg=cfg,
        n_gws=2,
    )
    xp = out["xp"]
    injured = xp[(xp["element"] % 100 == 2) & (xp["gw"] == 8)]
    assert len(injured) and (injured["p_play"] == 0).all()
