"""Frozen prospective cohorts: seeded draws, activity rule, freeze with mocked HTTP, immutability, scorer
idempotence, staged publish, within-cohort percentile."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_rocket import cohorts
from fpl_rocket.api import FPLClient

FREEZE_GW = 9


class FakeResp:
    def __init__(self, status: int, body=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _hist(events, chips=(), base_total=0, cost=None):
    cost = cost or {}
    rows, total = [], base_total
    for e in events:
        c = cost.get(e, 0)
        total += 50 + e - c
        rows.append(
            {
                "event": e,
                "points": 50 + e,
                "total_points": total,
                "overall_rank": 1000 * e,
                "bank": 5,
                "value": 1000 + e,
                "event_transfers": c // 4,
                "event_transfers_cost": c,
                "player_name": "must not be stored",
            }
        )
    return {
        "current": rows,
        "past": [{"season_name": "2025/26", "total_points": 2000}],
        "chips": [{"name": n, "event": g, "time": "x"} for n, g in chips],
    }


class World:
    """URL router for a tiny synthetic FPL: entries 1..30, ids divisible by 5 do not exist, ids divisible by
    3 joined at GW8 (inactive under the 3-GW rule), the others play GW1-10 (GW10 = after the freeze)."""

    def __init__(self, fail: set[int] | None = None, scored_up_to: int = 9, ties: bool = False):
        self.fail = set(fail or ())
        self.ties = ties  # top 100 in tie blocks of 20 whose inner order rotates with every page fetch
        self.page_fetches = 0
        self.calls: list[str] = []
        self.done = scored_up_to  # last finished and data-checked GW in the bootstrap

    def history(self, eid: int):
        if eid % 5 == 0:
            return None
        start = 8 if eid % 3 == 0 else 1
        return _hist(range(start, 11), chips=[("wildcard", 4), ("bboost", 10)], cost={10: 4 * (eid % 2)})

    def __call__(self, method, url, params=None, **kw):
        self.calls.append(url + (f"?{params}" if params else ""))
        path = url.split("/api/", 1)[1]
        if path == "bootstrap-static/":
            events = [
                {"id": g, "finished": g <= self.done, "data_checked": g <= self.done} for g in range(1, 12)
            ]
            return FakeResp(200, {"total_players": 30, "events": events})
        if path.startswith("entry/"):
            eid = int(path.split("/")[1])
            if eid in self.fail:
                return FakeResp(503, {})
            h = self.history(eid)
            return FakeResp(404, {"detail": "Not found."}) if h is None else FakeResp(200, h)
        if path.startswith("leagues-classic/314/standings/"):
            page = int(params["page_standings"])
            order = list(range(1, 101))
            if (
                self.ties
            ):  # rotate every tie block by the fetch count (as the live league does between fetches)
                k = self.page_fetches
                order = [b[k % 20 :] + b[: k % 20] for b in (order[i : i + 20] for i in range(0, 100, 20))]
                order = [x for b in order for x in b]
            self.page_fetches += 1
            results = []
            for pos in range((page - 1) * 50 + 1, min(page * 50, 100) + 1):
                q = order[pos - 1]
                rank = ((q - 1) // 20) * 20 + 1 if self.ties else q
                results.append({"entry": 100 + q, "rank": rank, "rank_sort": pos, "total": 900 - rank})
            return FakeResp(
                200, {"standings": {"results": results}, "last_updated_data": f"t{self.page_fetches}"}
            )
        return FakeResp(404, {})


def _client(world: World) -> FPLClient:
    s = cohorts.PoliteSession(min_interval=0, cooldown=0)
    s.request = world  # type: ignore[method-assign]
    return FPLClient(session=s, backoff=0)


# ------------------------------------------------------------------------------------------ draws
def test_random_draw_is_deterministic_distinct_and_in_range():
    a = [x for _, x in zip(range(25), cohorts.random_candidates(7, 30), strict=False)]
    b = [x for _, x in zip(range(25), cohorts.random_candidates(7, 30), strict=False)]
    c = [x for _, x in zip(range(25), cohorts.random_candidates(8, 30), strict=False)]
    assert a == b and a != c
    assert len(set(a)) == 25 and all(1 <= x <= 30 for x in a)
    assert sorted(cohorts.random_candidates(7, 30)) == list(range(1, 31))  # exhausts without repeats


def test_top_positions_deterministic_and_pages():
    p = cohorts.top_positions(7, 1500)
    assert p == cohorts.top_positions(7, 1500) and p != cohorts.top_positions(8, 1500)
    assert len(set(p)) == 1500 and p == sorted(p) and p[0] >= 1 and p[-1] <= 10_000
    assert cohorts.pages_for([1, 50, 51, 100, 101]) == [1, 2, 3]


# ------------------------------------------------------------------------------------------ rules
def test_activity_rule_last_three_finished_gws():
    assert cohorts.is_active(_hist(range(1, 10)), 9)
    assert cohorts.is_active(_hist([7, 8, 9]), 9)
    assert not cohorts.is_active(_hist([8, 9]), 9)
    assert not cohorts.is_active(_hist([1, 2, 3, 4, 5, 6, 7, 9]), 9)
    assert not cohorts.is_active(_hist([]), 9)
    assert not cohorts.is_active(None, 9)


def test_member_record_reads_only_gws_up_to_the_freeze():
    h = cohorts.compact_history(_hist(range(1, 11), chips=[("wildcard", 4), ("bboost", 10)]))
    assert "past" not in h and "player_name" not in h["current"][0]
    rec = cohorts.member_record(42, h, FREEZE_GW)
    gw9 = next(r for r in h["current"] if r["event"] == 9)
    assert rec["total_points"] == gw9["total_points"] and rec["overall_rank"] == 9000
    assert rec["value"] == 1009 and rec["bank"] == 5
    assert rec["chips_used"] == [{"name": "wildcard", "gw": 4}]  # the GW10 bench boost is after the freeze


# ------------------------------------------------------------------------------------------ freeze
def test_freeze_random_counts_rejections_and_resumes(tmp_path):
    world = World()
    cache = cohorts.JsonlCache(tmp_path / "rnd.jsonl")
    members, stats = cohorts.freeze_random(_client(world), 3, 30, FREEZE_GW, cache, size=5)
    assert len(members) == 5 and stats["accepted"] == 5
    ids = [m["entry_id"] for m in members]
    assert all(i % 5 and i % 3 for i in ids)
    drawn = [
        x for _, x in zip(range(stats["candidates_drawn"]), cohorts.random_candidates(3, 30), strict=False)
    ]
    assert stats["rejected_not_found"] == sum(1 for i in drawn if i % 5 == 0)
    assert stats["rejected_inactive"] == sum(1 for i in drawn if i % 5 and i % 3 == 0)
    assert stats["candidates_drawn"] == 5 + stats["rejected_not_found"] + stats["rejected_inactive"]
    assert [i for i in drawn if i % 5 and i % 3] == ids  # first qualifying in draw order
    # resume from the cache: same cohort, no request at all
    world2 = World()
    again, stats2 = cohorts.freeze_random(
        _client(world2), 3, 30, FREEZE_GW, cohorts.JsonlCache(tmp_path / "rnd.jsonl"), size=5
    )
    assert again == members and stats2 == stats and world2.calls == []
    # fresh cache, same seed: identical membership
    fresh, _ = cohorts.freeze_random(
        _client(World()), 3, 30, FREEZE_GW, cohorts.JsonlCache(tmp_path / "other.jsonl"), size=5
    )
    assert fresh == members
    assert "must not be stored" not in (tmp_path / "rnd.jsonl").read_text(encoding="utf-8")


def _top(world, tmp_path, snapshots=2, size=4):
    return cohorts.freeze_top(
        _client(world),
        3,
        FREEZE_GW,
        [cohorts.JsonlCache(tmp_path / f"p{i}.jsonl") for i in range(snapshots)],
        cohorts.JsonlCache(tmp_path / "h.jsonl"),
        size=size,
        top=100,
    )


def test_freeze_top_canonical_list(tmp_path):
    world = World()
    members, stats = _top(world, tmp_path)
    pos = cohorts.top_positions(3, 4, 100)
    assert [m["position"] for m in members] == pos
    assert [m["entry_id"] for m in members] == [100 + p for p in pos]
    assert [m["rank_at_freeze"] for m in members] == pos
    page_calls = [c for c in world.calls if "standings" in c]
    assert len(page_calls) == stats["page_requests"] == 2 * 2
    assert stats["canonical_list_size"] == 100 and stats["standings_last_updated_data"] == [
        "t1",
        "t2",
        "t3",
        "t4",
    ]
    # resume: no request, same cohort
    world2 = World()
    again, _ = _top(world2, tmp_path)
    assert again == members and world2.calls == []


def test_freeze_top_is_robust_to_tie_order_changing_between_fetches(tmp_path):
    # one pass: the tie block 41-60 straddles pages 1 and 2 and rotates between the two fetches, so a pass
    # repeats some entries and misses others
    c = _client(World(ties=True))
    rows = [r for page in (1, 2) for r in c.league_classic_standings(314, page)["standings"]["results"]]
    assert len(rows) == 100 and len({r["entry"] for r in rows}) < 100
    # the canonical list over 2 snapshots: distinct entries ordered by (rank, entry); every member unique
    members, stats = _top(World(ties=True), tmp_path, snapshots=2, size=60)
    assert stats["rows_seen"] == 200 and stats["distinct_entries_seen"] > len({r["entry"] for r in rows})
    assert stats["canonical_list_size"] == stats["distinct_entries_seen"] <= 100
    assert stats["distinct_members"] == 60 and stats["entries_rank_or_total_inconsistent"] == 0
    by_pos = sorted(members, key=lambda m: m["position"])
    keys = [(m["rank_at_freeze"], m["entry_id"]) for m in by_pos]
    assert keys == sorted(keys)


def _doc(members=None):
    members = members or {
        cohorts.RANDOM: [
            cohorts.member_record(e, cohorts.compact_history(_hist(range(1, 10))), 9) for e in (1, 2)
        ],
        cohorts.TOP10K: [
            cohorts.member_record(e, cohorts.compact_history(_hist(range(1, 10))), 9) for e in (4,)
        ],
    }
    return {"freeze_id": "20990101", "freeze_gw": FREEZE_GW, "seed": 3, "cohorts": members}


def test_write_freeze_and_immutability_check(tmp_path):
    manifest = {"seed": 3, "last_finished_gw": FREEZE_GW}
    cpath, mpath, sha = cohorts.write_freeze(tmp_path, "20990101", _doc(), manifest)
    assert sha == cohorts.sha256_bytes(cpath.read_bytes())
    assert cohorts.cohort_bytes(_doc()) == cpath.read_bytes()  # deterministic serialisation
    d = tmp_path / cohorts.COHORTS_DIR
    [(m, doc)] = cohorts.load_frozen(d, known={})
    assert m["sha256"] == sha and doc["cohorts"][cohorts.RANDOM][0]["entry_id"] == 1
    # rewriting the same content is allowed; different content is refused
    cohorts.write_freeze(tmp_path, "20990101", _doc(), manifest)
    other = _doc()
    other["cohorts"][cohorts.RANDOM].pop()
    with pytest.raises(cohorts.CohortTampered):
        cohorts.write_freeze(tmp_path, "20990101", other, manifest)
    # a second, independent record (code constant) must agree too
    with pytest.raises(cohorts.CohortTampered):
        cohorts.load_frozen(d, known={"20990101": "0" * 64})
    # a CRLF checkout of the same file (Windows autocrlf) is the same freeze
    original = cpath.read_bytes()
    cpath.write_bytes(original.replace(b"\n", b"\r\n"))
    assert cohorts.load_frozen(d, known={"20990101": sha})[0][0]["sha256"] == sha
    # any other byte change in the cohort file is refused
    cpath.write_bytes(original.replace(b'"entry_id": 2', b'"entry_id": 3'))
    with pytest.raises(cohorts.CohortTampered):
        cohorts.load_frozen(d, known={})


# ------------------------------------------------------------------------------------------ score
def _load_score_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "cohort_score.py"
    spec = importlib.util.spec_from_file_location("cohort_score_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frozen_dir(tmp_path, world: World):
    c = _client(world)
    members = {
        cohorts.RANDOM: [cohorts.member_record(e, cohorts.fetch_history(c, e), 9) for e in (1, 2, 3)],
        cohorts.TOP10K: [cohorts.member_record(e, cohorts.fetch_history(c, e), 9) for e in (4, 7)],
    }
    members[cohorts.RANDOM].append(cohorts.member_record(10, None, 9))  # vanished entry: kept
    cohorts.write_freeze(tmp_path, "20990101", _doc(members), {"last_finished_gw": FREEZE_GW})
    return tmp_path / cohorts.COHORTS_DIR


def test_score_rows_never_drop_members_and_net_of_hits(tmp_path):
    world = World()
    d = _frozen_dir(tmp_path, world)
    [(manifest, doc)] = cohorts.load_frozen(d, known={})
    c = _client(world)
    hist = {e: cohorts.fetch_history(c, e) for e in (1, 2, 3, 4, 7, 10)}
    rows = cohorts.score_rows("20990101", doc["cohorts"], hist, [10], scored_at="t")
    by = {(r["cohort"], r["entry_id"]): r for r in rows}
    assert len(rows) == 6
    assert by[(cohorts.RANDOM, 10)]["status"] == "not_found"
    r1 = by[(cohorts.RANDOM, 1)]  # odd id: one 4-point hit in GW10
    assert (r1["points"], r1["hit_cost"], r1["hits"], r1["net_points"]) == (60, 4, 1, 56)
    assert r1["chip"] == "bboost" and r1["cum_net_points"] == 56
    r2 = by[(cohorts.RANDOM, 2)]  # even id: no hit
    assert (r2["net_points"], r2["hits"]) == (60, 0)
    r3 = by[(cohorts.RANDOM, 3)]  # joined GW8 (odd: one hit), still a member
    assert r3["status"] == "ok" and r3["net_points"] == 56
    # no row for a GW (e.g. GW11 not played): kept with empty points
    rows11 = cohorts.score_rows("20990101", doc["cohorts"], hist, [11], scored_at="t")
    assert {r["status"] for r in rows11} == {"no_row", "not_found"} and len(rows11) == 6


def test_scorer_staged_publish_and_idempotence(tmp_path):
    script = _load_score_script()
    world = World()
    d = _frozen_dir(tmp_path, world)
    stage = tmp_path / "stage"
    # 1) one member fails: nothing is published, the others stay staged
    failing = World(fail={7}, scored_up_to=10)
    code = script.run(d, stage, _client(failing), retry_pause=0)
    assert code == 1
    assert not (d / cohorts.SCORES_NAME).exists() and not (d / cohorts.SUMMARY_NAME).exists()
    # 2) the next run fetches only the missing member and publishes the whole GW
    ok = World(scored_up_to=10)
    assert script.run(d, stage, _client(ok), retry_pause=0) == 0
    entry_calls = [c for c in ok.calls if "/entry/" in c]
    assert len(entry_calls) == 1 and "/entry/7/" in entry_calls[0]
    scores = cohorts.read_scores(d / cohorts.SCORES_NAME)
    assert len(scores) == 6 and set(scores["gw"]) == {10}
    summary = pd.read_csv(d / cohorts.SUMMARY_NAME)
    first = (d / cohorts.SCORES_NAME).read_bytes(), (d / cohorts.SUMMARY_NAME).read_bytes()
    # 3) idempotent: a rerun scores nothing new and leaves both files byte-identical
    again = World(scored_up_to=10)
    assert script.run(d, stage, _client(again), retry_pause=0) == 0
    assert [c for c in again.calls if "/entry/" in c] == []
    assert ((d / cohorts.SCORES_NAME).read_bytes(), (d / cohorts.SUMMARY_NAME).read_bytes()) == first
    # publishing the same rows again adds nothing
    added, _ = cohorts.publish(d, scores.to_dict("records"))
    assert added == 0 and len(cohorts.read_scores(d / cohorts.SCORES_NAME)) == 6
    rnd = summary[summary["cohort"] == cohorts.RANDOM].iloc[0]
    assert rnd["n"] == 4 and rnd["n_scored"] == 3 and rnd["share_bboost"] == 0.75
    # a tampered cohort file stops the scorer before any request
    cfile = d / "cohort_20990101.json"
    cfile.write_bytes(cfile.read_bytes() + b" ")
    assert script.run(d, stage, _client(World()), retry_pause=0) == 2


def test_summary_quantiles_and_percentile_of():
    rows = [
        {
            "freeze_id": "f",
            "gw": 10,
            "cohort": "random",
            "entry_id": i,
            "status": "ok",
            "net_points": p,
            "cum_net_points": p * 2,
            "chip": "",
        }
        for i, p in enumerate([10, 20, 30, 40, 50], 1)
    ]
    rows.append(
        {"freeze_id": "f", "gw": 10, "cohort": "random", "entry_id": 9, "status": "not_found", "chip": ""}
    )
    df = pd.DataFrame(rows).reindex(columns=list(cohorts.SCORE_COLUMNS))
    s = cohorts.summarize(df).iloc[0]
    assert (s["n"], s["n_scored"], s["mean"], s["median"], s["p25"], s["p75"]) == (
        6,
        5,
        30.0,
        30.0,
        20.0,
        40.0,
    )
    assert s["cum_median"] == 60.0
    assert cohorts.percentile_of(30, 10, "random", scores=df) == 50.0
    assert cohorts.percentile_of(5, 10, "random", scores=df) == 0.0
    assert cohorts.percentile_of(55, 10, "random", scores=df) == 100.0
    assert cohorts.percentile_of(45, 10, "random", scores=df) == 80.0
    assert cohorts.percentile_of(60, 10, "random", cumulative=True, scores=df) == 50.0
    with pytest.raises(ValueError):
        cohorts.percentile_of(30, 11, "random", scores=df)


def test_polite_session_throttles_counts_and_cools_down():
    slept: list[float] = []
    s = cohorts.PoliteSession(min_interval=5.0, cooldown=30.0, sleep=slept.append)
    replies = iter([FakeResp(429), FakeResp(200, {})])
    s.request = lambda method, url, **kw: next(replies)  # type: ignore[method-assign]
    c = FPLClient(session=s, backoff=0)
    assert c.get("bootstrap-static/") == {}
    assert s.n_requests == 2 and s.by_status == {"429": 1, "200": 1}
    assert 30.0 in slept and any(4.0 < x <= 5.0 for x in slept)
