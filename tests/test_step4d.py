"""Step 4d: frozen baseline record, independent ledger audit, diagnostic variants, robustness statistics and
the real final-rank mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_rocket import ledger_audit as LA
from fpl_rocket import rankmap
from fpl_rocket.replay import baseline, chips, engine, policies, robustness, variants
from fpl_rocket.replay.engine import Package
from fpl_rocket.replay.forecast import Forecasts
from fpl_rocket.rules import season_rules

# synthetic world: GK 1-2, DEF 3-7, MID 8-12, FWD 13-15, one club each; pool 16-20
TYPES = {1: 1, 2: 1, **{e: 2 for e in range(3, 8)}, **{e: 3 for e in range(8, 13)}, 13: 4, 14: 4, 15: 4}
TYPES |= {16: 2, 17: 3, 18: 4, 19: 1, 20: 3}
XI = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
BENCH = [2, 7, 12, 15]
LIMITS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}


# ================================================================================= baseline record
def test_canonical_form_drops_only_wall_clock_fields(tmp_path):
    rec = {
        "event": "decision",
        "gw": 1,
        "decide_seconds": 3.2,
        "package": {"xi": [1], "solver": {"status": "optimal", "seconds": 1.5, "reason": "x"}},
    }
    c = baseline.canonical_decision(rec)
    assert "decide_seconds" not in c and c["package"]["solver"] == {"status": "optimal", "reason": "x"}
    score = {"event": "score", "gw": 1, "row": {"net": 50}}
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(rec) + "\n" + json.dumps(score) + "\n", encoding="utf-8")
    rec2 = {
        **rec,
        "decide_seconds": 9.9,
        "package": {**rec["package"], "solver": {**rec["package"]["solver"], "seconds": 7}},
    }
    b.write_text(json.dumps(rec2) + "\n" + json.dumps(score) + "\n", encoding="utf-8")
    ra, rb = baseline.log_record(a), baseline.log_record(b)
    assert ra["sha256"] != rb["sha256"]
    assert {k: ra[k] for k in ("packages", "decisions", "scores", "total_net")} == {
        k: rb[k] for k in ("packages", "decisions", "scores", "total_net")
    }
    assert baseline.compare_logs(a, b)["identical"]
    rec3 = {**rec, "package": {**rec["package"], "xi": [2]}}
    b.write_text(json.dumps(rec3) + "\n" + json.dumps(score) + "\n", encoding="utf-8")
    cmp = baseline.compare_logs(a, b)
    assert not cmp["identical"] and cmp["packages"]["different"] == [1]


def test_baseline_v0_record_matches_the_frozen_defaults():
    rec = baseline.load("baseline_v0")
    assert rec["tag"] == "baseline-v0" and len(rec["commit"]) == 40
    assert rec["planner"] == {
        "horizon": 3,
        "decay": 0.85,
        "time_limit": 60.0,
        "restrained": "transfers <= free transfers, never a hit",
    }
    # the code's defaults are still the frozen ones (a change here needs a new baseline)
    assert rec["chip_rule"] == chips.CHIP_RULE.describe()
    from dataclasses import asdict

    assert rec["planner"] == asdict(policies.PLANNER)
    assert rec["wide_pool"] == [list(x) for x in policies.WIDE_POOL]
    assert rec["model"]["version"] == "v0.2.1"
    assert set(rec["policies"]) == set(policies.POLICY_NAMES)
    totals = {s: {p: v["total_net"] for p, v in d["policies"].items()} for s, d in rec["logs"].items()}
    assert totals["2024-25"]["minimal_model"] == 2444 and totals["2024-25"]["form_restrained"] == 2128
    for d in rec["logs"].values():
        for v in d["policies"].values():
            assert len(v["packages"]) == 64 and len(v["sha256"]) == 64


# ================================================================================= ledger audit
def test_bench_driven_substitutions_keep_formation_and_goalkeeper_rule():
    et = TYPES
    mins = {e: 90 for e in TYPES} | {1: 0, 3: 0, 4: 0}  # GK and two defenders absent
    # bench: GK 2, DEF 7, MID 12, FWD 15 -> GK 2 for 1; DEF 7 for 3; then the XI has 3 DEF only after 4 is
    # replaced by a non-defender? 4 DEF -> 3 DEF is legal, so MID 12 replaces 4
    team, subs = LA.substitutions(XI, BENCH, mins, et, LIMITS)
    assert subs == [(1, 2), (3, 7), (4, 12)]
    # three defenders out and only one defender on the bench: formation keeps 3 DEF
    xi = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]  # 3 DEF
    mins = {e: 90 for e in TYPES} | {3: 0}
    team, subs = LA.substitutions(xi, [2, 15, 6, 7], mins, et, LIMITS)
    assert subs == [(3, 6)]  # FWD 15 cannot replace a defender in a 3-DEF XI


def test_score_lineup_armband_triple_and_bench_boost():
    pts = {e: 2 for e in TYPES} | {13: 10, 8: 6}
    mins = {e: 90 for e in TYPES} | {13: 0}
    r = LA.score_lineup(XI, BENCH, 13, 8, None, pts, mins, TYPES, LIMITS)
    assert r["armband"] == 8 and r["subs"] == [(13, 7)]  # first bench outfielder: 5-4-1 stays legal
    assert r["points"] == 2 * 9 + 6 + 2 + 6  # 9 others, vice doubled, DEF 7 in
    r3 = LA.score_lineup(XI, BENCH, 8, 9, "3xc", pts, mins, TYPES, LIMITS)
    assert r3["points"] == 2 * 9 + 2 + 3 * 6
    rb = LA.score_lineup(XI, BENCH, 8, 9, "bboost", pts, mins, TYPES, LIMITS)
    assert rb["subs"] == [] and rb["points"] == sum(pts[e] for e in XI + BENCH) + 6


@pytest.mark.parametrize(
    "season,gw,f_in,n,chip,want",
    [
        ("2024-25", 1, 1, 0, None, 1),
        ("2024-25", 10, 1, 0, None, 2),
        ("2024-25", 10, 5, 0, None, 5),
        ("2024-25", 10, 3, 2, "wildcard", 3),  # carry, no +1
        ("2023-24", 10, 2, 9, "wildcard", 1),  # reset_one
        ("2023-24", 10, 2, 0, None, 2),  # cap 2 before 2024-25
        ("2022-23", 17, 1, 0, None, 2),  # unlimited week, no transfer
        ("2022-23", 17, 2, 4, None, 1),
        ("2025-26", 15, 1, 1, None, 5),  # AFCON top-up entering GW16
    ],
)
def test_free_transfer_state_machine_from_the_rules_data(season, gw, f_in, n, chip, want):
    assert LA.free_after(season_rules(season), gw, f_in, n, chip) == want


def test_selling_price_rounds_the_gain_down():
    assert (
        LA.sell_value(50, 53, 0.5) == 51
        and LA.sell_value(50, 48, 0.5) == 48
        and LA.sell_value(50, 51, 0.5) == 50
    )


class FakeData:
    def __init__(self, gws, prices):
        self.issue = {g: pd.Timestamp("2025-08-01", tz="UTC") + pd.Timedelta(days=7 * g) for g in gws}
        self.prices = prices  # {gw: {element: cost}}
        self.points = {g: {e: 2 for e in TYPES} for g in gws}
        self.minutes = {g: {e: 90 for e in TYPES} for g in gws}

    def players(self, gw):
        rows = [
            {
                "element": e,
                "element_type": t,
                "team": e,
                "now_cost": self.prices[gw].get(e, 50),
                "removed": False,
                "can_select": True,
            }
            for e, t in TYPES.items()
        ]
        return pd.DataFrame(rows).set_index("element", drop=False)

    def name(self, e):
        return str(e)


def _engine_log(tmp_path: Path, rules, data: FakeData, packages: list[Package]) -> Path:
    """Execute packages with the ENGINE and write a decision log like run_season."""
    st = engine.new_state(rules, data.issue[1])
    path = tmp_path / "log.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for p in packages:
            engine.sync_chips(st, rules, data.issue[p.gw])
            pl = data.players(p.gw).reset_index(drop=True).assign(buyable=True)
            res = pd.DataFrame(
                {
                    "points": [data.points[p.gw][e] for e in TYPES],
                    "minutes": [data.minutes[p.gw][e] for e in TYPES],
                },
                index=list(TYPES),
            )
            fh.write(
                json.dumps(
                    {"event": "decision", "gw": p.gw, "state_before": st.to_json(), "package": p.to_json()}
                )
                + "\n"
            )
            st, row = engine.step(rules, st, pl, p, res)
            row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            fh.write(json.dumps({"event": "score", "gw": p.gw, "row": row}) + "\n")
    return path


def test_ledger_audit_agrees_with_the_engine_and_catches_tampering(tmp_path):
    season = "2024-25"
    rules = season_rules(season)
    squad = sorted(e for e in TYPES if e <= 15)
    prices = {1: {}, 2: {3: 54}, 3: {3: 54, 16: 47}}
    data = FakeData([1, 2, 3], prices)
    data.minutes[2][13] = 0  # captain absent in GW2: vice takes the armband
    data.points[3][9] = 12
    pk = [
        Package(season, 1, squad, [], XI, BENCH, 13, 8, "bboost", 0, {e: 1.0 for e in squad}),
        Package(season, 2, [], [], XI, BENCH, 13, 8, None, 0, {e: 1.0 for e in squad}),
        # sell DEF 3 (bought 50, now 54 -> sells for 52), buy DEF 16 at 47: two free transfers, one used
        Package(
            season,
            3,
            [16],
            [3],
            [1, 16, 4, 5, 6, 8, 9, 10, 11, 13, 14],
            BENCH,
            9,
            8,
            "3xc",
            0,
            {e: 1.0 for e in squad + [16]},
        ),
    ]
    log = _engine_log(tmp_path, rules, data, pk)
    res = LA.audit_log(season, log, data, rules)
    assert res["issues"] == [] and res["disagreements"] == []
    assert [r["net"] for r in res["rows"]] == [30 + 2, 2 * 10 + 2 + 2, 2 * 10 + 3 * 12]
    assert res["rows"][2]["bank"] == 1000 - 750 + 52 - 47
    # tamper with a recorded score: the auditor reports it
    lines = log.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[3])
    rec["row"]["net"] += 4
    lines[3] = json.dumps(rec)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = LA.audit_log(season, log, data, rules)
    assert [(d["gw"], d["field"]) for d in res["disagreements"]] == [(2, "net")]
    # an over-budget purchase is a legality issue
    data.prices[3][16] = 400
    res = LA.audit_log(season, log, data, rules)
    assert any("over budget" in i["issue"] for i in res["issues"])


# ================================================================================= variants
def _fc() -> Forecasts:
    els = sorted(TYPES)
    p = pd.DataFrame(
        {
            "element": els,
            "element_type": [TYPES[e] for e in els],
            "team": els,
            "now_cost": [50] * len(els),
            "buyable": [True] * len(els),
            "unavailable": [e == 9 for e in els],
            "code": els,
        }
    )
    base = {(e, 1): (0.0 if e == 9 else 2.0 + e / 100) for e in els}
    model = {(e, 1): 3.0 + e / 100 for e in els}
    return Forecasts(
        "2024-25", 1, [1], p, {(e, 1): 1 for e in els}, frozenset({20}), {"baseline": base, "model": model}
    )


def test_noise_is_deterministic_and_bounded():
    t = {(e, 1): 1.0 for e in range(1, 50)}
    a = variants.perturb(t, 3, "2024-25", 7, 1e-6)
    assert a == variants.perturb(t, 3, "2024-25", 7, 1e-6)
    assert a != variants.perturb(t, 4, "2024-25", 7, 1e-6)
    assert max(abs(a[k] - 1.0) for k in t) <= 1e-6


def test_variant_forecasts_switch_one_factor():
    fc = _fc()
    assert variants.apply(variants.BASELINE, fc, None) is fc
    assert variants.apply(variants.by_name("chips_off"), fc, None) is fc
    with pytest.raises(ValueError):
        variants.apply(variants.by_name("model_flags_off"), fc, None)
    tabs = {"model_known_uncapped": {k: v + 1 for k, v in fc.tables["model"].items()}}
    out = variants.apply(variants.by_name("model_flags_off"), fc, None, tabs)
    assert (
        out.tables["model"] == tabs["model_known_uncapped"]
        and out.tables["baseline"] == fc.tables["baseline"]
    )
    assert fc.tables["model"][(1, 1)] == 3.01  # the frozen forecasts are not mutated


def test_variant_policy_without_chips_and_with_an_imposed_start():
    v = variants.by_name("flags_off_chips_off")
    pols = variants.make_variant_policies(v)
    assert all(p.allowed_chips == () for p in pols.values())
    with pytest.raises(ValueError):
        variants.make_variant_policies(variants.by_name("imposed_start"))
    imposed = sorted(e for e in TYPES if e <= 15)
    pol = variants.make_variant_policies(variants.by_name("imposed_start"), imposed)["minimal_model"]
    fc = _fc()
    rules = season_rules("2024-25")
    st = engine.new_state(rules)
    p = pol._imposed_gw1(fc, st, rules)
    assert sorted(p.transfers_in) == imposed and p.captain == 15 and p.vice == 14
    facts = engine.validate(rules, st, fc.players, p)  # legal GW1 package
    assert facts["bank"] == 1000 - 750


# ================================================================================= robustness
def test_robustness_statistics():
    d = {s: np.full(38, 2.0) for s in robustness.UNTOUCHED}
    b = robustness.pooled_bootstrap(d, robustness.UNTOUCHED, 4, resamples=200)
    assert b["mean_per_gw"] == 2.0 and b["ci_low"] == pytest.approx(2.0) and b["total"] == 228
    x = np.zeros(38)
    x[:3] = [10, 5, -1]
    t = robustness.top_share({"2021-22": x}, ("2021-22",), top=2)
    assert t["top_sum"] == 15 and t["share"] == pytest.approx(15 / 14)
    s = robustness.sign_rows({"2021-22": x}, ("2021-22",))[0]
    assert (s["a_ahead"], s["b_ahead"], s["ties"]) == (2, 1, 35)


# ================================================================================= rank mapping
def test_rank_mapping_pairs_curve_and_placement():
    assert rankmap.compact_past(None) is None
    body = {"past": [{"season_name": "2024/25", "total_points": 2401, "rank": 566255}], "current": []}
    assert rankmap.compact_past(body) == [{"season": "2024-25", "points": 2401, "rank": 566255}]
    ids = rankmap.random_ids(1, 1000, 50, exclude=range(1, 11))
    assert ids == rankmap.random_ids(1, 1000, 50, exclude=range(1, 11)) and len(set(ids)) == 50
    assert not set(ids) & set(range(1, 11))
    pairs = pd.DataFrame(
        {
            "season": "2024-25",
            "points": [2600, 2500, 2400, 2400, 2200, 2000],
            "rank": [1000, 10_000, 100_000, 100_010, 1_000_000, 4_000_000],
        }
    )
    assert rankmap.monotonic_violations(pairs) == 0 and rankmap.tie_groups(pairs) == 1  # FPL breaks ties
    bad = pd.concat([pairs, pd.DataFrame({"season": ["2024-25"], "points": [2550], "rank": [50_000]})])
    assert rankmap.monotonic_violations(bad) == 1
    c = rankmap.fit_curve(pairs, "2024-25")
    assert c.n_pairs == 6
    pl = c.place(2450, entrants=10_000_000)
    assert pl["rank_best"] == 10_000 and pl["rank_worst"] == 100_000
    assert pl["rank_est"] == pytest.approx(10**4.5, rel=0.01) and pl["pct_est"] == pytest.approx(
        0.316, rel=0.01
    )
    exact = c.place(2400)
    assert (exact["rank_best"], exact["rank_worst"]) == (100_000, 100_010)
    assert not c.place(2700)["in_range"]
