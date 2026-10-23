"""Planning-state reconstruction: synthetic state transitions + a recorded live entry."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import synth as S
from fpl_rocket.rules.season_2026_27 import TRANSFERS
from fpl_rocket.state import (
    PlanningStateError,
    SquadPlayer,
    build_planning_state,
    chips_status,
    replay_free_transfers,
    validate_squad,
)

FIX = Path(__file__).parent / "fixtures"
SQ = S.initial_squad()
BANK0 = 1000 - 15 * S.BASE_PRICE  # 250


def _build(rows, *, upcoming, picks_by_gw, transfers=(), chips=(), started=1, prices=None, values=None):
    return build_planning_state(
        S.entry(started),
        S.history(rows, chips),
        picks_by_gw,
        list(transfers),
        S.bootstrap(upcoming, prices),
        values if values is not None else S.initial_values(SQ),
    )


def _ids(state):
    return sorted(p.element for p in state.squad)


# --------------------------------------------------------------------------------------------- basics
def test_season_start_only_gw1_published():
    st = _build([S.row(1, bank=BANK0)], upcoming=2, picks_by_gw={1: S.picks(SQ, 1, BANK0)})
    assert st.free_transfers.lo == st.free_transfers.hi == 1 and st.free_transfers.exact
    assert _ids(st) == sorted(SQ) and st.bank == BANK0
    assert all(p.purchase_source.startswith("initial GW1") for p in st.squad)
    assert st.warnings == []


def test_banking_caps_at_five():
    rows = [S.row(g, bank=BANK0) for g in range(1, 9)]
    st = _build(rows, upcoming=9, picks_by_gw={1: S.picks(SQ, 1), 8: S.picks(SQ, 8)})
    assert (st.free_transfers.lo, st.free_transfers.hi) == (5, 5)
    balances, _ = replay_free_transfers(rows[:3], [], 1, 4)
    assert balances == {3}


def test_multiple_transfers_with_hits_and_buy_back():
    out1, in1 = SQ[2], S.pid(15, 1)  # DEF swap
    t = [
        S.transfer(2, out1, in1, 50, 52, 0),
        S.transfer(3, in1, out1, 52, 48, 1),  # buy the original DEF back at 48
        S.transfer(3, SQ[7], S.pid(16, 3), 50, 55, 2),
        S.transfer(3, SQ[8], S.pid(17, 3), 50, 45, 3),
    ]
    rows = [S.row(1, bank=BANK0), S.row(2, 1, 0, BANK0 - 2), S.row(3, 3, 8, BANK0 - 2 + 4 + 5 - 5)]
    sq3 = [x for x in SQ if x not in (SQ[7], SQ[8])] + [S.pid(16, 3), S.pid(17, 3)]
    st = _build(rows, upcoming=4, picks_by_gw={1: S.picks(SQ, 1), 3: S.picks(sq3, 3)}, transfers=t)
    assert st.free_transfers.lo == st.free_transfers.hi == 1 and st.free_transfers.exact
    buy = {p.element: (p.purchase_price, p.purchase_source) for p in st.squad}
    assert buy[out1] == (48, "transfer GW3")  # latest transfer-in wins over the initial value
    assert buy[S.pid(16, 3)][0] == 55
    assert st.warnings == []


def test_inconsistent_hit_cost_gives_range_and_warning():
    rows = [S.row(1, bank=BANK0), S.row(2, 2, 0, BANK0)]  # 2 free transfers in GW2 is impossible
    t = [S.transfer(2, SQ[2], S.pid(15, 1), seq=0), S.transfer(2, SQ[3], S.pid(16, 1), seq=1)]
    sq2 = [S.pid(15, 1) if x == SQ[2] else S.pid(16, 1) if x == SQ[3] else x for x in SQ]
    st = _build(rows, upcoming=3, picks_by_gw={1: S.picks(SQ, 1), 2: S.picks(sq2, 2)}, transfers=t)
    assert not st.free_transfers.exact
    assert st.free_transfers.lo < st.free_transfers.hi
    assert any("inconsistent" in w for w in st.warnings)


# ------------------------------------------------------------------------------------------ wildcard
def _wildcard_case(pending_gw4: bool):
    wc_moves = [S.transfer(3, SQ[2 + i], S.pid(15 + i, 1), 50, 50, i) for i in range(4)]
    sq3 = [S.pid(15 + SQ.index(x) - 2, 1) if x in SQ[2:6] else x for x in SQ]
    rows = [S.row(1, bank=BANK0), S.row(2, 0, 0, BANK0), S.row(3, 0, 0, BANK0)]  # WC week shows 0 transfers
    picks = {1: S.picks(SQ, 1), 3: S.picks(sq3, 3, chip="wildcard")}
    moves = list(wc_moves)
    if pending_gw4:
        moves += [S.transfer(4, S.pid(15, 1), S.pid(19, 1), 50, 45, 10)]
    return _build(rows, upcoming=4, picks_by_gw=picks, transfers=moves, chips=[("wildcard", 3)]), sq3


def test_wildcard_week_carries_balance_without_accrual():
    st, sq3 = _wildcard_case(pending_gw4=False)
    # GW2 0 transfers -> 2 entering GW3; WC week neither consumes nor adds -> 2 entering GW4
    assert (st.free_transfers.lo, st.free_transfers.hi) == (2, 2)
    assert _ids(st) == sorted(sq3)
    assert {p.purchase_source for p in st.squad if p.element == S.pid(15, 1)} == {"transfer GW3"}
    assert st.warnings == []


def test_transfers_after_wildcard_are_pending_and_reduce_ft():
    st, _ = _wildcard_case(pending_gw4=True)
    assert st.transfers_since_publication == 1
    assert (st.free_transfers.lo, st.free_transfers.hi) == (1, 1)
    assert S.pid(19, 1) in _ids(st) and S.pid(15, 1) not in _ids(st)
    assert st.bank == BANK0 + 5


def test_transfers_with_hit_then_wildcard():
    t = [S.transfer(2, SQ[2], S.pid(15, 1), seq=0), S.transfer(2, SQ[3], S.pid(16, 1), seq=1)]
    sq2 = [S.pid(15, 1) if x == SQ[2] else S.pid(16, 1) if x == SQ[3] else x for x in SQ]
    rows = [S.row(1, bank=BANK0), S.row(2, 2, 4, BANK0), S.row(3, 0, 0, BANK0), S.row(4, 0, 0, BANK0)]
    picks = {1: S.picks(SQ, 1), 4: S.picks(sq2, 4)}
    st = _build(rows, upcoming=5, picks_by_gw=picks, transfers=t, chips=[("wildcard", 3)])
    # GW2: 1 FT, 2 transfers, -4 -> 0 left -> 1 entering GW3 (WC, carry) -> 1 entering GW4 -> 2 entering GW5
    assert (st.free_transfers.lo, st.free_transfers.hi) == (2, 2) and st.free_transfers.exact
    assert st.warnings == []


def test_chip_week_policy_is_data_driven():
    rows = [S.row(1), S.row(2), S.row(3), S.row(4, 3, 4)]
    chips = [{"name": "wildcard", "event": 3}]
    carry, w1 = replay_free_transfers(rows, chips, 1, 5)
    accrue, w2 = replay_free_transfers(rows, chips, 1, 5, replace(TRANSFERS, chip_week_policy="carry_accrue"))
    assert not w1 and carry == {1}  # 2 entering GW3 -> carried -> 2 entering GW4 -> 3 transfers, 1 hit
    assert w2  # carry_accrue predicts 3 FT entering GW4 => 0 hits, contradicting the observed 4


# ------------------------------------------------------------------------------------------ free hit
def _fh_moves():
    return [S.transfer(3, SQ[i], S.pid(15 + i, 3 if i >= 7 else 1), 50, 60, i) for i in (2, 3)]


def test_free_hit_last_published_reverts_squad():
    fh_sq = [S.pid(15 + i, 1) if i in (2, 3) else x for i, x in enumerate(SQ)]
    rows = [S.row(1, bank=BANK0), S.row(2, 0, 0, BANK0), S.row(3, 0, 0, BANK0 - 20)]
    picks = {1: S.picks(SQ, 1), 2: S.picks(SQ, 2), 3: S.picks(fh_sq, 3, chip="freehit")}
    pending = [S.transfer(4, SQ[8], S.pid(18, 3), 50, 47, 50)]
    st = _build(rows, upcoming=4, picks_by_gw=picks, transfers=_fh_moves() + pending, chips=[("freehit", 3)])
    expected = sorted([S.pid(18, 3) if x == SQ[8] else x for x in SQ])
    assert _ids(st) == expected  # free-hit players gone, pending transfer applied to the reverted squad
    assert st.bank == BANK0 + 3  # pre-free-hit bank + pending delta
    assert all(p.purchase_price == 50 for p in st.squad if p.element in SQ)
    assert (st.free_transfers.lo, st.free_transfers.hi) == (1, 1)  # 2 entering GW3, carried, 1 pending
    assert st.warnings == []


def test_free_hit_earlier_ignored_for_purchase_and_bank():
    rows = [
        S.row(1, bank=BANK0),
        S.row(2, 0, 0, BANK0),
        S.row(3, 0, 0, BANK0 - 20),
        S.row(4, 1, 0, BANK0 - 5),
    ]
    later = [S.transfer(4, SQ[2], S.pid(17, 1), 50, 55, 60)]  # permanent buy in GW4
    sq4 = [S.pid(17, 1) if x == SQ[2] else x for x in SQ]
    picks = {1: S.picks(SQ, 1), 4: S.picks(sq4, 4)}
    st = _build(rows, upcoming=5, picks_by_gw=picks, transfers=_fh_moves() + later, chips=[("freehit", 3)])
    buy = {p.element: p.purchase_price for p in st.squad}
    assert buy[S.pid(17, 1)] == 55 and buy[SQ[3]] == 50
    assert st.warnings == []  # bank ledger skips the free-hit week and excludes its transfers
    assert (st.free_transfers.lo, st.free_transfers.hi) == (2, 2)  # 2 -> FH carry 2 -> 1 used -> +1


# -------------------------------------------------------------------------------- late entry & errors
def test_late_entrant():
    values = S.initial_values(SQ, 52)
    bank = 1000 - 15 * 52
    rows = [S.row(5, bank=bank), S.row(6, 1, 0, bank)]
    t = [S.transfer(6, SQ[0], S.pid(18, 0), 52, 52)]
    sq6 = [S.pid(18, 0) if x == SQ[0] else x for x in SQ]
    st = _build(
        rows,
        upcoming=7,
        picks_by_gw={5: S.picks(SQ, 5), 6: S.picks(sq6, 6)},
        transfers=t,
        started=5,
        values=values,
    )
    assert (st.free_transfers.lo, st.free_transfers.hi) == (1, 1) and st.free_transfers.exact
    assert {p.purchase_price for p in st.squad} == {52}
    assert any("late entrant" in w for w in st.warnings)
    assert not validate_squad(st.squad, st.bank)


def test_budget_identity_solves_one_missing_initial_value():
    values = S.initial_values(SQ)
    values.pop(SQ[5])
    st = _build([S.row(1, bank=BANK0 - 3)], upcoming=2, picks_by_gw={1: S.picks(SQ, 1)}, values=values)
    p = next(p for p in st.squad if p.element == SQ[5])
    assert (p.purchase_price, p.purchase_source) == (53, "budget identity")


def test_selling_price_rounds_gain_down():
    prices = {SQ[0]: 55, SQ[1]: 45}
    st = _build([S.row(1, bank=BANK0)], upcoming=2, picks_by_gw={1: S.picks(SQ, 1)}, prices=prices)
    sell = {p.element: p.selling_price for p in st.squad}
    assert sell[SQ[0]] == 52 and sell[SQ[1]] == 45


def test_pending_transfers_beyond_balance_warn():
    t = [S.transfer(2, SQ[2 + i], S.pid(15 + i, 1), seq=i) for i in range(3)]
    st = _build([S.row(1, bank=BANK0)], upcoming=2, picks_by_gw={1: S.picks(SQ, 1)}, transfers=t)
    assert st.free_transfers.lo == 0 and any("pending" in w for w in st.warnings)


def test_invalid_entry_and_empty_history_raise():
    with pytest.raises(PlanningStateError):
        build_planning_state({}, S.history([]), {}, [], S.bootstrap(2), {})
    with pytest.raises(PlanningStateError):
        build_planning_state(S.entry(), S.history([]), {}, [], S.bootstrap(2), {})


def test_validate_squad_flags_illegal_club_count():
    squad = [
        SquadPlayer(i, et, 1 if i < 4 else i, 50, 50, 50, "x")
        for i, et in enumerate([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4])
    ]
    problems = validate_squad(squad, 0)
    assert any("team 1" in p for p in problems)


def test_chip_windows_reset_mid_season():
    b = S.bootstrap(21)
    avail = {(c["name"], c["start_event"]) for c in chips_status([("wildcard", 5), ("3xc", 25)], b, 21)}
    assert ("wildcard", 20) in avail and ("wildcard", 2) not in avail and ("3xc", 20) not in avail
    early = {(c["name"], c["start_event"]) for c in chips_status([("wildcard", 5)], b, 10)}
    assert ("wildcard", 2) not in early and ("freehit", 2) in early


# ---------------------------------------------------------------------------- recorded live entry
def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_recorded_entry_7630179_after_gw7():
    d = "entry_7630179/"
    hist = _load(d + "history.json")
    transfers = _load(d + "transfers.json")
    st = build_planning_state(
        _load(d + "entry.json"),
        hist,
        {1: _load(d + "picks_1.json"), 7: _load(d + "picks_7.json")},
        transfers,
        _load("bootstrap_trim.json"),
        {int(k): v for k, v in _load(d + "initial_values.json").items()},
    )
    assert st.bank == 7 and st.team_value == 1019
    hits = {h["event"]: h["event_transfers_cost"] for h in hist["current"] if h["event_transfers_cost"]}
    assert hits == {4: 16, 5: 8} and sum(hits.values()) == 24
    assert ("wildcard", 3) in st.chips_used
    assert len(transfers) == 17 and {t["event"] for t in transfers} == {2, 3, 4, 5, 6}
    assert len(st.squad) == 15 and validate_squad(st.squad, st.bank) == []
    assert st.last_published_gw == 7 and st.upcoming_gw == 8 and st.transfers_since_publication == 0
    assert (st.free_transfers.lo, st.free_transfers.hi) == (2, 2) and st.free_transfers.exact
    assert st.warnings == []  # replayed hit costs agree, bank ledger agrees, budget identity holds
