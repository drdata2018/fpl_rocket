"""Manager state, the decision package, legality checks, execution and scoring.

validate() REJECTS an illegal package with IllegalPackage (never a silent repair). Checks, against the
season's rules (fpl_rocket.rules.common.SeasonRules) and the players as known at T:
    GW1: no sales, 15 purchases, cost <= the budget (100.0).
    transfers: sold players owned, bought players not owned / not repeated, as many in as out, bought
        players exist at T and are buyable (not removed, can_select not False)
    squad after transfers: 15 players, 2/5/5/3; at most 3 per club - when the squad made transfers, no club
        may end above max(3, its count before the transfers) (a squad over the limit only because a player
        changed clubs need not be restored at once: KNOWN_UNCERTAIN)
    budget: bank + selling prices of sold players - now_cost of bought players >= 0 (free hit included)
    free transfers / hits: declared hits == rules.hits(gw, free transfers, n, chip); n <= transfers_cap
        outside free-transfer weeks
    chip: at most one; a held chip of that name whose window covers the GW (chips announced after T are not
        held yet); supported (2024-25 assistant manager is not); wildcard not in a blocked GW; no free hit in
        the GW after a free hit
    line-up: XI of 11 distinct squad players in a legal formation, bench = the other 4 with the goalkeeper
        first (sub_positions_locked [12]), captain and vice distinct and in the XI.
execute + score: transfers at selling / buying prices (purchase price recorded), free-hit squad and bank
restored after the GW, free transfers by rules.next_free_transfers, chip consumed (in a void GW free hit /
bench boost / triple captain are reinstated); points with automatic substitutions (formation from the rules),
the armband to the vice when the captain did not play, triple captain x3, bench boost all 15, minus hits.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from ..planner.lineup import XI_SIZE, bench_order, best_xi, captain_pair, realised_points
from ..rules.common import GK, ChipWindow, SeasonRules, selling_price


class IllegalPackage(ValueError):
    """A package that breaks the season's rules: the replay stops loudly."""


@dataclass
class ManagerState:
    season: str
    squad: dict[int, int] = field(default_factory=dict)  # element -> purchase price (tenths)
    bank: int = 1000
    free_transfers: int = 1
    chips: list[ChipWindow] = field(default_factory=list)  # held, known
    used_chips: list[dict[str, Any]] = field(default_factory=list)
    last_fh_gw: int | None = None
    meta: dict[int, dict[str, int]] = field(default_factory=dict)  # last known element_type / team / cost

    def copy(self) -> ManagerState:
        return copy.deepcopy(self)

    def held(self, name: str, gw: int) -> list[ChipWindow]:
        return sorted(
            (c for c in self.chips if c.name == name and c.covers(gw)), key=lambda c: (c.stop_event, c.number)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "squad": {str(k): v for k, v in sorted(self.squad.items())},
            "bank": self.bank,
            "free_transfers": self.free_transfers,
            "chips": [c.key for c in self.chips],
            "used_chips": self.used_chips,
            "last_fh_gw": self.last_fh_gw,
        }


def new_state(rules: SeasonRules, t: pd.Timestamp | None = None) -> ManagerState:
    st = ManagerState(rules.season, {}, rules.squad.budget, 1)
    sync_chips(st, rules, t)
    return st


def sync_chips(state: ManagerState, rules: SeasonRules, t: pd.Timestamp | None) -> None:
    """Add chips announced by `t` that the manager has never held (e.g. 2021-22's second free hit)."""
    seen = {c.key for c in state.chips} | {u["key"] for u in state.used_chips}
    for c in rules.chips_known_at(t):
        if c.key not in seen:
            state.chips.append(c)


@dataclass
class Package:
    season: str
    gw: int
    transfers_in: list[int]
    transfers_out: list[int]
    xi: list[int]
    bench: list[int]
    captain: int
    vice: int
    chip: str | None = None
    hits: int = 0
    scores: dict[int, float] = field(default_factory=dict)  # the policy's xP / ranking for this GW
    solver: dict[str, Any] = field(default_factory=dict)  # status, reason, seconds, fallback
    notes: list[str] = field(default_factory=list)

    def key(self) -> dict[str, Any]:
        """The decision itself (what the leakage test compares)."""
        return {
            "transfers_in": sorted(int(e) for e in self.transfers_in),
            "transfers_out": sorted(int(e) for e in self.transfers_out),
            "xi": [int(e) for e in self.xi],
            "bench": [int(e) for e in self.bench],
            "captain": int(self.captain),
            "vice": int(self.vice),
            "chip": self.chip,
            "hits": int(self.hits),
        }

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["scores"] = {str(k): round(float(v), 4) for k, v in sorted(self.scores.items())}
        return d


# ======================================================================================= legality
def _fail(pkg: Package, msg: str) -> None:
    raise IllegalPackage(f"{pkg.season} GW{pkg.gw}: {msg}")


def validate(rules: SeasonRules, state: ManagerState, players: pd.DataFrame, pkg: Package) -> dict[str, Any]:
    """Raise IllegalPackage when `pkg` breaks a rule; else the execution facts (squad after, bank after,
    hits, n). `players` = forecast.players_frame (element, element_type, team, now_cost, buyable)."""
    gw = pkg.gw
    info = players.set_index("element")
    meta = dict(state.meta)
    for e, r in info.iterrows():
        meta[int(e)] = {
            "element_type": int(r["element_type"]),
            "team": int(r["team"]),
            "cost": int(r["now_cost"]),
        }

    def et(e: int) -> int:
        if e not in meta:
            _fail(pkg, f"player {e} unknown at T")
        return meta[e]["element_type"]

    t_in, t_out = [int(e) for e in pkg.transfers_in], [int(e) for e in pkg.transfers_out]
    if len(set(t_in)) != len(t_in) or len(set(t_out)) != len(t_out):
        _fail(pkg, "a player is bought or sold twice")
    squad = dict(state.squad)
    if (gw == 1 or not squad) and t_out:
        _fail(pkg, "nothing to sell before the first deadline")
    for e in t_out:
        if e not in squad:
            _fail(pkg, f"sells {e}, not owned")
    for e in t_in:
        if e in squad and e not in t_out:
            _fail(pkg, f"buys {e}, already owned")
        if e in t_out:
            _fail(pkg, f"sells and buys {e} in the same GW")
        if e not in info.index:
            _fail(pkg, f"buys {e}, not in the game at T")
        if not bool(info.at[e, "buyable"]):
            _fail(pkg, f"buys {e}, not selectable at T")
    if squad and len(t_in) != len(t_out):
        _fail(pkg, f"{len(t_in)} in vs {len(t_out)} out")
    n = len(t_in) if squad else 0
    # chip
    chip = pkg.chip
    window = None
    if chip is not None:
        held = state.held(chip, gw)
        if not held:
            _fail(pkg, f"chip {chip} not held / not available in GW{gw}")
        window = held[0]
        if not window.supported:
            _fail(pkg, f"chip {chip} is unsupported in the replay")
        if not rules.chip_allowed(chip, gw):
            _fail(pkg, f"chip {chip} not allowed in GW{gw}")
        if chip == "freehit" and rules.special.fh_not_consecutive and state.last_fh_gw == gw - 1:
            _fail(pkg, "free hit in consecutive GWs")
    # squad after
    sell = {e: selling_price(p, meta[e]["cost"], rules.transfers.sell_on_fee) for e, p in squad.items()}
    after = {e: p for e, p in squad.items() if e not in t_out}
    for e in t_in:
        after[e] = int(info.at[e, "now_cost"])
    if len(after) != rules.squad.size:
        _fail(pkg, f"squad of {len(after)}")
    counts = Counter(et(e) for e in after)
    for t, need in rules.squad.per_position.items():
        if counts.get(t, 0) != need:
            _fail(pkg, f"{counts.get(t, 0)} players of type {t}, need {need}")
    if t_in:
        before = Counter(meta[e]["team"] for e in squad)
        for club, c in Counter(meta[e]["team"] for e in after).items():
            if c > max(rules.squad.max_per_team, before.get(club, 0) if squad else 0):
                _fail(pkg, f"{c} players from club {club}")
    if squad:
        bank = state.bank + sum(sell[e] for e in t_out) - sum(after[e] for e in t_in)
    else:
        bank = rules.squad.budget - sum(after.values())
    if bank < 0:
        _fail(pkg, f"over budget by {-bank / 10:.1f}")
    hits = rules.hits(gw, state.free_transfers, n, chip)
    if pkg.hits != hits:
        _fail(pkg, f"declared {pkg.hits} hits, rules say {hits}")
    if not rules.free_transfer_week(gw, chip) and n > rules.transfers.transfers_cap:
        _fail(pkg, f"{n} transfers > cap {rules.transfers.transfers_cap}")
    # line-up
    xi, bench = [int(e) for e in pkg.xi], [int(e) for e in pkg.bench]
    if len(xi) != XI_SIZE or len(set(xi)) != XI_SIZE or not set(xi) <= set(after):
        _fail(pkg, "XI is not 11 distinct squad players")
    if sorted(bench) != sorted(set(after) - set(xi)) or len(bench) != rules.squad.size - XI_SIZE:
        _fail(pkg, "bench is not the other 4 squad players")
    if et(bench[0]) != GK:
        _fail(pkg, "first bench slot must be the reserve goalkeeper")
    f = Counter(et(e) for e in xi)
    for t, (lo, hi) in rules.formation.items():
        if not lo <= f.get(t, 0) <= hi:
            _fail(pkg, f"formation: {f.get(t, 0)} of type {t} outside {lo}-{hi}")
    if pkg.captain == pkg.vice or pkg.captain not in xi or pkg.vice not in xi:
        _fail(pkg, "captain / vice must be two different XI players")
    return {"after": after, "bank": bank, "hits": hits, "n": n, "window": window, "meta": meta}


# ======================================================================================= execution
def lineup_by(squad: list[int], et: dict[int, int], score: dict[int, float], formation, low=frozenset()):
    xi = best_xi(squad, et, score, formation)
    cap, vice = captain_pair(xi, score, low)
    return xi, bench_order([e for e in squad if e not in xi], et, score), cap, vice


def step(
    rules: SeasonRules,
    state: ManagerState,
    players: pd.DataFrame,
    pkg: Package,
    results: pd.DataFrame,
) -> tuple[ManagerState, dict[str, Any]]:
    """Validate, execute and score one GW. Returns (state entering the next GW, score row).
    `results`: per element of this GW (points, minutes), as played."""
    facts = validate(rules, state, players, pkg)
    gw, chip = pkg.gw, pkg.chip
    void = gw in rules.special.void_gws
    new = state.copy()
    new.meta = facts["meta"]
    before_squad = dict(state.squad)
    new.squad, new.bank = dict(facts["after"]), int(facts["bank"])
    reinstated = void and chip in ("freehit", "bboost", "3xc")
    if chip is not None and not reinstated:
        w = facts["window"]
        new.chips = [c for c in new.chips if c.key != w.key]
        new.used_chips.append({"gw": gw, "chip": chip, "key": w.key})
    if chip == "freehit":
        new.last_fh_gw = gw
    et = {e: facts["meta"][e]["element_type"] for e in new.squad}
    pts = {int(e): int(v) for e, v in results["points"].items()} if not void else {}
    mins = {int(e): int(v) for e, v in results["minutes"].items()} if not void else {}
    eff_chip = None if reinstated else chip
    res = realised_points(
        pkg.xi, pkg.bench, pkg.captain, pkg.vice, pts, mins, et, chip=eff_chip, limits=rules.formation
    )
    gross = int(res["points"])
    net = gross - rules.transfers.hit_cost * int(facts["hits"])
    gain = None
    if eff_chip == "bboost":
        gain = sum(pts.get(e, 0) for e in pkg.bench)
    elif eff_chip == "3xc":
        gain = pts.get(res["armband"], 0) if res["armband"] is not None else 0
    elif eff_chip in ("freehit", "wildcard") and before_squad:
        old = sorted(before_squad)
        old_et = {e: facts["meta"][e]["element_type"] for e in old}
        score = {e: float(pkg.scores.get(e, 0.0)) for e in old}
        xi, bench, cap, vice = lineup_by(old, old_et, score, rules.formation)
        cf = realised_points(xi, bench, cap, vice, pts, mins, old_et, limits=rules.formation)
        gain = gross - int(cf["points"])
    row = {
        "season": pkg.season,
        "gw": gw,
        "points": gross,
        "hits": int(facts["hits"]),
        "net": net,
        "transfers": int(facts["n"]),
        "chip": chip,
        "chip_reinstated": bool(reinstated),
        "chip_gain": gain,
        "captain": int(pkg.captain),
        "captain_points": pts.get(int(pkg.captain), 0),
        "armband": res["armband"],
        "auto_subs": len(res["subs"]),
        "bank": new.bank,
        "free_transfers_in": state.free_transfers,
        "squad": ",".join(str(e) for e in sorted(new.squad)),
        "void_gw": void,
    }
    # after the GW: free-hit squad restored, free transfers for the next GW
    if chip == "freehit":
        new.squad, new.bank = before_squad, state.bank
    ft_chip = None if reinstated and chip != "freehit" else chip
    new.free_transfers = rules.next_free_transfers(gw, state.free_transfers, int(facts["n"]), ft_chip)
    return new, row
