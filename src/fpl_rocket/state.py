"""Planning-state reconstruction for any FPL entry, as a pure function over already-fetched JSON.

Inputs (all plain JSON as returned by the API; see fpl_rocket.api):
    entry          entry/{id}/
    history        entry/{id}/history/
    picks          {gw: entry/{id}/event/{gw}/picks/} - at least the last published GW; also the GW before
                   it when that GW was a free hit, and the started_event GW (initial squad)
    transfers      entry/{id}/transfers/
    bootstrap      bootstrap-static/
    initial_values {element_id: price (tenths) in round started_event} - from element-summary history
                   'value' for the players of the initial squad

Logic (docs/PLAN_v2.md section 2):
    squad      = last published picks (or the pre-free-hit picks when that GW was a free hit)
                 + transfers made after that GW's deadline (event > last published GW)
    bank       = history bank of that GW + sum(out_cost - in_cost) of the later transfers
    purchase   = element_in_cost of the latest permanent transfer-in of that player; players held since
                 the initial squad use the round-started_event value
    selling    = rules.selling_price(purchase, now_cost)
    free trans = replay of the rules state machine from started_event + 1, as a RANGE
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .rules import get_rules
from .rules.season_2026_27 import SQUAD, TRANSFERS, SquadRules, TransferRules, selling_price

JSON = dict[str, Any]


class PlanningStateError(ValueError):
    """The planning state cannot be built from the given data (e.g. invalid or empty entry)."""


@dataclass(frozen=True)
class SquadPlayer:
    element: int
    element_type: int
    team: int
    purchase_price: int  # tenths
    selling_price: int  # tenths
    current_price: int  # tenths
    purchase_source: str  # "transfer GWn" | "initial GWn value" | "budget identity" | "fallback"


@dataclass(frozen=True)
class FreeTransfers:
    lo: int
    hi: int
    exact: bool

    def __str__(self) -> str:
        return str(self.lo) if self.lo == self.hi else f"{self.lo}-{self.hi}"


@dataclass
class PlanningState:
    entry_id: int
    squad: list[SquadPlayer]
    bank: int  # tenths
    team_value: int  # FPL's official team value (incl. bank) at the last published GW, tenths
    market_value: int  # current prices of the reconstructed squad + bank, tenths
    selling_value: int  # sum of selling prices + bank, tenths (what the squad is worth to the manager)
    free_transfers: FreeTransfers  # available for the upcoming GW after pending transfers
    chips_used: list[tuple[str, int]]
    chips_available: list[JSON]
    last_published_gw: int
    upcoming_gw: int
    transfers_since_publication: int
    as_of: dict[str, str]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        chips = ",".join(f"{n}@{g}" for n, g in self.chips_used) or "-"
        return (
            f"entry {self.entry_id}: as of GW{self.last_published_gw} picks"
            f" + {self.transfers_since_publication} transfers; squad {len(self.squad)};"
            f" bank {self.bank / 10:.1f}; value {self.team_value / 10:.1f}"
            f" (now {self.market_value / 10:.1f});"
            f" selling value {self.selling_value / 10:.1f}; FT for GW{self.upcoming_gw} {self.free_transfers}"
            f" ({'exact' if self.free_transfers.exact else 'range'}); chips used {chips};"
            f" warnings {len(self.warnings)}"
        )


# --------------------------------------------------------------------------- free-transfer replay
def _after_chip_week(balances: set[int], rules: TransferRules) -> set[int]:
    if rules.chip_week_policy == "carry":
        return set(balances)
    if rules.chip_week_policy == "carry_accrue":
        return {min(rules.max_bank, b + rules.free_per_gw) for b in balances}
    if rules.chip_week_policy == "reset_one":
        return {rules.free_per_gw}
    raise ValueError(f"unknown chip_week_policy {rules.chip_week_policy!r}")


def _consistent(balance: int, n: int, cost: int, rules: TransferRules) -> bool:
    return max(0, n - balance) * rules.hit_cost == cost


def replay_free_transfers(
    history_current: list[JSON],
    chips: list[JSON],
    started_event: int,
    upcoming_gw: int,
    rules: TransferRules = TRANSFERS,
) -> tuple[set[int], list[str]]:
    """Replay the FT state machine; returns (possible balances entering `upcoming_gw`, warnings).

    Before the first deadline (started_event) changes are free and unlimited; the balance entering
    started_event + 1 is 1. Each normal GW: hits = max(0, transfers - balance) * 4 must equal the observed
    event_transfers_cost; balances that disagree are dropped. Then balance' = min(cap, max(0, b - n) + 1).
    A wildcard/free-hit week neither consumes nor (policy "carry") accrues. When no balance is consistent
    the replay restarts from every balance the observation alone allows, and a warning is returned.
    """
    by_gw = {int(h["event"]): h for h in history_current}
    chip_by_gw = {int(c["event"]): c["name"] for c in chips}
    all_balances = set(range(rules.max_bank + 1))
    possible: set[int] = {rules.free_per_gw}
    warnings: list[str] = []
    for gw in range(started_event + 1, upcoming_gw):
        row = by_gw.get(gw)
        if row is None:
            warnings.append(f"GW{gw}: no history row; free-transfer balance widened")
            possible = {min(rules.max_bank, b + rules.free_per_gw) for b in all_balances}
            continue
        n, cost = int(row["event_transfers"]), int(row["event_transfers_cost"])
        chip = chip_by_gw.get(gw)
        if chip in rules.chip_weeks_free:
            if cost:
                warnings.append(f"GW{gw}: {chip} week but transfer cost {cost}")
            possible = _after_chip_week(possible, rules)
            continue
        kept = {b for b in possible if _consistent(b, n, cost, rules)}
        if not kept:
            kept = {b for b in all_balances if _consistent(b, n, cost, rules)}
            warnings.append(
                f"GW{gw}: {n} transfers cost {cost} is inconsistent with replayed balance "
                f"{sorted(possible)}; balance reset to {sorted(kept) or 'unknown'}"
            )
            kept = kept or all_balances
        possible = {min(rules.max_bank, max(0, b - n) + rules.free_per_gw) for b in kept}
    return possible, warnings


# --------------------------------------------------------------------------- helpers
def _player_index(bootstrap: JSON) -> dict[int, JSON]:
    return {p["id"]: p for p in bootstrap["elements"]}


def _upcoming_gw(last_published: int, bootstrap: JSON) -> int:
    nxt = [e["id"] for e in bootstrap.get("events", []) if e.get("is_next")]
    return max(last_published + 1, nxt[0]) if nxt else last_published + 1


def _deadline(bootstrap: JSON, gw: int) -> str:
    for e in bootstrap.get("events", []):
        if e["id"] == gw:
            return str(e.get("deadline_time"))
    return "unknown"


def _apply_transfers(squad: list[int], transfers: list[JSON], warnings: list[str]) -> list[int]:
    current = list(squad)
    for t in sorted(transfers, key=lambda t: t["time"]):
        if t["element_out"] not in current:
            warnings.append(f"transfer out of {t['element_out']} (GW{t['event']}) not in reconstructed squad")
            continue
        current[current.index(t["element_out"])] = t["element_in"]
    return current


def chips_status(chips_used: list[tuple[str, int]], bootstrap: JSON, upcoming_gw: int) -> list[JSON]:
    """Chip windows still usable from `upcoming_gw` on (window not yet consumed, not yet expired)."""
    available = []
    for w in get_rules().chip_windows(bootstrap):
        start, stop = int(w["start_event"]), int(w["stop_event"])
        used = any(name == w["name"] and start <= gw <= stop for name, gw in chips_used)
        if not used and stop >= upcoming_gw:
            available.append({"name": w["name"], "start_event": start, "stop_event": stop})
    return available


def validate_squad(squad: list[SquadPlayer], bank: int, rules: SquadRules = SQUAD) -> list[str]:
    """Legality problems of a squad (empty list = legal)."""
    problems = []
    if len(squad) != rules.size:
        problems.append(f"squad has {len(squad)} players, expected {rules.size}")
    pos = Counter(p.element_type for p in squad)
    for et, need in rules.per_position.items():
        if pos.get(et, 0) != need:
            problems.append(f"position {et}: {pos.get(et, 0)} players, expected {need}")
    for team, n in Counter(p.team for p in squad).items():
        if n > rules.max_per_team:
            problems.append(f"{n} players from team {team} (max {rules.max_per_team})")
    if bank < 0:
        problems.append(f"negative bank {bank}")
    return problems


def _purchase_prices(
    squad_ids: list[int],
    permanent_transfers: list[JSON],
    initial_ids: list[int],
    initial_bank: int,
    initial_values: dict[int, int],
    started_event: int,
    players: dict[int, JSON],
    warnings: list[str],
    budget: int = SQUAD.budget,
) -> dict[int, tuple[int, str]]:
    """Purchase price and its source for each squad player."""
    latest_in: dict[int, JSON] = {}
    for t in sorted(permanent_transfers, key=lambda t: t["time"]):
        latest_in[t["element_in"]] = t
    initial: dict[int, tuple[int, str]] = {
        e: (int(initial_values[e]), f"initial GW{started_event} value")
        for e in initial_ids
        if e in initial_values
    }
    missing = [e for e in initial_ids if e not in initial_values]
    if len(missing) == 1:
        solved = budget - initial_bank - sum(v for v, _ in initial.values())
        initial[missing[0]] = (solved, "budget identity")
    elif missing:
        warnings.append(
            f"no round-{started_event} price for {missing}; purchase price approximated by now_cost"
        )
        for e in missing:
            initial[e] = (int(players.get(e, {}).get("now_cost", 0)), "fallback")
    elif initial_ids:
        total = sum(v for v, _ in initial.values()) + initial_bank
        if total != budget:
            warnings.append(
                f"initial squad round-{started_event} values + bank = {total}, expected {budget}: "
                "initial purchase prices are approximate"
            )
    out: dict[int, tuple[int, str]] = {}
    for e in squad_ids:
        if e in latest_in:
            out[e] = (int(latest_in[e]["element_in_cost"]), f"transfer GW{latest_in[e]['event']}")
        elif e in initial:
            out[e] = initial[e]
        else:
            warnings.append(f"no purchase record for {e}; using now_cost")
            out[e] = (int(players.get(e, {}).get("now_cost", 0)), "fallback")
    return out


def _check_bank_ledger(
    history_current: list[JSON],
    chips_by_gw: dict[int, str],
    permanent: list[JSON],
    started_event: int,
    warnings: list[str],
) -> None:
    """Bank of each non-free-hit GW must equal the initial bank plus the transfer cash flow so far."""
    rows = {int(h["event"]): h for h in history_current}
    if started_event not in rows:
        return
    bank = int(rows[started_event]["bank"])
    flows = Counter()
    for t in permanent:
        flows[int(t["event"])] += int(t["element_out_cost"]) - int(t["element_in_cost"])
    for gw in sorted(rows):
        if gw > started_event:
            bank += flows.get(gw, 0)
        if chips_by_gw.get(gw) == "freehit" or gw == started_event:
            continue
        if int(rows[gw]["bank"]) != bank:
            warnings.append(f"GW{gw}: history bank {rows[gw]['bank']} != ledger {bank}")
            bank = int(rows[gw]["bank"])


# --------------------------------------------------------------------------- main entry point
def build_planning_state(
    entry: JSON,
    history: JSON,
    picks: dict[int, JSON | None],
    transfers: list[JSON],
    bootstrap: JSON,
    initial_values: dict[int, int],
    *,
    prices_as_of: str | None = None,
    transfer_rules: TransferRules | None = None,
) -> PlanningState:
    """Reconstruct the planning state for the upcoming GW. Pure: no network access."""
    if not entry or "id" not in entry:
        raise PlanningStateError("invalid entry: no id")
    entry_id = int(entry["id"])
    current = sorted(history.get("current", []), key=lambda h: h["event"])
    if not current:
        raise PlanningStateError(f"entry {entry_id} has no published gameweeks yet")
    rules = transfer_rules or get_rules().transfer_rules_from_bootstrap(bootstrap)
    warnings: list[str] = []
    players = _player_index(bootstrap)
    started = int(entry.get("started_event") or current[0]["event"])
    last_pub = int(current[-1]["event"])
    upcoming = _upcoming_gw(last_pub, bootstrap)
    chips_used = [(c["name"], int(c["event"])) for c in history.get("chips", [])]
    chips_by_gw = {gw: name for name, gw in chips_used}
    fh_weeks = {gw for name, gw in chips_used if name in rules.reverting_chips}

    # squad: last published picks, or the pre-free-hit picks if that GW was a free hit
    base_gw = last_pub - 1 if last_pub in fh_weeks else last_pub
    base = picks.get(base_gw)
    if not base:
        raise PlanningStateError(f"entry {entry_id}: picks for GW{base_gw} are required")
    pending = [t for t in transfers if int(t["event"]) > last_pub]
    squad_ids = _apply_transfers([p["element"] for p in base["picks"]], pending, warnings)
    bank_row = next(h for h in current if int(h["event"]) == base_gw)
    bank = int(bank_row["bank"]) + sum(
        int(t["element_out_cost"]) - int(t["element_in_cost"]) for t in pending
    )

    # purchase prices: permanent transfers only (free-hit week transfers revert)
    permanent = [t for t in transfers if int(t["event"]) not in fh_weeks]
    initial = picks.get(started)
    if not initial:
        warnings.append(f"no GW{started} picks: initial-squad purchase prices unknown")
    initial_ids = [p["element"] for p in initial["picks"]] if initial else []
    initial_row = next((h for h in current if int(h["event"]) == started), None)
    initial_bank = int(initial_row["bank"]) if initial_row else 0
    purchase = _purchase_prices(
        squad_ids, permanent, initial_ids, initial_bank, initial_values, started, players, warnings
    )
    if started > 1:
        warnings.append(
            f"late entrant (started GW{started}): initial purchase prices use round-{started} values"
        )
    _check_bank_ledger(current, chips_by_gw, permanent, started, warnings)

    squad: list[SquadPlayer] = []
    for e in squad_ids:
        p = players.get(e)
        if p is None:
            warnings.append(f"player {e} not in bootstrap")
            continue
        buy, source = purchase[e]
        now = int(p["now_cost"])
        squad.append(
            SquadPlayer(e, int(p["element_type"]), int(p["team"]), buy, selling_price(buy, now), now, source)
        )

    # free transfers entering the upcoming GW, minus transfers already made for it
    balances, ft_warn = replay_free_transfers(current, history.get("chips", []), started, upcoming, rules)
    warnings += ft_warn
    remaining = {max(0, b - len(pending)) for b in balances}
    if pending and len(pending) > max(balances):
        warnings.append(
            f"{len(pending)} pending transfers exceed the FT balance {sorted(balances)}:"
            " hits or a pending chip"
        )
    ft = FreeTransfers(min(remaining), max(remaining), exact=len(remaining) == 1 and not ft_warn)

    for problem in validate_squad(squad, bank):
        if "players from team" in problem:
            problem += (
                " - FPL keeps a squad whose players changed club after selection; the next transfers"
                " must restore the club limit"
            )
        warnings.append(problem)
    latest_t = max((t["time"] for t in transfers), default="none")
    as_of = {
        "picks": f"GW{base_gw} (deadline {_deadline(bootstrap, base_gw)})",
        "latest_transfer": latest_t,
        "prices": prices_as_of or "bootstrap as supplied",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return PlanningState(
        entry_id=entry_id,
        squad=squad,
        bank=bank,
        team_value=int(current[-1]["value"]),
        market_value=sum(p.current_price for p in squad) + bank,
        selling_value=sum(p.selling_price for p in squad) + bank,
        free_transfers=ft,
        chips_used=chips_used,
        chips_available=chips_status(chips_used, bootstrap, upcoming),
        last_published_gw=last_pub,
        upcoming_gw=upcoming,
        transfers_since_publication=len(pending),
        as_of=as_of,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- fetching (thin, impure)
def fetch_inputs(
    client: Any, entry_id: int, round_prices: dict[int, dict[int, int]] | None = None
) -> dict[str, Any]:
    """Fetch everything build_planning_state needs for one entry (4-6 calls + initial-squad prices).

    round_prices ({element: {round: value}}, the bundle's round_prices table) supplies the initial-squad
    prices without API calls; a player absent from it falls back to one live element-summary call.
    """
    entry = client.entry(entry_id)
    history = client.entry_history(entry_id)
    current = history.get("current", [])
    if not current:
        raise PlanningStateError(f"entry {entry_id} has no published gameweeks yet")
    started = int(entry.get("started_event") or current[0]["event"])
    last_pub = int(current[-1]["event"])
    fh = {int(c["event"]) for c in history.get("chips", []) if c["name"] == "freehit"}
    gws = {started, last_pub} | ({last_pub - 1} if last_pub in fh else set())
    picks = {gw: client.entry_picks(entry_id, gw) for gw in sorted(gws)}
    transfers = client.entry_transfers(entry_id)
    bootstrap = client.bootstrap()
    initial_values: dict[int, int] = {}
    if picks.get(started):
        # all 15 initial players: their values also validate the budget identity (sum + bank == 1000)
        for e in initial_squad_ids(picks, started):
            value = initial_value(client, e, started, round_prices)
            if value is not None:
                initial_values[e] = value
    return {
        "entry": entry,
        "history": history,
        "picks": picks,
        "transfers": transfers,
        "bootstrap": bootstrap,
        "initial_values": initial_values,
    }


def initial_squad_ids(picks: dict[int, JSON | None], started: int) -> list[int]:
    first = picks.get(started)
    return [int(p["element"]) for p in first["picks"]] if first else []


def missing_round_prices(ids: list[int], round_prices: dict[int, dict[int, int]] | None) -> list[int]:
    """Players whose prices must come from a live element-summary call."""
    return [e for e in ids if round_prices is None or e not in round_prices]


def initial_value(
    client: Any, element: int, rnd: int, round_prices: dict[int, dict[int, int]] | None
) -> int | None:
    """Price of `element` in round `rnd`: from round_prices when the player is there, else live."""
    if round_prices is not None and element in round_prices:
        value = round_prices[element].get(rnd)
        return None if value is None else int(value)
    rows = [r for r in client.element_summary(element)["history"] if int(r["round"]) == rnd]
    return int(rows[0]["value"]) if rows else None
