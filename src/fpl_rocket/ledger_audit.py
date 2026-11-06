"""Independent ledger audit of blind-replay decision logs (step 4d, Part B).

Written WITHOUT the replay engine's scoring / legality code: nothing is imported from fpl_rocket.replay,
fpl_rocket.planner or the functions of fpl_rocket.rules.common (hits, next_free_transfers, chip_allowed,
selling_price, points_from_*). Used as DATA only: each season's rules tables (squad size / positions / club
limit / budget, formation limits, transfer parameters, chip windows, special events) through the SeasonRules
attributes, the as-known bootstrap snapshot at T (prices, positions, clubs, buyability; data/asknown), the
deadlines file (issue time T), and the raw per-GW points archive (data/archive/all_gws.parquet +
live_gws.parquet for the (season, GW) pairs the community archive lacks).

The auditor replays the decision log GW by GW from an empty manager and keeps its OWN state (squad with
purchase prices, bank, free transfers, chips held, last free hit). Per GW it checks the package and recomputes
the score, then compares with the engine's records (state_before of the decision, the score row):
    legality   GW1 15 purchases within 100.0; sells owned, buys new / existing / buyable at T, as many in as
               out; squad of 15 with 2/5/5/3; at most 3 per club (a club above 3 only when it did not grow
               through this GW's transfers); bank >= 0 at selling prices (purchase + floor(gain / 2) when the
               price rose, else the current price) and buying prices = now_cost at T; declared hits =
               transfers
               beyond the free ones (none in GW1, wildcard / free-hit weeks, unlimited weeks); transfers cap;
               chip held, announced by T, window covering the GW, supported, no wildcard in a blocked /
               unlimited week, no free hit right after a free hit, at most one; XI = 11 distinct squad players
               in a legal formation, bench = the other 4 with a goalkeeper first, captain != vice, both in XI
    points     bench-driven automatic substitutions (each bench player in priority order who played replaces
               the first non-playing starter he can replace keeping the formation legal; goalkeeper for
               goalkeeper), armband to the vice when the captain played 0 minutes, x3 with triple captain,
               all 15 with bench boost (no substitutions); a void GW scores 0 and returns free hit / bench
               boost / triple captain; net = points - hit_cost x hits
    state      free hit squad and bank restored after the GW; free transfers: GW1 -> 1; normal week
               min(cap, f - min(n, f) + 1); wildcard / free hit "carry" (balance kept, no +1) or "reset_one"
               (1); unlimited week: min(cap, f + 1) if no transfer else 1; top-ups raise the balance
    chip gain  recomputed with the engine's documented counterfactual (bench boost: bench points; triple
               captain: the armband player's points once more; wildcard / free hit: GW points minus the points
               of the squad held BEFORE the chip, lined up by the package's own scores - best legal XI,
               top two
               as captain / vice, no low-history exclusion - one GW only)
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GKP, DEF, MID, FWD = 1, 2, 3, 4
CHIP_WEEK_FREE = ("wildcard", "freehit")


# ================================================================================= data at T and results
class SeasonData:
    """As-known prices / positions / clubs at T per GW and the actual per-GW points."""

    def __init__(self, season: str, data_root: str | Path | None = None) -> None:
        root = Path(data_root or ROOT / "data")
        self.season = season
        dl = pd.read_parquet(root / "asknown" / "deadlines.parquet")
        dl = dl[(dl["season"] == season) & (dl["status"] == "ok")]
        self.issue = {int(g): pd.Timestamp(t) for g, t in zip(dl["gw"], dl["issue_time_utc"], strict=True)}
        self.root = root
        arch = pd.read_parquet(
            root / "archive" / "all_gws.parquet",
            columns=["season", "gw", "element", "player_name", "total_points", "minutes", "fixture"],
        )
        arch = arch[arch["season"] == season]
        live_p = root / "archive" / "live_gws.parquet"
        if live_p.exists():
            live = pd.read_parquet(live_p)
            live = live[live["season"] == season]
            have = set(arch["gw"].astype(int))
            live = live[~live["gw"].astype(int).isin(have)]
            arch = pd.concat([arch, live[[c for c in arch.columns if c in live.columns]]], ignore_index=True)
        g = arch.groupby(["gw", "element"]).agg(points=("total_points", "sum"), minutes=("minutes", "sum"))
        self.points: dict[int, dict[int, int]] = {}
        self.minutes: dict[int, dict[int, int]] = {}
        for (gw, e), r in g.iterrows():
            self.points.setdefault(int(gw), {})[int(e)] = int(r["points"])
            self.minutes.setdefault(int(gw), {})[int(e)] = int(r["minutes"])
        self.names = {
            int(e): str(n)
            for e, n in arch.drop_duplicates("element", keep="last")[["element", "player_name"]].itertuples(
                index=False
            )
        }
        self._players: dict[int, pd.DataFrame] = {}

    def players(self, gw: int) -> pd.DataFrame:
        if gw not in self._players:
            p = pd.read_parquet(
                self.root / "asknown" / "bootstrap" / "players" / self.season / f"gw{gw}.parquet"
            )
            p = p[pd.to_datetime(p["snapshot_ts"], utc=True) <= self.issue[gw]]
            p = p[p["element_type"].astype(int).isin((GKP, DEF, MID, FWD))]
            self._players[gw] = p.set_index(p["element"].astype(int))
            for e, n in zip(p["element"].astype(int), p["web_name"].astype(str), strict=True):
                self.names[int(e)] = n
        return self._players[gw]

    def name(self, e: int) -> str:
        return self.names.get(int(e), str(e))


# ================================================================================= line-ups (independent)
def legal_xi(types: list[int], limits: dict[int, tuple[int, int]]) -> bool:
    c = Counter(types)
    return len(types) == 11 and all(limits[t][0] <= c.get(t, 0) <= limits[t][1] for t in (GKP, DEF, MID, FWD))


def substitutions(
    xi: list[int], bench: list[int], minutes: dict[int, int], etype: dict[int, int], limits
) -> tuple[list[int], list[tuple[int, int]]]:
    """Bench-driven automatic substitutions (module docstring)."""
    team = list(xi)
    subs: list[tuple[int, int]] = []
    out_of_play = [e for e in xi if minutes.get(e, 0) <= 0]
    for b in bench:
        if minutes.get(b, 0) <= 0:
            continue
        for s in out_of_play:
            if s not in team:
                continue
            if (etype[s] == GKP) != (etype[b] == GKP):
                continue
            trial = [b if x == s else x for x in team]
            if legal_xi([etype[x] for x in trial], limits):
                team = trial
                subs.append((s, b))
                break
    return team, subs


def score_lineup(
    xi: list[int],
    bench: list[int],
    cap: int,
    vice: int,
    chip: str | None,
    pts: dict[int, int],
    mins: dict[int, int],
    etype: dict[int, int],
    limits,
) -> dict[str, Any]:
    if chip == "bboost":
        scorers, subs = list(xi) + list(bench), []
    else:
        scorers, subs = substitutions(xi, bench, mins, etype, limits)
    armband = cap if mins.get(cap, 0) > 0 else (vice if mins.get(vice, 0) > 0 else None)
    mult = 3 if chip == "3xc" else 2
    total = sum(pts.get(e, 0) for e in scorers)
    if armband is not None:
        total += (mult - 1) * pts.get(armband, 0)
    return {"points": total, "scorers": scorers, "subs": subs, "armband": armband, "mult": mult}


def counterfactual_lineup(squad: list[int], etype: dict[int, int], score: dict[int, float], limits):
    """Best legal XI by score (exhaustive over formations; ties: lower element id first), captain / vice =
    the top two of the XI, bench = goalkeeper then outfield by score."""
    ranked = sorted(squad, key=lambda e: (-float(score.get(e, 0.0)), e))
    best = None
    by_t = {t: [e for e in ranked if etype[e] == t] for t in (GKP, DEF, MID, FWD)}
    for d, m, f in product(range(3, 6), range(2, 6), range(1, 4)):
        if 1 + d + m + f != 11:
            continue
        need = {GKP: 1, DEF: d, MID: m, FWD: f}
        if any(not (limits[t][0] <= n <= limits[t][1]) or len(by_t[t]) < n for t, n in need.items()):
            continue
        xi = [e for t, n in need.items() for e in by_t[t][:n]]
        val = sum(float(score.get(e, 0.0)) for e in xi)
        if best is None or val > best[0] + 1e-12:
            best = (val, xi)
    xi = best[1]
    top = sorted(xi, key=lambda e: (-float(score.get(e, 0.0)), e))
    rest = [e for e in ranked if e not in xi]
    bench = [e for e in rest if etype[e] == GKP] + [e for e in rest if etype[e] != GKP]
    return xi, bench, top[0], top[1]


# ================================================================================= the ledger
@dataclass
class Ledger:
    squad: dict[int, int] = field(default_factory=dict)  # element -> purchase price
    bank: int = 0
    free: int = 1
    used: list[str] = field(default_factory=list)  # chip keys consumed
    last_fh: int | None = None
    meta: dict[int, dict[str, int]] = field(default_factory=dict)


def chip_key(c) -> str:
    return f"{c.name}#{c.number}@{c.start_event}-{c.stop_event}"


def held_chips(rules, ledger: Ledger, t: pd.Timestamp) -> list:
    out = []
    for c in rules.chips:
        if c.announced_at is not None and pd.Timestamp(c.announced_at) > t:
            continue
        if chip_key(c) in ledger.used:
            continue
        out.append(c)
    return out


def sell_value(purchase: int, current: int, fee: float) -> int:
    gain = current - purchase
    return purchase + int(gain * fee) if gain > 0 else current


def free_after(rules, gw: int, f_in: int, n: int, chip: str | None) -> int:
    tr, sp = rules.transfers, rules.special
    cap = tr.max_bank
    f = min(f_in, cap)
    if gw == 1:
        nxt = tr.free_per_gw
    elif chip in CHIP_WEEK_FREE and chip in tr.chip_weeks_free:
        nxt = {"carry": f, "carry_accrue": min(cap, f + tr.free_per_gw)}.get(
            tr.chip_week_policy, tr.free_per_gw
        )
    elif gw in sp.unlimited_free_transfer_gws:
        nxt = min(cap, f + tr.free_per_gw) if n == 0 else tr.free_per_gw
    else:
        nxt = min(cap, f - min(n, f) + tr.free_per_gw)
    if gw + 1 in sp.ft_topup:
        nxt = max(nxt, min(cap, int(sp.ft_topup[gw + 1])))
    return nxt


def audit_log(season: str, log_path: str | Path, data: SeasonData, rules) -> dict[str, Any]:
    """Replay one decision log independently. Returns {rows: [...], issues: [...], disagreements: [...]}."""
    decisions, scores = {}, {}
    with Path(log_path).open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            (decisions if rec["event"] == "decision" else scores)[int(rec["gw"])] = rec
    led = Ledger(bank=rules.squad.budget)
    limits = {int(k): tuple(v) for k, v in rules.formation.items()}
    issues: list[dict[str, Any]] = []
    disagree: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    sp = rules.special

    def issue(gw, msg):
        issues.append({"season": season, "gw": gw, "issue": msg})

    for gw in sorted(decisions):
        dec = decisions[gw]
        pkg = dec["package"]
        t = data.issue[gw]
        pl = data.players(gw)
        for e, r in pl.iterrows():
            led.meta[int(e)] = {
                "type": int(r["element_type"]),
                "team": int(r["team"]),
                "cost": int(r["now_cost"]),
            }
        held = held_chips(rules, led, t)
        # --- the engine's state entering the GW vs the ledger's
        sb = dec["state_before"]
        mine = {
            "squad": {str(k): v for k, v in sorted(led.squad.items())},
            "bank": led.bank,
            "free_transfers": led.free,
            "chips": sorted(chip_key(c) for c in held),
            "last_fh_gw": led.last_fh,
        }
        theirs = {**{k: sb[k] for k in ("squad", "bank", "free_transfers", "last_fh_gw")}}
        theirs["chips"] = sorted(sb["chips"])
        for k in mine:
            if mine[k] != theirs[k]:
                disagree.append(
                    {
                        "season": season,
                        "gw": gw,
                        "field": f"state.{k}",
                        "ledger": mine[k],
                        "engine": theirs[k],
                    }
                )
        t_in = [int(e) for e in pkg["transfers_in"]]
        t_out = [int(e) for e in pkg["transfers_out"]]
        chip = pkg.get("chip")
        first = not led.squad
        # --- transfers and squad
        if first and t_out:
            issue(gw, "sells before the first deadline")
        if len(set(t_in)) != len(t_in) or len(set(t_out)) != len(t_out):
            issue(gw, "duplicate transfer")
        for e in t_out:
            if e not in led.squad:
                issue(gw, f"sells {e}, not owned")
        for e in t_in:
            if e in led.squad:
                issue(gw, f"buys {e}, already owned")
            if e not in pl.index:
                issue(gw, f"buys {e}, not in the game at T")
            else:
                r = pl.loc[e]
                if str(r["removed"]).lower() == "true" or str(r["can_select"]).lower() == "false":
                    issue(gw, f"buys {e}, not buyable at T")
        if not first and len(t_in) != len(t_out):
            issue(gw, "transfers in != out")
        after = {e: p for e, p in led.squad.items() if e not in t_out}
        for e in t_in:
            after[e] = led.meta[e]["cost"]
        types = Counter(led.meta[e]["type"] for e in after)
        if len(after) != rules.squad.size or any(
            types.get(k, 0) != v for k, v in rules.squad.per_position.items()
        ):
            issue(gw, f"squad {len(after)} with positions {dict(types)}")
        if t_in:
            before_c = Counter(led.meta[e]["team"] for e in led.squad)
            for club, c in Counter(led.meta[e]["team"] for e in after).items():
                if c > rules.squad.max_per_team and c > before_c.get(club, 0):
                    issue(gw, f"{c} players from club {club}")
        fee = rules.transfers.sell_on_fee
        if first:
            bank = rules.squad.budget - sum(after.values())
        else:
            bank = (
                led.bank
                + sum(sell_value(led.squad[e], led.meta[e]["cost"], fee) for e in t_out)
                - sum(led.meta[e]["cost"] for e in t_in)
            )
        if bank < 0:
            issue(gw, f"over budget by {-bank}")
        n = 0 if first else len(t_in)
        free_week = gw == 1 or first or chip in CHIP_WEEK_FREE or gw in sp.unlimited_free_transfer_gws
        hits = 0 if free_week else max(0, n - led.free)
        if int(pkg["hits"]) != hits:
            issue(gw, f"declared hits {pkg['hits']} vs {hits}")
        if not free_week and n > rules.transfers.transfers_cap:
            issue(gw, f"{n} transfers above the cap")
        # --- chip
        window = None
        if chip is not None:
            cands = sorted(
                (c for c in held if c.name == chip and c.start_event <= gw <= c.stop_event),
                key=lambda c: (c.stop_event, c.number),
            )
            if not cands:
                issue(gw, f"chip {chip} not held in its window")
            else:
                window = cands[0]
                if not window.supported:
                    issue(gw, f"chip {chip} unsupported")
            if chip == "wildcard" and (gw in sp.wildcard_blocked_gws or gw in sp.unlimited_free_transfer_gws):
                issue(gw, "wildcard in a blocked GW")
            if chip == "freehit" and sp.fh_not_consecutive and led.last_fh == gw - 1:
                issue(gw, "free hit right after a free hit")
        # --- line-up
        xi, bench = [int(e) for e in pkg["xi"]], [int(e) for e in pkg["bench"]]
        cap, vice = int(pkg["captain"]), int(pkg["vice"])
        et = {e: led.meta[e]["type"] for e in after}
        if len(set(xi)) != 11 or not set(xi) <= set(after):
            issue(gw, "XI not 11 distinct squad players")
        elif not legal_xi([et[e] for e in xi], limits):
            issue(gw, "XI formation illegal")
        if sorted(bench) != sorted(set(after) - set(xi)):
            issue(gw, "bench is not the other 4")
        elif et[bench[0]] != GKP:
            issue(gw, "first bench slot is not the goalkeeper")
        if cap == vice or cap not in xi or vice not in xi:
            issue(gw, "captain / vice invalid")
        # --- points
        void = gw in sp.void_gws
        reinstated = void and chip in ("freehit", "bboost", "3xc")
        eff = None if reinstated else chip
        pts = {} if void else data.points.get(gw, {})
        mins = {} if void else data.minutes.get(gw, {})
        res = score_lineup(xi, bench, cap, vice, eff, pts, mins, et, limits)
        net = res["points"] - rules.transfers.hit_cost * hits
        gain = None
        if eff == "bboost":
            gain = sum(pts.get(e, 0) for e in bench)
        elif eff == "3xc":
            gain = pts.get(res["armband"], 0) if res["armband"] is not None else 0
        elif eff in CHIP_WEEK_FREE and led.squad:
            old = sorted(led.squad)
            oet = {e: led.meta[e]["type"] for e in old}
            sc = {e: float(pkg["scores"].get(str(e), 0.0)) for e in old}
            cxi, cb, cc, cv = counterfactual_lineup(old, oet, sc, limits)
            gain = res["points"] - score_lineup(cxi, cb, cc, cv, None, pts, mins, oet, limits)["points"]
        row = {
            "season": season,
            "gw": gw,
            "points": res["points"],
            "hits": hits,
            "net": net,
            "transfers": n,
            "chip": chip,
            "chip_reinstated": reinstated,
            "chip_gain": gain,
            "captain": cap,
            "armband": res["armband"],
            "auto_subs": len(res["subs"]),
            "bank": bank,
            "free_transfers_in": led.free,
            "squad": ",".join(str(e) for e in sorted(after)),
            "subs": res["subs"],
            "scorers": res["scorers"],
            "mult": res["mult"],
            "t_in": t_in,
            "t_out": t_out,
            "bank_before": led.bank,
        }
        rows.append(row)
        eng = scores.get(gw, {}).get("row")
        if eng is None:
            disagree.append(
                {"season": season, "gw": gw, "field": "score row", "ledger": "present", "engine": "missing"}
            )
        else:
            for k in (
                "points",
                "hits",
                "net",
                "transfers",
                "chip",
                "chip_reinstated",
                "captain",
                "armband",
                "auto_subs",
                "bank",
                "free_transfers_in",
                "squad",
                "chip_gain",
            ):
                a, b = row[k], eng.get(k)
                if k in ("armband", "chip_gain") and (a is None or b is None):
                    same = a is None and b is None
                elif k in ("armband", "chip_gain"):
                    same = int(a) == int(b)
                else:
                    same = a == b
                if not same:
                    disagree.append({"season": season, "gw": gw, "field": k, "ledger": a, "engine": b})
        # --- state for the next GW
        if chip is not None and window is not None and not reinstated:
            led.used.append(chip_key(window))
        ft_chip = None if (reinstated and chip != "freehit") else chip
        nxt = free_after(rules, gw, led.free, n, ft_chip)
        if chip == "freehit":
            led.last_fh = gw
        else:
            led.squad, led.bank = after, bank
        led.free = nxt
    extra = sorted(set(scores) - set(decisions))
    for gw in extra:
        disagree.append(
            {"season": season, "gw": gw, "field": "decision", "ledger": "missing", "engine": "score"}
        )
    return {"rows": rows, "issues": issues, "disagreements": disagree}


def audit_run(run_dir: str | Path, seasons: list[str], policies: list[str], data_root=None) -> dict[str, Any]:
    from .rules import season_rules  # the season rules DATA tables

    out: dict[str, Any] = {}
    for s in seasons:
        data = SeasonData(s, data_root)
        rules = season_rules(s)
        for p in policies:
            out[(s, p)] = audit_log(s, Path(run_dir) / s / f"{p}.jsonl", data, rules)
    return out
