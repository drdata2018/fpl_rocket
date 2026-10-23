"""Pure table builders for the app pages: plain API JSON in, pandas DataFrames out. No Streamlit, no network.

Every function here is deterministic over its inputs so the pages can be replayed from a recorded snapshot
and unit-tested offline. Prices are converted from tenths to millions (10.5), ownership to a float percent.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from .rules.season_2026_27 import POSITIONS

JSON = dict[str, Any]

STATUS_LABELS = {
    "a": "Available",
    "d": "Doubtful",
    "i": "Injured",
    "s": "Suspended",
    "u": "Unavailable",
    "n": "Not eligible",
}
CHIP_LABELS = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
}
POS_ORDER = ["GKP", "DEF", "MID", "FWD"]
TEMPLATE_SHAPE = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------------------------- gameweeks
def finished_gws(bootstrap: JSON) -> list[int]:
    """GWs whose results are final (events[].finished), ascending."""
    return sorted(int(e["id"]) for e in bootstrap.get("events", []) if e.get("finished"))


def last_finished_gw(bootstrap: JSON) -> int | None:
    gws = finished_gws(bootstrap)
    return gws[-1] if gws else None


def next_gw(bootstrap: JSON) -> int | None:
    """The upcoming GW (events[].is_next); None after the season ends."""
    nxt = [int(e["id"]) for e in bootstrap.get("events", []) if e.get("is_next")]
    return nxt[0] if nxt else None


def team_short(bootstrap: JSON) -> dict[int, str]:
    return {int(t["id"]): t["short_name"] for t in bootstrap.get("teams", [])}


# --------------------------------------------------------------------------------------------- players
def players_frame(bootstrap: JSON) -> pd.DataFrame:
    """One row per player with app-ready numeric columns."""
    teams = team_short(bootstrap)
    rows = []
    for p in bootstrap.get("elements", []):
        t_in, t_out = _i(p.get("transfers_in_event")), _i(p.get("transfers_out_event"))
        rows.append(
            {
                "id": int(p["id"]),
                "name": p.get("web_name", str(p["id"])),
                "team_id": _i(p.get("team")),
                "team": teams.get(_i(p.get("team")), "?"),
                "element_type": _i(p.get("element_type")),
                "pos": POSITIONS.get(_i(p.get("element_type")), "?"),
                "price": _i(p.get("now_cost")) / 10,
                "ownership": _f(p.get("selected_by_percent")),
                "total_points": _i(p.get("total_points")),
                "ppg": _f(p.get("points_per_game")),
                "form": _f(p.get("form")),
                "minutes": _i(p.get("minutes")),
                "goals": _i(p.get("goals_scored")),
                "assists": _i(p.get("assists")),
                "xg": _f(p.get("expected_goals")),
                "xa": _f(p.get("expected_assists")),
                "xgi": _f(p.get("expected_goal_involvements")),
                "xgc": _f(p.get("expected_goals_conceded")),
                "bonus": _i(p.get("bonus")),
                "bps": _i(p.get("bps")),
                "defcon": _i(p.get("defensive_contribution")),
                "cs": _i(p.get("clean_sheets")),
                "saves": _i(p.get("saves")),
                "status": p.get("status") or "a",
                "news": p.get("news") or "",
                "news_added": p.get("news_added"),
                "chance_next": p.get("chance_of_playing_next_round"),
                "cost_change_event": _i(p.get("cost_change_event")) / 10,
                "cost_change_start": _i(p.get("cost_change_start")) / 10,
                "transfers_in_event": t_in,
                "transfers_out_event": t_out,
                "net_transfers_event": t_in - t_out,
            }
        )
    return pd.DataFrame(rows)


def player_label(row: pd.Series | JSON) -> str:
    return f"{row['name']} ({row['team']}, {row['pos']}, {row['price']:.1f})"


# --------------------------------------------------------------------------------------------- live GWs
LIVE_STATS = (
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "bonus",
    "bps",
    "saves",
    "defensive_contribution",
)


def live_frame(lives: dict[int, JSON]) -> pd.DataFrame:
    """Stack event/{gw}/live payloads into rows (gw, element, minutes, total_points, ...)."""
    rows = []
    for gw, live in sorted(lives.items()):
        for el in live.get("elements", []):
            s = el.get("stats", {})
            rows.append({"gw": int(gw), "element": int(el["id"]), **{k: _i(s.get(k)) for k in LIVE_STATS}})
    cols = ["gw", "element", *LIVE_STATS]
    return pd.DataFrame(rows, columns=cols)


def points_lookup(live: pd.DataFrame) -> dict[tuple[int, int], int]:
    """{(gw, element): total_points}."""
    if live.empty:
        return {}
    return dict(zip(zip(live["gw"], live["element"], strict=True), live["total_points"], strict=True))


# --------------------------------------------------------------------------------------------- standings
def standings(fixtures: list[JSON], bootstrap: JSON) -> pd.DataFrame:
    """Premier League table from finished fixtures: P W D L GF GA GD Pts (sorted Pts, GD, GF, name)."""
    teams = {int(t["id"]): t["name"] for t in bootstrap.get("teams", [])}
    tab = {tid: dict(P=0, W=0, D=0, L=0, GF=0, GA=0) for tid in teams}
    for f in fixtures:
        if not f.get("finished") or f.get("team_h_score") is None or f.get("team_a_score") is None:
            continue
        h, a = int(f["team_h"]), int(f["team_a"])
        hs, as_ = int(f["team_h_score"]), int(f["team_a_score"])
        for tid, gf, ga in ((h, hs, as_), (a, as_, hs)):
            row = tab.setdefault(tid, dict(P=0, W=0, D=0, L=0, GF=0, GA=0))
            row["P"] += 1
            row["GF"] += gf
            row["GA"] += ga
            row["W" if gf > ga else "D" if gf == ga else "L"] += 1
    df = pd.DataFrame([{"Team": teams.get(t, str(t)), **r} for t, r in tab.items()])
    df["GD"] = df["GF"] - df["GA"]
    df["Pts"] = 3 * df["W"] + df["D"]
    df = df.sort_values(["Pts", "GD", "GF", "Team"], ascending=[False, False, False, True])
    df.insert(0, "Pos", range(1, len(df) + 1))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- leaders etc.
LEADER_METRICS = {
    "Points": "total_points",
    "Goals": "goals",
    "Assists": "assists",
    "Bonus": "bonus",
    "Clean sheets": "cs",
    "Saves": "saves",
    "Defensive contribution": "defcon",
}


def leaders(players: pd.DataFrame, metric: str, pos: str | None = None, n: int = 10) -> pd.DataFrame:
    col = LEADER_METRICS.get(metric, metric)
    df = players if not pos else players[players["pos"] == pos]
    df = df.sort_values([col, "total_points", "name"], ascending=[False, False, True]).head(n)
    return df[["name", "team", "pos", "price", "ownership", col, "minutes"]].reset_index(drop=True)


def gw_top_scorers(players: pd.DataFrame, live: pd.DataFrame, gw: int, n: int = 10) -> pd.DataFrame:
    g = live[live["gw"] == gw]
    df = g.merge(players[["id", "name", "team", "pos", "ownership"]], left_on="element", right_on="id")
    df = df.sort_values(["total_points", "bps"], ascending=False).head(n)
    out = df[
        ["name", "team", "pos", "total_points", "minutes", "goals_scored", "assists", "bonus", "ownership"]
    ]
    return out.rename(columns={"total_points": "points", "goals_scored": "goals"}).reset_index(drop=True)


def most_owned_returns(players: pd.DataFrame, live: pd.DataFrame, gw: int, n: int = 5) -> pd.DataFrame:
    """The n most-owned players and their GW points (ownership proxies captaincy: counts are not public)."""
    top = players.sort_values("ownership", ascending=False).head(n)
    pts = live[live["gw"] == gw].set_index("element")
    out = top[["id", "name", "team", "pos", "ownership"]].copy()
    out["points"] = [int(pts["total_points"].get(i, 0)) if not pts.empty else 0 for i in out["id"]]
    out["minutes"] = [int(pts["minutes"].get(i, 0)) if not pts.empty else 0 for i in out["id"]]
    return out.drop(columns="id").reset_index(drop=True)


def event_summary(bootstrap: JSON, gw: int) -> JSON:
    ev = next((e for e in bootstrap.get("events", []) if int(e["id"]) == gw), {})
    return {
        "average": ev.get("average_entry_score"),
        "highest": ev.get("highest_score"),
        "most_captained": ev.get("most_captained"),
        "most_vice_captained": ev.get("most_vice_captained"),
        "top_element": (ev.get("top_element_info") or {}).get("id"),
        "top_element_points": (ev.get("top_element_info") or {}).get("points"),
    }


def price_changes(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(risers, fallers) since the last deadline: cost_change_event != 0."""
    cols = [
        "name",
        "team",
        "pos",
        "price",
        "cost_change_event",
        "ownership",
        "transfers_in_event",
        "transfers_out_event",
        "net_transfers_event",
    ]
    ch = players[players["cost_change_event"] != 0]
    risers = ch[ch["cost_change_event"] > 0].sort_values(
        ["cost_change_event", "net_transfers_event"], ascending=False
    )
    fallers = ch[ch["cost_change_event"] < 0].sort_values(["cost_change_event", "net_transfers_event"])
    return risers[cols].reset_index(drop=True), fallers[cols].reset_index(drop=True)


def availability(players: pd.DataFrame) -> pd.DataFrame:
    """Players flagged (status != 'a') or with any news text, newest news first."""
    df = players[(players["status"] != "a") | (players["news"].str.len() > 0)].copy()
    df["status_label"] = df["status"].map(STATUS_LABELS).fillna(df["status"])
    df["news_added"] = pd.to_datetime(df["news_added"], utc=True, errors="coerce")
    df = df.sort_values("news_added", ascending=False, na_position="last")
    cols = ["name", "team", "pos", "status_label", "chance_next", "news", "news_added", "ownership"]
    return df[cols].reset_index(drop=True)


def template_squad(players: pd.DataFrame) -> pd.DataFrame:
    """Top-owned 2 GKP / 5 DEF / 5 MID / 3 FWD (not budget-constrained)."""
    parts = [
        players[players["pos"] == pos].sort_values("ownership", ascending=False).head(k)
        for pos, k in TEMPLATE_SHAPE.items()
    ]
    df = pd.concat(parts)
    return df[["pos", "name", "team", "price", "ownership", "total_points"]].reset_index(drop=True)


# --------------------------------------------------------------------------------------------- fixtures
def fixtures_by_team(fixtures: list[JSON], first_gw: int, last_gw: int) -> dict[int, dict[int, list[JSON]]]:
    """{team_id: {gw: [{opp, home, difficulty}, ...]}} for GWs in [first_gw, last_gw]. Unscheduled skipped."""
    out: dict[int, dict[int, list[JSON]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(fixtures, key=lambda f: (f.get("event") or 0, f.get("kickoff_time") or "", f["id"])):
        gw = f.get("event")
        if gw is None or not first_gw <= int(gw) <= last_gw:
            continue
        h, a = int(f["team_h"]), int(f["team_a"])
        out[h][int(gw)].append({"opp": a, "home": True, "difficulty": _i(f.get("team_h_difficulty"), 3)})
        out[a][int(gw)].append({"opp": h, "home": False, "difficulty": _i(f.get("team_a_difficulty"), 3)})
    return out


def _fx_text(fx: JSON, short: dict[int, str]) -> str:
    return f"{short.get(fx['opp'], '?')}({'H' if fx['home'] else 'A'})"


def fixture_grid(
    fixtures: list[JSON], bootstrap: JSON, first_gw: int, n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(text, difficulty) frames: one row per team, one column per GW. A double GW lists both fixtures
    ("ARS(H) + BOU(A)") and is coloured by the mean difficulty; a blank GW is "-" with no difficulty."""
    last_gw = first_gw + n - 1
    by_team = fixtures_by_team(fixtures, first_gw, last_gw)
    short = team_short(bootstrap)
    names = {int(t["id"]): t["name"] for t in bootstrap.get("teams", [])}
    gws = list(range(first_gw, last_gw + 1))
    text, diff = {}, {}
    for tid in sorted(names, key=lambda t: names[t]):
        text[names[tid]] = {}
        diff[names[tid]] = {}
        for gw in gws:
            fxs = by_team.get(tid, {}).get(gw, [])
            col = f"GW{gw}"
            text[names[tid]][col] = " + ".join(_fx_text(f, short) for f in fxs) if fxs else "-"
            diff[names[tid]][col] = sum(f["difficulty"] for f in fxs) / len(fxs) if fxs else float("nan")
    cols = [f"GW{g}" for g in gws]
    return (
        pd.DataFrame.from_dict(text, orient="index", columns=cols),
        pd.DataFrame.from_dict(diff, orient="index", columns=cols),
    )


def fixture_strip(
    team_id: int, by_team: dict[int, dict[int, list[JSON]]], gws: list[int], short: dict[int, str]
) -> str:
    """Compact next-fixtures text, e.g. "ARS(H) 4 · BOU(A) 2 · GW10 blank"."""
    parts = []
    for gw in gws:
        fxs = by_team.get(team_id, {}).get(gw, [])
        if not fxs:
            parts.append(f"GW{gw} blank")
        parts += [f"{_fx_text(f, short)} {f['difficulty']}" for f in fxs]
    return " · ".join(parts)


def fixture_strip_cells(
    team_id: int, by_team: dict[int, dict[int, list[JSON]]], gws: list[int], short: dict[int, str]
) -> tuple[dict[str, str], dict[str, float]]:
    """One cell per GW for a single team (text, difficulty) - used by the Compare page strip."""
    text, diff = {}, {}
    for gw in gws:
        fxs = by_team.get(team_id, {}).get(gw, [])
        text[f"GW{gw}"] = " + ".join(_fx_text(f, short) for f in fxs) if fxs else "-"
        diff[f"GW{gw}"] = sum(f["difficulty"] for f in fxs) / len(fxs) if fxs else float("nan")
    return text, diff


# --------------------------------------------------------------------------------------------- my team
def gw_frame(history: JSON, bootstrap: JSON) -> pd.DataFrame:
    """Per published GW: points, hits, bench, overall rank, GW average."""
    avg = {int(e["id"]): e.get("average_entry_score") for e in bootstrap.get("events", [])}
    rows = [
        {
            "gw": int(h["event"]),
            "points": int(h["points"]),
            "hits": int(h.get("event_transfers_cost") or 0),
            "net_points": int(h["points"]) - int(h.get("event_transfers_cost") or 0),
            "bench": int(h.get("points_on_bench") or 0),
            "transfers": int(h.get("event_transfers") or 0),
            "overall_rank": h.get("overall_rank"),
            "total_points": int(h.get("total_points") or 0),
            "gw_average": avg.get(int(h["event"])),
            "value": int(h.get("value") or 0) / 10,
            "bank": int(h.get("bank") or 0) / 10,
        }
        for h in sorted(history.get("current", []), key=lambda h: h["event"])
    ]
    cols = ["gw", "points", "hits", "net_points", "bench", "transfers", "overall_rank", "total_points"]
    return pd.DataFrame(rows, columns=[*cols, "gw_average", "value", "bank"])


def chips_used_frame(history: JSON) -> pd.DataFrame:
    rows = [
        {"chip": CHIP_LABELS.get(c["name"], c["name"]), "gw": int(c["event"])}
        for c in sorted(history.get("chips", []), key=lambda c: c["event"])
    ]
    return pd.DataFrame(rows, columns=["chip", "gw"])


def chips_available_frame(chips_available: list[JSON]) -> pd.DataFrame:
    rows = [
        {
            "chip": CHIP_LABELS.get(c["name"], c["name"]),
            "window": f"GW{c['start_event']}-GW{c['stop_event']}",
        }
        for c in sorted(chips_available, key=lambda c: (c["start_event"], c["name"]))
    ]
    return pd.DataFrame(rows, columns=["chip", "window"])


def captain_gap(
    picks_by_gw: dict[int, JSON | None], live: pd.DataFrame, players: pd.DataFrame
) -> pd.DataFrame:
    """Per GW: captain and his (un-multiplied) points vs the best starting-XI player's points that GW.

    Descriptive and hindsight-based. Uses the XI as picked (before automatic substitutions).
    """
    pts = points_lookup(live)
    mins = (
        dict(zip(zip(live["gw"], live["element"], strict=True), live["minutes"], strict=True))
        if len(live)
        else {}
    )
    names = dict(zip(players["id"], players["name"], strict=True))
    live_gws = set(live["gw"].unique()) if len(live) else set()
    rows = []
    for gw in sorted(picks_by_gw):
        pk = picks_by_gw[gw]
        if not pk or gw not in live_gws:  # unfinished GW: no final points yet
            continue
        xi = [p for p in pk.get("picks", []) if int(p["position"]) <= 11]
        cap = next((p for p in pk.get("picks", []) if p.get("is_captain")), None)
        if not xi or cap is None:
            continue
        best = max(xi, key=lambda p: (pts.get((gw, p["element"]), 0), -int(p["position"])))
        cap_pts = pts.get((gw, cap["element"]), 0)
        best_pts = pts.get((gw, best["element"]), 0)
        note = ""
        if mins.get((gw, cap["element"]), 0) == 0:
            vc = next((p for p in pk.get("picks", []) if p.get("is_vice_captain")), None)
            note = "captain did not play" + (
                f"; armband to {names.get(vc['element'], vc['element'])}" if vc else ""
            )
        rows.append(
            {
                "gw": gw,
                "captain": names.get(cap["element"], str(cap["element"])),
                "captain_points": cap_pts,
                "best_xi_player": names.get(best["element"], str(best["element"])),
                "best_points": best_pts,
                "gap": best_pts - cap_pts,
                "chip": CHIP_LABELS.get(pk.get("active_chip") or "", ""),
                "note": note,
            }
        )
    cols = ["gw", "captain", "captain_points", "best_xi_player", "best_points", "gap", "chip", "note"]
    return pd.DataFrame(rows, columns=cols)


def transfer_ledger(
    transfers: list[JSON], history: JSON, live: pd.DataFrame, players: pd.DataFrame, last_finished: int | None
) -> pd.DataFrame:
    """Per transfer: points of the player in minus the player out, summed over GWx..last finished GW.

    A free-hit transfer reverts after its week, so its window is that GW only. The hit column is the GW's
    total hit cost (FPL does not attribute a hit to an individual transfer). No causal claim is made.
    """
    pts = points_lookup(live)
    names = dict(zip(players["id"], players["name"], strict=True))
    chips = {int(c["event"]): c["name"] for c in history.get("chips", [])}
    cost = {int(h["event"]): int(h.get("event_transfers_cost") or 0) for h in history.get("current", [])}
    rows = []
    for t in sorted(transfers, key=lambda t: t["time"]):
        gw = int(t["event"])
        chip = chips.get(gw, "")
        end = gw if chip == "freehit" else last_finished
        if end is None or gw > end:
            window, p_in, p_out = "not started", None, None
        else:
            window = f"GW{gw}-GW{end}"
            p_in = sum(pts.get((g, t["element_in"]), 0) for g in range(gw, end + 1))
            p_out = sum(pts.get((g, t["element_out"]), 0) for g in range(gw, end + 1))
        rows.append(
            {
                "gw": gw,
                "out": names.get(t["element_out"], str(t["element_out"])),
                "out_price": int(t["element_out_cost"]) / 10,
                "in": names.get(t["element_in"], str(t["element_in"])),
                "in_price": int(t["element_in_cost"]) / 10,
                "chip": CHIP_LABELS.get(chip, ""),
                "gw_hit_cost": cost.get(gw, 0),
                "window": window,
                "points_in": p_in,
                "points_out": p_out,
                "in_minus_out": None if p_in is None else p_in - p_out,
            }
        )
    cols = [
        "gw",
        "out",
        "out_price",
        "in",
        "in_price",
        "chip",
        "gw_hit_cost",
        "window",
        "points_in",
        "points_out",
        "in_minus_out",
    ]
    return pd.DataFrame(rows, columns=cols)


def classic_leagues(entry: JSON) -> pd.DataFrame:
    """The entry's classic leagues with its rank, league size and its league total (overall phase)."""
    rows = []
    for lg in (entry.get("leagues") or {}).get("classic", []):
        phase1 = next((p for p in lg.get("active_phases") or [] if p.get("phase") == 1), {})
        rows.append(
            {
                "id": int(lg["id"]),
                "name": lg.get("name", str(lg["id"])),
                "type": "Private" if lg.get("league_type") == "x" else "Public/system",
                "rank": lg.get("entry_rank"),
                "last_rank": lg.get("entry_last_rank"),
                "size": lg.get("rank_count"),
                "my_total": phase1.get("total", entry.get("summary_overall_points")),
            }
        )
    return pd.DataFrame(rows, columns=["id", "name", "type", "rank", "last_rank", "size", "my_total"])


def league_gap(standings_page: JSON, my_entry: int, my_total: int | None) -> JSON:
    """Leader total and gap (leader total - my total) from standings page 1."""
    results = (standings_page.get("standings") or {}).get("results") or []
    if not results:
        return {"leader_total": None, "gap": None, "my_total": my_total}
    leader = min(results, key=lambda r: r.get("rank_sort") or r.get("rank") or 0)
    mine = next((r for r in results if int(r["entry"]) == int(my_entry)), None)
    total = int(mine["total"]) if mine else my_total
    lt = int(leader["total"])
    return {"leader_total": lt, "gap": None if total is None else lt - int(total), "my_total": total}


PITCH_ROWS = ("GKP", "DEF", "MID", "FWD")


def pitch(picks: JSON, players: pd.DataFrame, live: pd.DataFrame, points_gw: int | None) -> JSON:
    """{'GKP': [...], 'DEF': [...], 'MID': [...], 'FWD': [...], 'Bench': [...]} of display dicts."""
    pts = points_lookup(live)
    idx = players.set_index("id")
    out: JSON = {k: [] for k in (*PITCH_ROWS, "Bench")}
    for p in sorted(picks.get("picks", []), key=lambda p: int(p["position"])):
        e = int(p["element"])
        row = idx.loc[e] if e in idx.index else None
        card = {
            "name": row["name"] if row is not None else str(e),
            "team": row["team"] if row is not None else "?",
            "price": float(row["price"]) if row is not None else None,
            "points": pts.get((points_gw, e)) if points_gw else None,
            "badge": "C" if p.get("is_captain") else "VC" if p.get("is_vice_captain") else "",
            "pos": row["pos"] if row is not None else POSITIONS.get(int(p.get("element_type", 0)), "?"),
        }
        out["Bench" if int(p["position"]) > 11 else card["pos"]].append(card)
    return out


def planning_squad_frame(state: Any, players: pd.DataFrame) -> pd.DataFrame:
    """Planning-state squad as a table (prices in millions)."""
    idx = players.set_index("id")
    rows = []
    for sp in sorted(state.squad, key=lambda s: (s.element_type, -s.current_price)):
        known = sp.element in idx.index
        rows.append(
            {
                "pos": POSITIONS.get(sp.element_type, "?"),
                "name": idx.at[sp.element, "name"] if known else str(sp.element),
                "team": idx.at[sp.element, "team"] if known else "?",
                "purchase": sp.purchase_price / 10,
                "selling": sp.selling_price / 10,
                "current": sp.current_price / 10,
                "purchase_source": sp.purchase_source,
            }
        )
    return pd.DataFrame(
        rows, columns=["pos", "name", "team", "purchase", "selling", "current", "purchase_source"]
    )


def pending_transfers_frame(
    transfers: list[JSON], last_published: int, players: pd.DataFrame
) -> pd.DataFrame:
    names = dict(zip(players["id"], players["name"], strict=True))
    rows = [
        {
            "gw": int(t["event"]),
            "out": names.get(t["element_out"], str(t["element_out"])),
            "in": names.get(t["element_in"], str(t["element_in"])),
            "out_price": int(t["element_out_cost"]) / 10,
            "in_price": int(t["element_in_cost"]) / 10,
            "time": t["time"],
        }
        for t in sorted(transfers, key=lambda t: t["time"])
        if int(t["event"]) > last_published
    ]
    return pd.DataFrame(rows, columns=["gw", "out", "in", "out_price", "in_price", "time"])


# --------------------------------------------------------------------------------------------- compare
COMPARE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("Price", "price", "{:.1f}"),
    ("Ownership", "ownership", "{:.1f}%"),
    ("Total points", "total_points", "{:d}"),
    ("Points per game", "ppg", "{:.1f}"),
    ("Form", "form", "{:.1f}"),
    ("Minutes", "minutes", "{:,d}"),
    ("Goals", "goals", "{:d}"),
    ("Assists", "assists", "{:d}"),
    ("xG", "xg", "{:.2f}"),
    ("xA", "xa", "{:.2f}"),
    ("xGI", "xgi", "{:.2f}"),
    ("xGC", "xgc", "{:.2f}"),
    ("Bonus", "bonus", "{:d}"),
    ("BPS", "bps", "{:d}"),
    ("Defensive contribution", "defcon", "{:d}"),
    ("Clean sheets", "cs", "{:d}"),
)


def compare_table(players: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    """Metrics as rows, one column per selected player (formatted strings)."""
    idx = players.set_index("id")
    data: dict[str, list[str]] = {}
    for pid in ids:
        r = idx.loc[pid]
        col = f"{r['name']} ({r['team']})"
        vals = []
        for _, key, fmt in COMPARE_METRICS:
            v = r[key]
            vals.append(fmt.format(int(v)) if "d}" in fmt else fmt.format(float(v)))
        data[col] = vals
    return pd.DataFrame(data, index=[m[0] for m in COMPARE_METRICS])


def points_by_gw(live: pd.DataFrame, players: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    df = live[live["element"].isin(ids)][["gw", "element", "total_points", "minutes"]]
    labels = {
        int(r["id"]): f"{r['name']} ({r['team']})" for _, r in players[players["id"].isin(ids)].iterrows()
    }
    df = df.assign(player=df["element"].map(labels)).rename(columns={"total_points": "points"})
    return df.sort_values(["gw", "player"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------- differentials
def minutes_last_n(live: pd.DataFrame, n: int = 5) -> pd.Series:
    """Minutes per element over the last n finished GWs present in `live`."""
    if live.empty:
        return pd.Series(dtype="int64")
    gws = sorted(live["gw"].unique())[-n:]
    return live[live["gw"].isin(gws)].groupby("element")["minutes"].sum()


def differentials(
    players: pd.DataFrame,
    live: pd.DataFrame,
    fixtures: list[JSON],
    bootstrap: JSON,
    *,
    max_ownership: float = 10.0,
    pos: str | None = None,
    price_range: tuple[float, float] | None = None,
    clubs: list[str] | None = None,
    n_fixtures: int = 5,
    xp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Players under an ownership threshold with descriptive columns. Default sort: total points desc.
    `xp` (xp_next(): element, xp, xp3, p_play, low_history) adds the experimental columns xp_next (next GW),
    xp_next3 (sum over the next 3 GWs) and p_play (a band) as display strings ("insufficient history" for
    low-history players)."""
    df = players[players["ownership"] < max_ownership]
    if pos:
        df = df[df["pos"] == pos]
    if price_range:
        df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]
    if clubs:
        df = df[df["team"].isin(clubs)]
    df = df.copy()
    mins = minutes_last_n(live, 5)
    df["minutes_last5"] = df["id"].map(mins).fillna(0).astype(int)
    first = next_gw(bootstrap)
    short = team_short(bootstrap)
    if first is not None:
        gws = [g for g in range(first, first + n_fixtures) if g <= 38]
        by_team = fixtures_by_team(fixtures, first, first + n_fixtures - 1)
        df["next_fixtures"] = [fixture_strip(t, by_team, gws, short) for t in df["team_id"]]
    else:
        df["next_fixtures"] = ""
    df = df.sort_values(["total_points", "ppg"], ascending=False)
    cols = [
        "name",
        "team",
        "pos",
        "price",
        "ownership",
        "total_points",
        "ppg",
        "form",
        "minutes_last5",
        "next_fixtures",
        "cost_change_start",
        "net_transfers_event",
    ]
    if xp is not None:
        df = add_xp_columns(df, xp)
        cols[cols.index("ppg") : cols.index("ppg")] = ["xp_next", "xp_next3", "p_play"]
    return df[cols].reset_index(drop=True)


# --------------------------------------------------------------------------------------------- model (xP)
INSUFFICIENT = "insufficient history"


XP_WINDOW = 3  # the longest public xP sum (decision 1, GW9): next GW and next 3 GWs; no 6-GW column anywhere
P_PLAY_LABEL = "P(play)"
# P(play) is shown as a band, never a percentage (decision 1): lower edges, highest first
P_PLAY_BANDS: tuple[tuple[float, str], ...] = (
    (0.9, "nailed"),
    (0.7, "likely"),
    (0.4, "rotation risk"),
    (0.0, "doubtful"),
)


def xp_label(version: str | None) -> str:
    return f"xP next GW ({version or 'v?'}, experimental)"


def xp3_label(version: str | None) -> str:
    return f"xP next {XP_WINDOW} GWs ({version or 'v?'}, experimental, sum over listed fixtures)"


def xp_window(xp: pd.DataFrame | None, next_gw: int | None, n: int = XP_WINDOW) -> list[int]:
    """The GWs summed in "xP next n GWs": the first predicted GW >= next_gw and the n-1 calendar GWs after
    it (never past GW38). Empty when the xP table does not cover every one of them."""
    if xp is None or xp.empty:
        return []
    have = {int(g) for g in xp["gw"].unique()}
    gws = sorted(g for g in have if next_gw is None or g >= int(next_gw))
    if not gws:
        return []
    window = [g for g in range(gws[0], gws[0] + n) if g <= 38]
    return window if set(window) <= have else []


def xp_next(xp: pd.DataFrame, next_gw: int | None) -> pd.DataFrame:
    """One row per element for the first predicted GW >= next_gw: element, gw, xp, p_play, low_history
    (a missing low_history column = all False), plus xp3 = the sum of xp over xp_window(): a blank GW is a
    row with n_fixtures 0 and xp 0, a double GW's row already holds both fixtures; NaN when the table does
    not cover the window or the player has no row in it. Public display: next GW and next 3 GWs, never 6."""
    cols = ["element", "gw", "xp", "p_play", "low_history", "xp3"]
    if xp is None or xp.empty:
        return pd.DataFrame(columns=cols)
    gws = sorted(int(g) for g in xp["gw"].unique() if next_gw is None or int(g) >= int(next_gw))
    if not gws:
        return pd.DataFrame(columns=cols)
    x = xp[xp["gw"] == gws[0]].copy()
    if "low_history" not in x:
        x["low_history"] = False
    x["low_history"] = x["low_history"].fillna(False).astype(bool)
    if "p_play" not in x:
        x["p_play"] = float("nan")
    x = x.drop_duplicates("element")
    window = xp_window(xp, next_gw)
    if window:
        w = xp[xp["gw"].isin(window)].drop_duplicates(["element", "gw"])
        x["xp3"] = x["element"].map(w.groupby("element")["xp"].sum(min_count=1))
    else:
        x["xp3"] = float("nan")
    return x[cols].reset_index(drop=True)


def fmt_xp(value: Any, low_history: bool) -> str:
    if low_history:
        return INSUFFICIENT
    v = _f(value, float("nan"))
    return "-" if v != v else f"{v:.2f}"


def p_play_band(value: Any) -> str:
    """nailed >= 0.9, likely 0.7-0.9, rotation risk 0.4-0.7, doubtful < 0.4; "-" when unknown."""
    v = _f(value, float("nan"))
    if v != v:
        return "-"
    return next((label for edge, label in P_PLAY_BANDS if v >= edge), P_PLAY_BANDS[-1][1])


def fmt_p_play(value: Any, low_history: bool) -> str:
    """P(play) as a BAND (never a percentage); "insufficient history" for low-history players."""
    return INSUFFICIENT if low_history else p_play_band(value)


def add_xp_columns(df: pd.DataFrame, xpn: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """df + xp_next / xp_next3 / p_play display strings by player id ("-" when the model has no row)."""
    idx = xpn.set_index("element")
    out = df.copy()
    low = out[id_col].map(idx["low_history"]).fillna(False).astype(bool)
    xp3 = idx["xp3"] if "xp3" in idx else pd.Series(dtype=float)
    out["xp_next"] = [fmt_xp(v, lo) for v, lo in zip(out[id_col].map(idx["xp"]), low, strict=True)]
    out["xp_next3"] = [fmt_xp(v, lo) for v, lo in zip(out[id_col].map(xp3), low, strict=True)]
    out["p_play"] = [fmt_p_play(v, lo) for v, lo in zip(out[id_col].map(idx["p_play"]), low, strict=True)]
    return out


def compare_xp_rows(
    players: pd.DataFrame, ids: list[int], xpn: pd.DataFrame, version: str | None
) -> pd.DataFrame:
    """Three extra rows for the Compare table: xP next GW, xP next 3 GWs and the P(play) band; columns =
    the selected players."""
    idx = players.set_index("id")
    sub = add_xp_columns(pd.DataFrame({"id": ids}), xpn)
    data = {
        f"{idx.at[pid, 'name']} ({idx.at[pid, 'team']})": [x, x3, p]
        for pid, x, x3, p in zip(ids, sub["xp_next"], sub["xp_next3"], sub["p_play"], strict=True)
    }
    return pd.DataFrame(data, index=[xp_label(version), xp3_label(version), P_PLAY_LABEL])


# --------------------------------------------------------------------------------------------- planner
def plan_pitch(
    gp: Any, players: pd.DataFrame, xp: dict[tuple[int, int], float], low_history: frozenset[int] | set[int]
) -> JSON:
    """ui.pitch_html rows for one GW of a plan: XI by position (C / VC badges), bench in order (badges 1-4),
    points = one-GW xP ("insufficient history" for low-history players)."""
    idx = players.set_index("id")
    out: JSON = {k: [] for k in (*PITCH_ROWS, "Bench")}

    def card(e: int, badge: str) -> JSON:
        known = e in idx.index
        return {
            "name": idx.at[e, "name"] if known else str(e),
            "team": idx.at[e, "team"] if known else "?",
            "price": float(idx.at[e, "price"]) if known else None,
            "points": fmt_xp(xp.get((e, gp.gw), 0.0), e in low_history),
            "badge": badge,
            "pos": idx.at[e, "pos"] if known else "?",
        }

    for e in gp.xi:
        c = card(e, "C" if e == gp.captain else "VC" if e == gp.vice else "")
        out[c["pos"]].append(c)
    for i, e in enumerate(gp.bench):
        out["Bench"].append(card(e, str(i + 1)))
    return out


def plan_gw_frame(plan: Any, names: dict[int, str]) -> pd.DataFrame:
    """One row per GW of a plan: transfers, captain, hits, chip, expected points."""
    rows = []
    for gp in plan.gws:
        rows.append(
            {
                "gw": gp.gw,
                "out": ", ".join(names.get(e, str(e)) for e in gp.transfers_out) or "-",
                "in": ", ".join(names.get(e, str(e)) for e in gp.transfers_in) or "-",
                "free_transfers": gp.free_transfers,
                "hits": gp.hits,
                "captain": names.get(gp.captain, str(gp.captain)),
                "chip": CHIP_LABELS.get(gp.chip or "", ""),
                "bank_after": gp.bank_after / 10,
                "expected_points": round(gp.expected_points, 1),
            }
        )
    cols = ["gw", "out", "in", "free_transfers", "hits", "captain", "chip", "bank_after", "expected_points"]
    return pd.DataFrame(rows, columns=cols)


def chip_frame(scenarios: list[Any]) -> pd.DataFrame:
    rows = [
        {
            "chip": CHIP_LABELS.get(sc.chip, sc.chip),
            "window": sc.window,
            "evaluated": sc.evaluated,
            "played_in": f"GW{sc.gw}" if sc.gw else "-",
            "gain": round(sc.gain, 1) if sc.status == "solved" else None,
            "status": sc.status,
            "note": sc.reason,
        }
        for sc in scenarios
    ]
    return pd.DataFrame(rows, columns=["chip", "window", "evaluated", "played_in", "gain", "status", "note"])
