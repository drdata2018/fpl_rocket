"""FPL bootstrap-static AS KNOWN at each issue time, from github.com/Randdalf/fplcache (Unlicense).

Layout (README + cache.py of the repository, verified 2026-11-06):
`cache/{year}/{month}/{day}/{HHMM}.json.xz`,
one LZMA-compressed bootstrap-static JSON written four times a day by GitHub Actions since 2021-04-18. The
path time is `datetime.today()` on the Actions runner (UTC; it matches the commit time to the minute) taken
AFTER the HTTP response arrived, so a snapshot's content was public no later than its path time: that path
time is `snapshot_ts`. Files are only ever added (no file is modified or deleted in the history), so the
HEAD tree lists every snapshot.

Deadlines: for each season the deadline of GW t is the value of `events[t-1].deadline_time` in the LAST
snapshot whose time is <= the issue time T = deadline - 2 h (a fixed point: start from the season's final
deadline, re-pick if the snapshot then knew a different deadline). The final value (end-of-season snapshot)
and the first value seen are recorded too, with the number of distinct values across every snapshot read.

Player state at T: the last snapshot with snapshot_ts <= T (never a later one, however close). A gameweek
whose T precedes every snapshot of its season has NO row (stated in the report, never filled forward).

Everything parsed here is untrusted data: lzma + json only.
"""

from __future__ import annotations

import json
import lzma
import re
from typing import Any

import pandas as pd

from .common import ISSUE_LEAD, season_of

SOURCE = "fplcache"
REPO_URL = "https://github.com/Randdalf/fplcache"
_PATH_RE = re.compile(r"^cache/(\d{4})/(\d{1,2})/(\d{1,2})/(\d{2})(\d{2})\.json\.xz$")

PLAYER_FIELDS = (
    "id",
    "code",
    "web_name",
    "team",
    "team_code",
    "element_type",
    "now_cost",
    "status",
    "news",
    "news_added",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "selected_by_percent",
    "transfers_in_event",
    "transfers_out_event",
    "transfers_in",
    "transfers_out",
    "cost_change_event",
    "cost_change_start",
    "penalties_order",
    "direct_freekicks_order",
    "corners_and_indirect_freekicks_order",
    "form",
    "ep_next",
    "ep_this",
    "total_points",
    "event_points",
    "minutes",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "can_select",
    "removed",
)
FLOAT_FIELDS = (
    "selected_by_percent",
    "form",
    "ep_next",
    "ep_this",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
)
TEAM_FIELDS = (
    "id",
    "code",
    "name",
    "short_name",
    "strength",
    "strength_overall_home",
    "strength_overall_away",
    "strength_attack_home",
    "strength_attack_away",
    "strength_defence_home",
    "strength_defence_away",
    "unavailable",
)
EVENT_FIELDS = (
    "id",
    "deadline_time",
    "finished",
    "data_checked",
    "is_previous",
    "is_current",
    "is_next",
    "average_entry_score",
    "highest_score",
    "transfers_made",
)
# known only once the event is played: blanked for unfinished events
EVENT_AFTER_FIELDS = (
    "most_captained",
    "most_vice_captained",
    "most_selected",
    "most_transferred_in",
    "top_element",
    "chip_plays",
)


# ======================================================================================= snapshots
def snapshot_index(ls: pd.DataFrame) -> pd.DataFrame:
    """`git ls-tree` rows (blob, path) -> (path, blob, snapshot_ts), sorted by time; other files dropped."""
    rows = []
    for blob, path in zip(ls["blob"], ls["path"], strict=True):
        m = _PATH_RE.match(path)
        if not m:
            continue
        y, mo, d, hh, mm = (int(x) for x in m.groups())
        rows.append((path, blob, pd.Timestamp(year=y, month=mo, day=d, hour=hh, minute=mm, tz="UTC")))
    out = pd.DataFrame(rows, columns=["path", "blob", "snapshot_ts"])
    return out.sort_values("snapshot_ts", kind="stable").reset_index(drop=True)


def last_at_or_before(index: pd.DataFrame, t: pd.Timestamp) -> pd.Series | None:
    """The last snapshot with snapshot_ts <= t (None when there is none). Never a later one."""
    before = index[index["snapshot_ts"] <= t]
    return None if before.empty else before.iloc[-1]


def parse_snapshot(raw: bytes) -> dict[str, Any]:
    return json.loads(lzma.decompress(raw).decode("utf-8"))


def snapshot_season(d: dict[str, Any]) -> str | None:
    """Season of a bootstrap by its GW1 deadline (None when the game has no events, e.g. between seasons)."""
    ev = d.get("events") or []
    if not ev or not ev[0].get("deadline_time"):
        return None
    t = pd.Timestamp(ev[0]["deadline_time"])
    return season_of(t.year)


def event_deadlines(d: dict[str, Any]) -> dict[int, pd.Timestamp]:
    return {
        int(e["id"]): pd.Timestamp(e["deadline_time"]).tz_convert("UTC")
        for e in d.get("events") or []
        if e.get("deadline_time")
    }


# ======================================================================================= deadlines
def resolve_issue_snapshot(
    index: pd.DataFrame, gw: int, final_deadline: pd.Timestamp, deadline_of, max_iter: int = 4
) -> tuple[pd.Series | None, pd.Timestamp, int]:
    """Fixed point of (deadline as known at T, T = deadline - 2 h). `deadline_of(snapshot_row, gw)` reads the
    deadline the snapshot knew. Returns (snapshot row or None, deadline, iterations)."""
    deadline = final_deadline
    snap = None
    for it in range(1, max_iter + 1):
        snap = last_at_or_before(index, deadline - ISSUE_LEAD)
        if snap is None:
            return None, deadline, it
        known = deadline_of(snap, gw)
        if known is None or known == deadline:
            return snap, deadline, it
        deadline = known
    return snap, deadline, max_iter


def deadline_history(snaps: dict[str, dict[str, Any]], index: pd.DataFrame, season: str) -> pd.DataFrame:
    """(season, gw, snapshot_ts, deadline_utc) for every snapshot read that belongs to the season."""
    ts = dict(zip(index["path"], index["snapshot_ts"], strict=True))
    rows = []
    for path, d in snaps.items():
        if snapshot_season(d) != season:
            continue
        for gw, dl in event_deadlines(d).items():
            rows.append((season, gw, ts[path], dl))
    out = pd.DataFrame(rows, columns=["season", "gw", "snapshot_ts", "deadline_utc"])
    return out.sort_values(["gw", "snapshot_ts"], kind="stable").reset_index(drop=True)


def summarise_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Per (season, gw): first value seen, number of distinct values, number of changes in time order."""
    rows = []
    for (season, gw), g in hist.groupby(["season", "gw"], sort=True):
        vals = g.sort_values("snapshot_ts")["deadline_utc"].tolist()
        changes = sum(1 for a, b in zip(vals, vals[1:], strict=False) if a != b)
        rows.append(
            {
                "season": season,
                "gw": int(gw),
                "deadline_first_seen_utc": vals[0],
                "first_seen_snapshot_ts": g["snapshot_ts"].min(),
                "deadline_distinct_values": len(set(vals)),
                "deadline_changes": changes,
            }
        )
    return pd.DataFrame(rows)


# ========================================================================================= tables
def _frame(records: list[dict[str, Any]], fields: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([{f: r.get(f) for f in fields} for r in records], columns=list(fields))


def players_frame(d: dict[str, Any], season: str, gw: int, snapshot_ts, issue_time) -> pd.DataFrame:
    df = _frame(d.get("elements") or [], PLAYER_FIELDS).rename(columns={"id": "element"})
    for c in FLOAT_FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")
    for c in (
        "element",
        "code",
        "team",
        "team_code",
        "element_type",
        "now_cost",
        "transfers_in_event",
        "transfers_out_event",
        "transfers_in",
        "transfers_out",
        "cost_change_event",
        "cost_change_start",
        "penalties_order",
        "direct_freekicks_order",
        "corners_and_indirect_freekicks_order",
        "chance_of_playing_next_round",
        "chance_of_playing_this_round",
        "total_points",
        "event_points",
        "minutes",
        "starts",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("web_name", "status", "news"):
        df[c] = df[c].astype("string")
    df["news_added"] = pd.to_datetime(df["news_added"], utc=True, errors="coerce")
    for c in ("can_select", "removed"):
        df[c] = df[c].astype("boolean")
    return _stamp(df, season, gw, snapshot_ts, issue_time)


def teams_frame(d: dict[str, Any], season: str, gw: int, snapshot_ts, issue_time) -> pd.DataFrame:
    df = _frame(d.get("teams") or [], TEAM_FIELDS).rename(columns={"id": "team"})
    for c in TEAM_FIELDS[4:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["team"] = pd.to_numeric(df["team"]).astype("Int64")
    df["code"] = pd.to_numeric(df["code"]).astype("Int64")
    df["unavailable"] = df["unavailable"].astype("boolean")
    return _stamp(df, season, gw, snapshot_ts, issue_time)


def events_frame(d: dict[str, Any], season: str, gw: int, snapshot_ts, issue_time) -> pd.DataFrame:
    """Every event as known; the crowd fields (most_captained, chip_plays, ...) only for FINISHED events."""
    recs = []
    for e in d.get("events") or []:
        r = {f: e.get(f) for f in EVENT_FIELDS}
        done = bool(e.get("finished"))
        for f in EVENT_AFTER_FIELDS:
            v = e.get(f) if done else None
            r[f] = json.dumps(v, sort_keys=True) if f == "chip_plays" and v is not None else v
        recs.append(r)
    df = pd.DataFrame(recs, columns=[*EVENT_FIELDS, *EVENT_AFTER_FIELDS]).rename(columns={"id": "event"})
    df["deadline_time"] = pd.to_datetime(df["deadline_time"], utc=True)
    for c in (
        "event",
        "average_entry_score",
        "highest_score",
        "transfers_made",
        "most_captained",
        "most_vice_captained",
        "most_selected",
        "most_transferred_in",
        "top_element",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("finished", "data_checked", "is_previous", "is_current", "is_next"):
        df[c] = df[c].astype("boolean")
    df["chip_plays"] = df["chip_plays"].astype("string")
    return _stamp(df, season, gw, snapshot_ts, issue_time)


def _stamp(df: pd.DataFrame, season: str, gw: int, snapshot_ts, issue_time) -> pd.DataFrame:
    df.insert(0, "season", season)
    df.insert(1, "gw", int(gw))
    df["snapshot_ts"] = pd.Timestamp(snapshot_ts)
    df["issue_time_utc"] = pd.Timestamp(issue_time)
    df["lag_minutes"] = (pd.Timestamp(issue_time) - pd.Timestamp(snapshot_ts)).total_seconds() / 60.0
    return df


# ========================================================================================== rules
RULE_KEYS = ("chips", "game_settings", "game_config", "element_types", "phases")


def rules_record(d: dict[str, Any]) -> dict[str, Any]:
    """The season-rule parts of a bootstrap (chips, game_settings, game_config, squad limits per position)."""
    et = [
        {
            k: e.get(k)
            for k in (
                "id",
                "singular_name_short",
                "squad_select",
                "squad_min_play",
                "squad_max_play",
                "element_count",
                "sub_positions_locked",
            )
        }
        for e in d.get("element_types") or []
    ]
    for e in et:  # element_count moves every week (new players): not a rule
        e.pop("element_count", None)
    phases = [
        {k: p.get(k) for k in ("id", "name", "start_event", "stop_event")} for p in d.get("phases") or []
    ]
    return {
        "chips": d.get("chips"),
        "game_settings": _drop_volatile(d.get("game_settings")),
        "game_config": _drop_volatile(d.get("game_config")),
        "element_types": et,
        "phases": phases,
    }


_VOLATILE = ("featured_entries", "status")  # status: price-change timestamps, not a rule


def _drop_volatile(x):
    if isinstance(x, dict):
        return {k: _drop_volatile(v) for k, v in x.items() if k not in _VOLATILE}
    return x


def rules_hash(rec: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def flatten(x: Any, prefix: str = "") -> dict[str, Any]:
    """Nested dict/list -> {dotted.key: scalar}; chips keyed by name#number@start-stop (order-free)."""
    out: dict[str, Any] = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(x, list) and x and all(isinstance(i, dict) for i in x):
        for i, v in enumerate(x):
            key = v.get("name") or v.get("singular_name_short") or str(i)
            if "start_event" in v and "name" in v and prefix.endswith("chips"):
                key = f"{v['name']}@{v.get('start_event')}-{v.get('stop_event')}"
            elif "id" in v:
                key = f"{key}#{v['id']}"
            out.update(flatten(v, f"{prefix}[{key}]"))
    else:
        out[prefix] = json.dumps(x, sort_keys=True) if isinstance(x, list) else x
    return out


def diff_rules(a: dict[str, Any] | None, b: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """(key, before, after) for every flattened key that differs (added keys have before=<absent>)."""
    fa = flatten(a or {})
    fb = flatten(b)
    absent = "<absent>"
    return [
        (k, fa.get(k, absent), fb.get(k, absent))
        for k in sorted(set(fa) | set(fb))
        if fa.get(k, absent) != fb.get(k, absent)
    ]
