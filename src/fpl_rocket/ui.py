"""Shared Streamlit UI pieces: sidebar (team ID), per-feature error guard, formatting, colours, pitch."""

from __future__ import annotations

import html
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from . import data
from .api import EntryNotFound, FPLAPIError
from .config import APP_URL, PLACEHOLDER_TEAM_ID
from .state import PlanningStateError

log = logging.getLogger("fpl_rocket")

TEAM_KEY = "team_id"  # the loaded, validated team ID (int) for this browser session
INPUT_KEY = "team_id_input"
ERROR_KEY = "team_id_error"

# FPL's own difficulty palette (1 = easiest)
DIFFICULTY_BG = {1: "#375523", 2: "#01fc7a", 3: "#e7e7e7", 4: "#ff1751", 5: "#80072d"}
DIFFICULTY_FG = {1: "#ffffff", 2: "#111111", 3: "#111111", 4: "#ffffff", 5: "#ffffff"}


# ============================================================================================ errors
def reason(exc: BaseException) -> str:
    if isinstance(exc, EntryNotFound):
        return "team not found"
    if isinstance(exc, FPLAPIError):
        return (
            f"the FPL API did not answer (status {exc.status})"
            if exc.status
            else "the FPL API is unreachable"
        )
    if isinstance(exc, PlanningStateError):
        return str(exc)
    return f"unexpected {type(exc).__name__}"


@contextmanager
def guard(feature: str) -> Iterator[None]:
    """Render a warning instead of a traceback when one feature fails; the rest of the page continues.

    Streamlit's own control flow (rerun/stop) derives from BaseException and passes through untouched.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - per-feature degradation is the point
        log.warning("feature %r failed: %r", feature, exc)
        st.warning(f"{feature} is unavailable right now: {reason(exc)}.")


# ============================================================================================ team ID
def current_team_id() -> int | None:
    return st.session_state.get(TEAM_KEY)


def _init_from_query() -> None:
    """On the first run of a session only: accept ?id= from the URL (never written by the app)."""
    if TEAM_KEY in st.session_state:
        return
    st.session_state[TEAM_KEY] = None
    qid = st.query_params.get("id")
    if qid:
        tid, err = data.parse_team_id(qid)
        if tid is None:
            st.session_state[ERROR_KEY] = f"Link ignored: {err}"
            return
        st.session_state[INPUT_KEY] = str(tid)
        _try_load(tid)


def _try_load(tid: int) -> None:
    try:
        data.entry(tid)
    except EntryNotFound:
        st.session_state[ERROR_KEY] = f"No FPL team with ID {tid}."
        return
    except FPLAPIError as exc:
        st.session_state[ERROR_KEY] = f"Could not load team {tid}: {reason(exc)}."
        return
    st.session_state[TEAM_KEY] = tid
    st.session_state.pop(ERROR_KEY, None)


def _on_load() -> None:
    st.session_state.pop(ERROR_KEY, None)
    tid, err = data.parse_team_id(st.session_state.get(INPUT_KEY))
    if tid is None:
        st.session_state[ERROR_KEY] = err
        return
    if st.query_params.get("id") not in (None, str(tid)):
        del st.query_params["id"]  # a stale ?id= must not reload the previous team on refresh
    _try_load(tid)


def sidebar() -> Any:
    """Team ID box + loaded team caption + shareable link. Returns a footer container for timestamps."""
    _init_from_query()
    with st.sidebar:
        st.text_input("Team ID", key=INPUT_KEY, placeholder=PLACEHOLDER_TEAM_ID)
        st.button("Load", on_click=_on_load, type="primary")
        if st.session_state.get(ERROR_KEY):
            st.error(st.session_state[ERROR_KEY])
        tid = current_team_id()
        if tid is not None:
            try:
                e = data.entry(tid)
                rank = e.get("summary_overall_rank")
                rank_txt = f"overall rank {rank:,}" if rank else "no overall rank yet"
                st.caption(f"**{md_escape(e.get('name', tid))}** · {rank_txt}")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Team {tid}: {reason(exc)}")
            link = st.expander("Copy shareable link", on_change="rerun", key="share_link_open")
            if link.open:
                with link:
                    st.code(f"{_base_url()}?id={tid}", language=None)
                    st.caption("Anyone with this link sees this team's public FPL data.")
        footer = st.container()
    return footer


def _base_url() -> str:
    try:
        url = st.context.url or ""
    except Exception:  # noqa: BLE001
        url = ""
    base = url.split("?")[0] if url.startswith("http") else APP_URL
    return base if base.endswith("/") else base + "/"


def footer_timestamps(footer: Any) -> None:
    """Shared-data provenance (bundle id + creation time, or the live fetch time) and team-data time."""
    tid = current_team_id()
    lines = []
    status: dict[str, Any] = {}
    try:
        status = data.shared_status()
        when = fmt_ts(data.shared_as_of())
        if status.get("mode") == "bundle":
            lines.append(f"Shared data as of {when} · bundle {status['bundle_id']}")
        else:
            lines.append(f"Shared data as of {when} (live API)")
    except Exception:  # noqa: BLE001
        lines.append("Shared data: unavailable")
    if tid is not None:
        try:
            lines.append(f"Team data as of {fmt_ts(data.team_as_of(tid))}")
        except Exception:  # noqa: BLE001
            lines.append("Team data: unavailable")
    with footer:
        st.divider()
        if status.get("mode") == "fallback":
            st.warning(f"Shared data: live API fallback ({status.get('reason') or 'bundle unavailable'}).")
        st.caption("  \n".join(lines))


def require_team() -> int | None:
    tid = current_team_id()
    if tid is None:
        st.info("Enter a team ID in the sidebar and press **Load** to see this page.")
    return tid


# ============================================================================================ formatting
_MD_SPECIAL = set(r"\`*_{}[]<>()#+-.!|~$")


def md_escape(text: object) -> str:
    """Escape Markdown so a user-chosen team name renders literally."""
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in str(text))


def fmt_ts(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return str(iso)


def fmt_price(v: float | None) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}"


PRICE = st.column_config.NumberColumn("Price", format="%.1f")
OWNERSHIP = st.column_config.NumberColumn("Ownership", format="%.1f%%")

COMMON_COLUMNS: dict[str, Any] = {
    "name": st.column_config.TextColumn("Player"),
    "team": st.column_config.TextColumn("Team"),
    "pos": st.column_config.TextColumn("Pos"),
    "price": PRICE,
    "ownership": OWNERSHIP,
    "total_points": st.column_config.NumberColumn("Points", format="%d"),
    "points": st.column_config.NumberColumn("Points", format="%d"),
    "ppg": st.column_config.NumberColumn("PPG", format="%.1f"),
    "form": st.column_config.NumberColumn("Form", format="%.1f"),
    "minutes": st.column_config.NumberColumn("Minutes", format="%d"),
    "goals": st.column_config.NumberColumn("Goals", format="%d"),
    "assists": st.column_config.NumberColumn("Assists", format="%d"),
    "bonus": st.column_config.NumberColumn("Bonus", format="%d"),
    "cs": st.column_config.NumberColumn("Clean sheets", format="%d"),
    "saves": st.column_config.NumberColumn("Saves", format="%d"),
    "defcon": st.column_config.NumberColumn(
        "Defensive contribution", format="%d", help="Raw count (CBI + tackles [+ recoveries for MID/FWD])"
    ),
    "cost_change_event": st.column_config.NumberColumn("Change", format="%+.1f"),
    "cost_change_start": st.column_config.NumberColumn(
        "Price change (season)", format="%+.1f", help="cost_change_start"
    ),
    "transfers_in_event": st.column_config.NumberColumn("Transfers in (GW)", format="%d"),
    "transfers_out_event": st.column_config.NumberColumn("Transfers out (GW)", format="%d"),
    "net_transfers_event": st.column_config.NumberColumn("Net transfers (GW)", format="%+d"),
}


def table(df: pd.DataFrame, extra: dict[str, Any] | None = None, **kw: Any) -> None:
    cfg = {k: v for k, v in COMMON_COLUMNS.items() if k in df.columns}
    cfg.update(extra or {})
    st.dataframe(df, hide_index=kw.pop("hide_index", True), column_config=cfg, **kw)


# ============================================================================================ difficulty
def difficulty_style(diff: pd.DataFrame):
    """Return a function for DataFrame.style.apply(axis=None) colouring cells by FPL difficulty."""

    def css(v: float) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "background-color: transparent; color: #888888"
        d = max(1, min(5, round(v)))
        return f"background-color: {DIFFICULTY_BG[d]}; color: {DIFFICULTY_FG[d]}"

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        return diff.reindex(index=frame.index, columns=frame.columns).map(css)

    return apply


def difficulty_legend() -> None:
    chips = "".join(
        f'<span style="background:{DIFFICULTY_BG[d]};color:{DIFFICULTY_FG[d]};padding:2px 10px;'
        f'margin-right:4px;border-radius:3px;font-size:0.8rem">{d}</span>'
        for d in range(1, 6)
    )
    st.html(f'<div style="font-size:0.8rem">FPL difficulty (1 = easiest): {chips}</div>')


# ============================================================================================ pitch
_BADGE_CSS = (
    "position:absolute;top:-8px;right:-8px;background:#111;color:#fff;border-radius:50%;min-width:22px;"
    "height:22px;line-height:22px;font-size:11px;font-weight:700;padding:0 3px"
)
_CARD_CSS = (
    "position:relative;background:rgba(255,255,255,0.92);color:#111;border-radius:6px;padding:6px 4px;"
    "width:96px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.3)"
)
_NAME_CSS = "font-weight:700;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"
_ROW_CSS = "display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin:10px 0"


def pitch_html(rows: dict[str, list[dict[str, Any]]], points_label: str) -> str:
    """Squad on a pitch: GK / DEF / MID / FWD rows and a bench row; every text value is HTML-escaped."""

    def card(c: dict[str, Any]) -> str:
        badge = f'<span style="{_BADGE_CSS}">{html.escape(c["badge"])}</span>' if c.get("badge") else ""
        pts = "-" if c.get("points") is None else str(c["points"])
        team = html.escape(str(c["team"]))
        return (
            f'<div style="{_CARD_CSS}">{badge}'
            f'<div style="{_NAME_CSS}">{html.escape(str(c["name"]))}</div>'
            f'<div style="font-size:11px;color:#444">{team} · {fmt_price(c.get("price"))}</div>'
            f'<div style="font-size:12px;font-weight:600">{html.escape(points_label)} {pts}</div>'
            "</div>"
        )

    def row(cards: list[dict[str, Any]], label: str) -> str:
        if not cards:
            return f'<div style="text-align:center;color:#eee;font-size:12px">no {label}</div>'
        return f'<div style="{_ROW_CSS}">{"".join(card(c) for c in cards)}</div>'

    pitch = "".join(row(rows.get(k, []), k) for k in ("GKP", "DEF", "MID", "FWD"))
    bench = row(rows.get("Bench", []), "bench")
    return (
        '<div style="background:linear-gradient(#2e8b57,#237a49);border-radius:10px;padding:10px 6px">'
        f"{pitch}</div>"
        '<div style="background:#4a5a52;border-radius:10px;padding:6px;margin-top:6px">'
        '<div style="color:#fff;font-size:12px;text-align:center">Bench</div>'
        f"{bench}</div>"
    )
