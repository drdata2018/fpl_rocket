"""Cached data access for the Streamlit app.

Two cache layers (docs/PLAN_v2.md section 2), both in-process only (nothing written to disk in production):

* SHARED (ttl 600 s, same for every visitor): bootstrap, fixtures, live stats of finished GWs, element
  summaries.
* PER-ID (ttl 300 s, keyed by the team ID): entry, history, picks of a GW, transfers, and the planning state.
  The planning state is built ONCE per ID from the cached per-ID pieces (reusing state.fetch_inputs through a
  cache-backed client adapter), so a cache miss never fans out into duplicate API calls.

Each loader returns its payload; fetch times are kept alongside so the sidebar can show "shared data as of"
and "team data as of". Exceptions (FPLAPIError, EntryNotFound, PlanningStateError) propagate to the page,
which degrades per feature (ui.guard). Failed calls are never cached.

Environment (tests / offline replay only): FPL_ROCKET_CACHE_DIR = raw JSON cache directory,
FPL_ROCKET_OFFLINE=1 = never touch the network.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import streamlit as st

from . import views
from .api import FPLClient
from .config import FETCH_WORKERS, MAX_TEAM_ID, SHARED_TTL_S, TEAM_CACHE_ENTRIES, TEAM_TTL_S
from .state import PlanningState, build_planning_state, fetch_inputs

JSON = dict[str, Any]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_local = threading.local()
# One bounded pool for the whole process: its threads keep their HTTP connections alive, and it caps the
# number of parallel calls this app makes to the FPL API across all sessions.
_POOL = ThreadPoolExecutor(max_workers=FETCH_WORKERS, thread_name_prefix="fpl-fetch")


def client() -> FPLClient:
    """One client per thread (requests.Session is not thread-safe; reusing it keeps TLS connections warm)."""
    cache = os.environ.get("FPL_ROCKET_CACHE_DIR") or None
    offline = os.environ.get("FPL_ROCKET_OFFLINE") == "1"
    key = (cache, offline)
    if getattr(_local, "key", None) != key:
        _local.client = FPLClient(cache, offline=offline, backoff=0.5)
        _local.key = key
    return _local.client


def _parallel(fn, items: list[Any]) -> list[Any]:
    """Map fn over items on the shared pool. fn must not itself submit to the pool (no nesting)."""
    if len(items) <= 1:
        return [fn(i) for i in items]
    return list(_POOL.map(fn, items))


# ============================================================================================ shared layer
@st.cache_data(ttl=SHARED_TTL_S, show_spinner="Loading FPL data...", max_entries=2)
def _bootstrap() -> tuple[JSON, str]:
    return client().bootstrap(), _now()


def bootstrap() -> JSON:
    return _bootstrap()[0]


def shared_as_of() -> str:
    return _bootstrap()[1]


@st.cache_data(ttl=SHARED_TTL_S, show_spinner=False, max_entries=2)
def _players(as_of: str) -> pd.DataFrame:  # keyed by the bootstrap fetch time so both expire together
    return views.players_frame(bootstrap())


def players() -> pd.DataFrame:
    return _players(shared_as_of())


@st.cache_data(ttl=SHARED_TTL_S, show_spinner=False, max_entries=2)
def fixtures() -> list[JSON]:
    return client().fixtures()


@st.cache_data(ttl=SHARED_TTL_S, show_spinner="Loading gameweek stats...", max_entries=2)
def _live_table(gws: tuple[int, ...]) -> pd.DataFrame:
    def one(gw: int) -> JSON:
        return client().event_live(gw)

    return views.live_frame(dict(zip(gws, _parallel(one, list(gws)), strict=True)))


def live_table() -> pd.DataFrame:
    """Live stats of every finished GW (one row per GW x player)."""
    return _live_table(tuple(views.finished_gws(bootstrap())))


@st.cache_data(ttl=SHARED_TTL_S, show_spinner=False, max_entries=1000)
def element_summary(player_id: int) -> JSON:
    return client().element_summary(int(player_id))


# ============================================================================================ per-ID layer
def parse_team_id(text: str | None) -> tuple[int | None, str | None]:
    """(team_id, None) for a valid ID, else (None, error message). No network access."""
    raw = (text or "").strip()
    if not raw:
        return None, "Enter a team ID."
    if not raw.isdigit():
        return None, "A team ID contains digits only (find it in the URL of your Points page)."
    value = int(raw)
    if value <= 0 or value > MAX_TEAM_ID:
        return None, f"{raw} is not a valid team ID."
    return value, None


@st.cache_data(ttl=TEAM_TTL_S, show_spinner=False, max_entries=TEAM_CACHE_ENTRIES)
def _entry(team_id: int) -> tuple[JSON, str]:
    return client().entry(team_id), _now()


def entry(team_id: int) -> JSON:
    return _entry(team_id)[0]


def team_as_of(team_id: int) -> str:
    return _entry(team_id)[1]


@st.cache_data(ttl=TEAM_TTL_S, show_spinner=False, max_entries=TEAM_CACHE_ENTRIES)
def history(team_id: int) -> JSON:
    return client().entry_history(team_id)


@st.cache_data(ttl=TEAM_TTL_S, show_spinner=False, max_entries=TEAM_CACHE_ENTRIES)
def transfers(team_id: int) -> list[JSON]:
    return client().entry_transfers(team_id)


@st.cache_data(ttl=TEAM_TTL_S, show_spinner=False, max_entries=4 * TEAM_CACHE_ENTRIES)
def picks(team_id: int, gw: int) -> JSON | None:
    """Picks of one GW (None when not yet published)."""
    return client().entry_picks(team_id, gw)


def last_published_gw(team_id: int) -> int | None:
    cur = history(team_id).get("current") or []
    return int(max(h["event"] for h in cur)) if cur else None


class _CachedClient:
    """Adapter giving state.fetch_inputs the FPLClient interface, served from the caches above."""

    def entry(self, team_id: int) -> JSON:
        return entry(team_id)

    def entry_history(self, team_id: int) -> JSON:
        return history(team_id)

    def entry_picks(self, team_id: int, gw: int) -> JSON | None:
        return picks(team_id, gw)

    def entry_transfers(self, team_id: int) -> list[JSON]:
        return transfers(team_id)

    def bootstrap(self) -> JSON:
        return bootstrap()

    def element_summary(self, player_id: int) -> JSON:
        return element_summary(player_id)


def _prefetch_element_summaries(ids: list[int]) -> None:
    """Warm the shared element-summary cache in parallel (the planning state needs ~15 of them)."""
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

    ctx = get_script_run_ctx()

    def one(pid: int) -> None:
        if ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)
        element_summary(pid)

    _parallel(one, ids)


@st.cache_data(
    ttl=TEAM_TTL_S, show_spinner="Reconstructing the planning state...", max_entries=TEAM_CACHE_ENTRIES
)
def planning_state(team_id: int) -> PlanningState:
    """Planning state for the upcoming GW (raises PlanningStateError / FPLAPIError)."""
    hist = history(team_id)
    cur = hist.get("current") or []
    if cur:
        started = int(entry(team_id).get("started_event") or cur[0]["event"])
        first = picks(team_id, started)
        if first:
            _prefetch_element_summaries([int(p["element"]) for p in first["picks"]])
    inputs = fetch_inputs(_CachedClient(), team_id)
    return build_planning_state(**inputs, prices_as_of=f"bootstrap fetched {shared_as_of()}")


@st.cache_data(
    ttl=TEAM_TTL_S, show_spinner="Loading picks for every gameweek...", max_entries=TEAM_CACHE_ENTRIES
)
def all_picks(team_id: int, gws: tuple[int, ...]) -> dict[int, JSON | None]:
    """Picks for many GWs in one parallel fetch (used lazily by the captain-hindsight section)."""

    def one(gw: int) -> JSON | None:
        return client().entry_picks(team_id, gw)

    return dict(zip(gws, _parallel(one, list(gws)), strict=True))


@st.cache_data(ttl=TEAM_TTL_S, show_spinner=False, max_entries=TEAM_CACHE_ENTRIES)
def league_standings_page1(league_id: int) -> JSON:
    return client().league_classic_standings(league_id, 1)


def clear_all() -> None:
    """Drop every cached payload (tests and timing measurements)."""
    st.cache_data.clear()
