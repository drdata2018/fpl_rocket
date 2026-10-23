"""Client for the public (unauthenticated) Fantasy Premier League API.

All endpoints used here are public. The client sends a browser User-Agent (the API rejects some default
library agents from cloud hosts), applies timeouts, and retries 429/403/5xx with exponential backoff
(max 4 tries). An optional on-disk raw cache stores every JSON response by path so that tests and
scripts can replay recorded data offline.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://fantasy.premierleague.com/api/"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
RETRY_STATUSES = frozenset({403, 429, 500, 502, 503, 504})
MAX_TRIES = 4

JSON = dict[str, Any]


class FPLAPIError(RuntimeError):
    """Raised when the API returns an unrecoverable error."""

    def __init__(self, path: str, status: int | None, message: str = "") -> None:
        self.path = path
        self.status = status
        super().__init__(f"FPL API {path} failed (status={status}) {message}".strip())


class EntryNotFound(FPLAPIError):
    """The entry (team) ID does not exist."""


def _cache_name(path: str, params: dict[str, Any] | None) -> str:
    """Stable file name for a request path + params, e.g. 'entry_123_history.json'."""
    key = path.strip("/")
    if params:
        key += "__" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key) + ".json"


class FPLClient:
    """Thin typed wrapper over the FPL API.

    Args:
        cache_dir: if given, responses are read from / written to this directory.
        offline: if True, never touch the network (cache misses raise FPLAPIError).
        timeout: per-request timeout in seconds (connect, read).
        backoff: base backoff in seconds; sleep = backoff * 2**attempt.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        offline: bool = False,
        timeout: tuple[float, float] = (5.0, 20.0),
        backoff: float = 1.0,
        session: requests.Session | None = None,
        base_url: str = BASE_URL,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.timeout = timeout
        self.backoff = backoff
        self.base_url = base_url
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": BROWSER_UA, "Accept": "application/json"})

    # ------------------------------------------------------------------ core
    def get(self, path: str, params: dict[str, Any] | None = None, *, allow_404: bool = False) -> Any:
        """GET a JSON resource. Returns None on 404 when allow_404 is True."""
        cached = self._cache_read(path, params)
        if cached is not None:
            return cached.get("__body__") if isinstance(cached, dict) and "__404__" in cached else cached
        if self.offline:
            raise FPLAPIError(path, None, "offline cache miss")
        body, status = self._fetch(path, params)
        if status == 404:
            if allow_404:
                self._cache_write(path, params, {"__404__": True, "__body__": None})
                return None
            raise FPLAPIError(path, 404, "not found")
        self._cache_write(path, params, body)
        return body

    def _fetch(self, path: str, params: dict[str, Any] | None) -> tuple[Any, int]:
        url = self.base_url + path.lstrip("/")
        last_status: int | None = None
        for attempt in range(MAX_TRIES):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_status = None
                if attempt == MAX_TRIES - 1:
                    raise FPLAPIError(path, None, repr(exc)) from exc
            else:
                last_status = resp.status_code
                if resp.status_code == 404:
                    return None, 404
                if resp.ok:
                    return resp.json(), resp.status_code
                if resp.status_code not in RETRY_STATUSES or attempt == MAX_TRIES - 1:
                    raise FPLAPIError(path, resp.status_code, resp.text[:200])
            time.sleep(self.backoff * (2**attempt))
        raise FPLAPIError(path, last_status, "retries exhausted")

    def _cache_read(self, path: str, params: dict[str, Any] | None) -> Any:
        if not self.cache_dir:
            return None
        f = self.cache_dir / _cache_name(path, params)
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def _cache_write(self, path: str, params: dict[str, Any] | None, body: Any) -> None:
        if self.cache_dir:
            f = self.cache_dir / _cache_name(path, params)
            f.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------- endpoints
    def bootstrap(self) -> JSON:
        return self.get("bootstrap-static/")

    def fixtures(self, event: int | None = None) -> list[JSON]:
        return self.get("fixtures/", {"event": event} if event is not None else None)

    def event_live(self, gw: int) -> JSON:
        return self.get(f"event/{gw}/live/")

    def element_summary(self, player_id: int) -> JSON:
        return self.get(f"element-summary/{player_id}/")

    def entry(self, entry_id: int) -> JSON:
        """Entry header. Raises EntryNotFound for an unknown ID."""
        body = self.get(f"entry/{int(entry_id)}/", allow_404=True)
        if body is None:
            raise EntryNotFound(f"entry/{entry_id}/", 404, "no such team id")
        return body

    def entry_history(self, entry_id: int) -> JSON:
        return self.get(f"entry/{int(entry_id)}/history/")

    def entry_picks(self, entry_id: int, gw: int) -> JSON | None:
        """Picks for a GW; None when the GW is not yet published (404 before its deadline)."""
        return self.get(f"entry/{int(entry_id)}/event/{int(gw)}/picks/", allow_404=True)

    def entry_transfers(self, entry_id: int) -> list[JSON]:
        return self.get(f"entry/{int(entry_id)}/transfers/")

    def league_classic_standings(self, league_id: int, page: int = 1) -> JSON:
        return self.get(f"leagues-classic/{int(league_id)}/standings/", {"page_standings": page})
