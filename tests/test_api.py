"""API client: retries, 404 handling and cache replay, with a fake session (no network)."""

from __future__ import annotations

import pytest

from fpl_rocket.api import BROWSER_UA, EntryNotFound, FPLAPIError, FPLClient


class FakeResp:
    def __init__(self, status: int, body=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        return self.responses.pop(0)


def _client(responses, **kw):
    return FPLClient(session=FakeSession(responses), backoff=0, **kw)


def test_browser_user_agent_set():
    c = _client([])
    assert c.session.headers["User-Agent"] == BROWSER_UA and BROWSER_UA.startswith("Mozilla/5.0")


def test_retries_then_succeeds():
    c = _client([FakeResp(429), FakeResp(503), FakeResp(200, {"events": []})])
    assert c.bootstrap() == {"events": []}
    assert len(c.session.calls) == 3


def test_gives_up_after_four_tries():
    c = _client([FakeResp(403)] * 4)
    with pytest.raises(FPLAPIError) as e:
        c.bootstrap()
    assert e.value.status == 403 and len(c.session.calls) == 4


def test_upcoming_picks_404_returns_none():
    c = _client([FakeResp(404)])
    assert c.entry_picks(1, 8) is None


def test_unknown_entry_raises_entry_not_found():
    c = _client([FakeResp(404)])
    with pytest.raises(EntryNotFound):
        c.entry(999999999)


def test_non_retryable_error_raises_immediately():
    c = _client([FakeResp(400)])
    with pytest.raises(FPLAPIError):
        c.entry_history(1)
    assert len(c.session.calls) == 1


def test_cache_records_and_replays_offline(tmp_path):
    c = _client([FakeResp(200, {"current": [1]}), FakeResp(404)], cache_dir=tmp_path)
    assert c.entry_history(5) == {"current": [1]}
    assert c.entry_picks(5, 9) is None
    replay = FPLClient(tmp_path, offline=True)
    assert replay.entry_history(5) == {"current": [1]}
    assert replay.entry_picks(5, 9) is None
    with pytest.raises(FPLAPIError):
        replay.bootstrap()
