"""scripts/keepalive.py decision logic with a fake page and a fake clock (no browser, no network):
asleep -> presses the wake button and waits for the app; awake -> no press; never renders -> exit 1."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "keepalive.py"
_spec = importlib.util.spec_from_file_location("keepalive", _PATH)
keepalive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(keepalive)


class FakePage:
    """asleep: the sleep page shows the wake button; the app renders `wake_delay` seconds after a press.
    render_at: an awake app renders at this time (None = never)."""

    def __init__(self, *, asleep: bool, wake_delay: float = 30.0, render_at: float | None = 4.0):
        self.t = 0.0
        self.asleep = asleep
        self.wake_delay = wake_delay
        self.render_at = None if asleep else render_at
        self.clicks: list[float] = []
        self.opened: list[str] = []
        self.waited = 0.0

    def clock(self) -> float:
        return self.t

    def open(self, url: str) -> None:
        self.opened.append(url)
        self.t += 3.0

    def wake_button_visible(self) -> bool:
        return self.asleep and self.render_at is None

    def click_wake(self) -> None:
        self.clicks.append(self.t)
        if self.wake_delay is not None:
            self.render_at = self.t + self.wake_delay

    def rendered_title(self) -> str | None:
        return "FPL Rocket" if self.render_at is not None and self.t >= self.render_at else None

    def wait(self, seconds: float) -> None:
        self.t += seconds
        self.waited += seconds


def _visit(page: FakePage, **kw) -> dict:
    return keepalive.visit(page, "https://app.example", clock=page.clock, **kw)


def test_asleep_presses_the_wake_button_once_and_waits_for_the_app():
    page = FakePage(asleep=True, wake_delay=30.0)
    rec = _visit(page, dwell=20)
    assert page.opened == ["https://app.example"]
    assert len(page.clicks) == 1
    assert rec == {
        "was_asleep": True,
        "woke": True,
        "seconds": rec["seconds"],
        "title": "FPL Rocket",
        "error": None,
    }
    assert 3 + 30 + 20 <= rec["seconds"] <= 3 + 30 + 20 + keepalive.POLL_S  # rendered, then the dwell


def test_awake_app_is_never_clicked_and_still_dwells():
    page = FakePage(asleep=False, render_at=4.0)
    rec = _visit(page, dwell=20)
    assert page.clicks == []
    assert rec["was_asleep"] is False and rec["woke"] is False and rec["title"] == "FPL Rocket"
    assert rec["error"] is None and rec["seconds"] >= 20


def test_slow_wake_presses_again_but_never_renders_is_an_error():
    page = FakePage(asleep=True, wake_delay=None)  # the button never brings the app back
    rec = _visit(page, wake_timeout=180, dwell=20)
    assert rec["was_asleep"] is True and rec["woke"] is False and rec["title"] is None
    assert "did not render" in rec["error"] and "wake button" in rec["error"]
    assert 180 <= rec["seconds"] <= 180 + 3 + keepalive.POLL_S
    assert 2 <= len(page.clicks) <= 1 + 180 // keepalive.RECLICK_S  # re-pressed, not on every poll


def test_awake_page_that_never_renders_times_out():
    page = FakePage(asleep=False, render_at=None)
    rec = _visit(page, render_timeout=90)
    assert page.clicks == [] and rec["title"] is None and "no sleep page" in rec["error"]


def test_exceptions_are_reported_not_raised():
    class Broken(FakePage):
        def open(self, url: str) -> None:
            raise TimeoutError("navigation timed out")

    rec = _visit(Broken(asleep=False))
    assert rec["title"] is None and rec["error"] == "TimeoutError: navigation timed out"


@pytest.mark.parametrize(
    ("page", "code"),
    [
        (FakePage(asleep=True), 0),
        (FakePage(asleep=False), 0),
        (FakePage(asleep=True, wake_delay=None), 1),
    ],
)
def test_main_exit_code_and_one_json_line(monkeypatch, capsys, page, code):
    """main() with the browser replaced by the fake: exit 0 only when the app rendered."""

    class FakePlaywright:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        @property
        def chromium(self):
            return self

        def launch(self, **kw):
            return self

        def new_context(self, **kw):
            return self

        def new_page(self):
            return page

        def close(self):
            pass

    import sys
    import types

    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.sync_playwright = FakePlaywright
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    real_visit = keepalive.visit
    monkeypatch.setattr(keepalive, "PlaywrightPage", lambda p: p)
    monkeypatch.setattr(keepalive, "visit", lambda p, url, **kw: real_visit(p, url, clock=p.clock, **kw))
    assert keepalive.main(["--url", "https://app.example"]) == code
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert set(rec) == {"ts", "was_asleep", "woke", "seconds", "title", "error"}
    assert rec["woke"] is (code == 0 and page.asleep)
