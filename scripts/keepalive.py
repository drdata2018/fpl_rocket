"""Keep the free Streamlit Community Cloud app awake (workflow .github/workflows/keepalive.yml).

Community Cloud puts a free app to sleep after a period without visitors; the sleep page then shows a
"Yes, get this app back up!" button that only a real browser visit can press (an anonymous HTTP call to the
resume endpoint is refused). This script is that visitor: headless Chromium (Playwright) opens the app,
presses the wake button when the sleep page is shown, waits for the Streamlit app itself to render (the
sidebar "Team ID" box, or a "Season Overview" / "FPL Rocket" title) and stays on the page for a short dwell so
the visit counts as a real session. No credentials, no stored cookies (a fresh browser context every run).

Prints one JSON line {ts, was_asleep, woke, seconds, title, error}; exit code 0 when the app rendered, else 1.

    python scripts/keepalive.py [--url URL] [--wake-timeout 180] [--dwell 20]

Playwright is installed by the workflow only (pip install playwright; python -m playwright install
--with-deps chromium); it is not an app or model dependency.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

APP_URL = "https://fpl-rocket.streamlit.app"
WAKE_TEXT = "get this app back up"  # the sleep page's button: "Yes, get this app back up!"
RENDER_TEXT = "Team ID"  # the sidebar text input label (fpl_rocket.ui.sidebar)
RENDER_TITLES = ("Season Overview", "FPL Rocket")
WAKE_TIMEOUT_S = 180.0  # after pressing the wake button
RENDER_TIMEOUT_S = 90.0  # an awake app (or the sleep page) must show up within this
DWELL_S = 20.0
POLL_S = 2.0
RECLICK_S = 45.0  # press the wake button again if it is still shown this long after a press


class AppPage(Protocol):
    """The few page operations the decision logic needs (PlaywrightPage below; a fake in the tests)."""

    def open(self, url: str) -> None: ...

    def wake_button_visible(self) -> bool: ...

    def click_wake(self) -> None: ...

    def rendered_title(self) -> str | None:
        """A title (non-empty) once the Streamlit app itself has rendered, else None."""
        ...

    def wait(self, seconds: float) -> None: ...


def visit(
    page: AppPage,
    url: str = APP_URL,
    *,
    wake_timeout: float = WAKE_TIMEOUT_S,
    render_timeout: float = RENDER_TIMEOUT_S,
    dwell: float = DWELL_S,
    poll: float = POLL_S,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Open the app, wake it if asleep, wait for it to render, dwell. The result record (without ts)."""
    t0 = clock()
    was_asleep = False
    title: str | None = None
    error: str | None = None
    try:
        page.open(url)
        deadline = t0 + render_timeout
        last_click: float | None = None
        while True:
            title = page.rendered_title()
            if title:
                break
            now = clock()
            if page.wake_button_visible() and (last_click is None or now - last_click >= RECLICK_S):
                if not was_asleep:
                    was_asleep = True
                    deadline = now + wake_timeout
                page.click_wake()
                last_click = now
            if clock() >= deadline:
                phase = "after pressing the wake button" if was_asleep else "(no sleep page either)"
                error = f"the app did not render within {deadline - t0:.0f} s {phase}"
                break
            page.wait(poll)
        if title:
            page.wait(dwell)  # stay on the page so the visit counts as a session
    except Exception as exc:  # noqa: BLE001 - reported in the JSON line, exit 1
        error = f"{type(exc).__name__}: {exc}"
        title = None
    return {
        "was_asleep": was_asleep,
        "woke": bool(was_asleep and title),
        "seconds": round(clock() - t0, 1),
        "title": title,
        "error": error,
    }


class PlaywrightPage:
    """AppPage over a Playwright page. The Streamlit app runs inside an iframe of the Community Cloud page,
    so every frame is searched."""

    def __init__(self, page) -> None:
        self.page = page

    def open(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        # a websocket app may never be fully idle: polling continues either way
        with contextlib.suppress(Exception):
            self.page.wait_for_load_state("networkidle", timeout=30_000)

    def _wake_button(self):
        import re

        return self.page.get_by_role("button", name=re.compile(WAKE_TEXT, re.IGNORECASE))

    def wake_button_visible(self) -> bool:
        try:
            btn = self._wake_button()
            return btn.count() > 0 and btn.first.is_visible()
        except Exception:  # noqa: BLE001 - navigating
            return False

    def click_wake(self) -> None:
        self._wake_button().first.click(timeout=10_000)

    def rendered_title(self) -> str | None:
        for frame in self.page.frames:
            try:
                if frame.get_by_text(RENDER_TEXT, exact=True).count() > 0:
                    return self._title(frame) or RENDER_TEXT
                ftitle = frame.title()
                if any(t in ftitle for t in RENDER_TITLES) and frame.locator('[data-testid="stApp"]').count():
                    return ftitle
            except Exception:  # noqa: BLE001 - a frame detached while navigating
                continue
        return None

    def _title(self, frame) -> str:
        for t in (frame.title(), self.page.title()):
            if t:
                return t
        return ""

    def wait(self, seconds: float) -> None:
        self.page.wait_for_timeout(seconds * 1000)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=APP_URL)
    ap.add_argument("--wake-timeout", type=float, default=WAKE_TIMEOUT_S)
    ap.add_argument("--dwell", type=float, default=DWELL_S)
    args = ap.parse_args(argv)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        rec = {"was_asleep": False, "woke": False, "seconds": 0.0, "title": None, "error": f"{exc}"}
    else:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context(viewport={"width": 1280, "height": 900})
                rec = visit(
                    PlaywrightPage(context.new_page()),
                    args.url,
                    wake_timeout=args.wake_timeout,
                    dwell=args.dwell,
                )
                context.close()
            finally:
                browser.close()
    print(json.dumps({"ts": ts, **rec}, ensure_ascii=False), flush=True)
    return 0 if rec["title"] and not rec["error"] else 1


if __name__ == "__main__":
    sys.exit(main())
