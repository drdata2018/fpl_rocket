"""Source `premier_injuries`: the public EPL injury table at premierinjuries.com/injury-table.php.

Polite: ONE GET per build, 10 s timeout, browser User-Agent, no retries. Any failure (network, HTTP status, a
bot
challenge page, a layout change) returns [] with the reason in `last_error`; the build never fails because of
it.

Layout parsed (rule-based, stdlib html.parser; the page is untrusted data, nothing in it is executed or
followed):
    <table class="injury-table ..."> with, per club, <tr class="heading"> holding <div
    class="injury-team">Club</div>
    and then <tr class="player-row"> rows of 7 cells: Player | Reason | Further Detail (often a dated quote,
    "Oct 08: '...'") | Potential Return (dd/mm/yyyy or "No Return Date") | Condition | Status ("Ruled Out",
    "25%", "50%", "75%", "Suspended") | (track link). Each cell starts with a <div class="mob-title"> label.
    <div class="pi-table-timestamp">Last update: 13 Oct 26 10:46 <span>UK</span></div> dates the whole table.
FRAGILE: the site sits behind a Cloudflare managed challenge that answers non-browser clients with HTTP 403
("Just a moment..."); the parser depends on the class names above.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from html.parser import HTMLParser

import requests

from ..api import BROWSER_UA
from .items import NewsItem

log = logging.getLogger(__name__)

URL = "https://www.premierinjuries.com/injury-table.php"
SOURCE = "premier_injuries"
TIMEOUT_S = 10.0
MAX_TEXT = 1000
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_QUOTE_DATE = re.compile(r"^\s*([A-Za-z]{3,9})\.?\s+(\d{1,2})\s*:")
_STAMP = re.compile(r"Last update:\s*(\d{1,2})\s+([A-Za-z]{3})\w*\s+(\d{2,4})\s+(\d{1,2}):(\d{2})")
_SKIP_TAGS = ("a", "svg", "path", "div", "span")


class _TableParser(HTMLParser):
    """Collects (club, [cell texts], player-page href) for every player row, and the table timestamp text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str | None, list[str], str | None]] = []
        self.stamp_text = ""
        self.saw_table = False
        self._club: str | None = None
        self._in_club = False
        self._club_buf: list[str] = []
        self._row: list[str] | None = None
        self._href: str | None = None
        self._cell: list[str] | None = None
        self._skip = 0  # depth inside mob-title labels, links and svg icons (their text is ignored)
        self._in_stamp = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "table" and "injury-table" in cls:
            self.saw_table = True
        if self._cell is not None:
            if tag == "br":
                if not self._skip:
                    self._cell.append(" ")
                return
            if tag == "a" and (a.get("href") or "").startswith("/newsroom/epl/players/"):
                self._href = a.get("href")
            skipping = self._skip or (tag == "div" and "mob-title" in cls) or tag in ("a", "svg")
            if skipping and tag in _SKIP_TAGS:
                self._skip += 1
            return
        if tag == "div" and "injury-team" in cls:
            self._in_club, self._club_buf = True, []
        elif tag == "div" and "pi-table-timestamp" in cls:
            self._in_stamp = True
        elif tag == "tr":
            self._row = [] if "player-row" in cls else None
            self._href = None
        elif tag == "td" and self._row is not None:
            self._cell, self._skip = [], 0

    def handle_endtag(self, tag: str) -> None:
        if self._cell is not None:
            if tag == "td":
                assert self._row is not None
                self._row.append(" ".join("".join(self._cell).split()))
                self._cell, self._skip = None, 0
            elif self._skip and tag in _SKIP_TAGS:
                self._skip -= 1
            return
        if self._in_club and tag == "div":
            self._in_club = False
            self._club = " ".join("".join(self._club_buf).split()) or None
        elif self._in_stamp and tag == "div":
            self._in_stamp = False
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append((self._club, self._row, self._href))
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            if not self._skip:
                self._cell.append(data)
        elif self._in_club:
            self._club_buf.append(data)
        elif self._in_stamp:
            self.stamp_text += data


def parse_status(status: str | None) -> float | None:
    """Chance of playing (0-100) from the Status cell: "Ruled Out" / "Suspended" -> 0, "75%" -> 75."""
    s = (status or "").strip().lower()
    if not s:
        return None
    if "ruled out" in s or "suspend" in s:
        return 0.0
    m = re.search(r"(\d{1,3})\s*%", s)
    return float(min(int(m.group(1)), 100)) if m else None


def parse_return_date(text: str | None) -> str | None:
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text or "")
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat()
    except ValueError:
        return None


def parse_table_stamp(text: str) -> datetime | None:
    """'Last update: 13 Oct 26 10:46 UK' -> aware UTC datetime (UK local time; BST/GMT via zoneinfo)."""
    m = _STAMP.search(text or "")
    if not m:
        return None
    day, mon, yr, hh, mm = m.groups()
    month = MONTHS.get(mon[:3].lower())
    if month is None:
        return None
    year = int(yr) + (2000 if len(yr) == 2 else 0)
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Europe/London")
    except Exception:  # noqa: BLE001 - no tz database: treat UK time as UTC (<= 1 h error)
        tz = UTC
    try:
        return datetime(year, month, int(day), int(hh), int(mm), tzinfo=tz).astimezone(UTC)
    except ValueError:
        return None


def quote_date(text: str, ref: datetime) -> datetime | None:
    """The date a 'Further Detail' quote was given ("Oct 08: '...'"), year inferred from `ref` (a date more
    than 2 days after ref belongs to the previous year). Midday UTC."""
    m = _QUOTE_DATE.match(text or "")
    if not m:
        return None
    month = MONTHS.get(m.group(1)[:3].lower())
    if month is None:
        return None
    try:
        d = datetime(ref.year, month, int(m.group(2)), 12, 0, tzinfo=UTC)
    except ValueError:
        return None
    if (d - ref).days > 2:
        d = d.replace(year=ref.year - 1)
    return d


def parse_injury_table(html: str, fetched_at: datetime, url: str = URL) -> tuple[list[NewsItem], str | None]:
    """(items, table 'Last update' as ISO UTC or None). Raises ValueError when the page has no injury
    table."""
    p = _TableParser()
    p.feed(html)
    p.close()
    if not p.saw_table:
        raise ValueError("no injury table in the page (bot challenge or layout change)")
    stamp = parse_table_stamp(p.stamp_text)
    ref = stamp or fetched_at
    items = []
    for club, cells, href in p.rows:
        if len(cells) < 6 or not cells[0]:
            continue
        player, reason, detail, ret, condition, status = cells[:6]
        info = quote_date(detail, ref) or stamp
        text = f"{reason}. {condition}. Status: {status}. Potential return: {ret}. {detail}".strip()
        items.append(
            NewsItem(
                player_name_raw=player,
                team_hint=club,
                text=text[:MAX_TEXT],
                status_hint=status or None,
                chance_hint=parse_status(status),
                return_date_hint=parse_return_date(ret),
                source=SOURCE,
                fetched_at=fetched_at.isoformat(timespec="seconds"),
                url=("https://www.premierinjuries.com" + href) if href else url,
                published_at=info.isoformat(timespec="seconds") if info else None,
                source_updated_at=stamp.isoformat(timespec="seconds") if stamp else None,
            )
        )
    return items, stamp.isoformat(timespec="seconds") if stamp else None


class PremierInjuriesSource:
    """fetch() -> list[NewsItem]; `html` (a saved page) replaces the network fetch (tests, manual runs)."""

    name = SOURCE

    def __init__(
        self,
        url: str = URL,
        *,
        html: str | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        session: requests.Session | None = None,
    ) -> None:
        self.url = url
        self.html = html
        self.now = now
        self.session = session
        self.last_error: str | None = None
        self.last_status = "skipped"
        self.source_updated_at: str | None = None

    def _get(self) -> str:
        getter = self.session.get if self.session is not None else requests.get
        resp = getter(
            self.url,
            timeout=TIMEOUT_S,
            headers={
                "User-Agent": BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )
        if resp.status_code != 200:
            challenge = "just a moment" in resp.text[:2000].lower()
            raise RuntimeError(
                f"HTTP {resp.status_code}" + (" (Cloudflare bot challenge)" if challenge else "")
            )
        return resp.text

    def fetch(self) -> list[NewsItem]:
        fetched_at = self.now()
        try:
            html = self.html if self.html is not None else self._get()
            items, self.source_updated_at = parse_injury_table(html, fetched_at, self.url)
        except Exception as exc:  # noqa: BLE001 - a news source must never fail the build
            self.last_status, self.last_error = "error", f"{type(exc).__name__}: {exc}"
            log.warning("premier_injuries: %s", self.last_error)
            return []
        self.last_status, self.last_error = ("ok" if items else "empty"), None
        return items
