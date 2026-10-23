"""Name matching of news items to FPL elements. Conservative: an item is matched only when exactly ONE player
of
the item's club fits; otherwise it goes to news_unmatched with the reason. Nothing is guessed.

Normalisation: Unicode NFKD with combining marks removed (Hincapié -> hincapie, Ødegaard -> odegaard via an
explicit map for letters NFKD cannot split), lower case, punctuation -> space, whitespace collapsed.
Club: the item's team_hint must resolve to exactly one FPL team (name or short_name, else TEAM_ALIASES).
Player, within that club, in order (first step with exactly one candidate wins; two or more = ambiguous):
    1 full name     raw == first_name + second_name
    2 web_name      raw == web_name
    3 first+second  every token of second_name and the first token of first_name appear in raw
    4 web tokens    every token of web_name appears in raw
    5 second name   every token of second_name appears in raw
    6 first surname the first tokens of first_name and second_name appear in raw (double surnames:
                    "Julian Araujo" = Julián Araujo Zúñiga)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_EXTRA = str.maketrans(
    {
        "ø": "o",
        "Ø": "o",
        "ð": "d",
        "þ": "th",
        "ß": "ss",
        "æ": "ae",
        "Æ": "ae",
        "ł": "l",
        "Ł": "l",
        "đ": "d",
        "ı": "i",
        "œ": "oe",
    }
)

# common long / alternative club names -> FPL `name` (normalised); FPL names and short names match directly
TEAM_ALIASES = {
    "afc bournemouth": "bournemouth",
    "brighton and hove albion": "brighton",
    "brighton hove albion": "brighton",
    "manchester city": "man city",
    "manchester united": "man utd",
    "man united": "man utd",
    "newcastle united": "newcastle",
    "nottingham forest": "nott m forest",
    "nottm forest": "nott m forest",
    "tottenham hotspur": "spurs",
    "tottenham": "spurs",
    "west ham united": "west ham",
    "wolverhampton wanderers": "wolves",
    "wolverhampton": "wolves",
    "leeds united": "leeds",
    "leicester city": "leicester",
    "ipswich town": "ipswich",
    "coventry city": "coventry",
    "hull city": "hull",
    "sheffield united": "sheffield utd",
    "luton town": "luton",
    "norwich city": "norwich",
    "stoke city": "stoke",
    "swansea city": "swansea",
    "cardiff city": "cardiff",
    "west bromwich albion": "west brom",
    "queens park rangers": "qpr",
    "middlesbrough fc": "middlesbrough",
}


def normalise(s: Any) -> str:
    s = str(s or "").translate(_EXTRA)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


@dataclass(frozen=True)
class Match:
    element: int | None
    method: str  # full_name | web_name | first_second | web_tokens | second_name | element_id | -
    reason: str | None = None  # why unmatched: no_team | ambiguous_team | no_player | ambiguous_player


class Matcher:
    def __init__(self, elements: list[dict[str, Any]], teams: list[dict[str, Any]]) -> None:
        self.team_keys: dict[str, set[int]] = {}
        for t in teams:
            for k in (t.get("name"), t.get("short_name")):
                if k:
                    self.team_keys.setdefault(normalise(k), set()).add(int(t["id"]))
        self.by_team: dict[int, list[dict[str, Any]]] = {}
        self.ids = set()
        for e in elements:
            first, second, web = (
                normalise(e.get("first_name")),
                normalise(e.get("second_name")),
                normalise(e.get("web_name")),
            )
            row = {
                "id": int(e["id"]),
                "full": f"{first} {second}".strip(),
                "web": web,
                "first1": first.split()[0] if first else "",
                "second_t": set(second.split()),
                "second1": second.split()[0] if second else "",
                "web_t": set(web.split()),
            }
            self.by_team.setdefault(int(e["team"]), []).append(row)
            self.ids.add(int(e["id"]))

    def team(self, hint: str | None) -> tuple[int | None, str | None]:
        k = normalise(hint)
        if not k:
            return None, "no_team"
        ids = self.team_keys.get(k) or self.team_keys.get(TEAM_ALIASES.get(k, k), set())
        if not ids:
            return None, "no_team"
        if len(ids) > 1:
            return None, "ambiguous_team"
        return next(iter(ids)), None

    def match(self, name: str, team_hint: str | None, element_hint: int | None = None) -> Match:
        if element_hint is not None:
            return (
                Match(element_hint, "element_id")
                if element_hint in self.ids
                else Match(None, "-", "no_player")
            )
        team, why = self.team(team_hint)
        if team is None:
            return Match(None, "-", why)
        raw = normalise(name)
        toks = set(raw.split())
        cands = self.by_team.get(team, [])
        steps = (
            ("full_name", lambda c: c["full"] == raw),
            ("web_name", lambda c: c["web"] == raw),
            ("first_second", lambda c: bool(c["second_t"]) and c["second_t"] <= toks and c["first1"] in toks),
            ("web_tokens", lambda c: bool(c["web_t"]) and c["web_t"] <= toks),
            ("second_name", lambda c: bool(c["second_t"]) and c["second_t"] <= toks),
            ("first_surname", lambda c: bool(c["second1"]) and {c["first1"], c["second1"]} <= toks),
        )
        for method, ok in steps:
            hits = [c["id"] for c in cands if ok(c)]
            if len(hits) == 1:
                return Match(hits[0], method)
            if len(hits) > 1:
                return Match(None, "-", "ambiguous_player")
        return Match(None, "-", "no_player")
