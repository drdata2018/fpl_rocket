"""Phase A acceptance matrix against the LIVE FPL API (docs/PLAN_v2.md section 3), driven by AppTest.

Cases: the placeholder ID; a late entrant; an inactive manager (no transfers in the last 5+ GWs); a
wildcard + free-hit user; a manager with 3+ transfers in one non-chip GW; a team with no published history;
invalid IDs (0, 999999999, "abc"); two sessions with different IDs (no state leakage). Sampled managers come
from leagues-classic/314 (overall) standings pages, as in scripts/ft_hypotheses.py. Also measures cold and
warm load times. Prints a report; writes nothing.

Usage: uv run python scripts/acceptance.py [--pages 1 200 2000 20000 60000 100000] [--sleep 0.1]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_rocket.api import FPLAPIError, FPLClient  # noqa: E402
from fpl_rocket.config import PLACEHOLDER_TEAM_ID  # noqa: E402
from fpl_rocket.state import PlanningStateError, build_planning_state, fetch_inputs  # noqa: E402

APP = str(ROOT / "streamlit_app.py")
PAGES = (
    "app_pages/season_overview.py",
    "app_pages/my_team.py",
    "app_pages/compare_players.py",
    "app_pages/differentials.py",
    "app_pages/model_planner.py",
)


# ------------------------------------------------------------------------------------------------ sampling
def classify(entry: dict[str, Any], hist: dict[str, Any]) -> set[str]:
    cur = sorted(hist.get("current", []), key=lambda h: h["event"])
    chips = {c["name"]: int(c["event"]) for c in hist.get("chips", [])}
    chip_gws = {g for n, g in chips.items() if n in ("wildcard", "freehit")}
    tags = set()
    if not cur:
        tags.add("no_history")
        return tags
    if int(entry.get("started_event") or 1) > 1:
        tags.add("late_entrant")
    if len(cur) >= 5 and all(int(h["event_transfers"]) == 0 for h in cur[-5:]):
        tags.add("inactive")
    if "wildcard" in chips and "freehit" in chips:
        tags.add("wc_fh")
    if any(int(h["event_transfers"]) >= 3 and int(h["event"]) not in chip_gws for h in cur):
        tags.add("three_plus")
    return tags


def sample(client: FPLClient, pages: list[int], sleep: float, want: set[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for page in pages:
        try:
            rows = client.league_classic_standings(314, page)["standings"]["results"]
        except FPLAPIError as exc:
            print(f"  page {page}: {exc}")
            continue
        for r in rows:
            eid = int(r["entry"])
            try:
                tags = classify(client.entry(eid), client.entry_history(eid))
            except FPLAPIError:
                continue
            time.sleep(sleep)
            for t in tags & (want - set(found)):
                found[t] = eid
            if want <= set(found):
                return found
    return found


def find_no_history(client: FPLClient, total_players: int) -> int | None:
    """Newest registered IDs (just above the ranked count) usually have no published GW yet."""
    for eid in range(total_players + 2000, total_players - 2000, -97):
        try:
            if not client.entry_history(eid).get("current"):
                return eid
        except FPLAPIError:
            continue
    return None


# ------------------------------------------------------------------------------------------------ app checks
def run_pages(tid: str | None) -> tuple[list[str], dict[str, float], AppTest]:
    """Run every page for one session; returns (problems, seconds per page, the AppTest)."""
    at = AppTest.from_file(APP, default_timeout=180)
    if tid is not None:
        at.query_params["id"] = tid
    problems: list[str] = []
    times: dict[str, float] = {}
    for page in PAGES:
        t0 = time.perf_counter()
        at.switch_page(page)
        at.run()
        times[page] = time.perf_counter() - t0
        if at.exception:
            problems.append(f"{page}: EXCEPTION {[e.value for e in at.exception][:1]}")
        for w in at.warning:
            problems.append(f"{page}: warning {w.value}")
    # open the lazy sections once (captain hindsight, first league) to render them too
    at.switch_page("app_pages/my_team.py")
    at.session_state["cap_open"] = True
    at.run()
    if at.exception:
        problems.append(f"my_team (captain open): EXCEPTION {[e.value for e in at.exception][:1]}")
    return problems, times, at


def state_check(client: FPLClient, eid: int) -> str:
    try:
        st = build_planning_state(**fetch_inputs(client, eid))
    except (FPLAPIError, PlanningStateError) as exc:
        return f"planning state: {type(exc).__name__}: {exc}"
    incons = [w for w in st.warnings if "inconsistent" in w]
    ft = st.free_transfers
    return (
        f"FT GW{st.upcoming_gw} = {ft} ({'exact' if ft.exact else 'RANGE'}), bank {st.bank / 10:.1f}, "
        f"pending {st.transfers_since_publication}, chips used {[f'{n}@{g}' for n, g in st.chips_used]}, "
        f"hit-cost replay {'AGREES' if not incons else 'DISAGREES: ' + incons[0]}, "
        f"warnings {len(st.warnings)}{': ' + '; '.join(st.warnings[:3]) if st.warnings else ''}"
    )


def invalid_ids() -> list[str]:
    out = []
    for raw in ("0", "999999999", "abc"):
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        at.sidebar.text_input[0].input(raw)
        at.sidebar.button[0].click()
        at.run()
        errs = [e.value for e in at.sidebar.error]
        tid = at.session_state["team_id"]
        exc = bool(at.exception)
        out.append(f"{raw!r}: sidebar error {errs}; team_id={tid}; exception={exc}")
    return out


def leakage(a: str, b: str) -> list[str]:
    at_a = AppTest.from_file(APP, default_timeout=120)
    at_b = AppTest.from_file(APP, default_timeout=120)
    at_a.query_params["id"] = a
    at_b.query_params["id"] = b
    res = []
    for _ in range(2):  # interleave runs
        for at in (at_a, at_b):
            at.switch_page("app_pages/my_team.py")
            at.run()
    for name, at, tid in (("A", at_a, a), ("B", at_b, b)):
        cap = [c.value for c in at.caption if "team ID" in c.value]
        res.append(
            f"session {name}: team_id={at.session_state['team_id']} (expected {tid}); header {cap[:1]}"
        )
    ok = (
        at_a.session_state["team_id"] == int(a)
        and at_b.session_state["team_id"] == int(b)
        and any(f"team ID {a}" in c.value for c in at_a.caption)
        and any(f"team ID {b}" in c.value for c in at_b.caption)
        and not any(f"team ID {b}" in c.value for c in at_a.caption)
        and not any(f"team ID {a}" in c.value for c in at_b.caption)
    )
    res.append("NO LEAKAGE" if ok else "LEAKAGE OR MISMATCH")
    return res


# ------------------------------------------------------------------------------------------------ timings
def timings(ids: list[str]) -> list[str]:
    import streamlit as st

    out = []
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    t0 = time.perf_counter()
    at.run()  # Season Overview, no ID: bootstrap + fixtures + live GWs
    cold_shared = time.perf_counter() - t0
    t0 = time.perf_counter()
    at.run()
    warm_shared = time.perf_counter() - t0
    out.append(f"cold shared load (Season Overview, empty cache): {cold_shared:.2f}s")
    out.append(f"warm shared load (Season Overview rerun): {warm_shared:.2f}s")
    first, warm = [], []
    for tid in ids:
        at = AppTest.from_file(APP, default_timeout=180)
        at.query_params["id"] = tid
        at.switch_page("app_pages/my_team.py")
        t0 = time.perf_counter()
        at.run()
        first.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        at.run()
        warm.append(time.perf_counter() - t0)
    out.append(
        f"per-ID first load (My Team, shared cache warm, ID not cached) over {len(ids)} IDs: "
        f"median {statistics.median(first):.2f}s, max {max(first):.2f}s"
    )
    out.append(
        f"per-ID warm load (My Team rerun, ID cached): "
        f"median {statistics.median(warm):.2f}s, max {max(warm):.2f}s"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+", default=[1, 200, 2000, 20000, 60000, 100000, 140000])
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args(argv)
    client = FPLClient(backoff=0.5)
    boot = client.bootstrap()

    print("== sampling managers from leagues-classic/314")
    want = {"late_entrant", "inactive", "wc_fh", "three_plus"}
    found = sample(client, args.pages, args.sleep, want)
    nohist = find_no_history(client, int(boot["total_players"]))
    cases = {"owner (placeholder)": PLACEHOLDER_TEAM_ID}
    cases.update({k: str(v) for k, v in found.items()})
    if nohist:
        cases["no_history"] = str(nohist)
    for k in sorted(want - set(found)):
        print(f"  NOT FOUND in sample: {k}")

    print("== no ID")
    problems, _, _ = run_pages(None)
    print("  pages render:", problems or "OK")

    bad = 0
    for label, tid in cases.items():
        print(f"== {label}: {tid}")
        problems, times, _ = run_pages(tid)
        bad += any("EXCEPTION" in p for p in problems)
        print("  pages:", problems or "OK, no exceptions/warnings")
        print("  page times:", {Path(k).stem: round(v, 2) for k, v in times.items()})
        print("  ", state_check(client, int(tid)))

    print("== invalid IDs")
    for line in invalid_ids():
        print("  ", line)
    print("== two sessions")
    ids = [v for v in cases.values()][:2]
    for line in leakage(*ids):
        print("  ", line)
    print("== timings")
    for line in timings([v for k, v in cases.items() if k != "no_history"]):
        print("  ", line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
