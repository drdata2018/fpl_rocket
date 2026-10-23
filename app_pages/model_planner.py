"""Model & Planner: placeholder describing what is coming, with project status read from the repo."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import fpl_rocket

st.title("Model & Planner")
st.info("Coming in the next phase. Nothing on this page is a prediction yet.")

st.subheader("What is coming")
st.markdown(
    """
**Points for one gameweek.** An expected-points (xP) model per player per fixture - minutes, team goals,
attacking share, bonus and defensive contributions, scored by the same rules engine that already reproduces
every official point - and a planner that picks the best legal XI, captain and transfers for the next deadline
from your confirmed squad, bank and free transfers.

**Expected points over the next few gameweeks.** The default plan: a transfer path over a 5-gameweek horizon
with free-transfer banking and 4-point hits, discounted for uncertainty, always legal and affordable for the
entered team. Two or three variants (more or less template) are shown next to it with their simulated points
range, labelled experimental.

**Chips.** For each chip still available, the gain of playing it inside the evaluated window, with the window
stated. No "play it now" verb until its season-level opportunity cost can be shown.

**Beat a rival (later).** Pick a league and a rival; candidate plans are scored by the simulated chance of
being ahead of that rival after the horizon, with the assumptions about the rival disclosed on the page.
"""
)

st.subheader("Status")
status_file = Path(fpl_rocket.__file__).resolve().parent / "project_status.json"
try:
    status = json.loads(status_file.read_text(encoding="utf-8"))
    for line in status["lines"]:
        st.markdown(f"- **{line['item']}:** {line['status']}")
    st.caption(f"Status file updated {status.get('updated', '-')}.")
except (OSError, ValueError, KeyError) as exc:
    st.warning(f"Project status unavailable ({type(exc).__name__}).")
