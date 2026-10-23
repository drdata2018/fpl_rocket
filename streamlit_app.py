"""FPL Rocket - Streamlit Community Cloud entry point: navigation, sidebar (team ID), timestamps.

Pages live in app_pages/ (all but hosting_spike.py are in the navigation). Community Cloud installs
requirements.txt only (not the package), so src/ and
scripts/ are put on sys.path here before any page imports fpl_rocket.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fpl_rocket import ui  # noqa: E402

st.set_page_config(page_title="FPL Rocket", page_icon=":rocket:", layout="wide")

PAGES = [
    st.Page(
        ROOT / "app_pages" / "season_overview.py",
        title="Season Overview",
        icon=":material/stadium:",
        default=True,
    ),
    st.Page(ROOT / "app_pages" / "my_team.py", title="My Team", icon=":material/groups:"),
    st.Page(
        ROOT / "app_pages" / "compare_players.py", title="Compare Players", icon=":material/compare_arrows:"
    ),
    st.Page(ROOT / "app_pages" / "differentials.py", title="Differentials", icon=":material/filter_alt:"),
    st.Page(ROOT / "app_pages" / "model_planner.py", title="Model & Planner", icon=":material/construction:"),
]
# app_pages/hosting_spike.py (the day-1 hosting probe) is kept for diagnosing the host but is no longer in the
# public navigation: it calls the FPL API from the server on every visit. Add it back here when needed.

page = st.navigation(PAGES)
footer = ui.sidebar()
page.run()
ui.footer_timestamps(footer)
