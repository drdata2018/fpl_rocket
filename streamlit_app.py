"""FPL Rocket - hosting spike page (Streamlit Community Cloud entry point).

On load it calls the same four public FPL endpoints as scripts/spike_probe.py from the Streamlit host and
shows status / latency / bytes with the server UTC time. Nothing else yet.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # Community Cloud installs requirements.txt only, not the package
sys.path.insert(0, str(ROOT / "scripts"))

from spike_probe import probe_all  # noqa: E402

st.set_page_config(page_title="FPL Rocket — hosting spike", layout="centered")
st.title("FPL Rocket — hosting spike")
st.caption("Calls four public FPL API endpoints from this server with a browser User-Agent.")

if "runs" not in st.session_state or st.button("Probe again"):
    st.session_state["runs"] = probe_all()

st.write(f"Server UTC time: **{datetime.now(UTC).isoformat(timespec='seconds')}**")
df = pd.DataFrame(st.session_state["runs"])
st.dataframe(
    df[[c for c in ("endpoint", "status", "ms", "bytes", "error", "ts", "host_hint") if c in df]],
    hide_index=True,
    use_container_width=True,
)
ok = int((df["status"] == 200).sum())
st.write(f"{ok}/{len(df)} calls returned 200.")
