"""Hosting spike page: calls the same four public FPL endpoints as scripts/spike_probe.py from this host
and shows status / latency / bytes with the server UTC time (kept for diagnosing the host)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from spike_probe import probe_all

st.title("Hosting spike")
st.caption("Calls four public FPL API endpoints from this server with a browser User-Agent.")

if "spike_runs" not in st.session_state or st.button("Probe again"):
    st.session_state["spike_runs"] = probe_all()

st.write(f"Server UTC time: **{datetime.now(UTC).isoformat(timespec='seconds')}**")
df = pd.DataFrame(st.session_state["spike_runs"])
st.dataframe(
    df[[c for c in ("endpoint", "status", "ms", "bytes", "error", "ts", "host_hint") if c in df]],
    hide_index=True,
)
ok = int((df["status"] == 200).sum())
st.write(f"{ok}/{len(df)} calls returned 200.")
