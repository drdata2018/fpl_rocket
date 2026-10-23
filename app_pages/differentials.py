"""Differentials: low-ownership players with descriptive columns, plus the EXPERIMENTAL xP for the next GW
and summed over the next 3 GWs, and the P(play) band, when the model bundle is available ("insufficient
history" for newcomers). No recommendations; nothing beyond 3 GWs (decision 1, GW9)."""

from __future__ import annotations

import streamlit as st

from fpl_rocket import data, ui, views

st.title("Differentials")
st.info(
    "Descriptive columns plus an experimental xP (next gameweek and next 3 gameweeks) and a P(play) band "
    "(model backtested retrospectively only; its prospective record starts at the GW9 deadline). Low "
    "ownership, form and xP are not recommendations."
)

players = boot = None
with ui.guard("FPL season data"):
    boot = data.bootstrap()
    players = data.players()

if players is not None:
    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1.6])
    threshold = c1.slider("Ownership below (%)", 0.5, 30.0, 10.0, 0.5, key="dif_own")
    pos = c2.selectbox("Position", ["All", *views.POS_ORDER], key="dif_pos")
    lo, hi = float(players["price"].min()), float(players["price"].max())
    price = c3.slider("Price range", lo, hi, (lo, hi), 0.1, format="%.1f", key="dif_price")
    clubs = c4.multiselect("Club", sorted(players["team"].unique()), key="dif_clubs")

    with ui.guard("Differentials table"):
        try:
            live = data.live_table()
        except Exception as exc:  # noqa: BLE001 - the table still works without minutes
            st.warning(f"Minutes of the last 5 gameweeks unavailable: {ui.reason(exc)}.")
            live = views.live_frame({})
        try:
            fixtures = data.fixtures()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Fixtures unavailable: {ui.reason(exc)}.")
            fixtures = []
        model = data.model_status()
        xpn = None
        if model.get("available"):
            xpn = views.xp_next(data.xp_table(), views.next_gw(boot))
        df = views.differentials(
            players,
            live,
            fixtures,
            boot,
            max_ownership=threshold,
            pos=None if pos == "All" else pos,
            price_range=price,
            clubs=clubs or None,
            xp=xpn,
        )
        st.caption(f"{len(df)} players. Sorted by total points (click a header to re-sort).")
        ui.table(
            df,
            {
                "minutes_last5": st.column_config.NumberColumn("Minutes, last 5 GWs", format="%d"),
                "next_fixtures": st.column_config.TextColumn(
                    "Next 5 fixtures (FPL difficulty)", width="large"
                ),
                **ui.xp_columns(model.get("version")),
            },
            height=600,
        )
        if xpn is None:
            st.caption(f"xP columns unavailable: {model.get('reason') or 'no model bundle'}.")
        else:
            st.caption(
                ui.model_caption(model)
                + " xP for the next gameweek and summed over the next 3 (the first three gameweeks of the"
                " fixture list); P(play) as a band. Hover the column headers for the caveats."
            )
