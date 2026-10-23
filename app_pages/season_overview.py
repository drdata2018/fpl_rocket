"""Season Overview: league table, leaders, last-GW review, prices, availability, template, fixture grid."""

from __future__ import annotations

import streamlit as st

from fpl_rocket import data, ui, views

st.title("Season Overview")

boot = players = live = fixtures = None
with ui.guard("FPL season data"):
    boot = data.bootstrap()
    players = data.players()

if boot is not None:
    last_gw = views.last_finished_gw(boot)
    nxt = views.next_gw(boot)

    # ----------------------------------------------------------------------------------- standings
    st.subheader("Premier League table")
    with ui.guard("The league table"):
        fixtures = data.fixtures()
        tab = views.standings(fixtures, boot)
        st.caption("Computed from finished fixtures in the FPL API.")
        ui.table(tab, height=35 * 21 + 3)

    # ----------------------------------------------------------------------------------- leaders
    st.subheader("Season leaders")
    c1, c2 = st.columns([1, 2])
    metric = c1.selectbox("Metric", list(views.LEADER_METRICS), key="so_metric")
    pos = c2.radio("Position", ["All", *views.POS_ORDER], horizontal=True, key="so_pos")
    with ui.guard("Season leaders"):
        lead = views.leaders(players, metric, None if pos == "All" else pos, n=10)
        ui.table(lead)
        if metric == "Defensive contribution":
            st.caption(
                "Defensive contribution is the raw season count (CBI + tackles, plus recoveries for "
                "MID/FWD); points are awarded per match at 10 (DEF) / 12 (MID/FWD)."
            )

    # ----------------------------------------------------------------------------------- last GW review
    if last_gw:
        st.subheader(f"Gameweek {last_gw} review")
        with ui.guard(f"GW{last_gw} review"):
            live = data.live_table()
            ev = views.event_summary(boot, last_gw)
            names = dict(zip(players["id"], players["name"], strict=True))
            m1, m2, m3 = st.columns(3)
            m1.metric("GW average", ev["average"] if ev["average"] is not None else "-")
            m2.metric("Highest score", ev["highest"] if ev["highest"] is not None else "-")
            m3.metric(
                "Most captained (FPL)",
                names.get(ev["most_captained"], "-") if ev["most_captained"] else "-",
                help="FPL publishes only the single most-captained player, not captaincy counts.",
            )
            a, b = st.columns([3, 2])
            with a:
                st.markdown(f"**Top 10 scorers, GW{last_gw}**")
                ui.table(views.gw_top_scorers(players, live, last_gw))
            with b:
                st.markdown(f"**5 most owned and their GW{last_gw} points**")
                ui.table(views.most_owned_returns(players, live, last_gw))
                st.caption(
                    "Captaincy counts are not in the public API: current ownership "
                    "(selected_by_percent) is shown as a proxy."
                )

    # ----------------------------------------------------------------------------------- prices
    st.subheader("Price changes since the last deadline")
    with ui.guard("Price changes"):
        risers, fallers = views.price_changes(players)
        a, b = st.columns(2)
        with a:
            st.markdown(f"**Risers ({len(risers)})**")
            if len(risers):
                ui.table(risers)
            else:
                st.caption("No risers yet this gameweek.")
        with b:
            st.markdown(f"**Fallers ({len(fallers)})**")
            if len(fallers):
                ui.table(fallers)
            else:
                st.caption("No fallers yet this gameweek.")

    # ----------------------------------------------------------------------------------- availability
    st.subheader("Availability and news")
    with ui.guard("Availability feed"):
        av = views.availability(players)
        st.caption(f"{len(av)} players flagged or with news, newest first.")
        ui.table(
            av,
            {
                "status_label": st.column_config.TextColumn("Status"),
                "chance_next": st.column_config.NumberColumn("Chance next GW", format="%d%%"),
                "news": st.column_config.TextColumn("News", width="large"),
                "news_added": st.column_config.DatetimeColumn("News added", format="YYYY-MM-DD HH:mm"),
            },
            height=400,
        )

    # ----------------------------------------------------------------------------------- template
    st.subheader("Template squad by ownership")
    with ui.guard("Template squad"):
        tpl = views.template_squad(players)
        st.caption(
            "The most-owned 2 GKP, 5 DEF, 5 MID and 3 FWD. Not budget-checked: "
            f"total price {tpl['price'].sum():.1f}."
        )
        ui.table(tpl)

    # ----------------------------------------------------------------------------------- fixture grid
    st.subheader("Fixture grid")
    if nxt is None:
        st.caption("No upcoming gameweeks.")
    else:
        n = st.slider("Gameweeks ahead", 3, 8, 5, key="so_grid_n")
        with ui.guard("Fixture grid"):
            fixtures = fixtures if fixtures is not None else data.fixtures()
            n = min(n, 38 - nxt + 1)
            text, diff = views.fixture_grid(fixtures, boot, nxt, n)
            ui.difficulty_legend()
            st.dataframe(text.style.apply(ui.difficulty_style(diff), axis=None), height=35 * 21 + 3)
            st.caption(
                "Colours use FPL difficulty (team_h_difficulty / team_a_difficulty). "
                "Double gameweeks show both fixtures, coloured by their mean; '-' is a blank."
            )
