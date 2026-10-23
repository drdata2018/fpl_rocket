"""Compare Players: 2-4 players side by side, per-GW points, 5-GW fixture strip (FPL difficulty), plus the
EXPERIMENTAL xP (next GW, next 3 GWs) and P(play) band rows when the model bundle is available."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from fpl_rocket import data, ui, views

st.title("Compare Players")

players = boot = None
with ui.guard("FPL season data"):
    boot = data.bootstrap()
    players = data.players()

if players is not None:
    by_owner = players.sort_values(["ownership", "total_points"], ascending=False)
    labels = {int(r["id"]): views.player_label(r) for _, r in by_owner.iterrows()}
    ids = st.multiselect(
        "Players (type to search by name)",
        options=list(labels),
        default=list(labels)[:2],
        format_func=lambda i: labels.get(i, str(i)),
        max_selections=4,
        key="cmp_ids",
    )
    if len(ids) < 2:
        st.info("Pick 2 to 4 players.")
    else:
        st.subheader("Season metrics")
        with ui.guard("Metric table"):
            table = views.compare_table(players, ids)
            model = data.model_status()
            if model.get("available"):
                xpn = views.xp_next(data.xp_table(), views.next_gw(boot))
                table = pd.concat([table, views.compare_xp_rows(players, ids, xpn, model.get("version"))])
            st.dataframe(table)
            st.caption("Ownership = selected_by_percent. Defensive contribution is the raw season count.")
            if model.get("available"):
                st.caption(ui.model_caption(model))
                st.caption(
                    ui.XP_HELP.format(version=model.get("model_version") or "")
                    + " "
                    + ui.XP3_HELP.format(version=model.get("model_version") or "")
                )
                st.caption(f"P(play): {ui.P_PLAY_HELP}")
            else:
                st.caption(f"xP rows unavailable: {model.get('reason') or 'no model bundle'}.")

        st.subheader("Points per gameweek")
        with ui.guard("Per-GW points"):
            live = data.live_table()
            pg = views.points_by_gw(live, players, ids)
            if pg.empty:
                st.caption("No finished gameweeks yet.")
            else:
                chart = (
                    alt.Chart(pg)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("gw:O", title="Gameweek", sort="ascending"),
                        y=alt.Y("points:Q", title="Points"),
                        color=alt.Color("player:N", title="Player"),
                        tooltip=[
                            alt.Tooltip("player:N", title="Player"),
                            alt.Tooltip("gw:O", title="GW"),
                            alt.Tooltip("points:Q", title="Points"),
                            alt.Tooltip("minutes:Q", title="Minutes"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart)

        st.subheader("Next 5 fixtures")
        with ui.guard("Fixture strip"):
            nxt = views.next_gw(boot)
            if nxt is None:
                st.caption("No upcoming gameweeks.")
            else:
                gws = [g for g in range(nxt, nxt + 5) if g <= 38]
                by_team = views.fixtures_by_team(data.fixtures(), gws[0], gws[-1])
                short = views.team_short(boot)
                idx = players.set_index("id")
                text_rows, diff_rows = {}, {}
                for pid in ids:
                    r = idx.loc[pid]
                    label = f"{r['name']} ({r['team']})"
                    text_rows[label], diff_rows[label] = views.fixture_strip_cells(
                        int(r["team_id"]), by_team, gws, short
                    )
                text = pd.DataFrame.from_dict(text_rows, orient="index")
                diff = pd.DataFrame.from_dict(diff_rows, orient="index")
                ui.difficulty_legend()
                st.dataframe(text.style.apply(ui.difficulty_style(diff), axis=None))
                st.caption("FPL difficulty. Double gameweeks list both fixtures; '-' is a blank.")
