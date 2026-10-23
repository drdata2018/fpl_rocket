"""My Team: header, rank trajectory, points per GW, captain hindsight, transfer ledger, value, chips,
leagues, squad on a pitch, planning-state panel. Works for any loaded team ID."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from fpl_rocket import data, ui, views
from fpl_rocket.state import chips_status

st.title("My Team")
tid = ui.require_team()

if tid is not None:
    entry = hist = boot = players = live = None
    with ui.guard("Team header"):
        entry = data.entry(tid)
        hist = data.history(tid)
    with ui.guard("FPL season data"):
        boot = data.bootstrap()
        players = data.players()
    with ui.guard("Gameweek stats"):
        live = data.live_table()
    if live is None:
        live = views.live_frame({})
    last_finished = views.last_finished_gw(boot) if boot else None

    # ----------------------------------------------------------------------------------- header
    if entry is not None:
        total_players = (boot or {}).get("total_players")
        rank = entry.get("summary_overall_rank")
        st.subheader(ui.md_escape(entry.get("name", tid)))
        st.caption(f"Manager: {entry.get('player_first_name') or '-'} · team ID {tid}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total points", f"{entry.get('summary_overall_points') or 0:,}")
        c2.metric("Overall rank", f"{rank:,}" if rank else "-")
        c3.metric(
            "Percentile",
            f"top {100 * rank / total_players:.1f}%" if rank and total_players else "-",
            help="overall_rank / total_players",
        )
        c4.metric(
            f"GW{entry.get('current_event') or '-'} points",
            entry.get("summary_event_points") if entry.get("summary_event_points") is not None else "-",
        )

    gwdf = views.gw_frame(hist, boot or {}) if hist is not None else pd.DataFrame()
    if hist is not None and gwdf.empty:
        st.info(
            "This team has no published gameweeks yet: history, ledger and squad appear after its "
            "first deadline."
        )

    # ----------------------------------------------------------------------------------- rank + points charts
    if not gwdf.empty:
        a, b = st.columns(2)
        with a, ui.guard("Rank trajectory"):
            st.markdown("**Overall rank per gameweek** (up = better)")
            r = gwdf.dropna(subset=["overall_rank"])
            chart = (
                alt.Chart(r)
                .mark_line(point=True)
                .encode(
                    x=alt.X("gw:O", title="Gameweek", sort="ascending"),
                    y=alt.Y(
                        "overall_rank:Q", title="Overall rank", scale=alt.Scale(reverse=True, zero=False)
                    ),
                    tooltip=[
                        alt.Tooltip("gw:O", title="GW"),
                        alt.Tooltip("overall_rank:Q", title="Overall rank", format=","),
                        alt.Tooltip("total_points:Q", title="Total points"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(chart)
        with b, ui.guard("Points per gameweek"):
            st.markdown("**Points per gameweek vs GW average** (hits in red, bench below)")
            base = alt.Chart(gwdf).encode(x=alt.X("gw:O", title="Gameweek", sort="ascending"))
            tips = [
                alt.Tooltip("gw:O", title="GW"),
                alt.Tooltip("points:Q", title="Points"),
                alt.Tooltip("hits:Q", title="Hit cost"),
                alt.Tooltip("bench:Q", title="Bench points"),
                alt.Tooltip("gw_average:Q", title="GW average"),
            ]
            bars = base.mark_bar(color="#4c78a8").encode(y=alt.Y("points:Q", title="Points"), tooltip=tips)
            avg = base.mark_line(color="#f58518", point=True, strokeDash=[4, 3]).encode(
                y="gw_average:Q", tooltip=tips
            )
            hit_txt = (
                alt.Chart(gwdf[gwdf["hits"] > 0].assign(label=lambda d: "-" + d["hits"].astype(str)))
                .mark_text(dy=-8, color="#d62728", fontWeight="bold")
                .encode(x=alt.X("gw:O", sort="ascending"), y="points:Q", text="label:N")
            )
            bench = (
                base.mark_bar(color="#9e9e9e")
                .encode(y=alt.Y("bench:Q", title="Bench"), tooltip=tips)
                .properties(height=70)
            )
            st.altair_chart(alt.vconcat((bars + avg + hit_txt).properties(height=200), bench))
            st.caption(
                f"Hits this season: {int(gwdf['hits'].sum())} points · "
                f"points left on bench: {int(gwdf['bench'].sum())}"
            )

    # ------------------------------------------------------------------- planning state (built once per ID)
    state = state_error = None
    if hist is not None and not gwdf.empty:
        try:
            state = data.planning_state(tid)
        except Exception as exc:  # noqa: BLE001 - shown in the panel below
            state_error = ui.reason(exc)

    # ----------------------------------------------------------------------------------- value / bank / chips
    if not gwdf.empty:
        st.subheader("Value, bank and chips")
        v1, v2, v3, v4 = st.columns(4)
        last = gwdf.iloc[-1]
        v1.metric("Team value (FPL, last deadline)", f"{last['value']:.1f}", help="Includes the bank")
        v2.metric("Bank", f"{(state.bank if state else last['bank'] * 10) / 10:.1f}")
        v3.metric(
            "Selling value",
            f"{state.selling_value / 10:.1f}" if state else "-",
            help="Sum of selling prices + bank: what the squad is worth to this manager now",
        )
        v4.metric(
            "Market value",
            f"{state.market_value / 10:.1f}" if state else "-",
            help="Today's prices of the squad + bank",
        )
        c_used, c_av = st.columns(2)
        with c_used, ui.guard("Chips used"):
            st.markdown("**Chips used**")
            used = views.chips_used_frame(hist)
            if used.empty:
                st.caption("None yet.")
            else:
                ui.table(used, {"gw": st.column_config.NumberColumn("GW", format="%d")})
        with c_av, ui.guard("Chips available"):
            st.markdown("**Chips available**")
            if state is not None:
                avail = state.chips_available
            else:
                used_list = [(c["name"], int(c["event"])) for c in hist.get("chips", [])]
                upcoming = views.next_gw(boot) or (int(gwdf["gw"].max()) + 1)
                avail = chips_status(used_list, boot, upcoming)
            af = views.chips_available_frame(avail)
            if af.empty:
                st.caption("No chips left.")
            else:
                ui.table(af)

    # ----------------------------------------------------------------------------------- squad on a pitch
    if not gwdf.empty and players is not None:
        pub = int(gwdf["gw"].max())
        st.subheader(f"Squad (GW{pub} picks)")
        with ui.guard("Squad"):
            pk = data.picks(tid, pub)
            if not pk:
                st.caption(f"GW{pub} picks are not published.")
            else:
                pts_gw = pub if last_finished and pub <= last_finished else None
                rows = views.pitch(pk, players, live, pts_gw)
                label = f"GW{pts_gw}" if pts_gw else "pts"
                st.html(ui.pitch_html(rows, label))
                chip = views.CHIP_LABELS.get(pk.get("active_chip") or "", "")
                st.caption(
                    (f"Chip played: {chip}. " if chip else "")
                    + (
                        f"Points are each player's GW{pts_gw} total (before captaincy)."
                        if pts_gw
                        else f"GW{pub} is not finished: points appear once it is."
                    )
                    + " Prices are current."
                )

    # ----------------------------------------------------------------------------------- captain hindsight
    if not gwdf.empty and players is not None:
        st.subheader("Captain hindsight gap")
        st.caption("descriptive, uses hindsight")
        exp = st.expander("Show per-gameweek captain vs best XI player", on_change="rerun", key="cap_open")
        if exp.open:
            with exp, ui.guard("Captain hindsight gap"):
                gws = tuple(int(g) for g in gwdf["gw"])
                cg = views.captain_gap(data.all_picks(tid, gws), live, players)
                if cg.empty:
                    st.caption("No finished gameweeks with picks yet.")
                else:
                    ui.table(
                        cg,
                        {
                            "gw": st.column_config.NumberColumn("GW", format="%d"),
                            "captain": "Captain",
                            "captain_points": st.column_config.NumberColumn("Captain pts", format="%d"),
                            "best_xi_player": "Best XI player",
                            "best_points": st.column_config.NumberColumn("His pts", format="%d"),
                            "gap": st.column_config.NumberColumn("Gap", format="%d"),
                            "chip": "Chip",
                            "note": "Note",
                        },
                    )
                    st.caption(
                        f"Total gap {int(cg['gap'].sum())} points over {len(cg)} gameweeks. "
                        "Points are un-multiplied; "
                        "XI as picked, before automatic substitutions."
                    )

    # ----------------------------------------------------------------------------------- transfer ledger
    if not gwdf.empty and players is not None:
        st.subheader("Transfer ledger")
        with ui.guard("Transfer ledger"):
            led = views.transfer_ledger(data.transfers(tid), hist, live, players, last_finished)
            if led.empty:
                st.caption("No transfers made.")
            else:
                ui.table(
                    led,
                    {
                        "gw": st.column_config.NumberColumn("GW", format="%d"),
                        "out": "Out",
                        "out_price": st.column_config.NumberColumn("Sold at", format="%.1f"),
                        "in": "In",
                        "in_price": st.column_config.NumberColumn("Bought at", format="%.1f"),
                        "chip": "Chip",
                        "gw_hit_cost": st.column_config.NumberColumn(
                            "GW hit cost",
                            format="%d",
                            help="Total hit cost of that GW (not attributable per transfer)",
                        ),
                        "window": "Window",
                        "points_in": st.column_config.NumberColumn("Pts in", format="%d"),
                        "points_out": st.column_config.NumberColumn("Pts out", format="%d"),
                        "in_minus_out": st.column_config.NumberColumn("In minus out", format="%+d"),
                    },
                )
                st.caption(
                    "Points of the player bought minus the player sold over the window shown (to the last "
                    "finished gameweek; a free-hit transfer covers its own week only)."
                    " Descriptive: it ignores "
                    "what happened to either player after later transfers."
                )

    # ----------------------------------------------------------------------------------- leagues
    if entry is not None:
        st.subheader("Classic leagues")
        with ui.guard("Leagues"):
            lg = views.classic_leagues(entry)
            if lg.empty:
                st.caption("No classic leagues.")
            else:
                ui.table(
                    lg.drop(columns=["id"]),
                    {
                        "name": "League",
                        "type": "Type",
                        "rank": st.column_config.NumberColumn("Rank", format="%d"),
                        "last_rank": st.column_config.NumberColumn("Last rank", format="%d"),
                        "size": st.column_config.NumberColumn("Size", format="%d"),
                        "my_total": st.column_config.NumberColumn("My total", format="%d"),
                    },
                )
                st.caption("Open a league to load its leader (standings page 1) and the gap.")
                for _, row in lg.iterrows():
                    size = f" of {int(row['size']):,}" if pd.notna(row["size"]) else ""
                    rank_txt = f"{int(row['rank']):,}" if pd.notna(row["rank"]) else "-"
                    lx = st.expander(
                        f"{row['name']} - rank {rank_txt}{size}",
                        on_change="rerun",
                        key=f"league_{int(row['id'])}",
                    )
                    if lx.open:
                        with lx, ui.guard("League standings"):
                            my_total = int(row["my_total"]) if pd.notna(row["my_total"]) else None
                            gap = views.league_gap(data.league_standings_page1(int(row["id"])), tid, my_total)
                            if gap["leader_total"] is None:
                                st.caption("No standings published yet.")
                            else:
                                g1, g2, g3 = st.columns(3)
                                g1.metric("Leader total", f"{gap['leader_total']:,}")
                                g2.metric(
                                    "My total", f"{gap['my_total']:,}" if gap["my_total"] is not None else "-"
                                )
                                g3.metric(
                                    "Gap to leader",
                                    f"{gap['gap']:,}" if gap["gap"] is not None else "-",
                                    help="leader total - my total",
                                )

    # ----------------------------------------------------------------------------------- planning state panel
    if hist is not None and not gwdf.empty:
        st.subheader("Planning state")
        with st.container(border=True):
            if state is None:
                st.warning(f"The planning state could not be established: {state_error}.")
            else:
                st.markdown(
                    f"**Status: awaiting confirmation** · as of {state.as_of['picks']} picks + "
                    f"{state.transfers_since_publication} transfer(s) since · "
                    f"built {ui.fmt_ts(state.as_of['built_at'])}"
                )
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Bank", f"{state.bank / 10:.1f}")
                ft = state.free_transfers
                s2.metric(
                    f"Free transfers for GW{state.upcoming_gw}",
                    str(ft),
                    help="Exact"
                    if ft.exact
                    else "A range: several balances are consistent with the observed hit costs",
                )
                s3.metric("Selling value", f"{state.selling_value / 10:.1f}")
                s4.metric("Squad size", len(state.squad))
                if players is not None:
                    ui.table(
                        views.planning_squad_frame(state, players),
                        {
                            "purchase": st.column_config.NumberColumn("Purchase", format="%.1f"),
                            "selling": st.column_config.NumberColumn("Selling", format="%.1f"),
                            "current": st.column_config.NumberColumn("Current", format="%.1f"),
                            "purchase_source": "Purchase price source",
                        },
                    )
                    with ui.guard("Pending transfers"):
                        pend = views.pending_transfers_frame(
                            data.transfers(tid), state.last_published_gw, players
                        )
                        st.markdown(f"**Pending transfers since GW{state.last_published_gw}: {len(pend)}**")
                        if len(pend):
                            ui.table(
                                pend,
                                {
                                    "gw": st.column_config.NumberColumn("GW", format="%d"),
                                    "out_price": st.column_config.NumberColumn("Sold at", format="%.1f"),
                                    "in_price": st.column_config.NumberColumn("Bought at", format="%.1f"),
                                },
                            )
                if state.warnings:
                    st.markdown("**Warnings**")
                    for w in state.warnings:
                        st.markdown(f"- {w}")
                else:
                    st.caption(
                        "No warnings: squad legal, hit costs consistent with the free-transfer replay."
                    )
                st.caption(
                    f"Prices: {state.as_of['prices']}. Latest transfer: {state.as_of['latest_transfer']}."
                )
            b1, b2, _ = st.columns([1, 1, 4])
            b1.button("Confirm state", disabled=True, help="Confirm/edit arrives with the planner (Phase B)")
            b2.button("Edit state", disabled=True, help="Confirm/edit arrives with the planner (Phase B)")
            st.caption("The planner will only plan from a state you have confirmed or edited.")
