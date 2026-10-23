"""Model & Planner - SHADOW mode (docs/PLAN_v2.md section 4; decision 1 of the GW9 review).

For a loaded team ID: the planning state with confirm / edit (session-only corrections of bank, free
transfers and squad). While config.SHOW_SHADOW_PLANS is False (the public setting) NO plan is computed or
shown: the page explains that the planner's proposals are recorded before each deadline and scored afterwards
(the prospective registry, fpl_rocket.registry) and shows the registry's status. With the flag True the
existing plan UI renders: (a) the next-GW plan without transfers (XI, captain, vice, bench order,
explanation, edits stamped "user-edited") and, in a collapsed experimental section, (b) transfers, (c) the
multi-GW path and (d) chip scenarios.

The planner (PuLP + HiGHS) is imported in a guard: if the solver cannot be installed on the host, this page
still renders the planning state and the registry status, and no other page imports the planner.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from fpl_rocket import config, data, registry, ui, views
from fpl_rocket.rules.season_2026_27 import SQUAD

PLANNER_ERROR: str | None = None
try:
    from fpl_rocket.planner import (
        DECAY_OPTIONS,
        DEFAULT_DECAY,
        DEFAULT_HORIZON,
        MULTI_GW,
        ONE_GW_NO_TRANSFERS,
        ONE_GW_WITH_TRANSFERS,
        PlanSettings,
        StateEdits,
        from_planning_state,
        plan_key,
    )
except Exception as exc:  # noqa: BLE001 - ImportError, or OSError from the solver's native library
    PLANNER_ERROR = f"{type(exc).__name__}: {exc}"

SHOW_PLANS = bool(config.SHOW_SHADOW_PLANS)

st.title("Model & Planner")
if SHOW_PLANS:
    st.info(
        "Shadow mode: the planner runs on an EXPERIMENTAL xP model (backtested retrospectively only). Its "
        "plans are recorded and scored before any of them is offered as advice."
    )
else:
    st.info(
        "Shadow mode: the planner's proposals are recorded before every deadline and scored after the "
        "gameweek; none is shown here until that record exists. Below: the planning state the planner starts "
        "from for your team, and the record so far."
    )

model = data.model_status()
if model.get("available"):
    st.caption(
        f"Model {model.get('model_version')} · bundle {model.get('bundle_id')} "
        f"(created {ui.fmt_ts(model.get('created_at'))})"
    )


def _bullets(lines: list[str]) -> None:
    st.markdown("\n".join(f"- {ui.md_escape(x)}" for x in lines))


def _state_panel(tid: int, state, players: pd.DataFrame) -> dict | None:
    """Planning state + confirm/edit form. Returns the confirmed (possibly edited) values or None."""
    key = f"ps_confirmed_{tid}"
    st.subheader("Planning state")
    with st.container(border=True):
        st.markdown(
            f"As of {state.as_of['picks']} picks + {state.transfers_since_publication} transfer(s) since · "
            f"built {ui.fmt_ts(state.as_of['built_at'])}"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bank", f"{state.bank / 10:.1f}")
        m2.metric(f"Free transfers for GW{state.upcoming_gw}", str(state.free_transfers))
        m3.metric("Selling value", f"{state.selling_value / 10:.1f}")
        m4.metric("Squad size", len(state.squad))
        if state.warnings:
            st.markdown("**Reconstruction warnings**")
            _bullets(state.warnings)
        idx = players.set_index("id")
        order = players.sort_values(["element_type", "price", "name"], ascending=[True, False, True])
        labels = {int(r["id"]): views.player_label(r) for _, r in order.iterrows()}
        with st.form(f"ps_form_{tid}"):
            st.markdown("**Confirm the state, or correct it for this session only**")
            c1, c2 = st.columns(2)
            bank = c1.number_input(
                "Bank",
                min_value=0.0,
                max_value=100.0,
                value=state.bank / 10,
                step=0.1,
                format="%.1f",
                key=f"ps_bank_{tid}",
            )
            ft = c2.number_input(
                f"Free transfers for GW{state.upcoming_gw}",
                min_value=0,
                max_value=5,
                value=int(state.free_transfers.lo),
                step=1,
                key=f"ps_ft_{tid}",
                help=f"Reconstructed: {state.free_transfers}. A range means several balances fit the "
                "observed hit costs; the plan uses the low end unless you change it.",
            )
            squad = st.multiselect(
                "Squad (15 players)",
                options=list(labels),
                default=[p.element for p in state.squad if p.element in labels],
                format_func=lambda i: labels.get(i, str(i)),
                max_selections=SQUAD.size,
                key=f"ps_squad_{tid}",
            )
            submitted = st.form_submit_button("Confirm and plan" if SHOW_PLANS else "Confirm", type="primary")
        if submitted:
            st.session_state[key] = {"bank": round(bank * 10), "ft": int(ft), "squad": tuple(sorted(squad))}
        if st.session_state.get(key) and st.button("Discard my edits", key=f"ps_reset_{tid}"):
            st.session_state.pop(key, None)
            for k in (f"ps_bank_{tid}", f"ps_ft_{tid}", f"ps_squad_{tid}"):
                st.session_state.pop(k, None)
            st.rerun()
        st.caption("Edits stay in this browser session only and are stamped 'user-edited' in the plan.")
    conf = st.session_state.get(key)
    if conf is None:
        st.info(
            "Confirm (or correct and confirm) the planning state to see the plan."
            if SHOW_PLANS
            else "Confirm (or correct and confirm) the planning state."
        )
        return None
    counts = Counter(int(idx.at[e, "element_type"]) for e in conf["squad"] if e in idx.index)
    if len(conf["squad"]) != SQUAD.size or dict(counts) != SQUAD.per_position:
        st.error(
            f"The corrected squad must have {SQUAD.size} players: 2 GKP, 5 DEF, 5 MID, 3 FWD "
            f"(it has {len(conf['squad'])}: "
            + ", ".join(f"{views.POS_ORDER[t - 1]} {n}" for t, n in sorted(counts.items()))
            + ")."
        )
        return None
    return conf


def _edited_fields(state, conf: dict) -> list[str]:
    owned = tuple(sorted(p.element for p in state.squad))
    out = []
    if conf["bank"] != state.bank:
        out.append(f"bank {conf['bank'] / 10:.1f}")
    if conf["ft"] != state.free_transfers.lo:
        out.append(f"free transfers {conf['ft']}")
    if conf["squad"] != owned:
        out.append(f"squad ({len(set(conf['squad']) - set(owned))} player(s) changed)")
    return out


def _edits(state, conf: dict) -> StateEdits:
    owned = tuple(sorted(p.element for p in state.squad))
    return StateEdits(
        bank=conf["bank"] if conf["bank"] != state.bank else None,
        free_transfers=conf["ft"] if conf["ft"] != state.free_transfers.lo else None,
        squad=conf["squad"] if conf["squad"] != owned else None,
    )


def _plan_section(tid: int, inp, players: pd.DataFrame) -> None:
    bundle_id = str(model.get("bundle_id"))
    key = plan_key(inp)
    plan = data.plan(tid, key, bundle_id, PlanSettings(objective=ONE_GW_NO_TRANSFERS), inp)
    gp = plan.first()
    st.subheader(f"GW{gp.gw} plan: best XI from your squad (no transfers)")
    if plan.status != "optimal":
        st.warning(f"Fallback plan: {plan.reason}.")
    names = inp.names
    c1, c2, c3 = st.columns(3)
    c1.metric("Captain", names.get(gp.captain, str(gp.captain)))
    c2.metric("Vice-captain", names.get(gp.vice, str(gp.vice)))
    c3.metric(
        "Expected points (XI + captain)",
        f"{gp.expected_points:.1f}",
        help=ui.XP_HELP.format(version=model.get("model_version") or ""),
    )
    st.html(ui.pitch_html(views.plan_pitch(gp, players, inp.xp, inp.low_history), "xP"))
    st.markdown("**Why**")
    _bullets(plan.explanation)
    if plan.warnings:
        st.markdown("**Warnings**")
        _bullets(plan.warnings)
    st.caption(
        f"Experimental xP model {model.get('model_version')} · bundle {bundle_id} · solved in "
        f"{plan.solve_seconds:.2f} s. One-gameweek mean xP only; bench order by xP (reserve goalkeeper "
        "first)."
    )

    exp = st.expander(
        "Experimental — multiweek and transfer plans (shadow mode)", on_change="rerun", key="planner_exp_open"
    )
    if not exp.open:
        return
    with exp:
        st.info(
            "NOT VALIDATED. These plans are computed in shadow mode and recorded for scoring only; they are "
            "not advice and are not shown publicly."
        )
        n_gws = len(inp.gws)
        c1, c2, c3 = st.columns(3)
        horizon = (
            c1.slider("Horizon (GWs)", 1, max(1, n_gws), min(DEFAULT_HORIZON, n_gws), key="planner_h")
            if n_gws > 1
            else 1
        )
        decay = c2.radio(
            "Decay",
            DECAY_OPTIONS,
            index=DECAY_OPTIONS.index(DEFAULT_DECAY),
            horizontal=True,
            key="planner_decay",
            help="Weight of GW i in the objective = decay^i; the first action can change with it.",
        )
        k = c3.selectbox("Max transfers per GW", [1, 2, 3], index=1, key="planner_k")
        b = data.plan(
            tid, key, bundle_id, PlanSettings(objective=ONE_GW_WITH_TRANSFERS, max_transfers=k), inp
        )
        st.markdown(f"**(b) Next GW with up to {k} transfer(s), hits allowed**")
        ui.table(views.plan_gw_frame(b, names), _PLAN_COLS)
        _bullets(b.explanation)
        settings = PlanSettings(objective=MULTI_GW, horizon=horizon, decay=decay, max_transfers=k)
        c = data.plan(tid, key, bundle_id, settings, inp)
        st.markdown(f"**(c) {horizon}-GW horizon, decay {decay}, FT banking, 4-point hits**")
        if c.status != "optimal":
            st.caption(f"Fallback: {c.reason}.")
        ui.table(views.plan_gw_frame(c, names), _PLAN_COLS)
        _bullets(c.explanation)
        if st.toggle("Evaluate chip scenarios (slow: one plan per chip)", key="planner_chips"):
            _, scenarios = data.chip_plans(tid, key, bundle_id, settings, inp)
            st.markdown("**(d) Chips: conditional gain inside the evaluated window**")
            ui.table(
                views.chip_frame(scenarios),
                {"gain": st.column_config.NumberColumn("Gain (xP, net of hits)", format="%+.1f")},
            )
            st.caption(
                "Gain = expected points of the plan with the chip minus the plan without it, inside the "
                "evaluated window only (not the season-level opportunity cost). Not a recommendation."
            )


_PLAN_COLS = {
    "gw": st.column_config.NumberColumn("GW", format="%d"),
    "free_transfers": st.column_config.NumberColumn("FT entering", format="%d"),
    "bank_after": st.column_config.NumberColumn("Bank after", format="%.1f"),
    "expected_points": st.column_config.NumberColumn("xP (net of hits)", format="%.1f"),
}


def _shadow_section() -> None:
    n = len(registry.REGISTRY_ENTRY_IDS)
    st.subheader("How the planner is being tested")
    st.markdown(
        "- **Before every deadline** the shared-data bundle built 30-90 minutes before it is kept as the "
        "scored snapshot: the xP table as issued and, for a fixed roster of "
        f"{n} public teams, the planner's proposals from their planning state at that moment: "
        "(a) best XI, captain and vice without transfers, (b) the next gameweek with up to 2 transfers "
        "(4-point hits allowed), (c) a 5-gameweek transfer path with free-transfer banking.\n"
        "- **After the gameweek** they are scored against the real points: xP error and ranking against a "
        "frozen baseline (form per 90 x availability); per team, the captain's points against the best "
        "captain in the squad and the manager's own captain, and the planned line-up's points (automatic "
        "substitutions, armband, net of hits for (b) and (c)) against the baseline and the manager.\n"
        "- **Every week counts:** a deadline without a snapshot is recorded as missing, never skipped. The "
        "proposals stay hidden until this record shows whether they help."
    )
    if PLANNER_ERROR:
        st.caption(f"The solver is not available on this host ({PLANNER_ERROR}); recording runs elsewhere.")


def _registry_section() -> None:
    st.subheader("Prospective registry")
    st.caption(
        "The scored snapshot is the bundle published 30-90 minutes before each deadline: its xP and the "
        f"plans (a), (b) and (c) of a fixed roster of {len(registry.REGISTRY_ENTRY_IDS)} teams are scored "
        "after the gameweek against a frozen baseline (form per 90 x availability). Missing weeks count."
    )
    scores, reason = data.registry_scores()
    latest = registry.latest_scores(scores)
    if latest is None:
        why = f" ({reason})" if reason and reason != "no scores yet" else ""
        st.caption(
            f"No scores yet{why}; the first scored deadline is GW{registry.REGISTRY_START_GW}, scored once "
            "that gameweek is finished."
        )
        return
    xp_row = latest["xp"]
    if latest["status"] != "ok":
        st.markdown(f"Last scored GW{latest['gw']}: **{latest['status']}** - {xp_row.get('reason') or ''}")
        return
    st.markdown(f"Last scored: **GW{latest['gw']}** (bundle {xp_row.get('bundle_id')})")
    st.caption(
        f"Scored snapshot: model {xp_row.get('model_version') or '?'} · "
        f"bundle {xp_row.get('bundle_id') or '?'}."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("xP RMSE (model)", f"{xp_row.get('rmse_model', float('nan')):.2f}")
    m2.metric("RMSE (baseline)", f"{xp_row.get('rmse_baseline', float('nan')):.2f}")
    m3.metric("Spearman, played (model)", f"{xp_row.get('spearman_played_model', float('nan')):.2f}")
    m4.metric("Spearman, played (baseline)", f"{xp_row.get('spearman_played_baseline', float('nan')):.2f}")
    e = latest["entries"]
    e = e[e["status"] == "ok"]
    if len(e):

        def mean(col: str) -> float | None:
            v = pd.to_numeric(e[col], errors="coerce") if col in e else pd.Series(dtype=float)
            return None if v.isna().all() else round(float(v.mean()), 1)

        rows = {
            "Captain: planner (a)": mean("captain_model_pts"),
            "Captain: baseline": mean("captain_baseline_pts"),
            "Captain: best in squad": mean("captain_best_pts"),
            "Captain: manager's own": mean("captain_manager_pts"),
            "Line-up: planner (a), no transfers": mean("xi_model_pts"),
            "Line-up: planner (b), with transfers, net of hits": mean("xi_model_b_pts"),
            "Line-up: planner (c), 5-GW path, first GW net of hits": mean("xi_model_c_pts"),
            "Line-up: baseline (a)": mean("xi_baseline_pts"),
            "Manager's GW points": mean("manager_pts"),
        }
        st.dataframe(pd.DataFrame({"mean points": rows}))
        st.caption(
            f"Means over {len(e)} roster teams; line-up points include automatic substitutions and the "
            "armband. Blank = not recorded for that week."
        )


def _plans(tid: int, state, players: pd.DataFrame, boot, conf: dict) -> None:
    """The plan UI (config.SHOW_SHADOW_PLANS only)."""
    if PLANNER_ERROR:
        st.info(f"The planner is not available on this host ({PLANNER_ERROR}).")
        return
    if not model.get("available"):
        st.info(f"The planner needs the xP model bundle, which is unavailable: {model.get('reason')}.")
        return
    with ui.guard("Planner"):
        xp = data.xp_table()
        inp = from_planning_state(state, players, xp, bootstrap=boot, edits=_edits(state, conf))
        if not inp.gws:
            st.info(f"No xP for GW{state.upcoming_gw} or later in this bundle.")
        else:
            _plan_section(tid, inp, players)


tid = ui.require_team()
if tid is not None:
    players = boot = state = None
    with ui.guard("FPL season data"):
        boot = data.bootstrap()
        players = data.players()
    with ui.guard("Planning state"):
        state = data.planning_state(tid)
    if state is not None and players is not None:
        conf = _state_panel(tid, state, players)
        if conf is not None:
            if SHOW_PLANS:
                _plans(tid, state, players, boot, conf)
            else:
                edited = _edited_fields(state, conf)
                st.success(
                    f"Planning state for GW{state.upcoming_gw} confirmed for this session"
                    + (f" (user-edited: {', '.join(edited)})." if edited else ".")
                )
if not SHOW_PLANS:
    _shadow_section()
with ui.guard("Prospective registry"):
    _registry_section()
