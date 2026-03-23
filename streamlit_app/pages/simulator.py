"""
streamlit_app/pages/simulator.py
================================
Full possession-level NBA game simulator UI.  Allows the user to select
two teams, configure rosters, run Monte Carlo simulations, and inspect
per-player PRA distributions with P(over line) calculations and edge
detection against sportsbook lines.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from simulation import (
    CoachArchetype,
    GameEngine,
    PlayerDistribution,
    PlayerProfile,
    SimulationConfig,
    SimulationOutput,
    SimulationValidator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_roster(prefix: str, team_name: str) -> List[Dict]:
    """Build a default 10-man roster as a list of dicts for the editor."""
    positions = ["PG", "SG", "SF", "PF", "C", "PG", "SG", "SF", "PF", "C"]
    rows = []
    for i, pos in enumerate(positions):
        is_starter = i < 5
        rows.append({
            "Name": f"{team_name} Player {i+1}",
            "Position": pos,
            "Starter": is_starter,
            "Age": 25 + (i % 6),
            "Usage": round(0.22 - i * 0.012 if is_starter else 0.12, 3),
            "2PT%": round(0.52 + (0.02 if is_starter else -0.01), 3),
            "3PT%": round(0.36 + (0.02 if is_starter else -0.01), 3),
            "FT%": round(0.78, 3),
            "AST Rate": round(0.20 if pos == "PG" else 0.10, 3),
            "REB Rate": round(0.15 if pos in ("C", "PF") else 0.07, 3),
            "STL Rate": round(0.018 if pos in ("PG", "SG") else 0.012, 3),
            "BLK Rate": round(0.025 if pos == "C" else 0.008, 3),
            "Height (in)": 72 + (i % 5) * 2 + (3 if pos == "C" else 0),
        })
    return rows


def _roster_df_to_profiles(df: pd.DataFrame, prefix: str) -> List[PlayerProfile]:
    """Convert an edited roster DataFrame into PlayerProfile objects."""
    profiles = []
    for i, row in df.iterrows():
        profiles.append(PlayerProfile(
            name=str(row.get("Name", f"Player_{i}")),
            player_id=f"{prefix}_{i}",
            position=str(row.get("Position", "SF")),
            age=int(row.get("Age", 27)),
            height_inches=int(row.get("Height (in)", 78)),
            rest_days=1,
            two_pt_pct=float(row.get("2PT%", 0.52)),
            three_pt_pct=float(row.get("3PT%", 0.36)),
            ft_pct=float(row.get("FT%", 0.78)),
            usage_rate=float(row.get("Usage", 0.15)),
            assist_rate=float(row.get("AST Rate", 0.10)),
            rebound_rate=float(row.get("REB Rate", 0.10)),
            steal_rate=float(row.get("STL Rate", 0.015)),
            block_rate=float(row.get("BLK Rate", 0.010)),
            turnover_rate=0.12,
            foul_rate=0.03,
            foul_draw_rate=0.03,
            is_starter=bool(row.get("Starter", i < 5)),
            rotation_order=int(i),
        ))
    return profiles


def _build_histogram(
    dist: PlayerDistribution,
    sportsbook_line: Optional[float] = None,
) -> go.Figure:
    """Build a plotly histogram for a single stat distribution."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=dist.values,
        nbinsx=40,
        marker_color="#4A90D9",
        opacity=0.75,
        name=dist.stat_name,
    ))
    # Mean line
    fig.add_vline(x=dist.mean, line_dash="dash", line_color="#E8A838",
                  annotation_text=f"Mean: {dist.mean:.1f}")
    # Sportsbook line
    if sportsbook_line is not None:
        p_over = dist.prob_over(sportsbook_line)
        fig.add_vline(x=sportsbook_line, line_dash="dot", line_color="#E85555",
                      annotation_text=f"Line: {sportsbook_line} | P(Over)={p_over:.1%}")
    fig.update_layout(
        title=f"{dist.player_name} — {dist.stat_name.upper()}",
        xaxis_title=dist.stat_name,
        yaxis_title="Frequency",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _build_correlation_heatmap(
    output: SimulationOutput,
    player_id: str,
) -> Optional[go.Figure]:
    """Build a correlation heatmap for a single player's stats."""
    dists = output.distributions.get(player_id, {})
    stats_to_show = ["points", "rebounds", "assists", "steals", "blocks", "turnovers"]
    available = [s for s in stats_to_show if s in dists]
    if len(available) < 2:
        return None

    data = {s: dists[s].values for s in available}
    df = pd.DataFrame(data)
    corr = df.corr()

    fig = px.imshow(
        corr,
        x=available,
        y=available,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title=f"Stat Correlations — {dists[available[0]].player_name}",
    )
    fig.update_layout(height=400, margin=dict(l=40, r=20, t=50, b=40))
    return fig


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render() -> None:
    st.markdown(
        "<div style='font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;"
        "text-transform:uppercase;margin-bottom:1rem;'>"
        "POSSESSION-LEVEL GAME SIMULATOR</div>",
        unsafe_allow_html=True,
    )

    # ---- Sidebar-style controls ----
    st.subheader("Game Setup")
    col1, col2, col3 = st.columns(3)
    with col1:
        home_name = st.text_input("Home Team", value="Home", key="sim_home")
    with col2:
        away_name = st.text_input("Away Team", value="Away", key="sim_away")
    with col3:
        n_sims = st.number_input(
            "Simulations", min_value=100, max_value=100_000,
            value=1000, step=500, key="sim_n",
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        home_pace = st.slider("Home Pace", 88.0, 115.0, 100.0, 0.5, key="sim_hpace")
    with col5:
        away_pace = st.slider("Away Pace", 88.0, 115.0, 100.0, 0.5, key="sim_apace")
    with col6:
        spread = st.number_input("Pre-game Spread (neg = Home fav)", -20.0, 20.0, 0.0, 0.5, key="sim_spread")

    arch_options = ["balanced", "starter_heavy", "deep_bench"]
    col7, col8 = st.columns(2)
    with col7:
        home_arch = st.selectbox("Home Coach Style", arch_options, key="sim_harch")
    with col8:
        away_arch = st.selectbox("Away Coach Style", arch_options, key="sim_aarch")

    arch_map = {
        "balanced": CoachArchetype.BALANCED,
        "starter_heavy": CoachArchetype.STARTER_HEAVY,
        "deep_bench": CoachArchetype.DEEP_BENCH,
    }

    # ---- Roster editors ----
    st.subheader("Rosters")
    tab_home, tab_away = st.tabs([f"{home_name} Roster", f"{away_name} Roster"])

    with tab_home:
        home_default = _default_roster("h", home_name)
        home_df = st.data_editor(
            pd.DataFrame(home_default),
            num_rows="dynamic",
            key="sim_home_roster",
            use_container_width=True,
        )

    with tab_away:
        away_default = _default_roster("a", away_name)
        away_df = st.data_editor(
            pd.DataFrame(away_default),
            num_rows="dynamic",
            key="sim_away_roster",
            use_container_width=True,
        )

    # ---- Run simulation ----
    st.divider()
    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        home_profiles = _roster_df_to_profiles(home_df, "h")
        away_profiles = _roster_df_to_profiles(away_df, "a")

        engine = GameEngine(
            config=SimulationConfig(default_simulations=int(n_sims)),
            home_profiles=home_profiles,
            away_profiles=away_profiles,
            home_name=home_name,
            away_name=away_name,
            home_pace=home_pace,
            away_pace=away_pace,
            pre_game_spread=spread,
            home_archetype=arch_map.get(home_arch, CoachArchetype.BALANCED),
            away_archetype=arch_map.get(away_arch, CoachArchetype.BALANCED),
        )

        with st.spinner(f"Simulating {n_sims} games..."):
            output = engine.run_simulation(n=int(n_sims))

        st.session_state["sim_output"] = output
        st.success(f"Completed {output.n_simulations} simulations.")

    # ---- Display results ----
    output: Optional[SimulationOutput] = st.session_state.get("sim_output")
    if output is None:
        st.info("Configure teams and click 'Run Simulation' to see results.")
        return

    # --- Game-level summary ---
    st.subheader("Game Results Summary")
    results_df = pd.DataFrame(output.game_results)
    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    with gcol1:
        st.metric("Avg Home Score", f"{results_df['home_score'].mean():.1f}")
    with gcol2:
        st.metric("Avg Away Score", f"{results_df['away_score'].mean():.1f}")
    with gcol3:
        home_win_pct = (results_df["margin"] > 0).mean()
        st.metric("Home Win %", f"{home_win_pct:.1%}")
    with gcol4:
        blowout_pct = (results_df["margin"].abs() >= 20).mean()
        st.metric("Blowout Rate", f"{blowout_pct:.1%}")

    # Score distribution
    fig_scores = go.Figure()
    fig_scores.add_trace(go.Histogram(x=results_df["home_score"], name="Home", opacity=0.6, marker_color="#4A90D9"))
    fig_scores.add_trace(go.Histogram(x=results_df["away_score"], name="Away", opacity=0.6, marker_color="#E85555"))
    fig_scores.update_layout(
        title="Team Score Distributions", barmode="overlay",
        template="plotly_white", height=300,
    )
    st.plotly_chart(fig_scores, use_container_width=True)

    # --- Player distributions ---
    st.subheader("Player Stat Distributions")

    # Player selector
    all_player_ids = list(output.distributions.keys())
    all_player_names = {
        pid: output.distributions[pid].get("points", next(iter(output.distributions[pid].values()))).player_name
        for pid in all_player_ids
        if output.distributions[pid]
    }
    name_to_id = {v: k for k, v in all_player_names.items()}

    selected_name = st.selectbox(
        "Select Player",
        list(name_to_id.keys()),
        key="sim_player_select",
    )
    if not selected_name:
        return

    selected_pid = name_to_id[selected_name]
    player_dists = output.distributions.get(selected_pid, {})

    # Stat selector
    stat_options = ["points", "rebounds", "assists", "pra", "pr", "pa", "ra",
                    "steals", "blocks", "turnovers", "minutes"]
    stat_cols = st.columns(2)
    with stat_cols[0]:
        selected_stat = st.selectbox("Stat", stat_options, key="sim_stat_select")
    with stat_cols[1]:
        sbook_line = st.number_input(
            "Sportsbook Line (optional)", value=0.0, step=0.5, key="sim_line",
        )

    dist = player_dists.get(selected_stat)
    if dist is None:
        st.warning(f"No data for {selected_stat}")
        return

    # P(over line) display
    if sbook_line > 0:
        p_over = dist.prob_over(sbook_line)
        p_under = dist.prob_under(sbook_line)
        implied_edge = p_over - 0.5  # simple vs 50/50
        edge_col1, edge_col2, edge_col3 = st.columns(3)
        with edge_col1:
            st.metric("P(Over)", f"{p_over:.1%}")
        with edge_col2:
            st.metric("P(Under)", f"{p_under:.1%}")
        with edge_col3:
            delta_color = "normal" if implied_edge > 0 else "inverse"
            st.metric("Edge vs -110", f"{implied_edge:+.1%}", delta_color=delta_color)

    # Histogram
    line_val = sbook_line if sbook_line > 0 else None
    fig_hist = _build_histogram(dist, sportsbook_line=line_val)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Percentile table
    st.markdown("**Percentile Breakdown**")
    pct_data = {
        "Percentile": ["5th", "10th", "25th", "Median", "75th", "90th", "95th"],
        "Value": [dist.p5, dist.p10, dist.p25, dist.median, dist.p75, dist.p90, dist.p95],
    }
    st.dataframe(pd.DataFrame(pct_data), use_container_width=True, hide_index=True)

    # Full summary stats
    with st.expander("Full Distribution Stats"):
        summary = dist.to_dict()
        st.json(summary)

    # Correlation heatmap
    st.subheader("Stat Correlations")
    corr_fig = _build_correlation_heatmap(output, selected_pid)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Not enough stat categories for a correlation heatmap.")

    # --- All players summary table ---
    st.subheader("All Players Summary")
    summary_rows = []
    for pid, stat_map in output.distributions.items():
        row = {"Player": stat_map.get("points", next(iter(stat_map.values()))).player_name}
        for s in ["points", "rebounds", "assists", "pra", "minutes"]:
            d = stat_map.get(s)
            if d:
                row[f"{s}_mean"] = round(d.mean, 1)
                row[f"{s}_std"] = round(d.std, 1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # --- Validation ---
    with st.expander("Simulation Validation"):
        validator = SimulationValidator(output)
        report = validator.run_all()
        if report.overall_pass:
            st.success(report.summary())
        else:
            st.warning(report.summary())
