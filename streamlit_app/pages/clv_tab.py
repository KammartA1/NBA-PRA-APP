"""
streamlit_app/pages/clv_tab.py
================================
CLV System dashboard — the TRUTH ENGINE.

Shows:
  - Average CLV, beat-close %, CLV trend over time
  - CLV by segment (market type, direction, day, hour)
  - Integrity report with score
  - Line movement charts for any selected player/market
  - Bet-time snapshot details
  - Closing line capture status

Without trustworthy CLV, nothing else matters.
"""

from __future__ import annotations

import json
import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

log = logging.getLogger(__name__)


def _get_calculator():
    from services.clv_system.clv_calculator import CLVCalculator
    return CLVCalculator(sport="NBA")


def _get_integrity():
    from services.clv_system.integrity import CLVIntegrity
    return CLVIntegrity(sport="NBA")


def _get_line_storage():
    from services.clv_system.line_storage import LineStorage
    return LineStorage(sport="NBA")


def _get_snapshot_service():
    from services.clv_system.snapshot import BetTimeSnapshotService
    return BetTimeSnapshotService(sport="NBA")


def _get_closing_capture():
    from services.clv_system.closing_capture import ClosingLineCapture
    return ClosingLineCapture(sport="NBA")


def render():
    """Render the CLV System tab."""
    st.markdown("""
    <h2 style='font-family: Chakra Petch, monospace; color: #00FFB2; margin-bottom: 0;
    font-size: 1.3rem; letter-spacing: 0.1em;'>
    CLV SYSTEM — TRUTH ENGINE
    </h2>
    <p style='font-family: Fira Code, monospace; color: #4A607A; font-size: 0.65rem;
    margin-top: 0.2rem;'>
    If CLV cannot be trusted, the entire system is invalid
    </p>
    """, unsafe_allow_html=True)

    tab_overview, tab_segments, tab_integrity, tab_lines, tab_details = st.tabs([
        "OVERVIEW", "SEGMENTS", "INTEGRITY", "LINE MOVEMENTS", "DETAILS",
    ])

    with tab_overview:
        _render_overview()

    with tab_segments:
        _render_segments()

    with tab_integrity:
        _render_integrity()

    with tab_lines:
        _render_line_movements()

    with tab_details:
        _render_details()


# ── OVERVIEW ─────────────────────────────────────────────────────────────

def _render_overview():
    """CLV dashboard: avg CLV, beat-close %, trend chart."""
    try:
        calc = _get_calculator()
        summary = calc.compute_clv_summary()
    except Exception as exc:
        st.info(f"CLV data not yet available: {exc}")
        _render_empty_state()
        return

    # Key metrics row
    clv_100 = summary.get("clv_100", {})
    n_bets = clv_100.get("n_bets", 0)

    if n_bets < 10:
        _render_empty_state()
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_clv = clv_100.get("avg_clv_cents", 0)
        color = "#00FFB2" if avg_clv > 0 else "#FF3358"
        st.metric(
            "AVG CLV (CENTS)",
            f"{avg_clv:+.2f}",
            delta=f"{clv_100.get('trend', 'N/A')}",
        )
    with col2:
        beat_pct = clv_100.get("beat_close_pct", 0) * 100
        st.metric("BEAT CLOSE %", f"{beat_pct:.1f}%")
    with col3:
        st.metric("SAMPLE SIZE", f"{n_bets}")
    with col4:
        p_val = clv_100.get("p_value", 1.0)
        sig = "YES" if p_val < 0.05 else "NO"
        st.metric("SIGNIFICANT", sig, delta=f"p={p_val:.4f}")

    st.markdown("---")

    # Multi-window summary table
    st.markdown("##### CLV Across Windows")
    window_data = []
    for key in ["clv_50", "clv_100", "clv_250", "clv_500"]:
        w = summary.get(key, {})
        if w.get("n_bets", 0) >= 10:
            window_data.append({
                "Window": w.get("window", "?"),
                "N Bets": w.get("n_bets", 0),
                "Avg CLV (cents)": f"{w.get('avg_clv_cents', 0):+.3f}",
                "Beat Close %": f"{w.get('beat_close_pct', 0) * 100:.1f}%",
                "CLV Positive": w.get("clv_positive", False),
                "T-Stat": f"{w.get('t_stat', 0):.3f}",
                "P-Value": f"{w.get('p_value', 1):.4f}",
                "Trend": w.get("trend", "N/A"),
            })

    if window_data:
        df = pd.DataFrame(window_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # CLV trend chart
    st.markdown("##### CLV Trend Over Time")
    try:
        trend_data = calc.compute_clv_trend(window=500, bucket_size=25)
        if trend_data:
            df_trend = pd.DataFrame(trend_data)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df_trend))),
                y=df_trend["avg_clv_cents"],
                mode="lines+markers",
                name="Avg CLV (cents)",
                line=dict(color="#00FFB2", width=2),
                marker=dict(size=4),
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#FF3358", opacity=0.5)
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Bucket (25 bets each)",
                yaxis_title="CLV (cents)",
                height=350,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cumulative CLV
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(range(len(df_trend))),
                y=df_trend["cumulative_clv"],
                mode="lines",
                name="Cumulative CLV",
                fill="tozeroy",
                line=dict(color="#00FFB2", width=2),
            ))
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Bucket",
                yaxis_title="Cumulative CLV",
                height=300,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough data for trend chart (need 25+ CLV records)")
    except Exception as exc:
        st.warning(f"Could not generate trend chart: {exc}")


# ── SEGMENTS ─────────────────────────────────────────────────────────────

def _render_segments():
    """CLV broken down by market type, direction, day, hour."""
    try:
        calc = _get_calculator()
    except Exception:
        st.info("CLV system not initialized yet")
        return

    segment_options = {
        "Market Type": "stat_type",
        "Direction": "direction",
        "Bet Type": "bet_type",
        "Day of Week": "day_of_week",
        "Hour of Day": "hour_of_day",
    }

    selected = st.selectbox("Segment By", list(segment_options.keys()))
    segment_key = segment_options[selected]

    try:
        data = calc.compute_clv_by_segment(segment_key=segment_key, window=500)
    except Exception as exc:
        st.info(f"Not enough data for segmentation: {exc}")
        return

    if not data:
        st.info("No CLV data available for segmentation")
        return

    # Build table
    rows = []
    for seg_val, metrics in sorted(data.items(), key=lambda x: x[1].get("avg_clv_cents", 0), reverse=True):
        rows.append({
            selected: seg_val,
            "N Bets": metrics["n_bets"],
            "Avg CLV (cents)": f"{metrics['avg_clv_cents']:+.3f}",
            "Beat Close %": f"{metrics['beat_close_pct'] * 100:.1f}%",
            "CLV Positive": metrics["clv_positive"],
            "Std Dev": f"{metrics['std_clv_cents']:.3f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart
    fig = go.Figure()
    seg_vals = [r[selected] for r in rows]
    clv_vals = [data[s]["avg_clv_cents"] for s in [r[selected] for r in rows]]
    colors = ["#00FFB2" if v > 0 else "#FF3358" for v in clv_vals]

    fig.add_trace(go.Bar(
        x=seg_vals,
        y=clv_vals,
        marker_color=colors,
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=selected,
        yaxis_title="Avg CLV (cents)",
        height=350,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── INTEGRITY ────────────────────────────────────────────────────────────

def _render_integrity():
    """Data integrity report and score."""
    try:
        integrity = _get_integrity()
    except Exception:
        st.info("CLV integrity system not initialized yet")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        # Generate report button
        if st.button("Generate Integrity Report", type="primary"):
            with st.spinner("Running integrity checks..."):
                report_text = integrity.generate_integrity_report()
            st.success("Report generated")

        # Show latest score
        latest = integrity.get_latest_report()
        if latest:
            score = latest.get("integrity_score", 0)
            if score >= 90:
                color = "#00FFB2"
                label = "EXCELLENT"
            elif score >= 80:
                color = "#FFB800"
                label = "ACCEPTABLE"
            else:
                color = "#FF3358"
                label = "UNTRUSTWORTHY"

            st.markdown(f"""
            <div style='text-align: center; padding: 20px; border: 2px solid {color};
            border-radius: 12px; margin: 10px 0;'>
                <div style='font-size: 2.5rem; font-weight: bold; color: {color};
                font-family: Chakra Petch, monospace;'>{score:.0f}/100</div>
                <div style='font-size: 0.8rem; color: {color};
                font-family: Fira Code, monospace;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            | Metric | Count |
            |--------|-------|
            | Total Bets | {latest.get('total_bets', 0)} |
            | Missing Opening | {latest.get('missing_opening_lines', 0)} |
            | Missing Closing | {latest.get('missing_closing_lines', 0)} |
            | Missing Snapshots | {latest.get('missing_bet_snapshots', 0)} |
            | Data Errors | {latest.get('suspected_data_errors', 0)} |
            """)
        else:
            st.info("No integrity report generated yet. Click the button above.")

    with col2:
        if latest and latest.get("report_text"):
            st.code(latest["report_text"], language="text")

    # Report history
    history = integrity.get_report_history(limit=20)
    if history:
        st.markdown("##### Integrity Score History")
        df = pd.DataFrame([{
            "Date": r.get("generated_at", ""),
            "Score": r.get("integrity_score", 0),
            "Bets": r.get("total_bets", 0),
        } for r in history])

        if len(df) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Date"],
                y=df["Score"],
                mode="lines+markers",
                line=dict(color="#00FFB2", width=2),
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="#FF3358", opacity=0.7,
                          annotation_text="Minimum Threshold")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Integrity Score",
                yaxis_range=[0, 105],
                height=300,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── LINE MOVEMENTS ───────────────────────────────────────────────────────

def _render_line_movements():
    """Line movement charts for any selected player/event."""
    try:
        storage = _get_line_storage()
    except Exception:
        st.info("Line storage not initialized yet")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        player = st.text_input("Player Name", placeholder="e.g. LeBron James")
    with col2:
        market = st.text_input("Market Type", placeholder="e.g. Points")
    with col3:
        hours = st.slider("Hours Back", min_value=1, max_value=168, value=48)

    if not player or not market:
        # Show table stats instead
        stats = storage.get_table_stats()
        st.markdown("##### Line Movement Database Stats")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Observations", f"{stats.get('total_rows', 0):,}")
        with col_b:
            st.metric("Books Tracked", stats.get("n_books", 0))
        with col_c:
            st.metric("Players Tracked", stats.get("n_players", 0))

        if stats.get("oldest_record"):
            st.caption(f"Data range: {stats['oldest_record']} to {stats.get('newest_record', 'now')}")
        if stats.get("archive_rows", 0) > 0:
            st.caption(f"Archived rows: {stats['archive_rows']:,}")
        return

    # Fetch line history
    history = storage.get_line_history(player, market, hours=hours)
    if not history:
        st.info(f"No line data found for {player} / {market}")
        return

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Movement stats
    stats = storage.compute_movement_stats(player, market, hours=hours)
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Opening", f"{stats.get('opening_line', 'N/A')}")
    with col_b:
        st.metric("Current", f"{stats.get('current_line', 'N/A')}")
    with col_c:
        mv = stats.get("total_movement", 0)
        st.metric("Movement", f"{mv:+.2f}")
    with col_d:
        st.metric("Direction", stats.get("direction", "unknown").upper())

    # Chart by book
    fig = go.Figure()
    for book in df["book"].unique():
        book_df = df[df["book"] == book]
        fig.add_trace(go.Scatter(
            x=book_df["timestamp"],
            y=book_df["line"],
            mode="lines+markers",
            name=book,
            marker=dict(size=3),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Time",
        yaxis_title="Line",
        height=400,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw data table
    with st.expander("Raw Line Data"):
        st.dataframe(df, use_container_width=True, hide_index=True)


# ── DETAILS ──────────────────────────────────────────────────────────────

def _render_details():
    """Detailed CLV records, bet-time snapshots, closing lines."""
    try:
        calc = _get_calculator()
    except Exception:
        st.info("CLV system not initialized yet")
        return

    st.markdown("##### Recent CLV Records")
    records = calc.get_all_clv_records(limit=100)
    if records:
        df = pd.DataFrame(records)
        # Color-code CLV
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No CLV records yet. Bets need to be settled with closing lines.")

    # Recent snapshots
    st.markdown("##### Recent Bet-Time Snapshots")
    try:
        snap_svc = _get_snapshot_service()
        snapshots = snap_svc.get_recent_snapshots(limit=20)
        if snapshots:
            snap_df = pd.DataFrame(snapshots)
            cols_to_show = [c for c in snap_df.columns if c != "lines_json"]
            st.dataframe(snap_df[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info("No bet-time snapshots captured yet")
    except Exception as exc:
        st.info(f"Snapshot data unavailable: {exc}")

    # Closing line stats
    st.markdown("##### Closing Line Capture Stats")
    try:
        closing = _get_closing_capture()
        cl_count = closing.closing_line_count()
        st.metric("Closing Lines Captured", cl_count)
    except Exception as exc:
        st.info(f"Closing line data unavailable: {exc}")


# ── Empty state ──────────────────────────────────────────────────────────

def _render_empty_state():
    """Shown when there's not enough CLV data."""
    st.markdown("""
    <div style='text-align: center; padding: 40px; border: 1px dashed #1E2D3D;
    border-radius: 12px; margin: 20px 0;'>
        <div style='font-size: 1.2rem; color: #4A607A; font-family: Chakra Petch, monospace;
        margin-bottom: 10px;'>CLV SYSTEM INITIALIZING</div>
        <div style='font-size: 0.75rem; color: #2A3A4A; font-family: Fira Code, monospace;'>
            The CLV system needs at least 10 settled bets with closing lines to produce metrics.<br><br>
            <b>How it works:</b><br>
            1. Odds are captured every 5 minutes from all sources<br>
            2. When a bet signal fires, a market snapshot is taken<br>
            3. At tip-off, closing lines are captured automatically<br>
            4. After settlement, CLV is computed for each bet<br><br>
            This is fully autonomous — zero manual input required.
        </div>
    </div>
    """, unsafe_allow_html=True)
