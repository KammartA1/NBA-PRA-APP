"""
streamlit_app/pages/data_quality_tab.py
========================================
Data Quality Audit dashboard.

Continuous validation of all data flowing through the system.
If any critical finding exists, edge may be fabricated.
"""

from __future__ import annotations

import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

log = logging.getLogger(__name__)


def _get_report():
    from services.data_audit.report import DataQualityReport
    return DataQualityReport(sport="NBA")


def _score_color(score: float) -> str:
    if score >= 90:
        return "#00FFB2"
    elif score >= 80:
        return "#FFB800"
    else:
        return "#FF3358"


def _score_emoji(score: float) -> str:
    if score >= 90:
        return "EXCELLENT"
    elif score >= 80:
        return "ACCEPTABLE"
    else:
        return "CRITICAL"


def _render_score_gauge(label: str, score: float, col):
    """Render a score gauge in the given Streamlit column."""
    color = _score_color(score)
    col.markdown(
        f"""<div style='text-align:center;padding:0.5rem;
        border:1px solid {color}33;border-radius:8px;background:{color}08;'>
        <div style='font-size:0.65rem;color:#4A607A;text-transform:uppercase;
        letter-spacing:0.1em;'>{label}</div>
        <div style='font-size:2rem;font-weight:700;color:{color};
        font-family:Chakra Petch,monospace;'>{score:.0f}</div>
        <div style='font-size:0.55rem;color:{color};'>{_score_emoji(score)}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render():
    """Render the Data Quality Audit tab."""
    st.markdown(
        """<h2 style='font-family:Chakra Petch,monospace;font-size:1.1rem;
        color:#00FFB2;margin-bottom:0.2rem;'>
        DATA QUALITY AUDIT</h2>
        <p style='font-size:0.65rem;color:#4A607A;margin-bottom:1rem;'>
        Continuous validation of all data flowing through the system.
        If data quality is poor, your edge may be fabricated.</p>""",
        unsafe_allow_html=True,
    )

    # Sub-tabs
    audit_tabs = st.tabs([
        "OVERVIEW", "TIMESTAMPS", "ODDS", "CLOSING LINES",
        "COMPLETENESS", "HISTORY", "RAW REPORT",
    ])

    # ── Generate or load report ───────────────────────────────────
    report_service = _get_report()

    with audit_tabs[0]:
        _render_overview(report_service)

    with audit_tabs[1]:
        _render_timestamp_detail(report_service)

    with audit_tabs[2]:
        _render_odds_detail(report_service)

    with audit_tabs[3]:
        _render_closing_detail(report_service)

    with audit_tabs[4]:
        _render_completeness_detail(report_service)

    with audit_tabs[5]:
        _render_history(report_service)

    with audit_tabs[6]:
        _render_raw_report(report_service)


def _render_overview(report_service):
    """Render the overview with composite score and dimension gauges."""
    col_btn, col_status = st.columns([1, 3])

    with col_btn:
        run_audit = st.button("Run Full Audit", type="primary", use_container_width=True)

    if run_audit:
        with st.spinner("Running data quality audit across all dimensions..."):
            data = report_service.generate_dict()
            st.session_state["_dq_audit_data"] = data
    else:
        data = st.session_state.get("_dq_audit_data")

    if not data:
        # Try loading last stored report
        latest = report_service.get_latest_report()
        if latest:
            with col_status:
                st.info(
                    f"Showing last audit from {latest.get('generated_at', 'unknown')} "
                    f"(score: {latest.get('integrity_score', '?')}). "
                    f"Click 'Run Full Audit' for fresh results."
                )
            # Show basic info from stored report
            score = latest.get("integrity_score", 0)
            color = _score_color(score)
            st.markdown(
                f"""<div style='text-align:center;padding:1.5rem;
                border:2px solid {color}55;border-radius:12px;
                background:{color}08;margin:1rem 0;'>
                <div style='font-size:0.7rem;color:#4A607A;'>COMPOSITE DATA QUALITY SCORE</div>
                <div style='font-size:3.5rem;font-weight:700;color:{color};
                font-family:Chakra Petch,monospace;'>{score:.0f}</div>
                <div style='font-size:0.7rem;color:{color};'>{_score_emoji(score)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if latest.get("report_text"):
                st.code(latest["report_text"], language=None)
        else:
            st.warning("No audit data available. Click 'Run Full Audit' to generate.")
        return

    # ── Composite score ───────────────────────────────────────────
    composite = data.get("composite_score", 0)
    color = _score_color(composite)
    is_trustworthy = data.get("is_trustworthy", False)

    st.markdown(
        f"""<div style='text-align:center;padding:1.5rem;
        border:2px solid {color}55;border-radius:12px;
        background:{color}08;margin:1rem 0;'>
        <div style='font-size:0.7rem;color:#4A607A;'>COMPOSITE DATA QUALITY SCORE</div>
        <div style='font-size:3.5rem;font-weight:700;color:{color};
        font-family:Chakra Petch,monospace;'>{composite:.0f}</div>
        <div style='font-size:0.7rem;color:{color};'>{_score_emoji(composite)}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Dimension scores ──────────────────────────────────────────
    cols = st.columns(4)
    ts_data = data.get("timestamp", {})
    odds_data = data.get("odds", {})
    closing_data = data.get("closing_line", {})
    comp_data = data.get("completeness", {})

    _render_score_gauge("Timestamp Accuracy", ts_data.get("score", 0), cols[0])
    _render_score_gauge("Odds Availability", odds_data.get("score", 0), cols[1])
    _render_score_gauge("Closing Line Accuracy", closing_data.get("score", 0), cols[2])
    _render_score_gauge("Data Completeness", comp_data.get("score", 0), cols[3])

    # ── Critical findings ─────────────────────────────────────────
    critical = data.get("critical_findings", [])
    if critical:
        st.markdown("---")
        st.markdown(
            """<div style='background:#FF335815;border:1px solid #FF3358;
            border-radius:8px;padding:1rem;margin:0.5rem 0;'>
            <div style='font-size:0.75rem;font-weight:700;color:#FF3358;
            font-family:Chakra Petch,monospace;margin-bottom:0.5rem;'>
            CRITICAL FINDINGS — YOUR EDGE MAY BE FABRICATED</div>""",
            unsafe_allow_html=True,
        )
        for finding in critical:
            st.markdown(
                f"<div style='font-size:0.65rem;color:#FF3358;padding:0.2rem 0;'>"
                f"!! {finding}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    elif is_trustworthy:
        st.success("No critical findings. Data quality is trustworthy.")

    # ── Quick stats ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#4A607A;font-weight:600;'>QUICK STATS</div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Bets", comp_data.get("total_bets", 0))
    m2.metric("Complete Bets", comp_data.get("complete_bets", 0))
    m3.metric("Stale Lines", odds_data.get("stale_lines_count", 0))
    m4.metric("Phantom Lines", odds_data.get("phantom_lines_count", 0))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Missed Polls", ts_data.get("missed_polls", 0))
    m6.metric("Timezone Issues", ts_data.get("timezone_issues", 0))
    m7.metric("Stale Closes", closing_data.get("stale_close_count", 0))
    m8.metric("Future Timestamps", ts_data.get("future_timestamps", 0))


def _render_timestamp_detail(report_service):
    """Render detailed timestamp audit results."""
    st.markdown(
        "<div style='font-size:0.8rem;font-weight:600;color:#00FFB2;'>"
        "TIMESTAMP AUDIT</div>",
        unsafe_allow_html=True,
    )

    data = st.session_state.get("_dq_audit_data", {}).get("timestamp")
    if not data:
        st.info("Run a full audit from the Overview tab to see timestamp details.")
        return

    score = data.get("score", 0)
    _render_score_gauge("Timestamp Score", score, st)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Second Precision", f"{data.get('second_precision_pct', 0):.1f}%")
    c2.metric("Ingestion Regularity", f"{data.get('ingestion_regularity_pct', 0):.1f}%")
    c3.metric("Avg Closing Delta", f"{data.get('avg_closing_delta_sec', 0):.0f}s")

    c4, c5, c6 = st.columns(3)
    c4.metric("Future Timestamps", data.get("future_timestamps", 0))
    c5.metric("Pre-Season Timestamps", data.get("pre_season_timestamps", 0))
    c6.metric("Late Closing Captures", data.get("late_closing_captures", 0))

    issues = data.get("issues", [])
    if issues:
        st.markdown("**Issues Found:**")
        for issue in issues:
            st.warning(issue)
    else:
        st.success("No timestamp issues detected.")


def _render_odds_detail(report_service):
    """Render detailed odds audit results."""
    st.markdown(
        "<div style='font-size:0.8rem;font-weight:600;color:#00FFB2;'>"
        "ODDS AUDIT</div>",
        unsafe_allow_html=True,
    )

    data = st.session_state.get("_dq_audit_data", {}).get("odds")
    if not data:
        st.info("Run a full audit from the Overview tab to see odds details.")
        return

    score = data.get("score", 0)
    _render_score_gauge("Odds Score", score, st)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Availability Verified", f"{data.get('availability_verified_pct', 0):.1f}%")
    c2.metric("Stale Sequences", data.get("stale_lines_count", 0))
    c3.metric("Phantom Lines", data.get("phantom_lines_count", 0))

    c4, c5 = st.columns(2)
    c4.metric("Bets Verified", f"{data.get('bets_verified', 0)}/{data.get('bets_checked', 0)}")
    c5.metric("Unreasonable Odds", data.get("unreasonable_odds_count", 0))

    # Stale lines detail
    stale_details = data.get("stale_lines_detail", [])
    if stale_details:
        st.markdown("**Stale Line Sequences:**")
        df = pd.DataFrame(stale_details)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Phantom lines detail
    phantom_details = data.get("phantom_lines_detail", [])
    if phantom_details:
        st.markdown("**Phantom Lines:**")
        df = pd.DataFrame(phantom_details)
        st.dataframe(df, use_container_width=True, hide_index=True)

    issues = data.get("issues", [])
    if issues:
        st.markdown("**Issues Found:**")
        for issue in issues:
            st.warning(issue)
    else:
        st.success("No odds quality issues detected.")


def _render_closing_detail(report_service):
    """Render detailed closing line audit results."""
    st.markdown(
        "<div style='font-size:0.8rem;font-weight:600;color:#00FFB2;'>"
        "CLOSING LINE AUDIT</div>",
        unsafe_allow_html=True,
    )

    data = st.session_state.get("_dq_audit_data", {}).get("closing_line")
    if not data:
        st.info("Run a full audit from the Overview tab to see closing line details.")
        return

    score = data.get("score", 0)
    _render_score_gauge("Closing Line Score", score, st)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Closes", f"{data.get('true_close_pct', 0):.1f}%")
    c2.metric("Capture Consistency", f"{data.get('capture_consistency_pct', 0):.1f}%")
    c3.metric("Cross-Source Match", f"{data.get('cross_source_match_pct', 0):.1f}%")
    c4.metric("Active Market", f"{data.get('active_market_pct', 0):.1f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Stale Closes", data.get("stale_close_count", 0))
    c6.metric("Total Closing Lines", data.get("total_closing_lines", 0))
    c7.metric("Timing Std Dev", f"{data.get('capture_std_dev_sec', 0):.0f}s")

    issues = data.get("issues", [])
    if issues:
        st.markdown("**Issues Found:**")
        for issue in issues:
            st.warning(issue)
    else:
        st.success("No closing line issues detected.")


def _render_completeness_detail(report_service):
    """Render detailed completeness audit results."""
    st.markdown(
        "<div style='font-size:0.8rem;font-weight:600;color:#00FFB2;'>"
        "COMPLETENESS AUDIT</div>",
        unsafe_allow_html=True,
    )

    data = st.session_state.get("_dq_audit_data", {}).get("completeness")
    if not data:
        st.info("Run a full audit from the Overview tab to see completeness details.")
        return

    score = data.get("score", 0)
    _render_score_gauge("Completeness Score", score, st)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bets", data.get("total_bets", 0))
    c2.metric("Settled Bets", data.get("settled_bets", 0))
    c3.metric("Complete Bets", data.get("complete_bets", 0))
    c4.metric("Event Coverage", f"{data.get('event_coverage_pct', 0):.1f}%")

    st.markdown("**Missing Data Breakdown:**")
    missing_data = {
        "Category": ["Opening Lines", "Bet-Time Snapshots", "Closing Lines", "CLV Calculations"],
        "Missing %": [
            data.get("missing_opening_pct", 0),
            data.get("missing_snapshot_pct", 0),
            data.get("missing_closing_pct", 0),
            data.get("missing_clv_pct", 0),
        ],
    }
    df = pd.DataFrame(missing_data)

    fig = go.Figure(go.Bar(
        x=df["Missing %"],
        y=df["Category"],
        orientation="h",
        marker_color=[
            "#FF3358" if v > 30 else "#FFB800" if v > 10 else "#00FFB2"
            for v in df["Missing %"]
        ],
        text=[f"{v:.1f}%" for v in df["Missing %"]],
        textposition="auto",
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Fira Code, monospace", size=11, color="#4A607A"),
        xaxis=dict(title="Missing %", gridcolor="#0E1E30"),
        yaxis=dict(gridcolor="#0E1E30"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Book coverage
    book_coverage = data.get("book_coverage", {})
    if book_coverage:
        st.markdown("**Book Coverage:**")
        book_df = pd.DataFrame([
            {"Book": book, "Observations": count}
            for book, count in sorted(book_coverage.items(), key=lambda x: -x[1])
        ])
        st.dataframe(book_df, use_container_width=True, hide_index=True)

    # Systematic gaps
    gaps = data.get("systematic_gaps", [])
    if gaps:
        st.markdown("**Systematic Gaps:**")
        for gap in gaps:
            st.warning(f"[{gap.get('type', '?')}] {gap.get('name', '?')}: {gap.get('detail', '')}")

    issues = data.get("issues", [])
    if issues:
        st.markdown("**Issues Found:**")
        for issue in issues:
            st.warning(issue)
    else:
        st.success("No completeness issues detected.")


def _render_history(report_service):
    """Render historical data quality score trends."""
    st.markdown(
        "<div style='font-size:0.8rem;font-weight:600;color:#00FFB2;'>"
        "AUDIT HISTORY</div>",
        unsafe_allow_html=True,
    )

    history = report_service.get_report_history(limit=50)
    if not history:
        st.info("No historical audit data. Run audits over time to see trends.")
        return

    df = pd.DataFrame(history)
    df["generated_at"] = pd.to_datetime(df["generated_at"])
    df = df.sort_values("generated_at")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["generated_at"],
        y=df["integrity_score"],
        mode="lines+markers",
        name="Quality Score",
        line=dict(color="#00FFB2", width=2),
        marker=dict(size=6),
    ))

    # Add threshold line
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#FF3358",
        annotation_text="Critical Threshold",
        annotation_position="top left",
        annotation_font_color="#FF3358",
    )
    fig.add_hline(
        y=90,
        line_dash="dot",
        line_color="#FFB800",
        annotation_text="Good Threshold",
        annotation_position="top left",
        annotation_font_color="#FFB800",
    )

    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Fira Code, monospace", size=11, color="#4A607A"),
        xaxis=dict(title="Date", gridcolor="#0E1E30"),
        yaxis=dict(title="Score", range=[0, 105], gridcolor="#0E1E30"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table of recent reports
    st.markdown("**Recent Reports:**")
    display_df = df[["generated_at", "integrity_score", "total_bets",
                      "missing_opening_lines", "missing_closing_lines",
                      "suspected_data_errors"]].copy()
    display_df.columns = ["Date", "Score", "Bets", "Missing Open",
                           "Missing Close", "Data Errors"]
    st.dataframe(
        display_df.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def _render_raw_report(report_service):
    """Render the full raw text report."""
    st.markdown(
        "<div style='font-size:0.8rem;font-weight:600;color:#00FFB2;'>"
        "RAW REPORT</div>",
        unsafe_allow_html=True,
    )

    if st.button("Generate Fresh Report", key="dq_raw_report"):
        with st.spinner("Generating full data quality report..."):
            report_text = report_service.generate()
            st.code(report_text, language=None)
    else:
        latest = report_service.get_latest_report()
        if latest and latest.get("report_text"):
            st.code(latest["report_text"], language=None)
        else:
            st.info("No report available. Click 'Generate Fresh Report'.")
