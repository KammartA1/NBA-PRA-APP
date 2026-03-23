"""
streamlit_app/pages/verdict_tab.py
====================================
Final Verdict report — the single most important page.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING


def render() -> None:
    st.markdown(
        f"<h2 style='font-family:{FONT_DISPLAY};color:{COLOR_PRIMARY};font-size:1.3rem;"
        f"letter-spacing:0.1em;margin-bottom:0;'>FINAL VERDICT</h2>"
        f"<p style='font-family:{FONT_MONO};color:#4A607A;font-size:0.65rem;margin-top:0.2rem;'>"
        f"Is this real edge or sophisticated illusion?</p>",
        unsafe_allow_html=True,
    )

    if st.button("Generate Full Report", type="primary"):
        with st.spinner("Analyzing all systems..."):
            try:
                from services.verdict.final_report import FinalVerdict
                fv = FinalVerdict(sport="NBA")
                report = fv.generate()
                st.session_state["_last_verdict_report"] = report
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                return

    report = st.session_state.get("_last_verdict_report")
    if not report:
        st.info("Click 'Generate Full Report' to produce the comprehensive verdict.")
        return

    # Main verdict banner
    has_edge = report.has_real_edge
    color = COLOR_PRIMARY if has_edge else COLOR_DANGER
    verdict_text = "REAL EDGE DETECTED" if has_edge else "NO PROVEN EDGE"

    st.markdown(f"""
    <div style='background:{"#00FFB210" if has_edge else "#FF335810"};
                border:3px solid {color};border-radius:12px;padding:2rem;
                text-align:center;margin:1rem 0;'>
        <div style='font-family:{FONT_DISPLAY};font-size:2rem;font-weight:700;
                    color:{color};letter-spacing:0.12em;'>{verdict_text}</div>
        <div style='font-family:{FONT_MONO};font-size:0.8rem;color:#8BA8BF;margin-top:0.5rem;'>
            Confidence: {report.edge_confidence_pct:.0f}% |
            Total Edge: {report.total_edge_pct:.2f}% |
            $1M Survival: {report.million_dollar_survival_pct:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Edge is Real", f"{report.pct_edge_is_real:.0f}%")
    with c2:
        st.metric("Edge is Illusion", f"{report.pct_edge_is_illusion:.0f}%")
    with c3:
        st.metric("Data Quality", f"{report.data_quality_score:.0f}/100")
    with c4:
        st.metric("Simulation Accuracy", f"{report.simulation_accuracy:.0f}/100")

    # Edge breakdown
    st.markdown("### Edge Breakdown")
    if report.edge_components:
        # Pie chart
        proven = [c for c in report.edge_components if c.category == "proven"]
        fake = [c for c in report.edge_components if c.category == "fake"]
        unproven = [c for c in report.edge_components if c.category == "unproven"]

        fig = go.Figure(go.Pie(
            labels=["Proven", "Fake/Overfit", "Unproven"],
            values=[
                sum(c.contribution_pct for c in proven),
                sum(c.contribution_pct for c in fake),
                sum(c.contribution_pct for c in unproven),
            ],
            marker=dict(colors=[COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING]),
            textinfo="label+percent",
        ))
        fig.update_layout(
            title="Edge Composition",
            template="plotly_dark", height=300,
            paper_bgcolor="transparent", plot_bgcolor="transparent",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Component table
        import pandas as pd
        comp_data = []
        for c in report.edge_components:
            comp_data.append({
                "Source": c.name,
                "Category": c.category.upper(),
                "Contribution": f"{c.contribution_pct:.1f}%",
                "Confidence": f"{c.confidence:.1%}",
                "Status": c.status.upper(),
                "Sample Size": c.sample_size,
            })
        st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

    # System status
    st.markdown("### System Architecture Status")
    for system, status in report.system_status.items():
        color = COLOR_PRIMARY if status == "operational" else (
            COLOR_WARNING if status == "unknown" else COLOR_DANGER
        )
        st.markdown(
            f"<span style='font-family:{FONT_MONO};color:{color};'>{status.upper()}</span> "
            f"<span style='font-family:{FONT_MONO};color:#8BA8BF;'>{system}</span>",
            unsafe_allow_html=True,
        )

    # Action items
    if report.features_to_delete:
        st.markdown("### Features to Delete Immediately")
        for f in report.features_to_delete:
            st.markdown(f"- {f}")

    if report.build_next_priorities:
        st.markdown("### Build-Next Priorities")
        for i, p in enumerate(report.build_next_priorities, 1):
            st.markdown(f"{i}. {p}")

    # Adversary's playbook
    if report.adversary_playbook:
        st.markdown("### Adversary's Playbook")
        st.caption("How a sportsbook would destroy this system")
        for entry in report.adversary_playbook:
            with st.expander(entry["attack"]):
                st.markdown(f"**Method:** {entry['method']}")
                st.markdown(f"**Defense:** {entry['defense']}")

    # Raw summary
    st.markdown("### Summary")
    st.code(report.summary_text, language=None)
