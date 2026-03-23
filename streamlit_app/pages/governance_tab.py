"""
streamlit_app/pages/governance_tab.py
=======================================
Model governance dashboard — versions, feature importance,
degradation detection, rollback history.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING


def render() -> None:
    st.markdown(
        f"<h2 style='font-family:{FONT_DISPLAY};color:{COLOR_PRIMARY};font-size:1.3rem;"
        f"letter-spacing:0.1em;margin-bottom:0;'>MODEL GOVERNANCE</h2>"
        f"<p style='font-family:{FONT_MONO};color:#4A607A;font-size:0.65rem;margin-top:0.2rem;'>"
        f"Version control, feature importance, degradation detection</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["VERSIONS", "PERFORMANCE", "FEATURES", "SIMPLICITY AUDIT", "ROLLBACK"])

    with tabs[0]:
        _render_versions()
    with tabs[1]:
        _render_performance()
    with tabs[2]:
        _render_features()
    with tabs[3]:
        _render_simplicity()
    with tabs[4]:
        _render_rollback()


def _render_versions() -> None:
    st.markdown("### Model Version History")

    try:
        from governance.version_control import ModelVersionManager
        mgr = ModelVersionManager(sport="NBA")
        history = mgr.get_version_history(limit=20)
    except Exception as e:
        st.warning(f"Could not load version history: {e}")
        return

    if not history:
        st.info("No model versions registered yet.")
        return

    active = mgr.get_active_version()
    if active:
        st.success(f"Active Version: **{active['version']}** (created: {active.get('created_at', 'N/A')})")

    import pandas as pd
    rows = []
    for v in history:
        metrics = v.get("performance_metrics", {})
        rows.append({
            "Version": v["version"],
            "Active": "YES" if v.get("is_active") else "",
            "Created": str(v.get("created_at", ""))[:19],
            "Brier": f"{metrics.get('brier_score', 'N/A')}",
            "ROI": f"{metrics.get('roi_pct', 'N/A')}",
            "Data Hash": str(v.get("training_data_hash", ""))[:12],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_performance() -> None:
    st.markdown("### Performance Tracking")

    try:
        from governance.performance_tracker import PerformanceTracker
        tracker = PerformanceTracker(sport="NBA")
        degradation = tracker.detect_degradation(window=100)
    except Exception as e:
        st.warning(f"Could not load performance data: {e}")
        return

    if degradation.get("degradation_detected"):
        st.error("Performance DEGRADATION detected!")
        st.markdown(f"**Recommendation:** {degradation.get('recommendation', '')}")
    else:
        reason = degradation.get("reason", "")
        if "insufficient" in reason:
            st.info(reason)
        else:
            st.success("Performance is stable. No degradation detected.")

    # Show degradation flags
    flags = degradation.get("flags", {})
    if flags:
        for metric, data in flags.items():
            if isinstance(data, dict):
                degraded = data.get("degraded", False)
                color = COLOR_DANGER if degraded else COLOR_PRIMARY
                st.markdown(
                    f"**{metric.upper()}**: Early={data.get('early', 'N/A')}, "
                    f"Recent={data.get('recent', 'N/A')} "
                    f"<span style='color:{color};font-weight:bold;'>"
                    f"{'DEGRADED' if degraded else 'OK'}</span>",
                    unsafe_allow_html=True,
                )

    # Rolling performance chart
    try:
        rolling = tracker.rolling_performance(window=50)
        if rolling.get("brier"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=rolling["brier"], mode="lines",
                name="Brier", line=dict(color="#4C9AFF", width=2)))
            fig.update_layout(
                title="Rolling Brier Score (50-bet window)",
                yaxis_title="Brier Score", template="plotly_dark", height=300,
                paper_bgcolor="transparent", plot_bgcolor="transparent",
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _render_features() -> None:
    st.markdown("### Feature Importance")

    if st.button("Run Permutation Importance", key="gov_perm"):
        with st.spinner("Computing permutation importance..."):
            try:
                from governance.feature_importance import FeatureImportanceAnalyzer
                analyzer = FeatureImportanceAnalyzer(sport="NBA")
                results = analyzer.permutation_importance()

                if results:
                    import pandas as pd
                    df = pd.DataFrame(results)
                    cols = ["feature", "importance", "p_value", "is_significant", "n_bets_with_feature"]
                    available = [c for c in cols if c in df.columns]
                    st.dataframe(df[available], hide_index=True, use_container_width=True)

                    # Bar chart
                    top_10 = results[:10]
                    if top_10:
                        fig = go.Figure(go.Bar(
                            x=[r["importance"] for r in top_10],
                            y=[r["feature"] for r in top_10],
                            orientation="h",
                            marker_color=[COLOR_PRIMARY if r["is_significant"] else "#4A607A" for r in top_10],
                        ))
                        fig.update_layout(
                            title="Top 10 Feature Importance",
                            xaxis_title="Importance (Brier increase x1000)",
                            template="plotly_dark", height=400,
                            paper_bgcolor="transparent", plot_bgcolor="transparent",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No feature data available.")
            except Exception as e:
                st.error(f"Feature importance analysis failed: {e}")

    if st.button("Run Marginal Brier Analysis", key="gov_brier"):
        with st.spinner("Computing marginal Brier contribution..."):
            try:
                from governance.feature_importance import FeatureImportanceAnalyzer
                analyzer = FeatureImportanceAnalyzer(sport="NBA")
                results = analyzer.marginal_brier_contribution()
                if results:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
                else:
                    st.info("No feature data available.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")


def _render_simplicity() -> None:
    st.markdown("### Simplicity Audit")
    st.caption("Every feature must justify its existence with p < 0.05")

    if st.button("Run Simplicity Audit", type="primary", key="gov_simplicity"):
        with st.spinner("Auditing all features..."):
            try:
                from governance.simplicity_audit import SimplicityAuditor
                auditor = SimplicityAuditor(sport="NBA")
                result = auditor.audit()

                if result["status"] == "completed":
                    summary = result["summary"]
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Total Features", summary["total"])
                    with c2:
                        st.metric("KEEP", summary["keep"])
                    with c3:
                        st.metric("DELETE", summary["delete"])
                    with c4:
                        st.metric("Complexity Score", f"{summary['complexity_score']:.0f}")

                    if result["results"]:
                        import pandas as pd
                        rows = []
                        for r in result["results"]:
                            rows.append({
                                "Feature": r["feature"],
                                "Decision": r["decision"],
                                "p-value": f"{r.get('p_value', 'N/A')}",
                                "Brier With": f"{r.get('brier_with', 'N/A')}",
                                "Brier Without": f"{r.get('brier_without', 'N/A')}",
                                "Reason": r["reason"],
                            })
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                    if summary.get("features_to_delete"):
                        st.warning(f"Recommended deletions: {', '.join(summary['features_to_delete'])}")
                else:
                    st.info(f"Audit status: {result['status']}")
            except Exception as e:
                st.error(f"Simplicity audit failed: {e}")


def _render_rollback() -> None:
    st.markdown("### Rollback Management")

    try:
        from governance.rollback import RollbackManager
        mgr = RollbackManager(sport="NBA")

        if st.button("Check for Auto-Rollback", key="gov_rollback"):
            result = mgr.check_and_rollback()
            if result.get("rollback_needed"):
                st.error(f"ROLLBACK EXECUTED: {result.get('action_taken', '')}")
                for d in result.get("degradations", []):
                    st.markdown(f"- {d}")
            else:
                st.success(f"No rollback needed. {result.get('reason', '')}")

        history = mgr.get_rollback_history()
        if history:
            st.markdown("#### Rollback History")
            import pandas as pd
            st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
        else:
            st.info("No rollback events recorded.")
    except Exception as e:
        st.warning(f"Rollback manager not available: {e}")
