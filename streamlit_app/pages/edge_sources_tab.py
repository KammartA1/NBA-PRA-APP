"""
streamlit_app/pages/edge_sources_tab.py
=======================================
EDGE SOURCES tab -- displays all edge signal sources ranked by standalone
value, independence heatmap, per-source validation metrics, and source
health dashboard.  All data from edge_analysis read-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app.config import (
    FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING,
    get_user_id,
)


def render() -> None:
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>"
        f"EDGE SOURCES</div>",
        unsafe_allow_html=True,
    )

    # Load edge analysis system
    edge_available = False
    edge_error = ""
    try:
        from edge_analysis.source_registry import SourceRegistry
        from edge_analysis.edge_sources import EdgeSourceCatalog
        edge_available = True
    except Exception as e:
        edge_error = str(e)

    if not edge_available:
        st.warning(f"Edge Analysis system not available: {edge_error}")
        return

    es_tabs = st.tabs([
        "Source Rankings",
        "Independence Matrix",
        "Source Details",
        "Health Dashboard",
        "Signal Catalog",
    ])

    # -- Source Rankings ---------------------------------------------------
    with es_tabs[0]:
        _render_rankings()

    # -- Independence Matrix -----------------------------------------------
    with es_tabs[1]:
        _render_independence_matrix()

    # -- Source Details -----------------------------------------------------
    with es_tabs[2]:
        _render_source_details()

    # -- Health Dashboard --------------------------------------------------
    with es_tabs[3]:
        _render_health_dashboard()

    # -- Signal Catalog ----------------------------------------------------
    with es_tabs[4]:
        _render_signal_catalog()


def _render_rankings() -> None:
    """Source rankings table sorted by standalone Sharpe ratio."""
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>SOURCE RANKINGS BY STANDALONE VALUE</div>",
        unsafe_allow_html=True,
    )

    try:
        from edge_analysis.source_registry import SourceRegistry

        registry = SourceRegistry()
        registry.load_sources()
        registry.compute_independence_matrix_synthetic()
        rankings = registry.validate_and_rank()

        if not rankings:
            st.info("No edge sources loaded.")
            return

        # Summary metrics
        active = [r for r in rankings if r.status == "active"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Sources", len(rankings))
        mc2.metric("Active Sources", len(active))
        mc3.metric(
            "Avg Sharpe",
            f"{np.mean([r.sharpe for r in active]):.2f}" if active else "--",
        )
        mc4.metric(
            "Avg Hit Rate",
            f"{np.mean([r.hit_rate for r in active]):.1%}" if active else "--",
        )

        # Rankings table
        rows = []
        for r in rankings:
            status_icon = {
                "active": "ACTIVE",
                "rejected_correlation": "REJECTED (corr)",
                "rejected_p_value": "REJECTED (p-val)",
                "rejected_public": "REJECTED (public)",
                "insufficient_data": "INSUFFICIENT DATA",
            }.get(r.status, r.status.upper())

            decay_short = r.decay_risk.split(".")[0].strip() if "." in r.decay_risk else r.decay_risk[:40]
            mechanism_short = r.mechanism[:120] + "..." if len(r.mechanism) > 120 else r.mechanism

            rows.append({
                "Source": r.name.replace("_", " ").title(),
                "Sharpe": f"{r.sharpe:.3f}",
                "p-value": f"{r.p_value:.4f}",
                "Hit Rate": f"{r.hit_rate:.1%}",
                "Sample": r.sample_size,
                "Decay Risk": decay_short,
                "Status": status_icon,
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Rejection details
        rejected = [r for r in rankings if r.status.startswith("rejected")]
        if rejected:
            st.markdown("---")
            st.markdown(
                f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:{COLOR_WARNING};"
                f"letter-spacing:0.12em;margin-bottom:0.4rem;'>REJECTED SOURCES</div>",
                unsafe_allow_html=True,
            )
            for r in rejected:
                st.markdown(
                    f"<div style='background:#060D18;border-left:3px solid {COLOR_DANGER};"
                    f"padding:0.4rem 0.6rem;margin-bottom:0.3rem;font-family:{FONT_MONO};"
                    f"font-size:0.58rem;color:#8BA8BF;'>"
                    f"<strong>{r.name}</strong>: {r.rejection_reason}</div>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Failed to load rankings: {e}")


def _render_independence_matrix() -> None:
    """Independence heatmap showing pairwise correlations."""
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>SIGNAL INDEPENDENCE MATRIX</div>",
        unsafe_allow_html=True,
    )

    try:
        from edge_analysis.source_registry import SourceRegistry
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        registry = SourceRegistry()
        registry.load_sources()
        corr = registry.compute_independence_matrix_synthetic()
        names = list(registry.get_all_sources().keys())

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0A1628")
        ax.set_facecolor("#0A1628")

        # Custom colormap: blue (negative) -> dark (zero) -> red (positive)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "edge_corr",
            ["#1565C0", "#0A1628", "#0A1628", "#C62828"],
            N=256,
        )

        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        # Labels
        display_names = [n.replace("_", "\n") for n in names]
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(display_names, fontsize=7, color="#8BA8BF", rotation=45, ha="right")
        ax.set_yticklabels(display_names, fontsize=7, color="#8BA8BF")

        # Add correlation values in cells
        for i in range(len(names)):
            for j in range(len(names)):
                val = corr[i, j]
                if abs(val) > 0.01:
                    color = "#FFFFFF" if abs(val) > 0.4 else "#6A8AAA"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=6, color=color, fontweight="bold" if abs(val) > 0.5 else "normal",
                    )

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#8BA8BF", labelsize=7)
        cbar.set_label("Pearson Correlation", color="#8BA8BF", fontsize=8)

        ax.set_title(
            "Pairwise Signal Correlation",
            color="#8BA8BF", fontsize=10, pad=12,
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Flag non-independent pairs
        st.markdown("---")
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:{COLOR_WARNING};"
            f"letter-spacing:0.12em;margin-bottom:0.4rem;'>HIGH CORRELATION PAIRS (|r| > 0.5)</div>",
            unsafe_allow_html=True,
        )

        high_corr_pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if abs(corr[i, j]) > 0.5:
                    high_corr_pairs.append({
                        "Source A": names[i].replace("_", " ").title(),
                        "Source B": names[j].replace("_", " ").title(),
                        "Correlation": f"{corr[i, j]:.3f}",
                        "Flag": "NOT INDEPENDENT",
                    })

        if high_corr_pairs:
            st.dataframe(
                pd.DataFrame(high_corr_pairs),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("All source pairs have |correlation| <= 0.5. Sources are sufficiently independent.")

    except Exception as e:
        st.error(f"Failed to render independence matrix: {e}")


def _render_source_details() -> None:
    """Expandable details for each source."""
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>PER-SOURCE VALIDATION DETAILS</div>",
        unsafe_allow_html=True,
    )

    try:
        from edge_analysis.source_registry import SourceRegistry

        registry = SourceRegistry()
        registry.load_sources()
        rankings = registry.validate_and_rank()
        sources = registry.get_all_sources()

        for r in rankings:
            source = sources.get(r.name)
            if not source:
                continue

            status_color = COLOR_PRIMARY if r.status == "active" else COLOR_DANGER
            header = (
                f"{r.name.replace('_', ' ').title()} — "
                f"Sharpe: {r.sharpe:.3f} | "
                f"Status: {r.status.upper()}"
            )

            with st.expander(header, expanded=False):
                # Metrics row
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Sharpe Ratio", f"{r.sharpe:.3f}")
                dc2.metric("p-value", f"{r.p_value:.4f}")
                dc3.metric("Hit Rate", f"{r.hit_rate:.1%}")
                dc4.metric("Sample Size", r.sample_size)

                # Mechanism
                st.markdown(
                    f"<div style='background:#060D18;border:1px solid #0E1E30;border-radius:4px;"
                    f"padding:0.6rem;margin:0.5rem 0;'>"
                    f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#2A6080;"
                    f"margin-bottom:0.3rem;'>MECHANISM (WHY MARKET IS WRONG)</div>"
                    f"<div style='font-family:{FONT_MONO};font-size:0.58rem;color:#8BA8BF;"
                    f"line-height:1.6;'>{source.get_mechanism()}</div></div>",
                    unsafe_allow_html=True,
                )

                # Decay risk
                st.markdown(
                    f"<div style='background:#060D18;border:1px solid #0E1E30;border-radius:4px;"
                    f"padding:0.6rem;margin:0.5rem 0;'>"
                    f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#2A6080;"
                    f"margin-bottom:0.3rem;'>DECAY RISK</div>"
                    f"<div style='font-family:{FONT_MONO};font-size:0.58rem;color:#8BA8BF;"
                    f"line-height:1.6;'>{source.get_decay_risk()}</div></div>",
                    unsafe_allow_html=True,
                )

                # Category and description
                st.markdown(
                    f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#4A607A;"
                    f"margin-top:0.5rem;'>"
                    f"Category: {source.category} | "
                    f"Module: edge_analysis.sources.{r.name}</div>",
                    unsafe_allow_html=True,
                )

                if r.rejection_reason:
                    st.markdown(
                        f"<div style='background:#1A0A0A;border-left:3px solid {COLOR_DANGER};"
                        f"padding:0.4rem 0.6rem;margin-top:0.5rem;font-family:{FONT_MONO};"
                        f"font-size:0.58rem;color:{COLOR_DANGER};'>"
                        f"Rejection: {r.rejection_reason}</div>",
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.error(f"Failed to load source details: {e}")


def _render_health_dashboard() -> None:
    """Overall source health dashboard."""
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>SOURCE HEALTH DASHBOARD</div>",
        unsafe_allow_html=True,
    )

    try:
        from edge_analysis.source_registry import SourceRegistry

        registry = SourceRegistry()
        registry.load_sources()
        registry.compute_independence_matrix_synthetic()
        registry.validate_and_rank()
        health = registry.get_health_summary()

        # Overall health indicator
        active_pct = health["active_sources"] / max(health["total_sources"], 1) * 100
        if active_pct >= 70:
            health_color = COLOR_PRIMARY
            health_label = "HEALTHY"
        elif active_pct >= 50:
            health_color = COLOR_WARNING
            health_label = "DEGRADED"
        else:
            health_color = COLOR_DANGER
            health_label = "CRITICAL"

        st.markdown(
            f"<div style='background:#060D18;border:1px solid #0E1E30;border-radius:4px;"
            f"padding:0.6rem 0.8rem;margin-bottom:0.8rem;display:flex;"
            f"align-items:center;gap:0.8rem;'>"
            f"<div style='width:10px;height:10px;border-radius:50%;background:{health_color};"
            f"box-shadow:0 0 8px {health_color};'></div>"
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.85rem;font-weight:700;"
            f"color:{health_color};'>{health_label}</div>"
            f"<div style='margin-left:auto;font-family:{FONT_MONO};font-size:0.55rem;"
            f"color:#4A607A;'>{health['active_sources']}/{health['total_sources']} sources active</div></div>",
            unsafe_allow_html=True,
        )

        # Key metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Active Sources", health["active_sources"])
        mc2.metric("Rejected Sources", health["rejected_sources"])
        mc3.metric("Avg Sharpe (Active)", f"{health['avg_active_sharpe']:.3f}")
        mc4.metric("Avg Hit Rate (Active)", f"{health['avg_active_hit_rate']:.1%}")

        # Source status breakdown
        st.markdown("---")
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.12em;margin-bottom:0.4rem;'>SOURCE STATUS BREAKDOWN</div>",
            unsafe_allow_html=True,
        )

        statuses = health.get("source_statuses", {})
        status_rows = []
        for name, status in statuses.items():
            color = COLOR_PRIMARY if status == "active" else COLOR_DANGER
            reason = health.get("rejection_reasons", {}).get(name, "")
            status_rows.append({
                "Source": name.replace("_", " ").title(),
                "Status": status.upper(),
                "Reason": reason if reason else "--",
            })

        st.dataframe(
            pd.DataFrame(status_rows),
            use_container_width=True,
            hide_index=True,
        )

        # Recommendations
        st.markdown("---")
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.12em;margin-bottom:0.4rem;'>RECOMMENDATIONS</div>",
            unsafe_allow_html=True,
        )

        recommendations = []
        rankings = registry.get_rankings()

        # Check for sources close to rejection thresholds
        for r in rankings:
            if r.status == "active" and r.p_value > 0.03:
                recommendations.append(
                    f"Source '{r.name}' has marginal significance (p={r.p_value:.4f}). "
                    f"Monitor closely for degradation."
                )
            if r.status == "active" and r.sharpe < 0.9:
                recommendations.append(
                    f"Source '{r.name}' has low standalone Sharpe ({r.sharpe:.3f}). "
                    f"Consider if it adds enough value to justify complexity."
                )

        if not recommendations:
            recommendations.append("All active sources are performing well. No action needed.")

        for rec in recommendations:
            st.markdown(
                f"<div style='background:#060D18;border-left:3px solid {COLOR_PRIMARY};"
                f"padding:0.4rem 0.6rem;margin-bottom:0.3rem;font-family:{FONT_MONO};"
                f"font-size:0.58rem;color:#8BA8BF;'>{rec}</div>",
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Failed to load health dashboard: {e}")


def _render_signal_catalog() -> None:
    """Full catalog of all ~53 signals in the codebase."""
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>COMPLETE SIGNAL CATALOG ({_get_signal_count()} SIGNALS)</div>",
        unsafe_allow_html=True,
    )

    try:
        from edge_analysis.edge_sources import EdgeSourceCatalog

        catalog = EdgeSourceCatalog()

        # Category filter
        categories = catalog.get_categories()
        selected_cat = st.selectbox(
            "Filter by Category",
            ["All"] + categories,
            key="edge_cat_filter",
        )

        if selected_cat == "All":
            signals = catalog.signals
        else:
            signals = catalog.get_by_category(selected_cat)

        st.markdown(
            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#4A607A;"
            f"margin-bottom:0.5rem;'>Showing {len(signals)} signals</div>",
            unsafe_allow_html=True,
        )

        # Summary table
        summary = catalog.get_summary_table()
        if selected_cat != "All":
            summary = [s for s in summary if s["category"] == selected_cat]

        df = pd.DataFrame(summary)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Expandable detail for each signal
        st.markdown("---")
        for sig in signals:
            with st.expander(f"{sig.name} ({sig.module})", expanded=False):
                st.markdown(
                    f"<div style='font-family:{FONT_MONO};font-size:0.58rem;color:#8BA8BF;"
                    f"line-height:1.8;'>"
                    f"<strong>Category:</strong> {sig.category}<br>"
                    f"<strong>Module:</strong> {sig.module}<br>"
                    f"<strong>Mechanism:</strong> {sig.mechanism}<br>"
                    f"<strong>Data Advantage:</strong> {sig.data_advantage}<br>"
                    f"<strong>Decay Risk:</strong> {sig.decay_risk}<br>"
                    f"<strong>Independence:</strong> {sig.independence_notes}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Failed to load signal catalog: {e}")


def _get_signal_count() -> int:
    """Get total signal count from catalog."""
    try:
        from edge_analysis.edge_sources import EdgeSourceCatalog
        return EdgeSourceCatalog().n_signals
    except Exception:
        return 53
