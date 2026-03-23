"""
streamlit_app/pages/calibration_tab.py
======================================
CALIBRATION tab -- predicted vs actual win rates from report_service.
Pure display, no computation.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services import report_service
from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY
from streamlit_app.components.charts import calibration_chart


def render() -> None:
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>"
        f"MODEL CALIBRATION ANALYSIS</div>",
        unsafe_allow_html=True,
    )

    # -- Get calibration data from DB --------------------------------------
    cal_report = report_service.get_latest_report(report_type="calibration")
    buckets = cal_report.get("buckets", []) if cal_report else []

    if not buckets:
        # Try raw calibration data
        buckets = report_service.get_calibration_data()

    if not buckets:
        st.info(
            "No calibration data available yet. Calibration snapshots are generated "
            "by the report_worker after enough settled bets accumulate."
        )
        return

    # -- Summary metrics ---------------------------------------------------
    total_bets = sum(int(b.get("n_bets", 0)) for b in buckets)
    errors = [abs(float(b.get("calibration_error", 0))) for b in buckets]
    n_bets_list = [int(b.get("n_bets", 0)) for b in buckets]
    weighted_error = (
        sum(e * n for e, n in zip(errors, n_bets_list)) / max(total_bets, 1)
    )
    overconfident_count = sum(
        int(b.get("n_bets", 0)) for b in buckets if b.get("is_overconfident")
    )
    overconfident_pct = overconfident_count / max(total_bets, 1) * 100

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Calibration Buckets", len(buckets))
    mc2.metric("Total Bets Analyzed", total_bets)
    mc3.metric("Avg Cal Error", f"{weighted_error:.4f}")
    mc4.metric("Overconfident %", f"{overconfident_pct:.1f}%")

    # Report date
    report_date = buckets[0].get("report_date") if buckets else None
    if report_date:
        st.caption(f"Report date: {report_date}")

    # -- Calibration chart -------------------------------------------------
    st.markdown("---")
    calibration_chart(buckets, title="Predicted Probability vs Actual Win Rate")

    # -- Bucket detail table -----------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:#2A5070;"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>BUCKET DETAIL</div>",
        unsafe_allow_html=True,
    )
    df = pd.DataFrame(buckets)
    display_cols = [
        "bucket_label", "prob_lower", "prob_upper", "predicted_avg",
        "actual_rate", "n_bets", "calibration_error", "is_overconfident",
    ]
    available = [c for c in display_cols if c in df.columns]
    if available:
        st.dataframe(df[available], use_container_width=True, hide_index=True)

    # -- Interpretation notes -----------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='font-family:{FONT_MONO};font-size:0.62rem;color:#4A607A;"
        f"line-height:1.8;'>"
        f"<b>Reading the chart:</b> Points above the diagonal = model is underconfident "
        f"(actual win rate higher than predicted). Points below = overconfident. "
        f"Ideal calibration hugs the 45-degree line.<br>"
        f"<b>Avg calibration error:</b> Lower is better. Under 0.03 is excellent. "
        f"0.03-0.06 is good. Over 0.06 suggests model needs retraining.</div>",
        unsafe_allow_html=True,
    )
