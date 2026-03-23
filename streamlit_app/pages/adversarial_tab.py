"""
streamlit_app/pages/adversarial_tab.py
========================================
Adversarial testing results — all tests with pass/fail verdicts.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING


def render() -> None:
    st.markdown(
        f"<h2 style='font-family:{FONT_DISPLAY};color:{COLOR_PRIMARY};font-size:1.3rem;"
        f"letter-spacing:0.1em;margin-bottom:0;'>ADVERSARIAL TESTS</h2>"
        f"<p style='font-family:{FONT_MONO};color:#4A607A;font-size:0.65rem;margin-top:0.2rem;'>"
        f"Trying to break the system to prove it is robust</p>",
        unsafe_allow_html=True,
    )

    col_all, col_single = st.columns([1, 1])
    with col_all:
        run_all = st.button("Run ALL Adversarial Tests", type="primary")
    with col_single:
        single_test = st.selectbox("Or run single test:", [
            "(select)", "probability_perturbation", "best_bet_removal",
            "noise_injection", "assumption_distortion",
        ])

    if run_all:
        with st.spinner("Running all adversarial tests — this may take a minute..."):
            try:
                from tests.adversarial.runner import AdversarialRunner
                runner = AdversarialRunner(sport="NBA")
                results = runner.run_all()
                st.session_state["_adversarial_results"] = results
            except Exception as e:
                st.error(f"Adversarial tests failed: {e}")
                return

    if single_test and single_test != "(select)":
        with st.spinner(f"Running {single_test}..."):
            try:
                from tests.adversarial.runner import AdversarialRunner
                runner = AdversarialRunner(sport="NBA")
                result = runner.run_single(single_test)
                st.session_state[f"_adv_{single_test}"] = result
            except Exception as e:
                st.error(f"Test failed: {e}")

    # Show full results if available
    results = st.session_state.get("_adversarial_results")
    if results:
        _render_full_results(results)
    else:
        # Show individual test results
        for test_name in ["probability_perturbation", "best_bet_removal", "noise_injection", "assumption_distortion"]:
            result = st.session_state.get(f"_adv_{test_name}")
            if result:
                _render_single_test(test_name, result)

    # Show last saved results
    if not results:
        try:
            from tests.adversarial.runner import AdversarialRunner
            last = AdversarialRunner(sport="NBA").get_last_results()
            if last:
                st.markdown("---")
                st.markdown("### Last Saved Results")
                summary = last.get("summary", {})
                verdict = summary.get("final_verdict", "UNKNOWN")
                color = COLOR_PRIMARY if verdict == "PASS" else (
                    COLOR_WARNING if verdict == "CONDITIONAL_PASS" else COLOR_DANGER
                )
                st.markdown(
                    f"<div style='font-family:{FONT_DISPLAY};font-size:1.2rem;color:{color};'>"
                    f"{verdict}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"Tests passed: {summary.get('tests_passed', 0)}/{summary.get('tests_total', 0)}")
                if last.get("failures"):
                    for f in last["failures"]:
                        st.warning(f"**{f['test']}**: {f.get('interpretation', 'Failed')}")
        except Exception:
            pass


def _render_full_results(results: dict) -> None:
    summary = results.get("summary", {})
    verdict = summary.get("final_verdict", "UNKNOWN")

    # Verdict banner
    if verdict == "PASS":
        color = COLOR_PRIMARY
        bg = "#00FFB210"
    elif verdict == "CONDITIONAL_PASS":
        color = COLOR_WARNING
        bg = "#FFB80010"
    else:
        color = COLOR_DANGER
        bg = "#FF335810"

    st.markdown(f"""
    <div style='background:{bg};border:3px solid {color};border-radius:12px;
                padding:2rem;text-align:center;margin:1rem 0;'>
        <div style='font-family:{FONT_DISPLAY};font-size:2rem;font-weight:700;
                    color:{color};letter-spacing:0.12em;'>{verdict}</div>
        <div style='font-family:{FONT_MONO};font-size:0.75rem;color:#8BA8BF;margin-top:0.5rem;'>
            {summary.get('interpretation', '')}</div>
        <div style='font-family:{FONT_MONO};font-size:0.7rem;color:#6A8AAA;margin-top:0.3rem;'>
            Tests passed: {summary.get('tests_passed', 0)}/{summary.get('tests_total', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Individual test results
    test_verdicts = summary.get("test_verdicts", {})
    for test_name, passed in test_verdicts.items():
        color = COLOR_PRIMARY if passed else COLOR_DANGER
        icon = "PASS" if passed else "FAIL"
        label = test_name.replace("_", " ").upper()
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:0.6rem 1rem;border:1px solid {color}40;border-radius:6px;margin:0.3rem 0;'>"
            f"<span style='font-family:{FONT_DISPLAY};color:#EEF4FF;'>{label}</span>"
            f"<span style='font-family:{FONT_MONO};color:{color};font-weight:700;'>{icon}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Detailed results per test
    individual_results = results.get("results", {})

    for test_name, test_result in individual_results.items():
        with st.expander(f"Details: {test_name.replace('_', ' ').upper()}"):
            _render_single_test(test_name, test_result)


def _render_single_test(test_name: str, result: dict) -> None:
    status = result.get("status", "unknown")
    if status == "error":
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        return
    if status == "insufficient_data":
        st.info(f"Insufficient data ({result.get('n_bets', 0)} bets)")
        return

    verdict = result.get("verdict", "UNKNOWN")
    color = COLOR_PRIMARY if verdict == "PASS" else COLOR_DANGER
    st.markdown(f"**Verdict:** <span style='color:{color};font-weight:bold;'>{verdict}</span>",
                unsafe_allow_html=True)

    interpretation = result.get("interpretation", "")
    if interpretation:
        st.markdown(f"*{interpretation}*")

    # Show results table
    test_results = result.get("results", [])
    if test_results and isinstance(test_results, list):
        import pandas as pd
        df = pd.DataFrame(test_results)
        st.dataframe(df, hide_index=True, use_container_width=True)

    # Special handling for assumption distortion
    if test_name == "assumption_distortion":
        critical = result.get("critical_assumptions", [])
        if critical:
            st.error(f"Critical assumptions: {', '.join(critical)}")
        high = result.get("high_sensitivity_assumptions", [])
        if high:
            st.warning(f"High sensitivity: {', '.join(high)}")

    # Special handling for noise injection
    if test_name == "noise_injection":
        bp = result.get("breakpoint_pct")
        if bp:
            st.metric("Breakpoint", f"{bp}% noise")
        else:
            st.success("No breakpoint found in tested range")
