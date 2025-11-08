import streamlit as st
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="NBA PRA Edge", page_icon="🏀", layout="centered")
st.title("🏀 NBA PRA Edge Calculator")

st.markdown(
    "Use this to estimate probability, EV, and suggested stake for PRA props "
    "based on your per-minute projections."
)

with st.form("pra_form"):
    col1, col2 = st.columns(2)
    with col1:
        player = st.text_input("Player Name", "Donovan Mitchell")
        line = st.number_input("PRA Line", value=33.5, step=0.5)
        proj_min = st.number_input("Projected Minutes", value=34.0, step=1.0)
    with col2:
        pra_per_min = st.number_input("PRA per Minute", value=1.10, step=0.01)
        sd_per_min = st.number_input("SD per Minute", value=0.15, step=0.01)

    st.markdown("### Bankroll & Payout")
    col3, col4, col5 = st.columns(3)
    with col3:
        payout = st.number_input("Payout Multiplier", value=3.0, step=0.1)
    with col4:
        bankroll = st.number_input("Bankroll ($)", value=30.0, step=1.0)
    with col5:
        kelly_frac = st.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)

    submitted = st.form_submit_button("Calculate")

if submitted:
    # Mean & SD for PRA
    mu = pra_per_min * proj_min
    sd = sd_per_min * np.sqrt(proj_min)
    if sd <= 0:
        sd = max(1.0, 0.15 * mu)

    # Probability OVER using Normal approximation
    p_hat = 1 - norm.cdf(line, mu, sd)

    # Market implied probability from payout
    if payout <= 1:
        st.error("Payout multiplier must be > 1.")
    else:
        p_market = 1.0 / payout

        # EV per $1
        ev = p_hat * (payout - 1.0) - (1.0 - p_hat)

        # Full Kelly fraction
        b = payout - 1.0
        full_kelly = max(0.0, (b * p_hat - (1.0 - p_hat)) / b)

        # Fractional Kelly stake, cap at 3% bankroll
        stake = bankroll * kelly_frac * full_kelly
        stake = min(stake, bankroll * 0.03)
        stake = max(0.0, round(stake, 2))

        st.subheader(f"Results for {player}")
        colA, colB, colC = st.columns(3)
        colA.metric("Prob OVER", f"{p_hat:.1%}")
        colB.metric("EV per $", f"{ev*100:.1f}%")
        colC.metric("Suggested Stake", f"${stake:.2f}")

        if ev > 0:
            st.success("✅ YES: This is +EV under your assumptions.")
        else:
            st.error("❌ NO: This is -EV under your assumptions.")

        st.caption(
            "Note: Uses your inputs for PRA/min, SD, and minutes. Be conservative if minutes or role are uncertain."
        )
