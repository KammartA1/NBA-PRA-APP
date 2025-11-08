# app.py — NBA PRA 2-Pick Combo Edge Calculator
import streamlit as st
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="NBA PRA 2-Pick Edge", page_icon="🏀", layout="centered")
st.title("🏀 NBA PRA 2-Pick Edge Calculator")

st.markdown(
    "Enter stats for two players to estimate probabilities, EV, and suggested stake for individual and combo plays."
)

# ----------- FUNCTIONS -----------
def compute_ev(line, pra_per_min, sd_per_min, minutes, payout, bankroll, kelly_frac):
    mu = pra_per_min * minutes
    sd = sd_per_min * np.sqrt(minutes)
    if sd <= 0:
        sd = max(1.0, 0.15 * mu)
    p_hat = 1 - norm.cdf(line, mu, sd)
    ev = p_hat * (payout - 1.0) - (1.0 - p_hat)
    b = payout - 1.0
    full_kelly = max(0.0, (b * p_hat - (1.0 - p_hat)) / b)
    stake = bankroll * kelly_frac * full_kelly
    stake = min(stake, bankroll * 0.03)
    return p_hat, ev, stake

# ----------- PLAYER 1 -----------
st.subheader("Player 1")
with st.form("player1_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        player1 = st.text_input("Name", "Donovan Mitchell")
        line1 = st.number_input("PRA Line", value=33.5, step=0.5)
        min1 = st.number_input("Projected Minutes", value=34.0, step=1.0)
    with col2:
        pra1 = st.number_input("PRA per Min", value=1.10, step=0.01)
        sd1 = st.number_input("SD per Min", value=0.15, step=0.01)
        payout = st.number_input("Payout Multiplier", value=3.0, step=0.1)
    with col3:
        bankroll = st.number_input("Bankroll ($)", value=30.0, step=1.0)
        kelly = st.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
        submitted1 = st.form_submit_button("Calculate Both")

# ----------- PLAYER 2 -----------
st.subheader("Player 2")
player2 = st.text_input("Name", "Jaylen Brown")
line2 = st.number_input("PRA Line", value=34.5, step=0.5)
min2 = st.number_input("Projected Minutes", value=32.0, step=1.0)
pra2 = st.number_input("PRA per Min", value=1.15, step=0.01)
sd2 = st.number_input("SD per Min", value=0.15, step=0.01)

# ----------- CALCULATE -----------
if submitted1:
    p1, ev1, stake1 = compute_ev(line1, pra1, sd1, min1, payout, bankroll, kelly)
    p2, ev2, stake2 = compute_ev(line2, pra2, sd2, min2, payout, bankroll, kelly)
    p_joint = p1 * p2
    ev_joint = payout * p_joint - 1
    b_joint = payout - 1
    full_k_joint = max(0.0, (b_joint * p_joint - (1 - p_joint)) / b_joint)
    stake_joint = bankroll * kelly * full_k_joint
    stake_joint = min(stake_joint, bankroll * 0.03)

    st.markdown("---")
    st.header("📊 Results")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"{player1}")
        st.metric("Prob OVER", f"{p1:.1%}")
        st.metric("EV per $", f"{ev1*100:.1f}%")
        st.metric("Stake", f"${stake1:.2f}")
        st.write("✅ +EV" if ev1 > 0 else "❌ -EV")

    with c2:
        st.subheader(f"{player2}")
        st.metric("Prob OVER", f"{p2:.1%}")
        st.metric("EV per $", f"{ev2*100:.1f}%")
        st.metric("Stake", f"${stake2:.2f}")
        st.write("✅ +EV" if ev2 > 0 else "❌ -EV")

    st.markdown("---")
    st.subheader("🎯 2-Pick Combo (Both Must Hit)")
    st.metric("Joint Prob", f"{p_joint:.1%}")
    st.metric("EV per $", f"{ev_joint*100:.1f}%")
    st.metric("Recommended Stake", f"${stake_joint:.2f}")
    if ev_joint > 0:
        st.success("✅ YES: Combo is +EV")
    else:
        st.error("❌ NO: Combo is -EV")

st.caption(
    "Tip: Use recent 10-20 games for PRA/min & SD/min. Adjust minutes for role, pace, and injury news."
)
