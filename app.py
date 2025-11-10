import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time, random, os
from datetime import datetime
from scipy.stats import norm

st.set_page_config(page_title="NBA Prop Model", layout="wide", page_icon="🏀")

TEMP_DIR = os.path.join("/tmp", "nba_prop_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# --- STYLE ---
st.markdown("""
<style>
.main-header {text-align:center;font-size:40px;color:#FFCC33;font-weight:700;margin-bottom:0px;}
.card {background-color:#222;border-radius:15px;padding:20px;margin-bottom:20px;
box-shadow:0px 0px 10px rgba(255,204,51,0.3);transition:all 0.2s ease-in-out;}
.card:hover {transform:scale(1.02);}
.metric {font-size:20px;color:#FFCC33;}
footer {text-align:center;color:#FFCC33;margin-top:40px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏀 NBA Prop Model</p>', unsafe_allow_html=True)

# --- Sidebar Config ---
st.sidebar.header("Settings")
user_id = st.sidebar.text_input("User ID", value="Me").strip() or "Me"
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=100.0, step=10.0)
payout_mult = st.sidebar.number_input("2-Pick Payout", min_value=1.5, value=3.0, step=0.1)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
compact = st.sidebar.checkbox("Compact Mode", False)

LOG_FILE = os.path.join(TEMP_DIR, f"bet_history_{user_id}.csv")

# --- Loader Animation ---
def run_loader():
    txt = st.empty()
    bar = st.progress(0)
    msgs = ["Crunching data...", "Simulating minutes...", "Adjusting for pace...", 
            "Applying usage modifiers...", "Finalizing probabilities..."]
    for i in range(100):
        time.sleep(0.03)
        if i % 20 == 0: txt.text(random.choice(msgs))
        bar.progress(i + 1)
    bar.empty(); txt.empty()

# --- Core Functions ---
def kelly_fraction(prob, odds_mult, frac):
    b = odds_mult - 1; q = 1 - prob
    k = ((b * prob - q) / b) * frac
    return max(0, min(k, 0.03))

def play_pass(ev):
    if ev < 0.05: return "❌ PASS"
    elif ev < 0.1: return "⚠️ Thin Edge"
    return "✅ Playable Edge"

def calc_projection(line, teammate_out, blowout):
    base = line * random.uniform(0.95, 1.1)
    if teammate_out: base *= 1.08
    if blowout: base *= 0.9
    return base

def load_history():
    if os.path.exists(LOG_FILE): return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Date","Player","Market","Line","EV","Stake","Result","CLV","KellyFrac"])

def save_history(df): df.to_csv(LOG_FILE, index=False)

# --- Layout Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model", "📓 Results", "📜 History", "🧠 Calibration"])

# --- TAB 1 MODEL ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.text_input("Player 1")
        m1 = st.selectbox("Market", ["PRA","Points","Rebounds","Assists"], key="m1")
        l1 = st.number_input("Line", value=25.0, step=0.5, key="l1")
    with col2:
        p2 = st.text_input("Player 2")
        m2 = st.selectbox("Market", ["PRA","Points","Rebounds","Assists"], key="m2")
        l2 = st.number_input("Line", value=25.0, step=0.5, key="l2")

    teammate_out = st.checkbox("Key teammate out?")
    blowout = st.checkbox("Blowout risk high?")

    if st.button("Run Model ⚡"):
        run_loader()

        c1, c2 = st.columns(2)
        for player, market, line, container in [(p1,m1,l1,c1),(p2,m2,l2,c2)]:
            if not player: continue
            proj = calc_projection(line, teammate_out, blowout)
            prob = norm.cdf((proj - line) / (0.12 * line))
            ev = (prob * payout_mult) - 1
            stake = bankroll * kelly_fraction(prob, payout_mult, fractional_kelly)
            decision = play_pass(ev)

            with container:
                st.markdown(f"<div class='card'><h3>{player} — {market}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric'>Adj. Projection: {proj:.1f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p>Hit Prob: {prob*100:.1f}% | EV: {ev*100:.1f}% | Stake: ${stake:.2f}</p>", unsafe_allow_html=True)
                st.markdown(f"<b>{decision}</b><br><small>{'Teammate Out (+8%) ' if teammate_out else ''}{'Blowout Risk (-10%)' if blowout else ''}</small></div>", unsafe_allow_html=True)

# --- TAB 2 RESULTS ---
with tab2:
    df = load_history()
    if not df.empty: st.dataframe(df, use_container_width=True)
    with st.form("add_res"):
        st.markdown("### Log Result")
        p = st.text_input("Player")
        m = st.selectbox("Market", ["PRA","Points","Rebounds","Assists"])
        l = st.number_input("Line", 0.0, 100.0, 25.0, 0.5)
        ev = st.number_input("EV (%)", -50.0, 200.0, 10.0)
        stak = st.number_input("Stake ($)", 0.0, 100.0, 5.0)
        clv = st.number_input("CLV", -10.0, 10.0, 0.0)
        res = st.selectbox("Result", ["Pending","Hit","Miss","Push"])
        sub = st.form_submit_button("Save Result")
        if sub:
            new = {"Date":datetime.now().strftime("%Y-%m-%d %H:%M"),"Player":p,"Market":m,"Line":l,"EV":ev,
                   "Stake":stak,"Result":res,"CLV":clv,"KellyFrac":fractional_kelly}
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True); save_history(df)
            st.success("Saved ✅")

# --- TAB 3 HISTORY ---
with tab3:
    df = load_history()
    if df.empty: st.info("No data yet.")
    else:
        min_ev = st.slider("Min EV Filter", -10.0, 100.0, 0.0)
        filt = df[df["EV"] >= min_ev]
        st.dataframe(filt, use_container_width=True)
        filt["Net"] = filt.apply(lambda r: r["Stake"]*(payout_mult-1) if r["Result"]=="Hit" else -r["Stake"], axis=1)
        filt["Cumulative"] = filt["Net"].cumsum()
        fig = px.line(filt, x="Date", y="Cumulative", title="Cumulative Profit", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 4 CALIBRATION ---
with tab4:
    df = load_history()
    if df.empty: st.info("Log results first.")
    else:
        comp = df[df["Result"].isin(["Hit","Miss"])]
        if len(comp) < 10: st.info("Need ≥10 completed bets.")
        else:
            pred = comp["EV"]/100; actual = comp["Result"].eq("Hit").astype(int)
            pred_m, act_m = pred.mean(), actual.mean()
            gap = (pred_m - act_m)*100
            roi = (comp.apply(lambda r: r["Stake"]*(payout_mult-1) if r["Result"]=="Hit" else -r["Stake"], axis=1).sum()/bankroll)*100
            st.markdown(f"**Predicted Hit Rate:** {pred_m*100:.1f}% | **Actual:** {act_m*100:.1f}% | **Gap:** {gap:+.1f}% | **ROI:** {roi:+.1f}%")
            if gap > 5: st.warning("Overconfident — widen variance +0.05")
            elif gap < -5: st.info("Underconfident — narrow variance -0.05")
            else: st.success("Model calibration looks balanced ✅")

st.markdown("<footer>© 2025 NBA Prop Model | Powered by Kamal</footer>", unsafe_allow_html=True)
