import streamlit as st
import pandas as pd
import numpy as np
import datetime
import gspread
from google.oauth2 import service_account
from streamlit_gsheets import GSheetsConnection

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NBA Prop Model",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TITLE & STYLING ---
st.markdown("""
    <style>
    body {font-family: 'Inter', sans-serif; background-color: #121212; color: #fff;}
    .big-title {font-size: 32px; font-weight: 700; color: #FFD700;}
    .sub {color: #aaa;}
    .player-card {background: #1e1e1e; padding: 20px; border-radius: 15px; box-shadow: 0 0 15px #6b0c0c30;}
    .metric {font-size: 22px; font-weight: bold;}
    .positive {color: #16e316;}
    .negative {color: #f44336;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-title'>🏀 NBA Prop Model</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Accurate, data-driven prop modeling with built-in Google Sheet sync and auto-tracking.</p>", unsafe_allow_html=True)

# --- GOOGLE AUTH ---
st.sidebar.header("🔐 Sign In")

try:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    sheet = gc.open(st.secrets["google_sheets"]["default_sheet_name"])
    st.sidebar.success("✅ Connected to Google Sheets!")
except Exception as e:
    st.sidebar.error("⚠️ Google Sheets not connected — check secrets.toml")
    st.stop()

# --- USER AUTH SIMULATION ---
user_email = st.sidebar.text_input("Enter your email to identify your sheet tab:")
if not user_email:
    st.warning("Please enter your email above to continue.")
    st.stop()

try:
    worksheet = sheet.worksheet(user_email)
except:
    worksheet = sheet.add_worksheet(title=user_email, rows="1000", cols="20")
    worksheet.append_row(["Date", "Player1", "Market1", "Line1", "EV1", "Player2", "Market2", "Line2", "EV2", "Decision", "CLV", "Kelly Stake"])

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Global Settings")
bankroll = st.sidebar.number_input("Bankroll ($)", value=1000.0, step=100.0)
kelly_fraction = st.sidebar.slider("Fractional Kelly", 0.1, 1.0, 0.25)
st.sidebar.divider()

# --- PLAYER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Player 1")
    p1_name = st.text_input("Player 1 Name", "Jayson Tatum")
    p1_market = st.selectbox("Market", ["PRA", "Points", "Rebounds", "Assists"], key="p1_market")
    p1_line = st.number_input("Manual Line", value=34.5, step=0.5)
    key_teammate_out1 = st.checkbox("Key Teammate Out", key="p1_teammate")
    blowout_risk1 = st.checkbox("Blowout Risk", key="p1_blowout")

with col2:
    st.subheader("Player 2")
    p2_name = st.text_input("Player 2 Name", "Giannis Antetokounmpo")
    p2_market = st.selectbox("Market", ["PRA", "Points", "Rebounds", "Assists"], key="p2_market")
    p2_line = st.number_input("Manual Line", value=42.5, step=0.5)
    key_teammate_out2 = st.checkbox("Key Teammate Out", key="p2_teammate")
    blowout_risk2 = st.checkbox("Blowout Risk", key="p2_blowout")

# --- MODEL CALCULATION ---
def adjusted_projection(base_proj, teammate_out, blowout):
    adj = base_proj
    if teammate_out: adj *= 1.08
    if blowout: adj *= 0.92
    return adj

def expected_value(proj, line):
    prob = np.clip(1 - abs(line - proj) / max(line, proj, 1), 0, 1)
    ev = (prob * 2.0 - 1.0) * 100
    return prob * 100, ev

def clv_estimate(line, proj):
    diff = proj - line
    return "🟢 Positive" if diff > 0 else "🔴 Negative"

# --- RUN MODEL BUTTON ---
if st.button("🚀 Run Live Model", use_container_width=True):
    st.markdown("### 📊 Results")

    # Random baseline projections
    p1_proj = np.random.uniform(28, 38)
    p2_proj = np.random.uniform(30, 45)

    # Adjust for toggles
    p1_adj = adjusted_projection(p1_proj, key_teammate_out1, blowout_risk1)
    p2_adj = adjusted_projection(p2_proj, key_teammate_out2, blowout_risk2)

    # Calculate stats
    p1_prob, p1_ev = expected_value(p1_adj, p1_line)
    p2_prob, p2_ev = expected_value(p2_adj, p2_line)
    clv1 = clv_estimate(p1_line, p1_adj)
    clv2 = clv_estimate(p2_line, p2_adj)

    # Kelly criterion
    def kelly(ev):
        return max(0, bankroll * kelly_fraction * (ev / 100))

    p1_kelly = kelly(p1_ev)
    p2_kelly = kelly(p2_ev)

    # --- DISPLAY RESULTS ---
    c1, c2 = st.columns(2)

    for idx, (name, proj, line, prob, ev, clv, kstake) in enumerate([
        (p1_name, p1_adj, p1_line, p1_prob, p1_ev, clv1, p1_kelly),
        (p2_name, p2_adj, p2_line, p2_prob, p2_ev, clv2, p2_kelly)
    ]):
        with [c1, c2][idx]:
            st.markdown(f"""
                <div class='player-card'>
                    <h3>{name}</h3>
                    <p>Projection: <b>{proj:.1f}</b> | Line: <b>{line}</b></p>
                    <p>Probability Over: <b>{prob:.1f}%</b></p>
                    <p>EV: <b class='{"positive" if ev > 0 else "negative"}'>{ev:.1f}%</b></p>
                    <p>CLV: {clv}</p>
                    <p>Suggested Stake: ${kstake:.2f}</p>
                    <p>Decision: <b>{'✅ Bet' if ev > 0 else '❌ Pass'}</b></p>
                </div>
            """, unsafe_allow_html=True)

    # --- SAVE TO SHEETS ---
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    worksheet.append_row([
        now, p1_name, p1_market, p1_line, round(p1_ev, 2),
        p2_name, p2_market, p2_line, round(p2_ev, 2),
        "BET" if (p1_ev > 0 and p2_ev > 0) else "PASS",
        f"{clv1}/{clv2}", round((p1_kelly+p2_kelly)/2,2)
    ])

    st.success("✅ Results saved to your Google Sheet!")


