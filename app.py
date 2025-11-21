# =========================================================
#  NBA QUANT ENGINE — TIER C STREAMLIT APPLICATION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

from engine.odds_api_client import fetch_odds
from engine.normalize import normalize_props
from engine.ensemble import ensemble_projection
from engine.injuries import injury_context
from engine.defensive import defensive_context
from engine.lineup import lineup_context
from engine.game_scripts import simulate_game_script
from engine.covariance_mc import estimate_correlation, joint_mc
from engine.scanner import scan_edges
from engine.clv_tracker import record_am_line, compute_clv
from engine.calibration import calibration_engine

st.set_page_config(page_title="NBA Quant Engine (Tier C)", layout="wide")

st.title("🏀 NBA Quant Engine — Tier C")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard", "Live Scanner", "Model", "2-Leg Simulator", "CLV", "Calibration"
])

# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------
with tab1:
    st.header("📊 Live Odds Dashboard")
    if st.button("Fetch Odds"):
        odds_raw = fetch_odds()
        if odds_raw:
            df = normalize_props(odds_raw)
            st.dataframe(df)

# ---------------------------------------------------------
# Live Scanner
# ---------------------------------------------------------
with tab2:
    st.header("🔍 Live Edge Scanner (>65% EV)")
    if st.button("Run Scanner"):
        odds_raw = fetch_odds()
        if odds_raw:
            df = normalize_props(odds_raw)

            def ctx_func(row):
                inj = injury_context(row['player'], row['home'])
                return {"ctx_mult": inj['ctx_mult'], "gs_prob": 0.5}

            def ens(samples, line, ctx_mult, gs_prob):
                return ensemble_projection(samples, line, ctx_mult, gs_prob)

            edges = scan_edges(df, ens, ctx_func, threshold=0.65)
            st.dataframe(edges)

# ---------------------------------------------------------
# Model Runner
# ---------------------------------------------------------
with tab3:
    st.header("🧠 Model Projection")
    player = st.text_input("Player")
    market = st.selectbox("Market", ["Points","Rebounds","Assists","PRA"])
    line = st.number_input("Line", 0.0)

    if st.button("Run Model"):
        samples = np.random.normal(line, 4, size=30)
        ctx_mult = 1.0
        prob = ensemble_projection(samples, line, ctx_mult, 0.5)
        st.write(f"Projected Prob: {prob:.3f}")

# ---------------------------------------------------------
# 2-Leg Simulator
# ---------------------------------------------------------
with tab4:
    st.header("🎲 2-Leg Correlation Monte Carlo")
    p1 = st.number_input("Leg1 Prob", 0.0, 1.0, 0.5)
    p2 = st.number_input("Leg2 Prob", 0.0, 1.0, 0.5)
    mu1 = st.number_input("mu1", 0.0)
    sd1 = st.number_input("sd1", 1.0)
    mu2 = st.number_input("mu2", 0.0)
    sd2 = st.number_input("sd2", 1.0)
    l1 = st.number_input("Line1", 0.0)
    l2 = st.number_input("Line2", 0.0)

    if st.button("Simulate"):
        corr = estimate_correlation({"team":"A","market":"Points"},{"team":"A","market":"Points"})
        res = joint_mc(mu1,sd1,l1,mu2,sd2,l2,corr)
        st.write(res)

# ---------------------------------------------------------
# CLV
# ---------------------------------------------------------
with tab5:
    st.header("💰 Closing Line Value")
    pl = st.text_input("Player (CLV)")
    mk = st.text_input("Market (CLV)")
    am_line = st.number_input("AM Line")
    model_am = st.number_input("Model AM Prob", 0.0,1.0)
    cur_line = st.number_input("Current Line")
    model_now = st.number_input("Model Now Prob",0.0,1.0)

    if st.button("Record AM Line"):
        record_am_line(pl,mk,am_line,model_am)
        st.success("AM Line Stored.")

    if st.button("Compute CLV"):
        res = compute_clv(pl,mk,cur_line,model_now)
        st.write(res)

# ---------------------------------------------------------
# Calibration
# ---------------------------------------------------------
with tab6:
    st.header("📏 Model Calibration")
    probs = st.text_input("Pred Probs CSV")
    outs = st.text_input("Outcomes CSV")

    if st.button("Calibrate"):
        try:
            pp = [float(x) for x in probs.split(',')]
            oo = [float(x) for x in outs.split(',')]
            res = calibration_engine(pp,oo)
            st.write(res)
        except:
            st.error("Invalid input.")
