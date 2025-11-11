# ============================================
# 🏀 NBA PROP MODEL (Professional Build)
# Auto advanced-stat pulls + heavy-tail engine
# ============================================

import os, time, random, difflib, requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from scipy.stats import norm, gamma
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import (
    PlayerGameLog,
    PlayerDashboardByGeneralSplits,
    LeagueDashTeamStats,
)

# --------------------
# CONFIG & STYLE
# --------------------
st.set_page_config(page_title="NBA Prop Model", layout="wide", page_icon="🏀")
TEMP_DIR = os.path.join("/tmp", "nba_prop_temp"); os.makedirs(TEMP_DIR, exist_ok=True)
PRIMARY_MAROON, GOLD, CARD_BG, BG = "#7A0019", "#FFCC33", "#17131C", "#0D0A12"

st.markdown(f"""
<style>
.stApp {{background-color:{BG};color:white;font-family:system-ui,sans-serif;}}
.main-header {{text-align:center;font-size:40px;color:{GOLD};font-weight:700;margin-bottom:0px;}}
.card {{background:{CARD_BG};border-radius:18px;padding:15px;margin-bottom:18px;
border:1px solid {GOLD}33;box-shadow:0 10px 25px rgba(0,0,0,0.6);transition:all .15s ease;}}
.card:hover {{transform:translateY(-3px) scale(1.015);}}
.metric {{font-size:18px;color:{GOLD};font-weight:600;}}
.adv {{font-size:13px;color:#e8e8e8;margin-left:6px;}}
.rec-play {{color:#4CAF50;font-weight:700;}}
.rec-thin {{color:#FFC107;font-weight:700;}}
.rec-pass {{color:#F44336;font-weight:700;}}
footer {{text-align:center;color:{GOLD};margin-top:25px;font-size:11px;}}
</style>""", unsafe_allow_html=True)
st.markdown('<p class="main-header">🏀 NBA Prop Model</p>', unsafe_allow_html=True)

# --------------------
# SIDEBAR SETTINGS
# --------------------
st.sidebar.header("Settings")
user_id = st.sidebar.text_input("User ID", value="Me").strip() or "Me"
bankroll = st.sidebar.number_input("Bankroll ($)", 10.0, 100000.0, 100.0, 10.0)
payout_mult = st.sidebar.number_input("2-Pick Payout", 1.5, 5.0, 3.0, 0.1)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
compact_mode = st.sidebar.checkbox("Compact Mode", False)
LOG_FILE = os.path.join(TEMP_DIR, f"bet_history_{user_id}.csv")

# --------------------
# UTILS
# --------------------
def _norm_name(n): return n.lower().replace(".", "").replace("'", "").replace("-", " ").strip()
@st.cache_data(ttl=900)
def all_players(): return nba_players.get_players()
def resolve_player(name):
    if not name: return None, None
    players = all_players(); target=_norm_name(name)
    for p in players:
        if _norm_name(p["full_name"])==target: return p["id"],p["full_name"]
    names=[_norm_name(p["full_name"]) for p in players]
    best=difflib.get_close_matches(target,names,n=1,cutoff=0.7)
    if best:
        for p in players:
            if _norm_name(p["full_name"])==best[0]: return p["id"],p["full_name"]
    return None,None
def headshot_url(pid): return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{pid}.png"

# --------------------
# TEAM CONTEXT
# --------------------
@st.cache_data(ttl=900)
def team_context():
    try:
        t=LeagueDashTeamStats(season="2024-25",per_mode_detailed="PerGame").get_data_frames()[0]
        pace=t[["TEAM_ABBREVIATION","PACE"]].set_index("TEAM_ABBREVIATION")["PACE"].to_dict()
        defr=t[["TEAM_ABBREVIATION","DEF_RATING"]].set_index("TEAM_ABBREVIATION")["DEF_RATING"].to_dict()
        lp,ld=np.mean(list(pace.values())),np.mean(list(defr.values()))
        ctx={k:{"PACE":pace[k],"DEF":defr[k]} for k in pace}
        return ctx,lp,ld
    except Exception: return {},100,110
TEAM_CTX,LEAGUE_PACE,LEAGUE_DEF=team_context()
def ctx_mult(opp):
    if not opp or opp not in TEAM_CTX: return 1.0
    p, d = TEAM_CTX[opp]["PACE"]/LEAGUE_PACE, LEAGUE_DEF/TEAM_CTX[opp]["DEF"]
    return float(np.clip((p*d)**0.5,0.85,1.15))

# --------------------
# ADVANCED STATS
# --------------------
def adv_stats(pid):
    try:
        dash=PlayerDashboardByGeneralSplits(player_id=pid,season="2024-25",per_mode_detailed="PerGame").get_data_frames()[0]
        return dict(
            MIN=float(dash["MIN"].iloc[0]),
            USG=float(dash["USG_PCT"].iloc[0])*100,
            REB=float(dash["REB_PCT"].iloc[0])*100,
            AST=float(dash["AST_PCT"].iloc[0])*100,
            FGA=float(dash["FGA"].iloc[0]),
            FTA=float(dash["FTA"].iloc[0]),
        )
    except Exception: return {}
def last_games(pid,n=10,market="PRA"):
    try:
        df=PlayerGameLog(player_id=pid,season="2024-25").get_data_frames()[0]
    except Exception: return None
    if df.empty: return None
    df["GAME_DATE"]=pd.to_datetime(df["GAME_DATE"])
    df=df.sort_values("GAME_DATE",ascending=False).head(n)
    if market=="PRA": df["VAL"]=df["PTS"]+df["REB"]+df["AST"]
    elif market=="Points": df["VAL"]=df["PTS"]
    elif market=="Rebounds": df["VAL"]=df["REB"]
    else: df["VAL"]=df["AST"]
    return float(df["VAL"].mean()), float(df["VAL"].std()), df

# --------------------
# PROJECTION ENGINE
# --------------------
def heavy_tail_prob(line,mu,sd):
    shape=(mu/sd)**2; scale=(sd**2)/mu
    p=1-gamma.cdf(line,shape,scale=scale)
    return float(np.clip(p,0.02,0.98))
def projection(player,market,line,opp,to,bo):
    pid,name=resolve_player(player)
    if not pid: return None,f"No player found for {player}"
    base,spread,_=last_games(pid,10,market) or (None,None,None)
    if base is None: return None,f"No recent data for {name}"
    a=adv_stats(pid); pace=ctx_mult(opp.strip().upper() if opp else None)
    mu,sd=base*pace,spread*pace if spread>0 else max(1.0,0.1*base)
    if to: mu*=1.07
    if bo: mu*=0.9
    p_over=heavy_tail_prob(line,mu,sd)
    ev_leg=p_over*1-(1-p_over)
    return dict(pid=pid,name=name,market=market,line=line,mu=mu,sd=sd,
                p=p_over,ev=ev_leg,adv=a,opp=opp,teammate=to,blow=bo),None

# --------------------
# KELLY / STAKE
# --------------------
def kelly(prob,payout,f=0.25): b=payout-1;q=1-prob;k=(b*prob-q)/b*f;return float(np.clip(k,0,0.03))
def decision(ev): return "✅ PLAY" if ev>=0.1 else "⚠️ Thin Edge" if ev>=0.05 else "❌ PASS"

# --------------------
# UI COMPONENTS
# --------------------
def loader():
    t=st.empty();b=st.progress(0)
    msgs=["Fetching advanced stats...","Blending pace & defense...",
          "Simulating distribution...","Sizing optimal Kelly stake..."]
    for i in range(100):
        time.sleep(0.02)
        if i%25==0:t.text(random.choice(msgs))
        b.progress(i+1)
    t.empty();b.empty()

def leg_card(leg,col):
    url=headshot_url(leg["pid"])
    with col:
        st.markdown("<div class='card'>",unsafe_allow_html=True)
        c1,c2=st.columns([1,3])
        with c1:
            if url: st.image(url,use_column_width=True)
        with c2:
            st.markdown(f"**{leg['name']} — {leg['market']}**")
            st.markdown(f"<span class='metric'>Line {leg['line']:.1f} | Model {leg['mu']:.1f}</span>",unsafe_allow_html=True)
        st.markdown(f"Hit Prob (Over): **{leg['p']*100:.1f}%**  |  EV: **{leg['ev']*100:+.1f}%**")
        st.markdown(f"<div class='adv'>USG {leg['adv'].get('USG','–'):.1f}% | REB% {leg['adv'].get('REB','–'):.1f} | AST% {leg['adv'].get('AST','–'):.1f} | FGA {leg['adv'].get('FGA','–'):.1f} | FTA {leg['adv'].get('FTA','–'):.1f}</div>",unsafe_allow_html=True)
        if leg['opp']: st.markdown(f"<div class='adv'>Opponent {leg['opp']} | Context mult {ctx_mult(leg['opp']):.3f}</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='adv'>{'Teammate out (+7%) ' if leg['teammate'] else ''}{'Blowout risk (−10%)' if leg['blow'] else ''}</div>",unsafe_allow_html=True)
        rec=decision(leg['ev']);cls='rec-play' if 'PLAY' in rec else 'rec-thin' if 'Thin' in rec else 'rec-pass'
        st.markdown(f"<div class='{cls}'>{rec}</div></div>",unsafe_allow_html=True)

# --------------------
# HISTORY LOG
# --------------------
def ensure_hist():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=["Date","Player","Market","Line","EV","Stake","Result","CLV","Kelly"]).to_csv(LOG_FILE,index=False)
def load_hist():
    ensure_hist();return pd.read_csv(LOG_FILE)
def save_hist(df): df.to_csv(LOG_FILE,index=False)

# --------------------
# MAIN APP TABS
# --------------------
tab1,tab2,tab3,tab4=st.tabs(["📊 Model","📓 Results","📜 History","🧠 Calibration"])

# ---- MODEL ----
with tab1:
    st.subheader("Auto Advanced Projection & EV Engine")
    c1,c2=st.columns(2)
    with c1:
        p1=st.text_input("Player 1");m1=st.selectbox("Market 1",["PRA","Points","Rebounds","Assists"])
        l1=st.number_input("Line 1",0.0,100.0,25.0,0.5);o1=st.text_input("Opponent 1 (abbr)");t1=st.checkbox("Key teammate out 1");b1=st.checkbox("Blowout risk 1")
    with c2:
        p2=st.text_input("Player 2");m2=st.selectbox("Market 2",["PRA","Points","Rebounds","Assists"])
        l2=st.number_input("Line 2",0.0,100.0,25.0,0.5);o2=st.text_input("Opponent 2 (abbr)");t2=st.checkbox("Key teammate out 2");b2=st.checkbox("Blowout risk 2")
    if st.button("Run Model ⚡"):
        loader()
        leg1,err1=projection(p1,m1,l1,o1,t1,b1)
        leg2,err2=projection(p2,m2,l2,o2,t2,b2)
        cL,cR=st.columns(2)
        if leg1: leg_card(leg1,cL)
        if leg2: leg_card(leg2,cR)
        if leg1 and leg2:
            corr=0.25 if leg1["opp"]==leg2["opp"] else 0
            joint=leg1["p"]*leg2["p"]+corr*(min(leg1["p"],leg2["p"])-leg1["p"]*leg2["p"])
            ev_combo=payout_mult*joint-1;kf=kelly(joint,payout_mult,fractional_kelly);stake=bankroll*kf
            st.markdown(f"### 🎯 2-Pick Combo")
            st.markdown(f"Joint prob **{joint*100:.1f}%**  |  EV **{ev_combo*100:+.1f}%**  |  Stake ${stake:.2f}")
            st.markdown(f"Recommendation: **{decision(ev_combo)}**")

# (results, history, calibration sections continue…)
# ---- RESULTS ----
with tab2:
    st.subheader("Results & Personal Tracking")
    df = load_hist()
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No bets logged yet. Log after you place entries.")
    with st.form("log_form"):
        c1, c2, c3 = st.columns(3)
        with c1: rp = st.text_input("Player / Combo")
        with c2: rm = st.selectbox("Market", ["PRA","Points","Rebounds","Assists","Combo"])
        with c3: rl = st.number_input("Line", 0.0, 200.0, 25.0, 0.5)
        c4, c5, c6 = st.columns(3)
        with c4: rev = st.number_input("Model EV (%)", -50.0, 200.0, 5.0)
        with c5: rstake = st.number_input("Stake ($)", 0.0, 10000.0, 5.0)
        with c6: rclv = st.number_input("CLV", -20.0, 20.0, 0.0)
        rres = st.selectbox("Result", ["Pending","Hit","Miss","Push"])
        sub = st.form_submit_button("Save Result")
        if sub:
            ensure_hist()
            new = {"Date":datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "Player":rp,"Market":rm,"Line":rl,"EV":rev,
                   "Stake":rstake,"Result":rres,"CLV":rclv,"Kelly":fractional_kelly}
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
            save_hist(df)
            st.success("Saved ✅")
    df = load_hist()
    comp = df[df["Result"].isin(["Hit","Miss"])]
    if not comp.empty:
        pnl = comp.apply(lambda r: r["Stake"]*(payout_mult-1) if r["Result"]=="Hit" else -r["Stake"], axis=1)
        hr = (comp["Result"]=="Hit").mean()*100
        roi = pnl.sum()/max(1,bankroll)*100
        st.markdown(f"**Completed:** {len(comp)} | **Hit Rate:** {hr:.1f}% | **ROI:** {roi:+.1f}%")
        comp["Net"]=pnl; comp["Cum"]=comp["Net"].cumsum()
        st.plotly_chart(px.line(comp,x="Date",y="Cum",title="Cumulative Profit",markers=True),use_container_width=True)

# ---- HISTORY ----
with tab3:
    st.subheader("Bet History and Filters")
    df = load_hist()
    if df.empty:
        st.info("No logged bets yet.")
    else:
        min_ev = st.slider("Min EV (%)", -20.0, 100.0, 0.0)
        filt = df[df["EV"] >= min_ev]
        st.markdown(f"**Filtered bets:** {len(filt)}")
        st.dataframe(filt,use_container_width=True)
        if not filt.empty:
            filt["Net"] = filt.apply(lambda r: r["Stake"]*(payout_mult-1) if r["Result"]=="Hit" else (-r["Stake"] if r["Result"]=="Miss" else 0),axis=1)
            filt["Cumulative"] = filt["Net"].cumsum()
            st.plotly_chart(px.line(filt,x="Date",y="Cumulative",title="Cumulative Profit (Filtered)",markers=True),use_container_width=True)

# ---- CALIBRATION ----
with tab4:
    st.subheader("Calibration & Edge Integrity Check")
    df = load_hist()
    comp = df[df["Result"].isin(["Hit","Miss"])]
    if len(comp) < 15:
        st.info("Log ≥15 completed bets to start calibration.")
    else:
        comp["EVf"] = pd.to_numeric(comp["EV"], errors="coerce")/100
        comp = comp.dropna(subset=["EVf"])
        pred_win = 0.5 + comp["EVf"].mean()
        act_win = (comp["Result"]=="Hit").mean()
        gap = (pred_win - act_win)*100
        pnl = comp.apply(lambda r: r["Stake"]*(payout_mult-1) if r["Result"]=="Hit" else -r["Stake"], axis=1)
        roi = pnl.sum()/max(1,bankroll)*100
        st.markdown(f"**Predicted Win:** {pred_win*100:.1f}% | **Actual:** {act_win*100:.1f}% | **Gap:** {gap:+.1f}% | **ROI:** {roi:+.1f}%")
        if gap > 5: st.warning("Model overconfident → increase variance.")
        elif gap < -5: st.info("Model conservative → reduce variance.")
        else: st.success("Model and results aligned ✅")

st.markdown("<footer>© 2025 NBA Prop Model | Powered by Kamal</footer>",unsafe_allow_html=True)
