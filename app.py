import os, time, random, difflib
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import norm

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(page_title="NBA Prop Model", page_icon="🏀", layout="wide")

TEMP_DIR = os.path.join("/tmp", "nba_prop_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

PRIMARY_MAROON = "#7A0019"
GOLD = "#FFCC33"
CARD_BG = "#17131C"
BG = "#0D0A12"

# =====================================================
# STYLE
# =====================================================

st.markdown(
    f"""
    <style>
    .main-header {{
        text-align:center;
        font-size:40px;
        color:{GOLD};
        font-weight:700;
        margin-bottom:0px;
    }}
    .card {{
        background-color:{CARD_BG};
        border-radius:18px;
        padding:14px;
        margin-bottom:18px;
        border:1px solid {GOLD}33;
        box-shadow:0 10px 24px rgba(0,0,0,0.75);
        transition:all 0.16s ease-in-out;
    }}
    .card:hover {{
        transform:translateY(-3px) scale(1.015);
        box-shadow:0 18px 40px rgba(0,0,0,0.9);
    }}
    .rec-play {{color:#4CAF50;font-weight:700;}}
    .rec-thin {{color:#FFC107;font-weight:700;}}
    .rec-pass {{color:#F44336;font-weight:700;}}
    .stApp {{
        background-color:{BG};
        color:white;
        font-family:system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
    }}
    section[data-testid="stSidebar"] {{
        background: radial-gradient(circle at top,{PRIMARY_MAROON} 0%,#2b0b14 55%,#12060a 100%);
        border-right:1px solid {GOLD}33;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🏀 NBA Prop Model</p>', unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("User & Bankroll")
user_id = st.sidebar.text_input("Your ID (for personal history)", "Me").strip() or "Me"
LOG_FILE = os.path.join(TEMP_DIR, f"bet_history_{user_id}.csv")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=100.0)
payout_mult = st.sidebar.number_input("2-Pick Payout (e.g. 3.0x)", min_value=1.5, value=3.0)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Recent Games Sample (N)", 5, 20, 10)
compact_mode = st.sidebar.checkbox("Compact Mode (mobile)", value=False)

# =====================================================
# HELPERS / GLOBAL CONFIG
# =====================================================

MARKET_OPTIONS = ["PRA", "Points", "Rebounds", "Assists"]
MARKET_METRICS = {"PRA": ["PTS","REB","AST"], "Points":["PTS"], "Rebounds":["REB"], "Assists":["AST"]}
HEAVY_TAIL = {"PRA":1.35,"Points":1.25,"Rebounds":1.25,"Assists":1.25}
MAX_KELLY_PCT = 0.03

def current_season():
    today = datetime.now()
    yr = today.year if today.month >= 10 else today.year - 1
    return f"{yr}-{str(yr+1)[-2:]}"

@st.cache_data(show_spinner=False)
def get_players_index(): return nba_players.get_players()

def _norm_name(s): return s.lower().replace(".", "").replace("'", "").replace("-", " ").strip()

@st.cache_data(show_spinner=False)
def resolve_player(name):
    if not name: return None, None
    players = get_players_index()
    target = _norm_name(name)
    for p in players:
        if _norm_name(p["full_name"]) == target:
            return p["id"], p["full_name"]
    names = [_norm_name(p["full_name"]) for p in players]
    best = difflib.get_close_matches(target, names, n=1, cutoff=0.7)
    if best:
        for p in players:
            if _norm_name(p["full_name"]) == best[0]:
                return p["id"], p["full_name"]
    return None, None

def get_headshot_url(name):
    pid, _ = resolve_player(name)
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{pid}.png" if pid else None

# =====================================================
# TEAM CONTEXT (ADVANCED MATCHUP METRICS)
# =====================================================

@st.cache_data(show_spinner=False, ttl=3600)
def get_team_context():
    try:
        # Standard per-game and advanced defensive stats
        base = LeagueDashTeamStats(season=current_season(), per_mode_detailed="PerGame").get_data_frames()[0]
        adv = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed="Advanced",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][["TEAM_ID","TEAM_ABBREVIATION","REB_PCT","OREB_PCT","DREB_PCT","AST_PCT","PACE"]]
        defn = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][["TEAM_ID","TEAM_ABBREVIATION","DEF_RATING"]]

        # Merge and calculate league averages
        df = base.merge(adv,on=["TEAM_ID","TEAM_ABBREVIATION"],how="left")
        df = df.merge(defn,on=["TEAM_ID","TEAM_ABBREVIATION"],how="left")

        league_avg = {col:df[col].mean() for col in ["PACE","DEF_RATING","REB_PCT","AST_PCT"]}
        ctx = {
            r["TEAM_ABBREVIATION"]: {
                "PACE":r["PACE"],
                "DEF_RATING":r["DEF_RATING"],
                "REB_PCT":r["REB_PCT"],
                "DREB_PCT":r["DREB_PCT"],
                "AST_PCT":r["AST_PCT"],
            }
            for _,r in df.iterrows()
        }
        return ctx, league_avg
    except Exception:
        return {}, {}

TEAM_CTX, LEAGUE_CTX = get_team_context()


def get_context_multiplier(opp_abbrev, market="PRA"):
    """Adjust player projections for opponent context."""
    if not opp_abbrev or opp_abbrev not in TEAM_CTX or not LEAGUE_CTX:
        return 1.0
    opp = TEAM_CTX[opp_abbrev]
    pace_f = opp["PACE"]/LEAGUE_CTX["PACE"]
    def_f = LEAGUE_CTX["DEF_RATING"]/opp["DEF_RATING"]
    reb_adj = LEAGUE_CTX["REB_PCT"]/opp["DREB_PCT"] if market=="Rebounds" else 1.0
    ast_adj = LEAGUE_CTX["AST_PCT"]/opp["AST_PCT"] if market=="Assists" else 1.0
    mult = (0.4*pace_f)+(0.3*def_f)+(0.3*(reb_adj if market=="Rebounds" else ast_adj))
    return float(np.clip(mult,0.8,1.2))

# =====================================================
# HISTORY HELPERS (moved up to fix NameError)
# =====================================================

def ensure_history():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV","Stake","Result","CLV","KellyFrac"
        ]).to_csv(LOG_FILE,index=False)

def load_history():
    ensure_history()
    try: return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV","Stake","Result","CLV","KellyFrac"
        ])

def save_history(df): df.to_csv(LOG_FILE,index=False)

# =====================================================
# PLAYER MODEL LOGIC
# =====================================================

@st.cache_data(show_spinner=False, ttl=900)
def get_player_rate_and_minutes(name,n_games,market):
    pid,label=resolve_player(name)
    if not pid: return None,None,None,None,f"No match for '{name}'."
    try:
        gl=PlayerGameLog(player_id=pid,season=current_season(),season_type_all_star="Regular Season").get_data_frames()[0]
    except Exception as e:
        return None,None,None,None,f"Log error: {e}"
    if gl.empty: return None,None,None,None,"No recent games."
    gl["GAME_DATE"]=pd.to_datetime(gl["GAME_DATE"])
    gl=gl.sort_values("GAME_DATE",ascending=False).head(n_games)
    cols=MARKET_METRICS[market]; per_min,mins=[],[]
    for _,r in gl.iterrows():
        m=0
        try:
            m_str=r.get("MIN","0")
            if isinstance(m_str,str) and ":" in m_str:
                mm,ss=m_str.split(":"); m=float(mm)+float(ss)/60
            else: m=float(m_str)
        except: m=0
        if m<=0: continue
        val=sum(float(r.get(c,0)) for c in cols)
        per_min.append(val/m); mins.append(m)
    if not per_min: return None,None,None,None,"Insufficient data."
    per_min,mins=np.array(per_min),np.array(mins)
    mu_per_min=float(np.mean(per_min))
    avg_min=float(np.mean(mins))
    sd_per_min=max(np.std(per_min,ddof=1),0.15*max(mu_per_min,0.5))
    team=gl["TEAM_ABBREVIATION"].mode().iloc[0]
    return mu_per_min,sd_per_min,avg_min,team,f"{label}: {len(per_min)} games • {avg_min:.1f} min"

def compute_leg_projection(player,market,line,opp,teammate_out,blowout,n_games):
    mu_min,sd_min,avg_min,team,msg=get_player_rate_and_minutes(player,n_games,market)
    if mu_min is None: return None,msg
    ctx_mult=get_context_multiplier(opp.strip().upper() if opp else None,market)
    ht=HEAVY_TAIL[market]; minutes=avg_min
    if teammate_out: minutes*=1.05; mu_min*=1.06
    if blowout: minutes*=0.9
    mu=mu_min*minutes*ctx_mult
    sd=max(1.0,sd_min*np.sqrt(max(minutes,1.0))*ht)
    p_over=1.0-norm.cdf(line,mu,sd); p_over=float(np.clip(p_over,0.05,0.95))
    ev_leg_even=p_over-(1-p_over)
    return {
        "player":player,"market":market,"line":line,"mu":mu,"sd":sd,
        "prob_over":p_over,"ev_leg_even":ev_leg_even,"team":team,
        "ctx_mult":ctx_mult,"msg":msg,"teammate_out":teammate_out,"blowout":blowout
    },None

def kelly_for_combo(p_joint,payout_mult,frac):
    b=payout_mult-1; q=1-p_joint; raw=(b*p_joint-q)/b; return float(np.clip(raw*frac,0,MAX_KELLY_PCT))

# =====================================================
# APP TABS
# =====================================================

tab_model, tab_results, tab_history, tab_calib = st.tabs(["📊 Model","📓 Results","📜 History","🧠 Calibration"])

# (→ You can paste your same UI + results/history/calibration sections from your working file below this point)

# =========================
# MODEL TAB
# =========================

with tab_model:
    st.subheader("2-Pick Projection & Edge (Auto stats, manual lines)")

    c1, c2 = st.columns(2)

    with c1:
        p1 = st.text_input("Player 1 Name", key="p1_name")
        m1 = st.selectbox("P1 Market", MARKET_OPTIONS, key="p1_market")
        l1 = st.number_input("P1 Line", min_value=0.0, value=25.0, step=0.5, key="p1_line")
        o1 = st.text_input("P1 Opponent (abbr, optional)", key="p1_opp", help="e.g. BOS, DEN")
        p1_teammate_out = st.checkbox("P1: Key teammate out?", key="p1_to")
        p1_blowout = st.checkbox("P1: Blowout risk high?", key="p1_bo")

    with c2:
        p2 = st.text_input("Player 2 Name", key="p2_name")
        m2 = st.selectbox("P2 Market", MARKET_OPTIONS, key="p2_market")
        l2 = st.number_input("P2 Line", min_value=0.0, value=25.0, step=0.5, key="p2_line")
        o2 = st.text_input("P2 Opponent (abbr, optional)", key="p2_opp", help="e.g. BOS, DEN")
        p2_teammate_out = st.checkbox("P2: Key teammate out?", key="p2_to2")
        p2_blowout = st.checkbox("P2: Blowout risk high?", key="p2_bo2")

    run = st.button("Run Model ⚡")

    if run:
        if payout_mult <= 1.0:
            st.error("Payout multiplier must be greater than 1.0")
        else:
            run_loader()

            leg1, err1 = (compute_leg_projection(p1, m1, l1, o1, p1_teammate_out, p1_blowout, games_lookback)
                          if p1 and l1 > 0 else (None, None))
            leg2, err2 = (compute_leg_projection(p2, m2, l2, o2, p2_teammate_out, p2_blowout, games_lookback)
                          if p2 and l2 > 0 else (None, None))

            if err1:
                st.error(f"P1: {err1}")
            if err2:
                st.error(f"P2: {err2}")

            col_L, col_R = st.columns(2)
            if leg1:
                render_leg_card(leg1, col_L, compact_mode)
            if leg2:
                render_leg_card(leg2, col_R, compact_mode)

            if leg1 and leg2:
                corr = 0.0
                if leg1["team"] and leg2["team"] and leg1["team"] == leg2["team"]:
                    corr = 0.25  # same-team positive correlation
                base_joint = leg1["prob_over"] * leg2["prob_over"]
                joint = base_joint + corr * (min(leg1["prob_over"], leg2["prob_over"]) - base_joint)
                joint = float(np.clip(joint, 0.0, 1.0))

                ev_combo = payout_mult * joint - 1.0
                k_frac = kelly_for_combo(joint, payout_mult, fractional_kelly)
                stake = round(bankroll * k_frac, 2)
                decision = combo_decision(ev_combo)

                st.markdown("### 🎯 2-Pick Combo (Both Must Hit)")
                st.markdown(f"- Corr adj: **{corr:+.2f}** (same team boosts correlation)" if corr else "- Corr adj: **0.00**")
                st.markdown(f"- Joint Hit Probability: **{joint*100:.1f}%**")
                st.markdown(f"- EV on 2-pick: **{ev_combo*100:+.1f}%** per $1")
                st.markdown(f"- Suggested Stake (Kelly-capped): **${stake:.2f}**")
                st.markdown(f"- Recommendation: **{decision}**")
# =====================================================
# Implied Probability & Market Benchmarking
# =====================================================
def implied_probability(payout_mult: float) -> float:
    """
    Converts a PrizePicks-style payout multiplier into an implied probability.
    Example: 3.0x 2-pick = 1/3.0 ≈ 33.3% implied win probability.
    """
    try:
        if payout_mult <= 1:
            return None
        return 1.0 / payout_mult
    except Exception:
        return None

imp_prob = implied_probability(payout_mult)
if imp_prob:
    st.markdown(f"- Market implied win probability per leg: **{imp_prob*100:.1f}%**")
    if leg1 and leg2:
        model_avg_prob = (leg1["prob_over"] + leg2["prob_over"]) / 2
        edge_vs_market = (model_avg_prob - imp_prob) * 100
        st.markdown(f"- Model avg prob vs market implied: **{model_avg_prob*100:.1f}% vs {imp_prob*100:.1f}%**")
        st.markdown(f"- Edge vs market: **{edge_vs_market:+.2f}%**")
        if edge_vs_market > 0:
            st.success("✅ Model edge detected over market pricing.")
        else:
            st.warning("⚠️ Model below market — market expects better performance.")

# =====================================================
# Update Market Library with current entries
# =====================================================
if p1 and l1 > 0:
    update_market_library(p1, m1, l1)
if p2 and l2 > 0:
    update_market_library(p2, m2, l2)

# Display baselines if available
for leg in [leg1, leg2]:
    if leg:
        mean_b, med_b = get_market_baseline(leg["player"], leg["market"])
        if mean_b:
            st.caption(f"📊 Historical market average for {leg['player']} {leg['market']}: mean={mean_b:.1f}, median={med_b:.1f}")

# =========================
# RESULTS TAB
# =========================

with tab_results:
    st.subheader("Results & Personal Tracking")

    df = load_history()
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No bets logged yet. Log after you place entries.")

    with st.form("log_result_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            r_player = st.text_input("Player / Combo")
        with c2:
            r_market = st.selectbox("Market", ["PRA", "Points", "Rebounds", "Assists", "Combo"])
        with c3:
            r_line = st.number_input("Line", 0.0, 200.0, 25.0, 0.5)

        c4, c5, c6 = st.columns(3)
        with c4:
            r_ev = st.number_input("Model EV (%)", -50.0, 200.0, 5.0, 0.1)
        with c5:
            r_stake = st.number_input("Stake ($)", 0.0, 10000.0, 5.0, 0.5)
        with c6:
            r_clv = st.number_input("CLV (Closing - Entry)", -20.0, 20.0, 0.0, 0.1)

        r_result = st.selectbox("Result", ["Pending", "Hit", "Miss", "Push"])
        submit_res = st.form_submit_button("Log Result")

        if submit_res:
            ensure_history()
            new_row = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Player": r_player,
                "Market": r_market,
                "Line": r_line,
                "EV": r_ev,
                "Stake": r_stake,
                "Result": r_result,
                "CLV": r_clv,
                "KellyFrac": fractional_kelly,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_history(df)
            st.success("Result logged ✅")

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]
    if not comp.empty:
        pnl = comp.apply(
            lambda r: r["Stake"] * (payout_mult - 1.0)
            if r["Result"] == "Hit"
            else -r["Stake"],
            axis=1,
        )
        hits = (comp["Result"] == "Hit").sum()
        total = len(comp)
        hit_rate = hits / total * 100 if total > 0 else 0.0
        roi = pnl.sum() / max(1.0, bankroll) * 100

        st.markdown(
            f"**Completed Bets:** {total} | **Hit Rate:** {hit_rate:.1f}% | **ROI:** {roi:+.1f}%"
        )

        trend = comp.copy()
        trend["Profit"] = pnl.values
        trend["Cumulative"] = trend["Profit"].cumsum()
        fig = px.line(
            trend,
            x="Date",
            y="Cumulative",
            title="Cumulative Profit (All Logged Bets)",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================
# HISTORY TAB
# =========================

with tab_history:
    st.subheader("History & Filters")

    df = load_history()
    if df.empty:
        st.info("No logged bets yet.")
    else:
        min_ev = st.slider("Min EV (%) filter", -20.0, 100.0, 0.0, 1.0)
        market_filter = st.selectbox(
            "Market filter",
            ["All", "PRA", "Points", "Rebounds", "Assists", "Combo"],
            index=0,
        )

        filt = df[df["EV"] >= min_ev]

        if market_filter != "All":
            filt = filt[filt["Market"] == market_filter]

        st.markdown(f"**Filtered Bets:** {len(filt)}")
        st.dataframe(filt, use_container_width=True)

        if not filt.empty:
            filt = filt.copy()
            filt["Net"] = filt.apply(
                lambda r: r["Stake"] * (payout_mult - 1.0)
                if r["Result"] == "Hit"
                else (-r["Stake"] if r["Result"] == "Miss" else 0.0),
                axis=1,
            )
            filt["Cumulative"] = filt["Net"].cumsum()
            fig = px.line(
                filt,
                x="Date",
                y="Cumulative",
                title="Cumulative Profit (Filtered View)",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================
# CALIBRATION TAB
# =========================

with tab_calib:
    st.subheader("Calibration & Edge Integrity Check")

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]
    if comp.empty or len(comp) < 15:
        st.info("Log at least 15 completed bets with EV to start calibration.")
    else:
        comp = comp.copy()
        comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
comp = comp.dropna(subset=["EV_float"])

if comp.empty:
    st.info("No valid EV values yet.")
else:
    # Approx predicted win prob from EV around 50% baseline
    pred_win_prob = 0.5 + comp["EV_float"].mean()
    actual_win_prob = (comp["Result"] == "Hit").mean()
    gap = (pred_win_prob - actual_win_prob) * 100

    pnl = comp.apply(
        lambda r: r["Stake"] * (payout_mult - 1.0)
        if r["Result"] == "Hit"
        else -r["Stake"],
        axis=1,
    )
    roi = pnl.sum() / max(1.0, bankroll) * 100

    # -------------------------------
    # Market vs Model Distribution Plot
    # -------------------------------
    st.markdown("---")
    st.subheader("Market vs Model Performance Trend")

    comp["Edge_vs_Market"] = comp["EV_float"] * 100

    fig2 = px.histogram(
        comp,
        x="Edge_vs_Market",
        nbins=20,
        title="Distribution of Model Edge vs Market (EV%)",
        color_discrete_sequence=["#FFCC33"]
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------
    # Calibration summary
    # -------------------------------
    st.markdown(
        f"**Predicted Avg Win Prob (approx):** {pred_win_prob*100:.1f}%"
    )
    st.markdown(
        f"**Actual Hit Rate:** {actual_win_prob*100:.1f}%"
    )
    st.markdown(
        f"**Calibration Gap:** {gap:+.1f}% | **ROI:** {roi:+.1f}%"
    )

    if gap > 5:
        st.warning(
            "Model appears overconfident → consider requiring higher EV before firing."
        )
    elif gap < -5:
        st.info(
            "Model appears conservative → thin edges may be slightly under-trusted."
        )
    else:
        st.success("Model and results are reasonably aligned ✅")
