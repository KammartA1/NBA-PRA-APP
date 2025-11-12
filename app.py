# =============================================================
#  NBA Prop Model – Advanced Edition (Full Production Build)
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import os
import json
import requests
import random
import math
from functools import lru_cache
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# ================================
# PAGE CONFIG MUST BE FIRST
# ================================
if "page_configured" not in st.session_state:
    st.set_page_config(
        page_title="NBA Prop Model",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state["page_configured"] = True

# ================================
# COLORWAY + UI STYLING
# ================================
GOPHER_MAROON = "#7A0019"
GOPHER_GOLD = "#FFCC33"
BACKGROUND = "#0F0F0F"
TEXT_COLOR = "#F5F5F"

st.markdown(
    f"""
    <style>
        body {{
            background-color: {BACKGROUND};
            color: {TEXT_COLOR};
            font-family: 'Inter', sans-serif;
        }}
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
        [data-testid="stMetricValue"] {{
            color: {GOPHER_GOLD} !important;
            font-weight: 600 !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: #DDDDDD !important;
        }}
        [data-testid="stMetricDelta"] {{
            display: none !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===================================
# CONFIG & MODEL PARAMETERS
# ===================================
_LAST_N = 15
_SIM_RUNS_DEFAULT = 1000
_BANKROLL_DEFAULT = 1000.0
_MAX_DAILY_LOSS_PCT = 0.05  # 5% max daily loss
_DATA_CACHE_TTL_HOURS = 12
_DATA_AUTO_REFRESH_HOURS = 6

# ===================================
# DATA SOURCES – BallDontLie API
# ===================================
_BDL_BASE = "https://api.balldontlie.io/v1"
_BDL_TIMEOUT = 20
HEADSHOT_FALLBACK = "https://static.thenounproject.com/png/363640-200.png"

def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def _to_minutes(min_str):
    if not min_str: return 0.0
    try:
        mm, ss = str(min_str).split(":")
        return float(mm) + float(ss)/60.0
    except:
        try:
            return float(min_str)
        except:
            return 0.0

def _possessions(row):
    return (row["fga"] + 0.44*row["fta"] - row["oreb"] + row["tov"])

@st.cache_data(ttl=_DATA_CACHE_TTL_HOURS*3600, show_spinner=False)
def _rget(url, params=None):
    try:
        r = requests.get(url, params=params or {}, timeout=_BDL_TIMEOUT,
                         headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=_DATA_CACHE_TTL_HOURS*3600, show_spinner=False)
def bdl_players_search(query: str):
    js = _rget(f"{_BDL_BASE}/players", params={"search": query, "per_page":50})
    return js.get("data",[]) if js else []

@st.cache_data(ttl=_DATA_CACHE_TTL_HOURS*3600, show_spinner=False)
def bdl_player_by_id(pid: int):
    js = _rget(f"{_BDL_BASE}/players/{pid}")
    return js or {}

@st.cache_data(ttl=_DATA_CACHE_TTL_HOURS*3600, show_spinner=False)
def bdl_stats_last_n_games(player_id: int, last_n: int = _LAST_N):
    stats = []
    page = 1
    per_page = 100
    while len(stats) < last_n:
        js = _rget(f"{_BDL_BASE}/stats",
                  params={"player_ids[]":player_id, "per_page":per_page,
                          "page":page, "postseason":"false"})
        if not js: break
        stats.extend(js.get("data",[]))
        if page >= js.get("meta",{}).get("total_pages",1): break
        page += 1
    return stats[:last_n]

def _frame_from_stats(stats):
    if not stats: return pd.DataFrame()
    rows = []
    for s in stats:
        ply = s.get("player",{}) or {}
        tm = s.get("team",{}) or {}
        g = s.get("game",{}) or {}
        stg = s.get("stats",{})
        rows.append({
            "game_id": g.get("id"),
            "date": g.get("date","")[:10],
            "home_team_id": g.get("home_team_id"),
            "visitor_team_id": g.get("visitor_team_id"),
            "team_id": tm.get("id"),
            "player_id": ply.get("id"),
            "min": _to_minutes(stg.get("min")),
            "pts": stg.get("pts",0),
            "reb": stg.get("reb",0),
            "ast": stg.get("ast",0),
            "tov": stg.get("turnover",0) or stg.get("tov",0),
            "fga": stg.get("fga",0),
            "fta": stg.get("fta",0),
            "oreb": stg.get("oreb",0),
            "dreb": stg.get("dreb",0),
        })
    return pd.DataFrame(rows)

def _team_game_totals(df_all):
    if df_all.empty: return pd.DataFrame()
    grp = df_all.groupby(["game_id","team_id"],as_index=False).agg({
        "pts":"sum","reb":"sum","ast":"sum","tov":"sum","fga":"sum","fta":"sum",
        "oreb":"sum","dreb":"sum","min":"sum"
    })
    grp["orb"] = grp["oreb"]
    grp["poss"] = grp.apply(_possessions,axis=1)
    return grp

def _opponent_id(row):
    if row["team_id"] == row["home_team_id"]:
        return row["visitor_team_id"]
    else:
        return row["home_team_id"]

def _merge_team_context(df_p, df_totals, df_meta):
    if df_p.empty: return df_p
    meta = df_meta[["id","home_team_id","visitor_team_id"]].rename(columns={"id":"game_id"})
    out = df_p.merge(meta,on="game_id",how="left")
    out["opp_team_id"] = out.apply(_opponent_id,axis=1)
    out = out.merge(df_totals.rename(columns={
        "pts":"team_pts","reb":"team_reb","ast":"team_ast","tov":"team_tov",
        "fga":"team_fga","fta":"team_fta","oreb":"team_orb","dreb":"team_drb",
        "min":"team_min","poss":"team_poss"
    }),on=["game_id","team_id"],how="left")
    opp = df_totals.rename(columns={
        "team_id":"opp_team_id",
        "pts":"opp_pts","reb":"opp_reb","ast":"opp_ast","tov":"opp_tov",
        "fga":"opp_fga","fta":"opp_fta","oreb":"opp_orb","dreb":"opp_drb",
        "poss":"opp_poss"
    })
    out = out.merge(opp[["game_id","opp_team_id","opp_pts","opp_reb","opp_tov","opp_fga","opp_fta","opp_orb","opp_drb","opp_poss"]],
                    on=["game_id","opp_team_id"],how="left")
    return out

def _estimate_usg(row):
    mp = row.get("min",0.0)
    if mp <= 0 or not row.get("team_fga"): return 0.0
    num = (row.get("fga",0)+0.44*row.get("fta",0)+row.get("tov",0)) * (row.get("team_min",0)/5.0)
    den = mp * (row.get("team_fga",0)+0.44*row.get("team_fta",0)+row.get("team_tov",0))
    return 100.0 * _safe_div(num, den)

def _team_def_rating(row):
    return 100.0 * _safe_div(row.get("opp_pts",0), row.get("team_poss",0))

def _team_pace(row):
    poss = 0.5 * (row.get("team_poss",0)+row.get("opp_poss",0))
    tm_min = max(row.get("team_min",240.0),1.0)
    return 48.0 * _safe_div(poss,(tm_min/5.0))

@st.cache_data(ttl=_DATA_CACHE_TTL_HOURS*3600, show_spinner=False)
def load_player_context(player_id: int, last_n: int = _LAST_N):
    raw = bdl_stats_last_n_games(player_id,last_n=last_n)
    if not raw:
        return pd.DataFrame(), {}, HEADSHOT_FALLBACK

    logs = _frame_from_stats(raw)
    game_ids = logs["game_id"].dropna().unique().tolist()
    games_meta = []
    for gid in game_ids:
        mj = _rget(f"{_BDL_BASE}/games/{gid}")
        if mj: games_meta.append(mj)
    games_meta = pd.DataFrame(games_meta) if games_meta else pd.DataFrame(columns=["id","home_team_id","visitor_team_id"])

    totals = []
    for gid in game_ids:
        page = 1
        while True:
            js = _rget(f"{_BDL_BASE}/stats", params={"game_ids[]":gid,"per_page":100,"page":page})
            if not js: break
            totals.append(_frame_from_stats(js.get("data",[])))
            if page >= js.get("meta",{}).get("total_pages",1): break
            page += 1
    allsub = pd.concat(totals,ignore_index=True) if totals else pd.DataFrame()
    team_totals = _team_game_totals(allsub)

    ctx = _merge_team_context(logs,team_totals,games_meta)
    if ctx.empty:
        return logs, {}, HEADSHOT_FALLBACK

    ctx["usg"] = ctx.apply(_estimate_usg,axis=1)
    ctx["team_def_rating"] = ctx.apply(_team_def_rating,axis=1)
    ctx["pace"] = ctx.apply(_team_pace,axis=1)

    def _agg(series):
        return pd.Series({
            "mean": float(np.nanmean(series)) if len(series) else 0.0,
            "std": float(np.nanstd(series,ddof=1)) if len(series)>1 else 0.0,
            "skew": float(skew(series,bias=False)) if len(series)>2 else 0.0,
            "kurt": float(kurtosis(series,bias=False)) if len(series)>3 else 0.0
        })

    agg = {
        "PTS": _agg(ctx["pts"]),
        "REB": _agg(ctx["reb"]),
        "AST": _agg(ctx["ast"]),
        "MIN": _agg(ctx["min"]),
        "USG": _agg(ctx["usg"]),
        "PACE": _agg(ctx["pace"]),
        "DEFRTG": _agg(ctx["team_def_rating"])
    }

    pmeta = bdl_player_by_id(player_id)
    nba_id = None
    for k in ("nba_id","person_id"):
        if k in pmeta and isinstance(pmeta[k],int):
            nba_id = str(pmeta[k])
            break
    if nba_id:
        headshot = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{nba_id}.png"
    else:
        headshot = HEADSHOT_FALLBACK

    return ctx.sort_values("date"), agg, headshot

# ===================================
# PLAYER CARD UI COMPONENT
# ===================================
def player_card(player_name, logs, agg, headshot_url):
    if logs.empty:
        st.warning(f"No data found for {player_name}.")
        return

    avg_pts = agg["PTS"]["mean"]
    avg_reb = agg["REB"]["mean"]
    avg_ast = agg["AST"]["mean"]
    avg_min = agg["MIN"]["mean"]
    avg_usg = agg["USG"]["mean"]
    avg_pace = agg["PACE"]["mean"]
    avg_def = agg["DEFRTG"]["mean"]

    c1, c2 = st.columns([1,3])
    with c1:
        st.image(headshot_url,use_container_width=True)
    with c2:
        st.markdown(f"""
            <div style="color:{TEXT_COLOR}; font-size:1.2rem; font-weight:600;">
                {player_name}
            </div>
            <div style="font-size:0.85rem; color:#BBBBBB;">
                Last {_LAST_N} Games
            </div>
        """, unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)
        colA.metric("PTS", f"{avg_pts:.1f}", help="Average points per game")
        colB.metric("REB", f"{avg_reb:.1f}", help="Average rebounds per game")
        colC.metric("AST", f"{avg_ast:.1f}", help="Average assists per game")

        colD, colE, colF = st.columns(3)
        colD.metric("MIN", f"{avg_min:.1f}", help="Average minutes played")
        colE.metric("USG%", f"{avg_usg:.1f}", help="Estimated usage rate %")
        colF.metric("PACE", f"{avg_pace:.1f}", help="Team pace (poss/48 min)")

        st.markdown(
            f"<div style='font-size:0.8rem; color:#999;'>Def Rating (opp): <b>{avg_def:.1f}</b></div>",
            unsafe_allow_html=True
        )

    logs_idx = logs.set_index("date")[["pts","reb","ast"]].tail(_LAST_N)
    fig, ax = plt.subplots(figsize=(4,1.2))
    logs_idx.plot(ax=ax, legend=False, color=GOPHER_GOLD)
    ax.set_facecolor(BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.set_title("Last 15 Game PTS/REB/AST", color=TEXT_COLOR, fontsize=9)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ===================================
# SIMULATION & MODEL FUNCTIONS
# ===================================
def compute_ev(probability: float, odds: float = 1.5):
    if probability <= 0 or odds <= 0:
        return 0.0
    return (probability * odds - 1) * 100

def compute_kelly(prob: float, odds: float = 1.5, fraction: float = 0.3):
    b = odds - 1
    edge = (prob * (b + 1) - 1) / b if b != 0 else 0.0
    return max(0.0, edge * fraction)

def compute_clv(proj: float, line: float):
    return proj - line

def bootstrap_simulation(series: np.ndarray, runs: int = _SIM_RUNS_DEFAULT):
    if len(series) == 0:
        return np.array([])
    idx = np.random.randint(0, len(series), size=(runs, len(series)))
    samples = np.take(series, idx)
    return samples.mean(axis=1)

@st.cache_data(ttl=_DATA_CACHE_TTL_HOURS*3600, show_spinner=False)
def run_player_simulation(series: np.ndarray, line: float, runs: int = _SIM_RUNS_DEFAULT):
    sims = bootstrap_simulation(series, runs=runs)
    if sims.size == 0:
        return None
    prob = (sims > line).mean()
    mean = float(np.mean(sims))
    var = float(np.var(sims, ddof=1))
    skewn = float(skew(sims, bias=False))
    return {"prob": prob, "mean": mean, "variance": var, "skew": skewn, "simulations": sims}

# ===================================
# SIDEBAR – Controls + Refresh Panel
# ===================================
st.sidebar.markdown("## 🔄 Controls")
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = dt.datetime.utcnow()
    
refresh_btn = st.sidebar.button("Refresh Data")
if refresh_btn or (dt.datetime.utcnow() - st.session_state["last_refresh"]).total_seconds() > _DATA_AUTO_REFRESH_HOURS*3600:
    st.session_state["last_refresh"] = dt.datetime.utcnow()
    # Clear cached player context so next call reruns
    load_player_context.clear()
    st.sidebar.success("✅ Data refreshed")

last_delta = dt.datetime.utcnow() - st.session_state["last_refresh"]
mins = int(last_delta.total_seconds()//60)
secs = int(last_delta.total_seconds() % 60)
st.sidebar.markdown(f"Last updated: {mins}m {secs}s ago")

bankroll = st.sidebar.number_input("Bankroll ($)", value=_BANKROLL_DEFAULT, step=100.0)
odds = st.sidebar.number_input("Default odds", value=1.5, step=0.1)
runs = st.sidebar.number_input("Simulation runs", value=_SIM_RUNS_DEFAULT, step=100)

# ===================================
# MAIN UI – Player Inputs
# ===================================
st.title("NBA Prop Model")
st.markdown("Dual player mode — analyze and simulate side-by-side")

p1_name = st.text_input("Player 1 Name")
p2_name = st.text_input("Player 2 Name (optional)")

if p1_name:
    p1_search = bdl_players_search(p1_name)
    p1_id = p1_search[0]["id"] if p1_search else None
else:
    p1_id = None

if p2_name:
    p2_search = bdl_players_search(p2_name)
    p2_id = p2_search[0]["id"] if p2_search else None
else:
    p2_id = None

if p1_id:
    logs1, agg1, img1 = load_player_context(p1_id)
    player_card(p1_name, logs1, agg1, img1)
if p2_id:
    logs2, agg2, img2 = load_player_context(p2_id)
    player_card(p2_name, logs2, agg2, img2)

# ===================================
# SIMULATION Section
# ===================================
if p1_id and logs1 is not None:
    st.markdown("---")
    st.markdown("### 🧮 Simulation / Model Output")

    line1 = st.number_input(f"{p1_name} PRA Line", value=33.5, step=0.5)
    core_series1 = logs1["pts"] + logs1["reb"] + logs1["ast"]
    sim_res1 = run_player_simulation(core_series1.to_numpy(), line1, runs=runs)

    if sim_res1:
        ev1 = compute_ev(sim_res1["prob"], odds)
        kelly1 = compute_kelly(sim_res1["prob"], odds)
        clv1 = compute_clv(sim_res1["mean"], line1)

        st.metric(f"{p1_name} Model EV", f"{ev1:.2f} %")
        st.metric(f"{p1_name} Model Kelly-Stake", f"${kelly1*bankroll:.2f}")
        st.metric(f"{p1_name} Model CLV", f"{clv1:.2f}")

    if p2_id and logs2 is not None:
        line2 = st.number_input(f"{p2_name} PRA Line", value=31.5, step=0.5)
        core_series2 = logs2["pts"] + logs2["reb"] + logs2["ast"]
        sim_res2 = run_player_simulation(core_series2.to_numpy(), line2, runs=runs)

        if sim_res2:
            ev2 = compute_ev(sim_res2["prob"], odds)
            kelly2 = compute_kelly(sim_res2["prob"], odds)
            clv2 = compute_clv(sim_res2["mean"], line2)

            st.metric(f"{p2_name} Model EV", f"{ev2:.2f} %")
            st.metric(f"{p2_name} Model Kelly-Stake", f"${kelly2*bankroll:.2f}")
            st.metric(f"{p2_name} Model CLV", f"{clv2:.2f}")

# ===================================
# RESULTS, CALIBRATION & INSIGHTS Tabs
# ===================================
tab1, tab2, tab3 = st.tabs(["Results Log", "Calibration", "Insights"])

with tab1:
    st.markdown("### 📋 Results Log")
    if os.path.exists("results_log.csv"):
        df = pd.read_csv("results_log.csv")
        st.dataframe(df.tail(25), use_container_width=True)
    else:
        st.info("No logs yet. Run a simulation to start logging.")

with tab2:
    st.markdown("### 📊 Calibration")
    st.write("Calibration & hit-rate tables will appear here once you have logged outcomes.")

with tab3:
    st.markdown("### 🔍 Insights")
    st.write("Insights and edge-analysis will appear here with time.")

# ===================================
# FOOTER
# ===================================
st.markdown("---")
st.caption(
    f"NBA Prop Model • Cached live data (last { _DATA_CACHE_TTL_HOURS }h) • Side-by-side dual-player • { _LAST_N }-game window"
)
