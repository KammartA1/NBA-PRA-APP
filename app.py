# ============================================================
# NBA PROP MODEL — QUANT ENGINE (The Odds API + NBA API)
# Single-file Streamlit app
# - Live lines (player props) via The Odds API v4 (events/{id}/odds)
# - Opponent/context via nba_api (enhancement only; never blocks)
# - Live Scanner finds single-leg edges above threshold
# - History logging to local CSV (per user id)
# ============================================================

import os
import re
import math
import time
import json
import difflib
from dataclasses import dataclass
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
import requests
import streamlit as st

# nba_api (stats endpoints)
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import playergamelog, scoreboardv2

# ------------------------------
# Global constants
# ------------------------------
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY_NBA = "basketball_nba"
REGION_US = "us"

# The Odds API market keys (player props) — from The Odds API docs
# See: betting markets list (player_points, player_rebounds, player_assists, etc.)
ODDS_MARKETS = {
    "Points": "player_points",
    "Rebounds": "player_rebounds",
    "Assists": "player_assists",
    "3PM": "player_threes",
    "PRA": "player_points_rebounds_assists",
    "PR": "player_points_rebounds",
    "PA": "player_points_assists",
    "RA": "player_rebounds_assists",
}

# Optional additional markets toggled in sidebar
ODDS_MARKETS_OPTIONAL = {
    "Blocks": "player_blocks",
    "Steals": "player_steals",
    "Turnovers": "player_turnovers",
}

# Simple stat mapping for projection from nba_api logs
STAT_FIELDS = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "3PM": "FG3M",
    "PRA": ("PTS", "REB", "AST"),
    "PR": ("PTS", "REB"),
    "PA": ("PTS", "AST"),
    "RA": ("REB", "AST"),
    "Blocks": "BLK",
    "Steals": "STL",
    "Turnovers": "TOV",
}

# ------------------------------
# Utilities
# ------------------------------
def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def normalize_name(name: str) -> str:
    """Canonical player name normalization shared across:
       - line lookup
       - scanner
       - projections
       - cards
    """
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"[\.\'\-]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def odds_api_key() -> str:
    # Never crash if missing; warn later
    return (
        st.secrets.get("ODDS_API_KEY", "")
        if hasattr(st, "secrets")
        else ""
    ) or os.getenv("ODDS_API_KEY", "")

def http_get_json(url: str, params: dict, timeout: int = 25):
    """Requests wrapper with robust error handling; never throws upstream without context."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        # store headers for credit tracking
        remaining = r.headers.get("x-requests-remaining") or r.headers.get("x-requests-remaining".title())
        used = r.headers.get("x-requests-used") or r.headers.get("x-requests-used".title())
        st.session_state["_odds_headers_last"] = {"remaining": remaining, "used": used, "ts": _now_iso()}
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        msg = f"HTTPError {getattr(e.response,'status_code',None)} on {url}"
        try:
            detail = e.response.text[:3000] if e.response is not None else ""
        except Exception:
            detail = ""
        return None, msg + (f"\n{detail}" if detail else "")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

@st.cache_data(ttl=60*10, show_spinner=False)
def get_team_maps():
    """Map between team full names and abbreviations/ids using nba_api static teams."""
    teams = nba_teams.get_teams()
    by_name = {}
    for t in teams:
        full = t.get("full_name", "")
        abbr = t.get("abbreviation", "")
        tid = t.get("id")
        by_name[normalize_name(full)] = {"abbr": abbr, "id": tid, "full_name": full}
    # common aliases (Odds API uses standard NBA names, but keep a few)
    alias = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
        "gs warriors": "golden state warriors",
    }
    for a, tgt in alias.items():
        if normalize_name(tgt) in by_name:
            by_name[normalize_name(a)] = by_name[normalize_name(tgt)]
    return by_name

def map_team_name_to_abbr(team_name: str) -> str | None:
    if not team_name:
        return None
    m = get_team_maps()
    rec = m.get(normalize_name(team_name))
    return rec["abbr"] if rec else None

@st.cache_data(ttl=60*60*24, show_spinner=False)
def lookup_player_id(full_name: str):
    """Best-effort NBA player id lookup (does not block)."""
    if not full_name:
        return None
    nm = normalize_name(full_name)
    # exact / close matching via nba_api static list
    plist = nba_players.get_players()
    # first exact normalized match
    for p in plist:
        if normalize_name(p.get("full_name", "")) == nm:
            return p.get("id")
    # fallback: difflib
    names = [p.get("full_name", "") for p in plist]
    cand = difflib.get_close_matches(full_name, names, n=1, cutoff=0.75)
    if cand:
        for p in plist:
            if p.get("full_name") == cand[0]:
                return p.get("id")
    return None

def nba_headshot_url(player_id: int | None) -> str | None:
    if not player_id:
        return None
    # NBA CDN pattern used broadly
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"

# ------------------------------
# Odds API ingestion
# ------------------------------
@st.cache_data(ttl=60*5, show_spinner=False)
def odds_get_events(date_iso: str | None = None):
    """Fetch NBA events. date_iso (YYYY-MM-DD) optional; if provided filters by that date on backend when supported."""
    key = odds_api_key()
    if not key:
        return [], "Missing ODDS_API_KEY in secrets or environment."
    url = f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events"
    params = {"apiKey": key}
    # Some Odds API deployments support dateFormat / commenceTimeFrom/To; keep minimal to reduce risk
    data, err = http_get_json(url, params=params)
    if err or not isinstance(data, list):
        return [], err or "Unexpected response for events."
    # Optional client-side filter by date if date_iso provided
    if date_iso:
        out = []
        for ev in data:
            ct = safe_get(ev, "commence_time")
            if not ct:
                continue
            try:
                d = ct[:10]
            except Exception:
                d = None
            if d == date_iso:
                out.append(ev)
        return out, None
    return data, None

def _markets_param(selected_market_keys: list[str]) -> str:
    # Odds API expects comma-separated market keys
    return ",".join(selected_market_keys)

@st.cache_data(ttl=60*5, show_spinner=False)
def odds_get_event_odds(event_id: str, market_keys: tuple, regions: str = REGION_US):
    """Fetch odds (including player props) for a given event."""
    key = odds_api_key()
    if not key:
        return None, "Missing ODDS_API_KEY in secrets or environment."
    url = f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events/{event_id}/odds"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": _markets_param(list(market_keys)),
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    data, err = http_get_json(url, params=params)
    return data, err

def extract_books_from_event_odds(event_odds: dict) -> list[str]:
    books = []
    for b in (event_odds or {}).get("bookmakers", []) or []:
        k = b.get("key")
        if k:
            books.append(k)
    books = sorted(list(dict.fromkeys(books)))
    return books

def _parse_player_prop_outcomes(event_odds: dict, market_key: str, book_filter: str | None = None):
    """
    Returns list of dict rows:
      {player, line, odds_price, book, side, market_key, event_id, home_team, away_team, commence_time}
    Supports:
      - book_filter=None => include all books
      - book_filter="consensus" => compute consensus line using median across books for each player/side
      - book_filter="<bookkey>" => only that book
    """
    if not event_odds:
        return [], "Empty event odds."
    event_id = event_odds.get("id")
    home = event_odds.get("home_team")
    away = event_odds.get("away_team")
    ct = event_odds.get("commence_time")
    books = event_odds.get("bookmakers", []) or []

    rows = []
    for b in books:
        bkey = b.get("key")
        if book_filter and book_filter not in ("consensus", "all") and bkey != book_filter:
            continue
        for mk in b.get("markets", []) or []:
            if mk.get("key") != market_key:
                continue
            for out in mk.get("outcomes", []) or []:
                player = out.get("description") or out.get("name")  # props use description for player in many payloads
                # For Over/Under props, name is typically "Over"/"Under", description is player
                side = out.get("name")
                point = out.get("point")
                price = out.get("price")
                if player and point is not None:
                    rows.append({
                        "player": player,
                        "player_norm": normalize_name(player),
                        "line": float(point),
                        "price": price,
                        "book": bkey or "",
                        "side": side or "",
                        "market_key": market_key,
                        "event_id": event_id,
                        "home_team": home,
                        "away_team": away,
                        "commence_time": ct,
                    })
    if book_filter == "consensus":
        # Build consensus by median line across books for each (player_norm, side)
        if not rows:
            return [], None
        df = pd.DataFrame(rows)
        # Keep only numeric lines
        df = df[pd.to_numeric(df["line"], errors="coerce").notna()].copy()
        if df.empty:
            return [], None
        g = df.groupby(["player_norm", "side"], dropna=False)["line"].median().reset_index()
        # use one representative player string (most common)
        name_map = df.groupby("player_norm")["player"].agg(lambda x: x.value_counts().index[0]).to_dict()
        out_rows = []
        for _, r in g.iterrows():
            out_rows.append({
                "player": name_map.get(r["player_norm"], r["player_norm"]),
                "player_norm": r["player_norm"],
                "line": float(r["line"]),
                "price": None,
                "book": "consensus",
                "side": r["side"],
                "market_key": market_key,
                "event_id": event_id,
                "home_team": home,
                "away_team": away,
                "commence_time": ct,
            })
        return out_rows, None

    return rows, None

def find_player_line_from_events(player_name: str, market_key: str, scan_date_iso: str, book_choice: str):
    """Find a player's line by searching events on a date and parsing event odds.
       Returns: (line, meta_dict, err)
    """
    evs, err = odds_get_events(scan_date_iso)
    if err:
        return None, None, err
    if not evs:
        return None, None, "No events returned for that date."
    target = normalize_name(player_name)
    # iterate events and look for player
    for ev in evs:
        eid = ev.get("id")
        if not eid:
            continue
        odds, oerr = odds_get_event_odds(eid, (market_key,))
        if oerr or not odds:
            continue
        rows, _ = _parse_player_prop_outcomes(odds, market_key, book_filter=book_choice)
        for r in rows:
            if r.get("player_norm") == target:
                return float(r["line"]), r, None
        # close match fallback
        norms = [r.get("player_norm","") for r in rows]
        close = difflib.get_close_matches(target, norms, n=1, cutoff=0.88)
        if close:
            rr = next((x for x in rows if x.get("player_norm")==close[0]), None)
            if rr:
                return float(rr["line"]), rr, None
    return None, None, "Player/market not found in Odds API props for that date/book."

# ------------------------------
# NBA API context (best-effort)
# ------------------------------
@st.cache_data(ttl=60*10, show_spinner=False)
def nba_scoreboard_games(game_date: date):
    """Fetch NBA scoreboard games and derive matchup mapping by team abbrev."""
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=game_date.strftime("%m/%d/%Y"))
        df = sb.get_data_frames()[0]  # game header
        # Columns include HOME_TEAM_ID, VISITOR_TEAM_ID, GAME_ID
        games = []
        for _, row in df.iterrows():
            games.append({
                "game_id": str(row.get("GAME_ID")),
                "home_team_id": int(row.get("HOME_TEAM_ID")),
                "away_team_id": int(row.get("VISITOR_TEAM_ID")),
            })
        # map team id -> opponent id for that day (if multiple games, keep first)
        return games, None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"

@st.cache_data(ttl=60*60*24, show_spinner=False)
def team_id_to_abbr_map():
    m = {}
    for t in nba_teams.get_teams():
        m[int(t["id"])] = t["abbreviation"]
    return m

def resolve_opponent_from_odds_event(meta: dict) -> tuple[str|None, str|None]:
    """From Odds API event meta (home_team, away_team), return (team_abbr, opp_abbr) if resolvable."""
    home = meta.get("home_team") if meta else None
    away = meta.get("away_team") if meta else None
    if not home or not away:
        return None, None
    home_abbr = map_team_name_to_abbr(home)
    away_abbr = map_team_name_to_abbr(away)
    # If player team is unknown here, return both; caller can pick
    return home_abbr, away_abbr

def resolve_player_team_abbr_from_nba(player_id: int, game_date: date) -> str | None:
    """Best-effort: infer player's team abbreviation from most recent game log up to date."""
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, date_from_nullable=None, date_to_nullable=game_date.strftime("%m/%d/%Y"))
        df = gl.get_data_frames()[0]
        if df.empty:
            return None
        # MATCHUP like "PHI vs. MIL" or "PHI @ MIL"
        matchup = str(df.iloc[0].get("MATCHUP",""))
        team_abbr = matchup.split(" ")[0].strip() if matchup else None
        return team_abbr or None
    except Exception:
        return None

def opponent_from_team_abbr(team_abbr: str, game_date: date) -> str | None:
    games, _ = nba_scoreboard_games(game_date)
    if not games or not team_abbr:
        return None
    tid_map = team_id_to_abbr_map()
    for g in games:
        ha = tid_map.get(g["home_team_id"])
        aa = tid_map.get(g["away_team_id"])
        if ha == team_abbr:
            return aa
        if aa == team_abbr:
            return ha
    return None

# ------------------------------
# Projection engine (robust + deterministic)
# ------------------------------
def compute_stat_from_gamelog(df: pd.DataFrame, market_name: str) -> pd.Series:
    f = STAT_FIELDS.get(market_name)
    if f is None:
        return pd.Series([], dtype=float)
    if isinstance(f, tuple):
        s = pd.Series(0.0, index=df.index)
        for col in f:
            s = s + pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
        return s
    return pd.to_numeric(df.get(f), errors="coerce")

def bootstrap_prob_over(stat_series: pd.Series, line: float, n_sims: int = 4000) -> tuple[float, float, float]:
    """Empirical bootstrap: returns (p_over, mu, sigma)."""
    x = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
    if x.size < 4:
        # fallback normal approximation with conservative sigma
        mu = float(np.nanmean(x)) if x.size else 0.0
        sigma = float(np.nanstd(x)) if x.size else max(1.0, 0.25*max(line,1.0))
        p = float(1.0 - 0.5*(1+math.erf((line-mu)/(sigma*math.sqrt(2)+1e-9))))
        return max(0.0, min(1.0, p)), mu, sigma
    rng = np.random.default_rng(7)  # deterministic
    sims = rng.choice(x, size=(n_sims, x.size), replace=True).mean(axis=1)
    p_over = float((sims > line).mean())
    return p_over, float(x.mean()), float(x.std(ddof=1) if x.size>1 else 0.0)

def estimate_blowout_risk(team_abbr: str | None, opp_abbr: str | None, spread_abs: float | None) -> float:
    """
    Conservative blowout risk estimator.
    - If we have spread magnitude, use it.
    - Else return baseline 0.10
    """
    if spread_abs is None:
        return 0.10
    try:
        s = float(abs(spread_abs))
    except Exception:
        return 0.10
    # map spread to probability bucket
    # 0-5 => 8%, 5-10 => 14%, 10-15 => 22%, 15+ => 30%
    if s < 5:
        return 0.08
    if s < 10:
        return 0.14
    if s < 15:
        return 0.22
    return 0.30

def context_multiplier(team_abbr: str | None, opp_abbr: str | None, key_teammate_out: bool) -> float:
    """Simple, stable context multiplier."""
    m = 1.00
    if key_teammate_out:
        m *= 1.04
    # placeholder for future defensive/pace adjustments
    return float(m)

def compute_leg_projection(player_name: str, market_name: str, line: float, meta: dict | None, n_games: int, key_teammate_out: bool):
    """Compute single leg model outputs with defensive coding.
       Returns: dict with projection, p_over, edge, opponent/team info, errors list.
    """
    errors = []
    player_id = lookup_player_id(player_name)
    if not player_id:
        errors.append("Could not resolve NBA player id (name mismatch).")
        # still return minimal structure
        return {
            "player": player_name, "market": market_name, "line": line,
            "proj": None, "p_over": None, "edge": None,
            "team": None, "opp": None, "headshot": None,
            "blowout_prob": 0.10, "context_mult": 1.00,
            "errors": errors,
        }

    # Pull gamelog (best-effort)
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season_nullable=None)
        gldf = gl.get_data_frames()[0]
    except Exception as e:
        errors.append(f"NBA API gamelog error: {type(e).__name__}")
        gldf = pd.DataFrame()

    if not gldf.empty:
        gldf = gldf.head(int(max(5, n_games))).copy()
        stat_series = compute_stat_from_gamelog(gldf, market_name)
    else:
        stat_series = pd.Series([], dtype=float)

    p_over, mu, sigma = bootstrap_prob_over(stat_series, float(line))

    # Team/opponent resolution:
    gdate = date.today()
    team_abbr = None
    opp_abbr = None

    # 1) Try to infer from NBA gamelog (doesn't require Odds meta)
    try:
        team_abbr = resolve_player_team_abbr_from_nba(player_id, gdate)
    except Exception:
        team_abbr = None

    # 2) Enhance using Odds event teams if available
    if meta:
        home_abbr, away_abbr = resolve_opponent_from_odds_event(meta)
        # if player team known, pick opponent from home/away
        if team_abbr and home_abbr and away_abbr:
            if team_abbr == home_abbr:
                opp_abbr = away_abbr
            elif team_abbr == away_abbr:
                opp_abbr = home_abbr
        # if we still don't know team, set both for display
        if not team_abbr and home_abbr and away_abbr:
            team_abbr = home_abbr
            opp_abbr = away_abbr

    # 3) If we have team but not opponent, try scoreboard
    if team_abbr and not opp_abbr:
        opp_abbr = opponent_from_team_abbr(team_abbr, gdate)

    # Spread not pulled in current minimal version; set None (future extension)
    blowout_prob = estimate_blowout_risk(team_abbr, opp_abbr, spread_abs=None)
    ctx = context_multiplier(team_abbr, opp_abbr, key_teammate_out)

    # Apply context multiplier to projection mean only (prob remains from empirical)
    proj = mu * ctx if mu is not None else None
    edge = (p_over - 0.5) if p_over is not None else None  # baseline edge vs 50/50

    return {
        "player": player_name,
        "market": market_name,
        "line": float(line),
        "proj": float(proj) if proj is not None else None,
        "p_over": float(p_over) if p_over is not None else None,
        "edge": float(edge) if edge is not None else None,
        "team": team_abbr,
        "opp": opp_abbr,
        "headshot": nba_headshot_url(player_id),
        "blowout_prob": float(blowout_prob),
        "context_mult": float(ctx),
        "errors": errors,
    }

# ------------------------------
# History persistence
# ------------------------------
def history_path(user_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip() or "default")
    return f"history_{safe}.csv"

def load_history(user_id: str) -> pd.DataFrame:
    fp = history_path(user_id)
    if os.path.exists(fp):
        try:
            return pd.read_csv(fp)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def append_history(user_id: str, row: dict):
    fp = history_path(user_id)
    df = load_history(user_id)
    df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df2.to_csv(fp, index=False)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="NBA Prop Model — Quant Engine", layout="wide")

# Styling close to prior look
st.markdown("""
<style>
/* App background */
.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(110,0,20,0.55), rgba(10,10,16,0.95) 55%, rgba(0,0,0,0.98) 100%);
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(110,0,20,0.92), rgba(40,0,12,0.92));
}
div[data-testid="stSidebarContent"] * { color: #f3f3f3; }
/* Cards */
.block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("## User & Bankroll")
user_id = st.sidebar.text_input("Your ID (for personal history)", value=st.session_state.get("user_id", "Me"))
st.session_state["user_id"] = user_id

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=0.0, value=float(st.session_state.get("bankroll", 100.0)), step=10.0)
st.session_state["bankroll"] = bankroll

payout_2pick = st.sidebar.number_input("2-Pick Payout (e.g. 3.0x)", min_value=1.0, value=float(st.session_state.get("payout_2pick", 3.0)), step=0.1)
st.session_state["payout_2pick"] = payout_2pick

frac_kelly = st.sidebar.slider("Fractional Kelly", min_value=0.0, max_value=1.0, value=float(st.session_state.get("frac_kelly", 0.25)), step=0.05)
st.session_state["frac_kelly"] = frac_kelly

n_games = st.sidebar.slider("Recent Games Sample (N)", min_value=5, max_value=30, value=int(st.session_state.get("n_games", 10)))
st.session_state["n_games"] = n_games

compact = st.sidebar.checkbox("Compact Mode (mobile)", value=bool(st.session_state.get("compact", False)))
st.session_state["compact"] = compact

max_daily_loss = st.sidebar.slider("Max Daily Loss % (stop)", 0, 50, int(st.session_state.get("max_daily_loss", 15)))
max_weekly_loss = st.sidebar.slider("Max Weekly Loss % (stop)", 0, 50, int(st.session_state.get("max_weekly_loss", 25)))
st.session_state["max_daily_loss"] = max_daily_loss
st.session_state["max_weekly_loss"] = max_weekly_loss

with st.sidebar.expander("The Odds API Settings", expanded=True):
    include_optional = st.checkbox("Include Rebs+Asts market", value=bool(st.session_state.get("include_ra", True)), key="include_ra")
    max_requests_per_day = st.number_input("Max Odds API requests per day", min_value=1, value=int(st.session_state.get("max_req_day", 60)), step=5)
    st.session_state["max_req_day"] = int(max_requests_per_day)

# Credits display
hdr = st.session_state.get("_odds_headers_last", {})
if hdr:
    st.sidebar.caption(f"Odds API usage — used: {hdr.get('used','?')} | remaining: {hdr.get('remaining','?')} | last: {hdr.get('ts','')}")

# Market list for UI
MARKET_OPTIONS = ["Points", "Rebounds", "Assists", "PRA", "PR", "PA", "RA", "3PM"]
if include_optional:
    MARKET_OPTIONS.append("RA")  # already included; harmless
# add optional
MARKET_OPTIONS += ["Blocks", "Steals", "Turnovers"]

# Tabs
tabs = st.tabs(["📊 Model", "📈 Results", "⚡ Live Scanner", "🗂️ History", "🧪 Calibration"])

# Shared session results container
if "last_results" not in st.session_state:
    st.session_state["last_results"] = []
if "last_board_meta" not in st.session_state:
    st.session_state["last_board_meta"] = {}

def get_sportsbook_choices(scan_date_iso: str):
    # derive from events -> first event odds (minimal)
    evs, err = odds_get_events(scan_date_iso)
    if err or not evs:
        return ["consensus"], err
    # use first event with odds
    for ev in evs[:6]:
        eid = ev.get("id")
        if not eid:
            continue
        mk = tuple({ODDS_MARKETS["Points"]})  # cheap call
        odds, oerr = odds_get_event_odds(eid, mk)
        if odds and not oerr:
            books = extract_books_from_event_odds(odds)
            return ["consensus"] + books, None
    return ["consensus"], "Could not enumerate sportsbooks (Odds API response missing bookmakers)."

with tabs[0]:
    st.markdown("### Up to 4-Leg Projection & Edge (Bootstrap + Context)")
    scan_date = st.date_input("Lines date", value=date.today())
    scan_date_iso = scan_date.isoformat()

    # sportsbook dropdown
    book_choices, book_err = get_sportsbook_choices(scan_date_iso)
    if book_err:
        st.info(book_err)
    sportsbook = st.selectbox("Sportsbook (live lines)", options=book_choices, index=0, help="Choose a book key or consensus (median across books).")

    colA, colB = st.columns(2)
    players = []
    with colA:
        p1 = st.text_input("Player 1 Name", value=st.session_state.get("p1",""))
        m1 = st.selectbox("P1 Market", options=MARKET_OPTIONS, index=MARKET_OPTIONS.index(st.session_state.get("m1","Points")) if st.session_state.get("m1","Points") in MARKET_OPTIONS else 0)
        o1 = st.checkbox("P1: Manual line override", value=False)
        l1 = st.number_input("P1 Line", min_value=0.0, value=float(st.session_state.get("l1", 25.0)), step=0.5)
        t1 = st.text_input("P1 Opponent (Team Abbrev, optional)", value=st.session_state.get("opp1",""))
        out1 = st.checkbox("P1 Key teammate OUT?", value=False)
        players.append(("P1", p1, m1, o1, l1, t1, out1))

        p3 = st.text_input("Player 3 Name", value=st.session_state.get("p3",""))
        m3 = st.selectbox("P3 Market", options=MARKET_OPTIONS, index=MARKET_OPTIONS.index(st.session_state.get("m3","PRA")) if st.session_state.get("m3","PRA") in MARKET_OPTIONS else 0)
        o3 = st.checkbox("P3: Manual line override", value=False)
        l3 = st.number_input("P3 Line", min_value=0.0, value=float(st.session_state.get("l3", 25.0)), step=0.5)
        t3 = st.text_input("P3 Opponent (Team Abbrev, optional)", value=st.session_state.get("opp3",""))
        out3 = st.checkbox("P3 Key teammate OUT?", value=False)
        players.append(("P3", p3, m3, o3, l3, t3, out3))

    with colB:
        p2 = st.text_input("Player 2 Name", value=st.session_state.get("p2",""))
        m2 = st.selectbox("P2 Market", options=MARKET_OPTIONS, index=MARKET_OPTIONS.index(st.session_state.get("m2","Points")) if st.session_state.get("m2","Points") in MARKET_OPTIONS else 0)
        o2 = st.checkbox("P2: Manual line override", value=False)
        l2 = st.number_input("P2 Line", min_value=0.0, value=float(st.session_state.get("l2", 25.0)), step=0.5)
        t2 = st.text_input("P2 Opponent (Team Abbrev, optional)", value=st.session_state.get("opp2",""))
        out2 = st.checkbox("P2 Key teammate OUT?", value=False)
        players.append(("P2", p2, m2, o2, l2, t2, out2))

        p4 = st.text_input("Player 4 Name", value=st.session_state.get("p4",""))
        m4 = st.selectbox("P4 Market", options=MARKET_OPTIONS, index=MARKET_OPTIONS.index(st.session_state.get("m4","PRA")) if st.session_state.get("m4","PRA") in MARKET_OPTIONS else 0)
        o4 = st.checkbox("P4: Manual line override", value=False)
        l4 = st.number_input("P4 Line", min_value=0.0, value=float(st.session_state.get("l4", 25.0)), step=0.5)
        t4 = st.text_input("P4 Opponent (Team Abbrev, optional)", value=st.session_state.get("opp4",""))
        out4 = st.checkbox("P4 Key teammate OUT?", value=False)
        players.append(("P4", p4, m4, o4, l4, t4, out4))

    run = st.button("Run Model ⚡")

    if run:
        results = []
        warnings = []
        for tag, pname, mkt, manual, manual_line, opp_in, teammate_out in players:
            pname = (pname or "").strip()
            if not pname:
                continue

            market_key = ODDS_MARKETS.get(mkt) or ODDS_MARKETS_OPTIONAL.get(mkt)
            if not market_key:
                warnings.append(f"{tag}: Unsupported market {mkt}")
                continue

            line = manual_line
            meta = None

            if not manual:
                line_found, meta, ferr = find_player_line_from_events(pname, market_key, scan_date_iso, sportsbook)
                if ferr:
                    warnings.append(f"{tag}: Could not auto-fetch The Odds API line ({ferr}). Enable manual override.")
                else:
                    line = float(line_found)
                    st.info(f"{tag} auto The Odds API line detected: {line}")

            if line is None or float(line) <= 0:
                warnings.append(f"{tag}: does not have a valid line.")
                continue

            leg = compute_leg_projection(
                player_name=pname,
                market_name=mkt,
                line=float(line),
                meta=meta,
                n_games=n_games,
                key_teammate_out=bool(teammate_out),
            )

            # user-provided opponent overrides do not block; they only override display
            if opp_in:
                leg["opp"] = opp_in.strip().upper()

            results.append(leg)

        if warnings:
            for w in warnings:
                st.warning(w)

        if not results:
            st.warning("No valid legs configured.")
        else:
            st.session_state["last_results"] = results
            st.success(f"Computed {len(results)} legs. See Results tab.")
            # persist inputs
            st.session_state["p1"], st.session_state["m1"], st.session_state["l1"], st.session_state["opp1"] = p1, m1, l1, t1
            st.session_state["p2"], st.session_state["m2"], st.session_state["l2"], st.session_state["opp2"] = p2, m2, l2, t2
            st.session_state["p3"], st.session_state["m3"], st.session_state["l3"], st.session_state["opp3"] = p3, m3, l3, t3
            st.session_state["p4"], st.session_state["m4"], st.session_state["l4"], st.session_state["opp4"] = p4, m4, l4, t4

    # Logging block (doesn't throw)
    st.markdown("---")
    st.markdown("#### 🗃️ Log This Bet?")
    placed = st.radio("Did you place this bet?", options=["No", "Yes"], horizontal=True, index=0)
    if st.button("Confirm Log Decision"):
        if placed != "Yes":
            st.info("Not logged (you selected No).")
        else:
            # Log the latest run results as one entry with legs list
            res = st.session_state.get("last_results") or []
            if not res:
                st.warning("Nothing to log yet. Run the model first.")
            else:
                entry = {
                    "ts": _now_iso(),
                    "user_id": user_id,
                    "legs": json.dumps(res),
                    "n_legs": len(res),
                    "notes": "",
                    "result": "Pending",
                }
                append_history(user_id, entry)
                st.success("Logged to History (Pending).")

with tabs[1]:
    st.markdown("### Results")
    res = st.session_state.get("last_results") or []
    if not res:
        st.info("Run the model to see projections.")
    else:
        # Player cards
        cols = st.columns(min(4, len(res)))
        for i, leg in enumerate(res):
            c = cols[i % len(cols)]
            with c:
                st.markdown(f"**{leg['player']} — {leg['market']}**")
                if leg.get("headshot"):
                    st.image(leg["headshot"], use_container_width=True)
                st.write(f"Line: {leg.get('line')}")
                st.write(f"Proj (ctx): {None if leg.get('proj') is None else round(leg['proj'],2)}")
                st.write(f"P(Over): {None if leg.get('p_over') is None else round(leg['p_over'],3)}")
                st.write(f"Edge vs 50%: {None if leg.get('edge') is None else round(leg['edge'],3)}")
                tm = leg.get("team") or "?"
                op = leg.get("opp") or "?"
                st.caption(f"Matchup: {tm} vs {op}")
                st.caption(f"Context mult: {round(leg.get('context_mult',1.0),3)} | Blowout risk: {round(leg.get('blowout_prob',0.1),3)}")
                if leg.get("errors"):
                    st.caption("Notes: " + "; ".join(leg["errors"]))

        st.markdown("---")
        st.dataframe(pd.DataFrame(res), use_container_width=True)

with tabs[2]:
    st.markdown("### ⚡ Live Scanner — Single-Leg Edges (The Odds API)")
    st.caption("Pulls player prop lines from The Odds API and ranks legs by model win probability (bootstrap).")

    scan_date2 = st.date_input("Scan date", value=date.today(), key="scan_date2")
    scan_date2_iso = scan_date2.isoformat()

    markets_sel = st.multiselect(
        "Markets",
        options=["Points", "Rebounds", "Assists", "PRA", "PR", "PA", "RA", "3PM", "Blocks", "Steals", "Turnovers"],
        default=["Points", "Rebounds", "Assists"],
    )

    min_prob = st.slider("Min model win prob", min_value=0.50, max_value=0.80, value=0.65, step=0.01)
    max_rows = st.slider("Max rows shown", min_value=10, max_value=200, value=60, step=10)

    # sportsbook choices
    book_choices2, book_err2 = get_sportsbook_choices(scan_date2_iso)
    if book_err2:
        st.info(book_err2)
    sportsbook2 = st.selectbox("Sportsbook (Scanner)", options=["all"] + book_choices2, index=0)

    # Fetch offers button
    if st.button("Fetch Live Lines"):
        st.session_state["scanner_offers"] = None
        st.session_state["scanner_reason"] = None

        selected_keys = []
        for m in markets_sel:
            mk = ODDS_MARKETS.get(m) or ODDS_MARKETS_OPTIONAL.get(m)
            if mk:
                selected_keys.append(mk)
        selected_keys = list(dict.fromkeys(selected_keys))
        if not selected_keys:
            st.warning("Select at least one supported market.")
        else:
            evs, err = odds_get_events(scan_date2_iso)
            if err:
                st.error(err)
            elif not evs:
                st.warning("No events returned for that date.")
            else:
                offers = []
                reasons = []
                for ev in evs:
                    eid = ev.get("id")
                    if not eid:
                        continue
                    odds, oerr = odds_get_event_odds(eid, tuple(selected_keys))
                    if oerr or not odds:
                        reasons.append({"event_id": eid, "reason": oerr or "No odds"})
                        continue
                    for m in markets_sel:
                        mk = ODDS_MARKETS.get(m) or ODDS_MARKETS_OPTIONAL.get(m)
                        if not mk:
                            continue
                        bf = sportsbook2 if sportsbook2 != "all" else None
                        parsed, perr = _parse_player_prop_outcomes(odds, mk, book_filter=bf)
                        if perr:
                            reasons.append({"event_id": eid, "reason": perr})
                        offers.extend([{**r, "market": m} for r in parsed])

                if not offers:
                    st.session_state["scanner_reason"] = reasons
                    st.warning("No offers returned. Confirm your plan includes NBA player props and that props are posted for the selected date.")
                else:
                    df = pd.DataFrame(offers)
                    st.session_state["scanner_offers"] = df
                    st.success(f"Fetched {len(df)} player prop outcomes (raw).")

    # Run scan button
    if st.button("Run Live Scan 🚀"):
        df = st.session_state.get("scanner_offers")
        if df is None or df.empty:
            st.warning("Fetch live lines first.")
        else:
            # Build per-player consensus line if sportsbook2 == all or consensus not used
            # For scanning, we treat each distinct (player, market, line) as a candidate
            out_rows = []
            dropped = []
            # Reduce duplicated Over/Under; we scan Over side by default (standard)
            df2 = df.copy()
            df2["side_norm"] = df2["side"].astype(str).str.lower()
            df2 = df2[df2["side_norm"].isin(["over", "o", "over "]) | (df2["side_norm"]=="over") | (df2["side"]=="Over")].copy()

            if df2.empty:
                st.warning("No Over-side outcomes found in offers. (Some books may name sides differently.)")
                df2 = df.copy()

            for _, r in df2.iterrows():
                pname = r.get("player")
                mkt = r.get("market")
                line = r.get("line")
                if not pname or pd.isna(line) or not mkt:
                    dropped.append({"player": pname, "market": mkt, "reason": "Missing player/market/line"})
                    continue
                meta = {
                    "event_id": r.get("event_id"),
                    "home_team": r.get("home_team"),
                    "away_team": r.get("away_team"),
                    "commence_time": r.get("commence_time"),
                }
                leg = compute_leg_projection(pname, mkt, float(line), meta, n_games=n_games, key_teammate_out=False)
                if leg.get("p_over") is None:
                    dropped.append({"player": pname, "market": mkt, "reason": "Projection failed"})
                    continue
                if leg["p_over"] >= float(min_prob):
                    out_rows.append({
                        "player": pname,
                        "market": mkt,
                        "line": float(line),
                        "p_over": float(leg["p_over"]),
                        "proj": leg.get("proj"),
                        "team": leg.get("team"),
                        "opp": leg.get("opp"),
                        "book": r.get("book"),
                        "event_id": r.get("event_id"),
                    })
                else:
                    dropped.append({"player": pname, "market": mkt, "reason": f"p_over<{min_prob:.2f}"})

            out_df = pd.DataFrame(out_rows).sort_values("p_over", ascending=False).head(int(max_rows))
            st.markdown("#### Scanner Results")
            if out_df.empty:
                st.warning("No legs met the threshold.")
            else:
                st.dataframe(out_df, use_container_width=True)

            with st.expander("Why legs were excluded (debug)", expanded=False):
                if dropped:
                    st.dataframe(pd.DataFrame(dropped).head(200), use_container_width=True)
                else:
                    st.write("No exclusions.")

with tabs[3]:
    st.markdown("### 🗂️ History")
    h = load_history(user_id)
    if h.empty:
        st.info("No bets logged yet.")
    else:
        # Expand legs JSON for display
        try:
            h_disp = h.copy()
            h_disp["legs_preview"] = h_disp["legs"].apply(lambda x: (json.loads(x) if isinstance(x,str) else x))
        except Exception:
            h_disp = h
        st.dataframe(h_disp, use_container_width=True)

        st.markdown("#### Update Result")
        idx = st.number_input("Row index to update", min_value=0, max_value=max(0, len(h)-1), value=0, step=1)
        new_res = st.selectbox("Result", options=["Pending", "HIT", "MISS", "PUSH"], index=0)
        if st.button("Update"):
            try:
                h2 = h.copy()
                h2.loc[int(idx), "result"] = new_res
                h2.to_csv(history_path(user_id), index=False)
                st.success("Updated.")
            except Exception as e:
                st.error(f"Update failed: {e}")

with tabs[4]:
    st.markdown("### 🧪 Calibration (stub)")
    st.info("Calibration UI is reserved for future iterations (CLV learning, drift calibration).")

# Footer
st.caption("© 2025 NBA Prop Quant Engine — Powered by Kamal")
