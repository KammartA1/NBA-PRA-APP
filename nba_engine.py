# ============================================================
# NBA PROP QUANT ENGINE v4.0 — Full Audit + Pro-Grade Signals
# v3.0: Win/Loss splits, Clutch, DNP prob, CI, FTA opp, Playoff,
#        Alt line EV, Middle detection, AI Deep Dive analysis
# v4.0 AUDIT FIXES (2026-03-08):
#   1. [BUG] Mean reversion logic: >= 0.90 was unreachable dead code
#   2. [BUG] Monte Carlo fixed seed=42 → dynamic data-derived seed
#   3. [BUG] Middle probability: zero-centered CDF → mean-centered
#   4. [BUG] Home/away factor: raw ratio → normalized vs season avg
#   5. [NEW] Alt line EV: uses actual bootstrap sigma (not line/4)
#   6. [NEW] Consecutive streak detector (L3/L5 over/under runs)
#   7. [NEW] Implied Team Total (ITT) for game-script context
#   8. [NEW] Lineup injury boost: usage-rate-scaled absorption model
#   9. [FIX] Calibrator: min samples 80→40, adaptive bins
# ============================================================
import os, re, math, time, json, difflib, hashlib, logging, threading, html as _html
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import requests
import streamlit as st
from streamlit_cookies_controller import CookieController
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

# ── Quant System v1.0 Integration ──
try:
    from quant_system.engine import QuantEngine
    from quant_system.core.types import Sport, BetType, SystemState
    from quant_system.risk.kelly_adaptive import KellyConfig
    _QUANT_AVAILABLE = True
except ImportError:
    _QUANT_AVAILABLE = False
# ── Possession-Level Simulation Integration ──
try:
    from simulation.game_engine import GameEngine, SimulationOutput
    from simulation.data_loader import SimulationDataLoader
    from simulation.config import SimulationConfig
    _SIM_AVAILABLE = True
except ImportError:
    _SIM_AVAILABLE = False
# ──────────────────────────────────────────────
# CLAUDE AI INTEGRATION
# ──────────────────────────────────────────────
def _get_anthropic_key():
    """Get Anthropic API key from session state, Streamlit secrets, or environment."""
    override = st.session_state.get("_anthropic_key_override", "")
    if override:
        return override
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")
def _anthropic_client():
    """Return an Anthropic client or None if no key configured."""
    key = _get_anthropic_key()
    if not key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=key)
    except ImportError:
        return None
@st.cache_data(ttl=60*60*4, show_spinner=False)
def ai_explain_edge(player, market, line, side, proj, p_cal, ev_pct,
                    edge_cat, hot_cold, rest_days, dnp_risk, b2b,
                    opp, vol_cv, n_games, errors_str, api_key="",
                    trend_label="Neutral", sharpness_score=None, sharpness_tier=None,
                    game_total=None, fatigue_label="Normal", opp_fatigue_label="Normal",
                    l3_avg=None, l5_avg=None, l10_avg=None):
    """Use Claude Haiku to generate a plain-English edge explanation for one leg."""
    client = _anthropic_client()
    if not client:
        return None
    try:
        import anthropic
        prob_pct = round((p_cal or 0) * 100, 1)
        direction = "OVER" if side.lower() != "under" else "UNDER"
        trend_str = f"Trend: {trend_label} (L3={f'{l3_avg:.1f}' if l3_avg else '--'}, L5={f'{l5_avg:.1f}' if l5_avg else '--'}, L10={f'{l10_avg:.1f}' if l10_avg else '--'})"
        sharp_str = f"Composite Sharpness: {f'{sharpness_score:.0f}' if sharpness_score is not None else '--'}/100 ({sharpness_tier or '--'})"
        game_ctx = f"Game total: {f'{game_total:.0f}' if game_total else 'N/A'} | Player schedule: {fatigue_label} | Opp schedule: {opp_fatigue_label}"
        prompt = f"""You are an elite NBA prop betting analyst. Explain this prop bet projection concisely in 3-4 sentences for a sophisticated quantitative bettor.
Player: {player}
Prop: {market} {direction} {line}
Model projection: {proj} {market} (vs line of {line})
Calibrated win probability: {prob_pct}%
Expected value: {ev_pct:+.1f}%
Edge category: {edge_cat}
{sharp_str}
Recent form: {hot_cold} | {trend_str}
Rest days: {rest_days} | Back-to-back: {b2b} | DNP risk: {dnp_risk}
{game_ctx}
Opponent: {opp}
Stat volatility (CV): {vol_cv}
Sample size: {n_games} games
Model flags: {errors_str or 'None'}
[v4.0 signals if available in data: consecutive streak pattern, implied team total context, lineup injury absorption boost, streak reversion signal]
Write a 3-4 sentence analysis covering: (1) why the model projects this outcome with trend, streak, and sharpness context, (2) key risk factors including fatigue/schedule/lineup changes, (3) one-line bet verdict with conviction level. Be specific, confident, and quantitative. No bullet points."""
        with client.messages.stream(
            model="claude-haiku-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            return stream.get_final_message().content[0].text.strip()
    except Exception as e:
        return f"AI analysis unavailable: {e}"
@st.cache_data(ttl=60*60*2, show_spinner=False)
def ai_slate_briefing(slate_json, api_key=""):
    """Use Claude Sonnet to generate a comprehensive slate analysis."""
    client = _anthropic_client()
    if not client:
        return None
    try:
        import anthropic
        prompt = f"""You are a world-class NBA prop betting quant analyst writing a pre-game slate briefing. Based on the model's edge scan results below, provide a sharp, actionable slate analysis.
SCANNER RESULTS (top edges found):
{slate_json}
Write a structured briefing covering:
1. **Top 3 Plays** — The strongest model edges with specific reasoning
2. **Injury/Context Alerts** — Players with DNP risk, B2B fatigue, or key teammate out
3. **Parlay Opportunity** — 1-2 correlated legs worth combining (same team or game-script)
4. **Fade Candidates** — Overvalued plays the market has wrong
5. **One-Liner Summary** — Today's overall slate quality (1-2 sentences)
Be specific with player names, lines, and percentages. Write like a quant fund's morning brief. Max 350 words."""
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            msg = stream.get_final_message()
            return next((b.text for b in msg.content if b.type == "text"), "").strip()
    except Exception as e:
        return f"AI briefing unavailable: {e}"
@st.cache_data(ttl=60*60*2, show_spinner=False)
def ai_prizepicks_helper(entry_json: str, legs_json: str, api_key: str = "") -> str | None:
    """
    Use Claude to recommend Power Play vs Flex for a PrizePicks entry.
    Inputs are sportsbook-calibrated (p_cal from -110 vig-removed model).
    The core model math is unchanged — this is purely a PP-mode decision layer.
    """
    client = _anthropic_client()
    if not client:
        return None
    try:
        import anthropic
        prompt = f"""You are a PrizePicks strategy expert. The legs below were selected using a sportsbook quantitative model — each player's win probability (p_cal) is calibrated against sharp -110 sportsbook lines (52.4% breakeven), meaning any p_cal above 52.4% already beats the market. On PrizePicks specifically, each leg is a flat 50/50 — so the sportsbook edge acts as a selection filter, not a pricing input.
ENTRY COMBINATIONS (sorted by best EV):
{entry_json}
INDIVIDUAL LEG DETAILS (sportsbook-calibrated):
{legs_json}
For each entry combination, analyze:
1. **Power Play vs Flex** — Which mode maximizes expected value? Consider: joint hit probability, payout multiplier, and whether the miss-insurance of Flex is worth the lower ceiling.
2. **Weakest Leg** — Which leg in the best entry is the most likely to miss (lowest p_cal or highest volatility)? Does this justify Flex?
3. **Correlation Risk** — Are any legs from the same team/game? If yes, does a bad game script hurt multiple legs simultaneously (correlation is a risk here, not a benefit)?
4. **Final Recommendation** — Pick ONE specific entry: the combo, the mode (Power Play or Flex), and the stake. Be decisive and quantitative.
Write in concise bullet format, max 250 words. Reference specific player names and percentages."""
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=450,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            return stream.get_final_message().content[0].text.strip()
    except Exception as e:
        return f"PP Helper unavailable: {e}"
@st.cache_data(ttl=60*60*2, show_spinner=False)
def ai_edge_deepdive(legs_json: str, api_key: str = "") -> str | None:
    """
    Use Claude Sonnet for an ultra-deep edge analysis on the current legs.
    Combines all v3.0 signals: W/L split, clutch, playoff context, CI, middle alerts.
    Returns a comprehensive analysis with ranked conviction levels.
    """
    client = _anthropic_client()
    if not client:
        return None
    try:
        prompt = f"""You are an elite NBA prop betting quant fund manager running a deep-dive edge analysis.
You have access to advanced v3.0 signals including win/loss performance splits, clutch statistics,
playoff implications, confidence intervals, and middle opportunity detection.
LEGS WITH FULL SIGNAL DATA:
{legs_json}
Provide a DEEP DIVE analysis structured as follows:
**CONVICTION RANKING** (rank legs 1-N by overall confidence)
For each leg: Conviction score 1-10 with specific reasoning referencing the data signals.
**EDGE DECOMPOSITION**
Break down WHERE the edge comes from for the top 2 legs:
- Statistical: Is the model projection significantly above/below the market line?
- Contextual: W/L splits, clutch factor, playoff situation, game environment
- Market Structure: Line movement, sharp book divergence, DNP risk adjustment
**RISK MATRIX**
For each leg, identify:
- Key Risk #1 (what kills this bet)
- Key Risk #2 (what makes it worse)
- Confidence Interval: Is the CI tight (high conviction) or wide (proceed with caution)?
**MIDDLE ALERTS**
If any middle opportunities detected, explain the exact strategy: which books, which lines, why it's profitable.
**FINAL SLATE STRATEGY**
1-2 sentences on optimal entry: which leg(s) to fire at full Kelly vs reduced Kelly vs pass.
Be specific, brutal, and quantitative. Max 450 words."""
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            return stream.get_final_message().content[0].text.strip()
    except Exception as e:
        return f"Deep dive unavailable: {e}"
@st.cache_data(ttl=60*60*2, show_spinner=False)
def ai_parlay_optimizer(legs_json, api_key=""):
    """Use Claude Sonnet to recommend optimal parlay combinations."""
    client = _anthropic_client()
    if not client:
        return None
    try:
        import anthropic
        prompt = f"""You are a professional NBA prop parlay optimizer. Given these model-validated legs, recommend the optimal 2-4 leg parlay combinations considering correlation, EV, and risk.
AVAILABLE LEGS (with model data):
{legs_json}
For each recommended parlay:
- List the legs (player + market + line + side)
- Explain WHY these legs correlate well (or are independent)
- Give a combined probability estimate
- Note any correlation risk (same team = shared risk)
- Rate: ⭐⭐⭐ (Best), ⭐⭐ (Good), ⭐ (Speculative)
Suggest 2-3 parlay combinations. Be concise and quantitative. Max 300 words."""
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            return stream.get_final_message().content[0].text.strip()
    except Exception as e:
        return f"AI parlay optimizer unavailable: {e}"
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import (
    playergamelog, scoreboardv2, LeagueDashTeamStats, CommonPlayerInfo
)
# ──────────────────────────────────────────────
# SAFE HELPERS
# ──────────────────────────────────────────────
def safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except Exception:
        return default
def safe_round(v, d=2):
    try:
        return round(float(v), d) if v is not None else None
    except Exception:
        return None
def _now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
def normalize_name(name: str) -> str:
    if not name:
        return ""
    import unicodedata
    # [AUDIT FIX] Normalize accented characters (Joël → joel, Nikola → nikola)
    # Prevents player lookup failures for international players with diacritics
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    s = re.sub(r"[\.\'\-]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()
def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur
# ══════════════════════════════════════════════
# RESEARCH-VALIDATED CONSTANTS (v3.0)
# Sources: VSiN HCA data, EVAnalytics, PubMed,
#          RefAnalytics.com, NBAstuffer referee DB
# ══════════════════════════════════════════════
# [Research] Team-specific Home Court Advantage margins (net pts vs league avg 3.0)
# Source: VSiN true HCA study — updated annually from 2023-25 seasons
TEAM_HCA_MARGIN = {
    # Strong home courts (>5.0 net margin)
    "OKC": 7.0, "BOS": 6.2, "MIL": 5.8, "CLE": 5.5, "MIN": 5.3,
    "DEN": 5.1, "IND": 5.0,
    # Average home courts (3.0-5.0)
    "NYK": 4.5, "PHX": 4.2, "GSW": 4.0, "DAL": 3.8, "SAC": 3.5,
    "CHI": 3.4, "HOU": 3.2, "NOP": 3.1, "LAL": 3.0, "MEM": 3.0,
    "POR": 3.0, "UTA": 3.0, "ORL": 2.9,
    # Weak home courts (<3.0)
    "ATL": 2.7, "BKN": 2.5, "TOR": 2.4, "WAS": 2.3, "PHI": 2.2,
    "SAS": 2.0, "DET": 1.8, "CHA": 1.6, "LAC": 1.4, "MIA": 0.8,
}
LEAGUE_AVG_HCA = 3.0  # league-average home court advantage in pts
# [Research] Position-specific B2B performance penalty.
# Source: PubMed 2020 study — Guards (5+ miles/game movement) hit hardest.
# Guards: -1% extra vs base B2B penalty. Centers: steady per-minute production.
POSITION_B2B_EXTRA_PENALTY = {
    "Guard": 0.008,   # extra 0.8% beyond base B2B penalty
    "Wing":  0.004,   # extra 0.4%
    "Big":   0.001,   # minimal (Centers most consistent on B2B)
    "Unknown": 0.004,
}
# [Research] Q1 scoring fraction of full-game.
# Source: EVAnalytics Q1 data — Q1 is 26-28% of full-game, not 25%.
# Q4 in close games trends under; Q1 overshoots slightly.
Q1_SCORING_FRACTION = 0.265  # 26.5% of full-game is more accurate than 25%
# [Research] Referee crew foul tendency database.
# Source: NBAstuffer 2024-25 referee stats, RefAnalytics.com, Covers.com.
# Format: {crew_chief_name_normalized: {fouls_per_100: float, o_u_lean: float}}
# o_u_lean: fraction of games going OVER (>0.52 = over-heavy crew)
# fouls_per_100: league average ~44 fouls per 100 possessions
REFEREE_FOUL_DB = {
    # Foul-heavy refs (>46 fouls/100) — good for FTA/FTM props
    "scott foster":      {"fouls_per_100": 48.2, "o_u_lean": 0.56, "tier": "Foul-Heavy"},
    "james capers":      {"fouls_per_100": 47.8, "o_u_lean": 0.55, "tier": "Foul-Heavy"},
    "tony brothers":     {"fouls_per_100": 47.1, "o_u_lean": 0.54, "tier": "Foul-Heavy"},
    "ed malloy":         {"fouls_per_100": 46.5, "o_u_lean": 0.53, "tier": "Foul-Heavy"},
    "kane fitzgerald":   {"fouls_per_100": 46.3, "o_u_lean": 0.53, "tier": "Foul-Heavy"},
    # Average refs (44-46 fouls/100)
    "marc davis":        {"fouls_per_100": 45.0, "o_u_lean": 0.51, "tier": "Average"},
    "zach zarba":        {"fouls_per_100": 44.8, "o_u_lean": 0.51, "tier": "Average"},
    "bill kennedy":      {"fouls_per_100": 44.5, "o_u_lean": 0.50, "tier": "Average"},
    "ken mauer":         {"fouls_per_100": 44.2, "o_u_lean": 0.50, "tier": "Average"},
    "derek richardson":  {"fouls_per_100": 44.0, "o_u_lean": 0.50, "tier": "Average"},
    # Foul-light refs (<44 fouls/100) — favors UNDER on FTA/FTM
    "bennie adams":      {"fouls_per_100": 42.8, "o_u_lean": 0.47, "tier": "Foul-Light"},
    "eric lewis":        {"fouls_per_100": 42.5, "o_u_lean": 0.47, "tier": "Foul-Light"},
    "j.t. orr":          {"fouls_per_100": 42.2, "o_u_lean": 0.46, "tier": "Foul-Light"},
    "jason phillips":    {"fouls_per_100": 41.9, "o_u_lean": 0.46, "tier": "Foul-Light"},
    "pat fraher":        {"fouls_per_100": 41.5, "o_u_lean": 0.46, "tier": "Foul-Light"},
}
_LEAGUE_AVG_FOULS_PER_100 = 44.5  # NBA 2024-25 season average
def get_referee_foul_factor(crew_chief_name, market):
    """
    Returns (ref_factor, ref_label, ref_tier) for foul-sensitive props.
    crew_chief_name: str (from NBA schedule API or manual entry).
    Only meaningful for: FTA, FTM, Stocks, Steals, Points.
    """
    if market not in ("FTA", "FTM", "Stocks", "Steals", "Points", "PRA"):
        return 1.0, "N/A", "N/A"
    if not crew_chief_name:
        return 1.0, "Avg Crew", "Unknown"
    nm = str(crew_chief_name).strip().lower()
    ref_data = REFEREE_FOUL_DB.get(nm)
    if ref_data is None:
        # Try partial match
        for k, v in REFEREE_FOUL_DB.items():
            if nm in k or k in nm:
                ref_data = v
                break
    if ref_data is None:
        return 1.0, "Unknown Crew", "Unknown"
    fouls = float(ref_data["fouls_per_100"])
    ratio = fouls / _LEAGUE_AVG_FOULS_PER_100
    # Market-specific weight: FTA/FTM most sensitive, Points partial
    market_weight = {"FTA": 1.0, "FTM": 1.0, "Stocks": 0.25, "Steals": 0.35, "Points": 0.20, "PRA": 0.12}
    w = market_weight.get(market, 0.15)
    factor = float(np.clip(1.0 + (ratio - 1.0) * w, 0.95, 1.05))
    tier = ref_data.get("tier", "Average")
    label = f"{tier} crew ({fouls:.1f}/100)"
    return factor, label, tier
# [Research] Team-specific HCA multiplier for home/away factor computation.
# Replaces generic 12% cap with research-validated team-specific margins.
def get_team_hca_factor(team_abbr, is_home, market):
    """
    Returns (hca_factor, hca_label) based on team-specific HCA margins.
    Research: OKC+7.0 vs MIA+0.8. Generic 3.0 league avg normalizes to 1.0.
    Converts margin difference to a proportion of typical game scoring (~115 pts).
    """
    if team_abbr is None or is_home is None:
        return 1.0, "N/A"
    try:
        team_hca = TEAM_HCA_MARGIN.get(str(team_abbr).upper(), LEAGUE_AVG_HCA)
        # How much above/below league average is this team's HCA?
        hca_premium = (team_hca - LEAGUE_AVG_HCA) / 115.0  # normalize to per-point fraction
        # Scoring props more sensitive to HCA; defensive stats less
        market_sensitivity = {
            "Points": 1.0, "PRA": 0.9, "PA": 0.85, "PR": 0.8, "Assists": 0.7,
            "Rebounds": 0.5, "RA": 0.5, "3PM": 0.8, "Blocks": 0.3, "Steals": 0.3,
            "Stocks": 0.3, "FTA": 0.9, "FTM": 0.9,
        }
        sensitivity = market_sensitivity.get(market, 0.6)
        factor = 1.0 + (hca_premium * sensitivity * (1.0 if is_home else -1.0))
        factor = float(np.clip(factor, 0.97, 1.03))
        if factor >= 1.015:
            label = f"HCA Boost ({team_abbr}+{team_hca:.1f})"
        elif factor <= 0.985:
            label = f"Away Drag ({team_abbr}+{team_hca:.1f})"
        else:
            label = "Avg HCA"
        return factor, label
    except Exception:
        return 1.0, "Avg HCA"
# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
ODDS_BASE       = "https://api.the-odds-api.com/v4"
SPORT_KEY_NBA   = "basketball_nba"
REGION_US       = "us"
MIN_MINUTES_THRESHOLD = 10  # [FIX 3] filter DNP/garbage-time
ODDS_MARKETS = {
    # ── Full-game standard ──────────────────────
    "Points":          "player_points",
    "Rebounds":        "player_rebounds",
    "Assists":         "player_assists",
    "3PM":             "player_threes",
    "PRA":             "player_points_rebounds_assists",
    "PR":              "player_points_rebounds",
    "PA":              "player_points_assists",
    "RA":              "player_rebounds_assists",
    "Blocks":          "player_blocks",
    "Steals":          "player_steals",
    "Turnovers":       "player_turnovers",
    "Stocks":          "player_blocks_steals",
    # ── 1st Half markets ────────────────────────
    # Odds API supports both q1q2 and first_half variants; q1q2 is most common
    "H1 Points":       "player_points_q1q2",
    "H1 Rebounds":     "player_rebounds_q1q2",
    "H1 Assists":      "player_assists_q1q2",
    "H1 3PM":          "player_threes_q1q2",
    "H1 PRA":          "player_points_rebounds_assists_q1q2",
    # ── 2nd Half markets ────────────────────────
    "H2 Points":       "player_points_q3q4",
    "H2 Rebounds":     "player_rebounds_q3q4",
    "H2 Assists":      "player_assists_q3q4",
    "H2 3PM":          "player_threes_q3q4",
    "H2 PRA":          "player_points_rebounds_assists_q3q4",
    # ── 1st Quarter markets ─────────────────────
    "Q1 Points":       "player_points_q1",
    "Q1 Rebounds":     "player_rebounds_q1",
    "Q1 Assists":      "player_assists_q1",
    # ── Alternate lines ─────────────────────────
    "Alt Points":      "player_points_alternate",
    "Alt Rebounds":    "player_rebounds_alternate",
    "Alt Assists":     "player_assists_alternate",
    "Alt 3PM":         "player_threes_alternate",
    # ── Fantasy score ────────────────────────────
    "Fantasy Score":   "player_fantasy_points",
    # ── Combo / special ─────────────────────────
    "Double Double":   "player_double_double",
    "Triple Double":   "player_triple_double",
    "First Basket":    "player_first_basket",
    # ── Shooting volume ───────────────────────────
    # Confirmed Odds API key: player_field_goals = FGM made.
    # FGA/FTM/FTA/3PA: no confirmed Odds API key; selectable for PP/UD/Sleeper source only.
    "FGM":             "player_field_goals",
    "FGA":             "player_field_goals_attempted",
    "3PA":             "player_three_point_field_goals_attempted",
    "FTM":             "player_free_throws_made",
    "FTA":             "player_free_throws_attempted",
}
# Markets with no confirmed Odds API key — available via PP/UD/Sleeper only.
# These will be skipped during Odds API fetches and the user will be warned.
ODDS_API_UNSUPPORTED_MARKETS = {
    "FGA", "3PA", "FTM", "FTA",
    # H1/H2 markets return HTTP 422 from Odds API — source from PrizePicks instead
    "H1 Points", "H1 Rebounds", "H1 Assists", "H1 3PM", "H1 PRA",
    "H2 Points", "H2 Rebounds", "H2 Assists", "H2 3PM", "H2 PRA",
}
# Same set but keyed by Odds API market key (for functions that work with raw API keys)
_UNSUPPORTED_API_KEYS = {ODDS_MARKETS[m] for m in ODDS_API_UNSUPPORTED_MARKETS if m in ODDS_MARKETS}
# ──────────────────────────────────────────────
# DFS PLATFORM PAYOUT STRUCTURES
# PrizePicks / Underdog / Sleeper use identical or similar payout tables.
# Individual legs are flat 50% breakeven — the EV lives in the multi-leg structure.
# ──────────────────────────────────────────────
DFS_PP_PAYOUTS = {        # Power Play — must hit ALL legs
    2: 3.0,
    3: 5.0,
    4: 10.0,
    5: 20.0,
    6: 40.0,
}
DFS_UD_PAYOUTS = {        # Underdog Pick'em — similar tiers
    2: 3.0,
    3: 6.0,
    4: 10.0,
    5: 20.0,
}
DFS_SLEEPER_PAYOUTS = {   # Sleeper default
    2: 3.0,
    3: 5.0,
    4: 10.0,
    5: 20.0,
}
DFS_PP_FLEX_PAYOUTS = {   # Flex — hit k-of-n for tiered payout
    3: {2: 1.25, 3: 2.25},
    4: {3: 1.5,  4: 5.0},
    5: {4: 2.0,  5: 10.0},
    6: {4: 0.4,  5: 2.0,  6: 25.0},
}
def dfs_power_play_ev(joint_prob: float, n_legs: int, platform: str = "prizepicks") -> float | None:
    """EV of a DFS power play entry (must hit all legs)."""
    tbl = {"prizepicks": DFS_PP_PAYOUTS, "underdog": DFS_UD_PAYOUTS, "sleeper": DFS_SLEEPER_PAYOUTS}
    payouts = tbl.get(platform.lower(), DFS_PP_PAYOUTS)
    mult = payouts.get(n_legs)
    return (mult * joint_prob - 1.0) if mult else None
def dfs_flex_ev(hit_dist: dict, n_legs: int) -> float | None:
    """EV of a DFS flex entry. hit_dist = {k: P(exactly k legs hit)}."""
    tier = DFS_PP_FLEX_PAYOUTS.get(n_legs, {})
    if not tier:
        return None
    ev = -1.0
    for k_correct, payout_mult in tier.items():
        ev += payout_mult * hit_dist.get(int(k_correct), 0.0)
    return ev
def dfs_entry_optimizer(legs: list, platform: str = "prizepicks",
                        max_legs: int = 4, n_sims: int = 4000,
                        bankroll: float = 1000.0, frac_kelly: float = 0.25) -> list:
    """
    Find optimal DFS power-play + flex combos using Gaussian copula Monte Carlo.
    Returns list of dicts sorted by best EV (Power Play or Flex, whichever is higher).
    """
    from itertools import combinations
    import scipy.stats as _sc
    valid = [l for l in legs if float(l.get("p_cal") or 0) > 0.50]
    if len(valid) < 2:
        return []
    nv = len(valid)
    full_corr = np.eye(nv)
    for _i in range(nv):
        for _j in range(_i + 1, nv):
            c = float(estimate_player_correlation(valid[_i], valid[_j]) or 0.0)
            full_corr[_i, _j] = full_corr[_j, _i] = c
    # [AUDIT FIX] Data-derived seed: fixed seed=7 produced identical noise patterns for
    # every DFS entry optimization call. Use player IDs + probs to ensure uniqueness.
    _dfs_seed = int(abs(hash(tuple(
        int(float(l.get("p_cal", 0.5)) * 1e6) + int(l.get("player_id") or 0)
        for l in valid
    ))) % (2**31))
    rng = np.random.default_rng(_dfs_seed)
    results = []
    tbl = {"prizepicks": DFS_PP_PAYOUTS, "underdog": DFS_UD_PAYOUTS, "sleeper": DFS_SLEEPER_PAYOUTS}
    payouts_tbl = tbl.get(platform.lower(), DFS_PP_PAYOUTS)
    for n in range(2, min(max_legs + 1, nv + 1)):
        if n not in payouts_tbl:
            continue
        for combo in combinations(range(nv), n):
            combo_legs = [valid[i] for i in combo]
            probs = np.array([float(l["p_cal"]) for l in combo_legs])
            # Build PSD correlation sub-matrix
            sub = full_corr[np.ix_(list(combo), list(combo))]
            evals, evecs = np.linalg.eigh(sub)
            evals = np.clip(evals, 1e-6, None)
            corr_psd = evecs @ np.diag(evals) @ evecs.T
            # [AUDIT FIX] Renormalize diagonal to 1.0 and enforce symmetry after eigenvalue
            # clipping — matches the PSD handling in kelly_parlay_optimizer and
            # monte_carlo_game_sim. Skipping this caused subtle numerical drift.
            _d = np.sqrt(np.maximum(np.diag(corr_psd), 1e-12))
            corr_psd = corr_psd / np.outer(_d, _d)
            np.fill_diagonal(corr_psd, 1.0)
            corr_psd = (corr_psd + corr_psd.T) / 2.0
            # Gaussian copula MC
            z = rng.multivariate_normal(np.zeros(n), corr_psd, n_sims)
            u = _sc.norm.cdf(z)
            hits = (u < probs).astype(int)          # shape (n_sims, n)
            per_sim_k = hits.sum(axis=1)             # how many legs hit per sim
            joint_prob = float((per_sim_k == n).mean())
            naive_joint = float(np.prod(probs))
            # Hit distribution P(exactly k hit)
            hit_dist = {k: float((per_sim_k == k).mean()) for k in range(n + 1)}
            # Power play EV
            pp_ev = dfs_power_play_ev(joint_prob, n, platform)
            # Flex EV (if available for this n)
            flex_ev = dfs_flex_ev(hit_dist, n)
            # Per-leg DFS edge (average probability above 50% floor)
            avg_edge = float(np.mean(probs) - 0.5)
            best_ev = max(
                pp_ev if pp_ev is not None else -99,
                flex_ev if flex_ev is not None else -99,
            )
            rec_mode = "PowerPlay" if (pp_ev or -99) >= (flex_ev or -99) else "Flex"
            # Kelly stake using best_ev / (payout - 1)
            payout_used = payouts_tbl.get(n, 3.0) if rec_mode == "PowerPlay" else (
                DFS_PP_FLEX_PAYOUTS.get(n, {}).get(n, 3.0)
            )
            kelly_f = max(0.0, best_ev / (payout_used - 1.0)) if payout_used > 1 else 0.0
            stake = min(bankroll * frac_kelly * kelly_f, bankroll * 0.05)
            results.append({
                "combo": " + ".join(
                    f"{l['player']} {l['market']} O{l.get('line','?')}" for l in combo_legs
                ),
                "n_legs": n,
                "platform": platform,
                "joint_prob_%": round(joint_prob * 100, 1),
                "naive_prob_%": round(naive_joint * 100, 1),
                "pp_payout_x": payouts_tbl.get(n, "—"),
                "pp_ev_%": round(pp_ev * 100, 1) if pp_ev is not None else "—",
                "flex_ev_%": round(flex_ev * 100, 1) if flex_ev is not None else "—",
                "rec_mode": rec_mode,
                "best_ev_%": round(best_ev * 100, 1) if best_ev > -99 else "—",
                "avg_leg_edge_%": round(avg_edge * 100, 1),
                "rec_stake_$": round(stake, 2),
            })
    return sorted(results, key=lambda x: x["best_ev_%"] if isinstance(x["best_ev_%"], (int, float)) else -99, reverse=True)[:30]
# Markets that require batching separately (not all books offer these)
# Only keys CONFIRMED to return data from the Odds API (tested 2026-03-07)
SPECIALTY_MARKET_KEYS = {
    # Half-game markets (open ~1-2h before tip-off on DK/FD)
    "player_points_q1q2", "player_rebounds_q1q2",
    "player_assists_q1q2", "player_threes_q1q2",
    "player_points_rebounds_assists_q1q2",
    "player_points_q3q4", "player_rebounds_q3q4", "player_assists_q3q4",
    "player_points_rebounds_assists_q3q4",
    # 1Q markets
    "player_points_q1", "player_rebounds_q1", "player_assists_q1",
    # Alt lines
    "player_points_alternate", "player_rebounds_alternate",
    "player_assists_alternate", "player_threes_alternate",
    # Shooting volume
    "player_field_goals",
    "player_field_goals_attempted",
    "player_free_throws_made",
    "player_free_throws_attempted",
    "player_three_point_field_goals_attempted",
    # Special / binary markets
    "player_double_double", "player_triple_double", "player_first_basket",
    # Fantasy
    "player_fantasy_points",
}
STAT_FIELDS = {
    "Points":          "PTS",
    "Rebounds":        "REB",
    "Assists":         "AST",
    "3PM":             "FG3M",
    "PRA":             ("PTS","REB","AST"),
    "PR":              ("PTS","REB"),
    "PA":              ("PTS","AST"),
    "RA":              ("REB","AST"),
    "Blocks":          "BLK",
    "Steals":          "STL",
    "Turnovers":       "TOV",
    "Stocks":          ("BLK","STL"),
    # Half markets map to full-game fields (adjusted via HALF_FACTOR)
    "H1 Points":       "PTS",
    "H1 Rebounds":     "REB",
    "H1 Assists":      "AST",
    "H1 3PM":          "FG3M",
    "H1 PRA":          ("PTS","REB","AST"),
    "H2 Points":       "PTS",
    "H2 Rebounds":     "REB",
    "H2 Assists":      "AST",
    "H2 PRA":          ("PTS","REB","AST"),
    # 1Q markets map to full-game fields (adjusted via Q1_FACTOR)
    "Q1 Points":       "PTS",
    "Q1 Rebounds":     "REB",
    "Q1 Assists":      "AST",
    # Alt lines use same fields as base
    "Alt Points":      "PTS",
    "Alt Rebounds":    "REB",
    "Alt Assists":     "AST",
    "Alt 3PM":         "FG3M",
    # Fantasy score: PTS + 1.2*REB + 1.5*AST + 3*(BLK+STL) - TOV (DK-style)
    "Fantasy Score":   ("PTS","REB","AST","BLK","STL","TOV"),
    # Combo / special
    "Double Double":   ("PTS","REB","AST","BLK","STL"),
    "Triple Double":   ("PTS","REB","AST","BLK","STL"),
    "First Basket":    "PTS",
    # Shooting volume
    "FGM":             "FGM",
    "FGA":             "FGA",
    "3PA":             "FG3A",
    "FTM":             "FTM",
    "FTA":             "FTA",
}
# Half-game projection scale factors (league-average baseline)
HALF_FACTOR = {
    "H1 Points": 0.52, "H1 Rebounds": 0.52, "H1 Assists": 0.52,
    "H1 3PM": 0.52, "H1 PRA": 0.52, "H1 PR": 0.52, "H1 PA": 0.52,
    "H1 FTM": 0.52, "H1 FTA": 0.52, "H1 FGM": 0.52, "H1 FGA": 0.52,
    "H2 Points": 0.48, "H2 Rebounds": 0.48, "H2 Assists": 0.48,
    "H2 3PM": 0.48, "H2 PRA": 0.48, "H2 PR": 0.48, "H2 PA": 0.48,
    "H2 FTM": 0.48, "H2 FTA": 0.48, "H2 FGM": 0.48, "H2 FGA": 0.48,
    # 1Q markets: Q1_SCORING_FRACTION of full-game (research-validated: 26.5%, not 25%)
    "Q1 Points": Q1_SCORING_FRACTION, "Q1 Rebounds": 0.25, "Q1 Assists": 0.25,
    "Q1 3PM": 0.25, "Q1 PRA": Q1_SCORING_FRACTION, "Q1 FTM": 0.24, "Q1 FGA": Q1_SCORING_FRACTION,
}
# [AUDIT FIX] Position-specific half-game adjustment deltas on top of HALF_FACTOR baseline.
# Guards attack more in H1 (faster pace, early shot attempts); Bigs grab more boards in H2
# (closeouts, putbacks in crunch time); Wings are near-neutral.
_HALF_POS_DELTA = {
    "Guard": {
        "H1 Points": +0.02, "H1 3PM": +0.02, "H1 FGA": +0.02, "H1 FGM": +0.01,
        "H2 Rebounds": -0.02, "H2 PR": -0.01,
    },
    "Big": {
        "H2 Rebounds": +0.03, "H2 PR": +0.02, "H2 RA": +0.02,
        "H1 FTM": -0.02, "H1 FTA": -0.02,   # fewer foul-drawing touches in H1
    },
    "Wing": {},   # near-neutral; real split captured by boxscore data when available
    "Unknown": {},
}
def get_half_factor(market_name, position_bucket="Unknown", spread_abs=None, game_total=None):
    """Return position-adjusted half-game scaling factor.
    [v5.0] Voulgaris close-game H2 boost:
    In close games (spread ≤ 5), Q4 sees intentional fouling + extra possessions,
    driving H2 scoring ~4-6% above the naïve 48% split. Blowouts see H2 < 48%
    (starters sit). Sportsbooks often price H2 totals at exactly ½ of game total
    — exploiting this asymmetry was one of Voulgaris's most profitable edges.
    [AUDIT FIX] Q1 dynamic factor: high-pace games (game_total 230+) overrepresent
    Q1 scoring; slow games (game_total ≤205) underrepresent. Adjusts the static 26.5%.
    Range: 0.235–0.280 depending on pace environment.
    """
    base = HALF_FACTOR.get(market_name, 1.0)
    if base == 1.0:
        return base   # not a half/Q1 market
    delta = _HALF_POS_DELTA.get(position_bucket, {}).get(market_name, 0.0)
    base_adj = float(np.clip(base + delta, 0.05, 0.95))
    # [v5.0] H2 close-game boost (Points, FTA markets specifically)
    if market_name.startswith("H2") and spread_abs is not None:
        try:
            s = float(spread_abs)
            if s <= 3.0:
                # Very close: +4% on H2 scoring due to intentional fouling + extra possessions
                close_boost = 0.04
            elif s <= 5.0:
                close_boost = 0.025  # Competitive: +2.5%
            elif s >= 12.0:
                close_boost = -0.025  # Expected blowout: H2 scoring deflated (starters sit)
            else:
                close_boost = 0.0
            if market_name in ("H2 Points", "H2 PRA", "H2 PA", "H2 FTM", "H2 FTA"):
                base_adj = float(np.clip(base_adj + close_boost, 0.05, 0.60))
        except Exception:
            pass
    # [AUDIT FIX] Q1 dynamic fraction: pace-adjusted (game_total drives Q1 scoring weight)
    if market_name.startswith("Q1") and game_total is not None:
        try:
            gt = float(game_total)
            # League avg ~220. Each 5 pts above → Q1 fraction goes up ~0.003 (mirrors pace effect)
            q1_adj = (gt - 220.0) / 5.0 * 0.003
            base_adj = float(np.clip(base_adj + q1_adj, 0.23, 0.29))
        except Exception:
            pass
    return base_adj
# Alt markets — same engine, different API key
ALT_MARKETS = {"Alt Points","Alt Rebounds","Alt Assists","Alt 3PM"}
# Fantasy score markets need custom stat computation
FANTASY_MARKETS = {"Fantasy Score"}
# DD/TD markets — probability from game log, not bootstrap
DD_TD_MARKETS = {"Double Double","Triple Double"}
BOOK_SHARPNESS = {
    "pinnacle":0.99,"circa":0.95,"bookmaker":0.90,"betcris":0.85,
    "draftkings":0.70,"fanduel":0.70,"betmgm":0.65,"caesars":0.65,
    "betrivers":0.60,"pointsbetus":0.55,
    "betonlineag":0.45,"mybookieag":0.30,
}
def book_sharpness(k):
    return float(BOOK_SHARPNESS.get((k or "").strip().lower(), 0.55))
POSITIONAL_PRIORS = {
    "Guard": {"Points":16.5,"Rebounds":3.4,"Assists":5.8,"3PM":2.1,
              "PRA":25.7,"PR":19.9,"PA":22.3,"RA":9.2,"Blocks":0.4,"Steals":1.2,"Turnovers":2.2,
              "Q1 Points":4.1,"Q1 Rebounds":0.9,"Q1 Assists":1.5,"Fantasy Score":31.2,
              "FGM":5.8,"FGA":13.5,"3PA":6.2,"FTM":3.2,"FTA":3.8},
    "Wing":  {"Points":14.8,"Rebounds":5.9,"Assists":2.9,"3PM":1.6,
              "PRA":23.6,"PR":20.7,"PA":17.7,"RA":8.8,"Blocks":0.8,"Steals":1.0,"Turnovers":1.7,
              "Q1 Points":3.7,"Q1 Rebounds":1.5,"Q1 Assists":0.7,"Fantasy Score":27.4,
              "FGM":5.4,"FGA":12.0,"3PA":4.5,"FTM":2.6,"FTA":3.2},
    "Big":   {"Points":13.2,"Rebounds":8.8,"Assists":2.1,"3PM":0.5,
              "PRA":24.1,"PR":22.0,"PA":15.3,"RA":10.9,"Blocks":1.4,"Steals":0.7,"Turnovers":2.0,
              "Q1 Points":3.3,"Q1 Rebounds":2.2,"Q1 Assists":0.5,"Fantasy Score":30.5,
              "FGM":5.0,"FGA":10.5,"3PA":1.4,"FTM":3.0,"FTA":4.0},
    "Unknown":{"Points":14.8,"Rebounds":5.5,"Assists":3.5,"3PM":1.4,
              "PRA":23.8,"PR":20.3,"PA":18.3,"RA":9.0,"Blocks":0.8,"Steals":0.9,"Turnovers":1.9,
              "Q1 Points":3.7,"Q1 Rebounds":1.5,"Q1 Assists":0.9,"Fantasy Score":29.7,
              "FGM":5.4,"FGA":12.0,"3PA":4.0,"FTM":2.9,"FTA":3.6},
}
# [AUDIT CALIBRATION] Rest multipliers re-calibrated to empirical NBA research:
# PubMed 2020 (Esteves et al.): B2B team performance declines ~1.9 pts/game (~2.7% of ~70-pt total).
# Individual scoring props: ~3-4% decline; rebounding/assists: ~1.5-2% decline.
# Previous 0.93 (-7%) was ~2x too aggressive vs. published data.
# 1-day rest: modest ~1.5% shortfall vs. 2+ days. 3-4 days: slight benefit from freshness.
REST_MULTIPLIERS = {0: 0.965, 1: 0.985, 2: 1.00, 3: 1.01, 4: 1.015}
# Exponential recency decay per stat (assists autocorrelate longer than blocks)
# [v6.0] Steeper recency decay — recent games matter MORE for edge detection.
# Research: NBA player performance has ~3-5 game autocorrelation windows.
# Lowering λ gives more weight to recent games for sharper projections.
LAMBDA_DECAY_BY_STAT = {
    "Points": 0.85, "Rebounds": 0.82, "Assists": 0.85,
    "3PM": 0.80, "PRA": 0.84, "PR": 0.83, "PA": 0.84, "RA": 0.82,
    "Blocks": 0.78, "Steals": 0.80, "Turnovers": 0.83, "Stocks": 0.78,
    "H1 Points": 0.85, "H1 Rebounds": 0.82, "H1 Assists": 0.85,
    "H2 Points": 0.85, "H2 Rebounds": 0.82, "H2 Assists": 0.85,
    "Q1 Points": 0.84, "Q1 Rebounds": 0.81, "Q1 Assists": 0.84,
    "Alt Points": 0.85, "Alt Rebounds": 0.82, "Alt Assists": 0.85, "Alt 3PM": 0.80,
    "Fantasy Score": 0.85,
    "default": 0.85,
}
# [v5.0] Count stats where Negative Binomial outperforms bootstrap (overdispersed count data)
# NegBin is theoretically superior for zero-inflated, overdispersed integer distributions.
# Research: BinomialBasketball.com Stan model; squared2020.com NegBin for 3PM
NEGBINOM_MARKETS = frozenset({
    "3PM", "Blocks", "Steals", "Stocks", "Assists",
    "Rebounds", "RA", "PR", "Turnovers", "FTA", "FTM",
    # FGM/FGA/3PA are also overdispersed integer counts — NegBin improves calibration
    "FGM", "FGA", "3PA",
})
# Persistent file paths
OPENING_LINES_PATH   = "opening_lines.json"
WATCHLIST_PATH_TPL   = "watchlist_{uid}.json"
SCANNER_CACHE_PATH   = "scanner_results_cache.pkl"
ALERT_HASHES_PATH    = "alert_hashes_cache.json"
PP_SETTINGS_PATH     = "pp_settings.json"
def load_pp_settings():
    """Load persisted PrizePicks settings (cookies/JSON, etc.) from disk."""
    try:
        if os.path.exists(PP_SETTINGS_PATH):
            with open(PP_SETTINGS_PATH) as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}
def save_pp_settings(**kwargs):
    """Persist PrizePicks settings to disk.  Pass keyword args to update individual keys."""
    existing = load_pp_settings()
    existing.update(kwargs)
    try:
        with open(PP_SETTINGS_PATH, "w") as f:
            json.dump(existing, f)
    except Exception:
        pass
def _save_scanner_cache():
    """Persist scanner results to disk so they survive server restarts / WebSocket drops."""
    import pickle
    try:
        _cache = {
            "scanner_results":       st.session_state.get("scanner_results"),
            "scanner_dropped":       st.session_state.get("scanner_dropped"),
            "scanner_scan_id":       st.session_state.get("scanner_scan_id"),
            "scanner_under_results": st.session_state.get("scanner_under_results"),
        }
        with open(SCANNER_CACHE_PATH, "wb") as _f:
            pickle.dump(_cache, _f)
    except Exception:
        pass
def _load_scanner_cache() -> dict:
    import pickle
    try:
        with open(SCANNER_CACHE_PATH, "rb") as _f:
            return pickle.load(_f)
    except Exception:
        return {}
def _save_alert_hashes(hashes: set):
    """Persist sent-alert hashes to disk so dedup survives session resets."""
    try:
        with open(ALERT_HASHES_PATH, "w") as _f:
            json.dump(list(hashes), _f)
    except Exception:
        pass
def _load_alert_hashes() -> set:
    try:
        with open(ALERT_HASHES_PATH) as _f:
            return set(json.load(_f))
    except Exception:
        return set()
# ──────────────────────────────────────────────
# PLAYER POSITION CACHE
# ──────────────────────────────────────────────
_POSITION_CACHE = {}  # {name_lower: pos_str}
_PID_POSITION_MAP = {}  # {player_id: pos_str} — populated by bulk fetch
@st.cache_data(ttl=60*60*6, show_spinner=False)
def _bulk_player_position_map():
    """Fetch ALL active player positions in one LeagueDashPlayerStats call.
    Returns {player_id: position_string}.  Falls back to {} on error.
    """
    try:
        from nba_api.stats.endpoints import LeagueDashPlayerStats
        df = LeagueDashPlayerStats(
            season=get_season_string(),
            per_mode_simple="PerGame",
            measure_type_detailed_defense="Base",
        ).get_data_frames()[0]
        if df.empty or "PLAYER_ID" not in df.columns:
            return {}
        # PLAYER_POSITION column if present
        pos_col = next((c for c in df.columns if "POSITION" in c.upper()), None)
        if pos_col:
            return {int(r["PLAYER_ID"]): str(r[pos_col]) for _, r in df.iterrows()}
        return {}
    except Exception:
        return {}
def _ensure_pid_position_map():
    global _PID_POSITION_MAP
    if not _PID_POSITION_MAP:
        _PID_POSITION_MAP = _bulk_player_position_map()
def get_player_position(name):
    key = (name or "").strip().lower()
    if not key:
        return ""
    if key in _POSITION_CACHE:
        return _POSITION_CACHE[key]
    # Try bulk map first (fast — no extra API call)
    try:
        matches = nba_players.find_players_by_full_name(name)
    except Exception:
        matches = []
    pos = ""
    if matches:
        pid = matches[0].get("id")
        if pid:
            _ensure_pid_position_map()
            if int(pid) in _PID_POSITION_MAP:
                pos = _PID_POSITION_MAP[int(pid)]
            else:
                # Fallback: single CommonPlayerInfo call (slow, only if not in bulk map)
                try:
                    info = CommonPlayerInfo(player_id=pid).get_data_frames()[0]
                    # H-8 audit fix: DataFrame.get() returns a Series; use .iloc[0] for scalar
                    def _scalar(col):
                        s = info.get(col)
                        return str(s.iloc[0]) if s is not None and not s.empty else ""
                    pos = _scalar("POSITION") or _scalar("POSITION_SHORT")
                except Exception:
                    pos = ""
    _POSITION_CACHE[key] = pos
    return pos
def get_position_bucket(pos):
    if not pos:
        return "Unknown"
    p = str(pos).upper()
    if p.startswith("G"): return "Guard"
    if p.startswith("F"): return "Wing"
    if p.startswith("C"): return "Big"
    if "G" in p and "F" in p: return "Wing"
    if "F" in p and "C" in p: return "Big"
    return "Unknown"
# ──────────────────────────────────────────────
# SEASON HELPER
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60, show_spinner=False)
def get_season_string(today=None):
    d = today or date.today()
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{(start+1)%100:02d}"
# ──────────────────────────────────────────────
# BULK GAME LOG — one API call for ALL players
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*6, show_spinner=False)  # 6h cache — large payload, refresh once per session
def _fetch_bulk_gamelogs():
    """LeagueGameLog: ONE call returns every player's game log for the season.
    Replaces ~200 individual PlayerGameLog calls for a full-slate scan.
    Returns a DataFrame sorted newest-first per player, or None on failure.
    Retries up to 3 times with increasing timeout on network errors.
    """
    from nba_api.stats.endpoints import LeagueGameLog
    for _attempt, _timeout in enumerate([60, 90, 120]):
        try:
            df = LeagueGameLog(
                player_or_team_abbreviation="P",
                season=get_season_string(),
                season_type_all_star="Regular Season",
                timeout=_timeout,
            ).get_data_frames()[0]
            if df.empty:
                return None
            # Normalize player ID column (LeagueGameLog uses PLAYER_ID)
            if "PLAYER_ID" not in df.columns and "Player_ID" in df.columns:
                df = df.rename(columns={"Player_ID": "PLAYER_ID"})
            df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce")
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
            # Sort newest → oldest within each player
            df = df.sort_values(["PLAYER_ID", "GAME_DATE"], ascending=[True, False])
            # LeagueGameLog returns MIN as float; convert to match PlayerGameLog "MM:SS" format
            if "MIN" in df.columns:
                df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
            return df
        except Exception:
            if _attempt == 2:
                return None
            continue
    return None
# ──────────────────────────────────────────────
# GAME LOG FETCHER
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_player_gamelog(player_id, max_games=15):
    """Per-player game log. Tries the bulk cache first (instant), falls back to individual API call."""
    # ── Fast path: bulk dataframe already in cache ──────────────
    bulk = _fetch_bulk_gamelogs()
    if bulk is not None:
        pid = int(player_id)
        player_df = bulk[bulk["PLAYER_ID"] == pid].head(int(max_games)).copy()
        if not player_df.empty:
            # Convert GAME_DATE back to string so downstream code (pd.to_datetime) still works
            player_df["GAME_DATE"] = player_df["GAME_DATE"].dt.strftime("%b %d, %Y")
            return player_df, []
    # ── Slow path: individual PlayerGameLog call (10s timeout) ──
    # [AUDIT FIX] Explicitly filter to Regular Season to exclude preseason/playoff contamination
    errs = []
    season_str = get_season_string()
    for params in [
        {"season": season_str, "season_type_all_star": "Regular Season"},
        {"season": season_str},
        {"season_nullable": season_str, "season_type_all_star": "Regular Season"},
        {"season_nullable": season_str},
        {},
    ]:
        try:
            gl = playergamelog.PlayerGameLog(player_id=player_id, timeout=10, **params)
            df = gl.get_data_frames()[0]
            if not df.empty:
                return df.head(int(max_games)).copy(), []
            errs.append(f"Empty log with params {params}")
        except TypeError as te:
            errs.append(f"TypeError {params}: {te}")
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}")
    return pd.DataFrame(), errs
# ──────────────────────────────────────────────
# REAL HALF-GAME BOXSCORE FETCHER
# Per-period stats via BoxScoreTraditionalV2.
# start_period/end_period parameters give cumulative splits
# (H1=1-2, H2=3-4, Q1=1-1) without extra joins.
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*3, show_spinner=False)  # 3h — games finish in ~3h; 12h was serving stale period stats
def _fetch_boxscore_halfgame(game_id, player_id_int, start_period, end_period):
    """Return {PTS, REB, AST, ...} for a player over a specific period range, or None."""
    try:
        from nba_api.stats.endpoints import BoxScoreTraditionalV2
        bx = BoxScoreTraditionalV2(
            game_id=str(game_id),
            start_period=int(start_period),
            end_period=int(end_period),
            timeout=20,
        )
        pstats = bx.get_data_frames()[0]
        if pstats.empty:
            return None
        pid_col = next((c for c in ["PLAYER_ID", "Player_ID"] if c in pstats.columns), None)
        if not pid_col:
            return None
        row = pstats[pstats[pid_col] == int(player_id_int)]
        if row.empty:
            return None
        row = row.iloc[0]
        result = {}
        for col in ["PTS","REB","AST","FG3M","FGM","FGA","FG3A","FTM","FTA","BLK","STL","TOV","OREB","DREB"]:
            if col in row.index:
                result[col] = safe_float(row[col], default=0.0)
        return result if result else None
    except Exception:
        return None
def fetch_player_halfgame_log(player_id, game_log_df, market_name, n_games=10):
    """
    Build a real H1/H2/Q1 stat series from per-period boxscores.
    Returns pd.Series (newest-first) if >=3 games fetched, else None (falls back to scaled full-game).
    """
    if game_log_df is None or game_log_df.empty or not player_id:
        return None
    # Period boundaries
    if market_name.startswith("H1"):
        start_p, end_p = 1, 2
    elif market_name.startswith("H2"):
        start_p, end_p = 3, 4
    elif market_name.startswith("Q1"):
        start_p, end_p = 1, 1
    else:
        return None
    # Get game IDs from log
    game_id_col = next((c for c in ["GAME_ID", "Game_ID"] if c in game_log_df.columns), None)
    if not game_id_col:
        return None
    game_ids = game_log_df.head(n_games)[game_id_col].dropna().astype(str).tolist()
    if not game_ids:
        return None
    # Fetch in parallel (3 workers to stay under rate limits)
    stats_list = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_fetch_boxscore_halfgame, gid, int(player_id), start_p, end_p)
                   for gid in game_ids]
        for fut in futures:
            try:
                result = fut.result(timeout=25)
                if result:
                    stats_list.append(result)
            except Exception:
                pass
    if len(stats_list) < 3:
        return None   # Not enough half-game data; caller falls back to scaled full-game
    stats_df = pd.DataFrame(stats_list)
    # Resolve base market (strip H1/H2/Q1 prefix)
    base_mkt = market_name.replace("H1 ","").replace("H2 ","").replace("Q1 ","")
    stat_field = STAT_FIELDS.get(base_mkt)
    if stat_field is None:
        return None
    if isinstance(stat_field, tuple):
        s = pd.Series(0.0, index=range(len(stats_df)))
        for col in stat_field:
            if col in stats_df.columns:
                s = s + pd.to_numeric(stats_df[col], errors="coerce").fillna(0)
        return s.reset_index(drop=True)
    if stat_field in stats_df.columns:
        return pd.to_numeric(stats_df[stat_field], errors="coerce").reset_index(drop=True)
    return None
# ──────────────────────────────────────────────
# [UPGRADE NEW] SCHEDULE FATIGUE — 3-IN-4 NIGHTS DETECTOR
# Beyond basic B2B: detects compressed 3-game stretches
# Sharp bettors know 3-in-4 nights crushes performance more than B2B alone
# ──────────────────────────────────────────────
def compute_schedule_fatigue(game_log_df, game_date):
    """
    Detect multi-game fatigue scenarios:
      - B2B (1 rest day): rest_days=0
      - 3-in-4 nights: 3 games within any 4-day window  → extra penalty
      - 4-in-6 nights: 4 games within 6 days → severe penalty
    Returns (fatigue_mult, fatigue_label).
    """
    if game_log_df is None or game_log_df.empty:
        return 1.0, "Normal"
    try:
        dates = pd.to_datetime(game_log_df["GAME_DATE"], errors="coerce").dropna()
        if dates.empty:
            return 1.0, "Normal"
        # Include today's game in the window
        all_dates = sorted([d.date() for d in dates] + [game_date])
        # Check 3-in-4 nights: player played 2 of last 3 days
        recent = [d for d in all_dates if (game_date - d).days <= 4]
        n_in_4 = len(recent)
        recent6 = [d for d in all_dates if (game_date - d).days <= 6]
        n_in_6 = len(recent6)
        if n_in_6 >= 4:
            return 0.91, "4-in-6"    # severe fatigue: ~9% penalty
        if n_in_4 >= 3:
            return 0.945, "3-in-4"   # compressed schedule: ~5.5% penalty
        return 1.0, "Normal"
    except Exception:
        return 1.0, "Normal"
# ──────────────────────────────────────────────
# REST / B2B FACTOR  [FIX 4: season-phase scaling]
# ──────────────────────────────────────────────
def compute_rest_factor(game_log_df, game_date):
    if game_log_df is None or game_log_df.empty:
        return 1.00, 2
    try:
        dates_raw = pd.to_datetime(game_log_df["GAME_DATE"], errors="coerce").dropna()
        if dates_raw.empty:
            return 1.00, 2
        last_game = dates_raw.max().date()
        rest = (game_date - last_game).days - 1
        rest = max(0, min(rest, 4))
        base_mult = REST_MULTIPLIERS.get(rest, 1.02)
        # Season-phase context: small additive adjustment to B2B mult only.
        # Research: playoff grind fatigue is real but already partially baked
        # into lower baseline REST_MULTIPLIERS. Phase adjustment is now conservative.
        try:
            _m = game_date.month
            if _m in (10, 11):    phase_mult = 0.993   # early: minor inconsistency
            elif _m in (12, 1):   phase_mult = 1.000   # mid: peak form
            elif _m == 2:         phase_mult = 0.995   # trade deadline: minor disruption
            else:                 phase_mult = 0.990   # Mar-Apr: playoff grind
            # Extra phase penalty on B2B only (additive to already-calibrated base)
            if rest == 0:
                base_mult *= phase_mult
        except Exception:
            pass
        return float(base_mult), int(rest)
    except Exception:
        return 1.00, 2
# ──────────────────────────────────────────────
# HOME / AWAY SPLIT
# ──────────────────────────────────────────────
def compute_home_away_factor(game_log_df, market, is_home):
    if game_log_df is None or game_log_df.empty or is_home is None:
        return 1.00
    try:
        df = game_log_df.copy()
        df["_home"] = df["MATCHUP"].str.contains("vs", case=False, na=False)
        stat_col = STAT_FIELDS.get(market)
        if stat_col is None:
            return 1.00
        if isinstance(stat_col, tuple):
            df["_stat"] = sum(pd.to_numeric(df.get(c), errors="coerce").fillna(0) for c in stat_col)
        else:
            df["_stat"] = pd.to_numeric(df.get(stat_col), errors="coerce")
        home_games = df[df["_home"]]["_stat"].dropna()
        away_games = df[~df["_home"]]["_stat"].dropna()
        # [FIX v4.0] Minimum 3 games each side; use season_avg as denominator
        # instead of the other split to avoid over-amplifying small samples.
        # Previous: ratio = home/away (extreme if away small); now: ratio vs season avg.
        if len(home_games) < 3 or len(away_games) < 3:
            return 1.00
        season_avg = df["_stat"].dropna().mean()
        if pd.isna(season_avg) or season_avg <= 1e-6:
            return 1.00
        target_split_avg = float(home_games.mean()) if is_home else float(away_games.mean())
        if pd.isna(target_split_avg):
            return 1.00
        ratio = target_split_avg / season_avg
        return float(np.clip(ratio, 0.88, 1.12))
    except Exception:
        return 1.00
# ──────────────────────────────────────────────
# BAYESIAN SHRINKAGE  [FIX 2: adaptive k]
# ──────────────────────────────────────────────
def bayesian_shrink(observed_mu, n_obs, market, position_bucket, custom_priors=None):
    # [v5.0] Use custom priors if available (from user's personal hit/miss calibration)
    if custom_priors and position_bucket in custom_priors and market in custom_priors[position_bucket]:
        prior = float(custom_priors[position_bucket][market])
    else:
        prior = POSITIONAL_PRIORS.get(position_bucket, POSITIONAL_PRIORS["Unknown"]).get(market)
    if prior is None or observed_mu is None:
        return observed_mu
    # [AUDIT IMPROVEMENT] Stat-specific shrinkage strength.
    # High-variance / zero-inflated stats (3PM, Blocks, Steals) need stronger prior pull
    # because small samples are dominated by noise. Points/Assists stabilize faster.
    # Research: empirical k calibrated to sport-specific Bayesian shrinkage literature
    # (Efron & Morris 1975 framework; NBA-specific: Deshpande & Jensen 2016)
    _STAT_K_NUMERATOR = {
        "3PM":     9.0,   # High variance (zero-inflated, binomial shooting)
        "Blocks":  9.0,   # Very high variance, zero-inflated
        "Steals":  8.0,   # High variance
        "Stocks":  8.0,   # Same as Steals+Blocks
        "FTA":     7.0,   # Foul-drawing varies game to game
        "FTM":     7.0,
        "Turnovers": 7.0,
        "Rebounds":  6.0,  # Moderate variance
        "RA":      6.0,
        "PR":      5.5,
        "Assists": 5.0,   # More stable role-driven
        "PA":      5.0,
        "Points":  5.0,   # Most stable, largest sample signal
        "PRA":     5.0,
        "FGM":     5.5,
        "FGA":     5.5,
    }
    k_num = _STAT_K_NUMERATOR.get(market, 5.0)
    k = max(1.0, k_num / (1.0 + math.log1p(max(n_obs, 1) / 5.0)))
    w_prior = k / (k + max(n_obs, 1))
    w_obs   = 1.0 - w_prior
    # Clip to >= 0: stat means can't be negative
    return max(0.0, float(w_prior * prior + w_obs * observed_mu))
# ──────────────────────────────────────────────
# STAT SERIES COMPUTATION
# ──────────────────────────────────────────────
def compute_stat_from_gamelog(df, market):
    # [UPGRADE] Fantasy Score: weighted DraftKings formula
    if market == "Fantasy Score":
        try:
            pts = pd.to_numeric(df.get("PTS"), errors="coerce").fillna(0)
            reb = pd.to_numeric(df.get("REB"), errors="coerce").fillna(0)
            ast = pd.to_numeric(df.get("AST"), errors="coerce").fillna(0)
            blk = pd.to_numeric(df.get("BLK"), errors="coerce").fillna(0)
            stl = pd.to_numeric(df.get("STL"), errors="coerce").fillna(0)
            tov = pd.to_numeric(df.get("TOV"), errors="coerce").fillna(0)
            # [AUDIT FIX] PrizePicks/DraftKings official formula:
            # PTS×1 + REB×1.2 + AST×1.5 + BLK×2 + STL×2 – TOV×1
            # Previous code used 3.0× for blk+stl (FanDuel formula) — WRONG.
            return pts + 1.2*reb + 1.5*ast + 2.0*blk + 2.0*stl - 1.0*tov
        except Exception:
            return pd.Series([], dtype=float)
    f = STAT_FIELDS.get(market)
    if f is None:
        return pd.Series([], dtype=float)
    if isinstance(f, tuple):
        s = pd.Series(0.0, index=df.index)
        for col in f:
            s = s + pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
        return s
    return pd.to_numeric(df.get(f), errors="coerce")
# ──────────────────────────────────────────────
# VOLATILITY ENGINE  [FIX 5: skewness helper]
# ──────────────────────────────────────────────
def compute_volatility(series):
    try:
        arr = pd.to_numeric(series, errors="coerce").dropna().values.astype(float)
    except Exception:
        return None, None
    if arr.size < 2:
        return None, None
    mean = arr.mean()
    if mean == 0:
        return None, None
    cv = float(arr.std(ddof=1) / mean)
    label = "Low" if cv < 0.15 else ("Moderate" if cv < 0.30 else "High")
    return cv, label
# [FIX 5/13] Skewness helper
def compute_skewness(series):
    try:
        arr = pd.to_numeric(series, errors="coerce").dropna().values.astype(float)
        # [AUDIT FIX] n<4 gives wildly unstable skewness; require >=6 for reliability
        if arr.size < 6:
            return None
        n = len(arr)
        m = arr.mean()
        s = arr.std(ddof=1)
        if s == 0:
            return 0.0
        return float((n / ((n-1)*(n-2))) * np.sum(((arr - m) / s)**3))
    except Exception:
        return None
# ──────────────────────────────────────────────
# [UPGRADE 3] OPPONENT-SPECIFIC HISTORICAL FACTOR
# ──────────────────────────────────────────────
def compute_opp_specific_factor(game_log_df, opp_abbr, market, n_min=2):
    """Return multiplier based on player's recorded performance vs this specific opponent.
    [AUDIT FIX] Only use current-season games vs this opponent.
    Games from prior seasons are stale (roster changes, system changes, coach changes).
    Filter by GAME_DATE >= Aug 1 of the current season year.
    """
    if game_log_df is None or game_log_df.empty or not opp_abbr:
        return 1.0, 0
    try:
        df = game_log_df.copy()
        # Filter to current season only (NBA season starts Oct, but use Aug cutoff for safety)
        if "GAME_DATE" in df.columns:
            try:
                _dates = pd.to_datetime(df["GAME_DATE"], errors="coerce")
                _season_start_yr = date.today().year if date.today().month >= 8 else date.today().year - 1
                _season_cutoff = pd.Timestamp(f"{_season_start_yr}-08-01")
                _season_mask = _dates >= _season_cutoff
                if _season_mask.sum() >= n_min:
                    df = df[_season_mask].copy()
            except Exception:
                pass
        opp_upper = str(opp_abbr).upper()
        mask = df["MATCHUP"].str.upper().str.contains(opp_upper, na=False)
        opp_df = df[mask]; rest_df = df[~mask]
        n_opp = len(opp_df)
        if n_opp < n_min or len(rest_df) < n_min:
            return 1.0, n_opp
        stat_col = STAT_FIELDS.get(market)
        if stat_col is None:
            return 1.0, n_opp
        if isinstance(stat_col, tuple):
            def _sum_cols(d):
                return sum(pd.to_numeric(d.get(c), errors="coerce").fillna(0) for c in stat_col).mean()
            opp_avg = _sum_cols(opp_df); rest_avg = _sum_cols(rest_df)
        else:
            opp_avg = pd.to_numeric(opp_df.get(stat_col), errors="coerce").mean()
            rest_avg = pd.to_numeric(rest_df.get(stat_col), errors="coerce").mean()
        if pd.isna(opp_avg) or pd.isna(rest_avg) or rest_avg <= 1e-6:
            return 1.0, n_opp
        ratio = opp_avg / rest_avg
        # 40% weight on opponent-specific (dampen for small sample)
        weight = min(0.40, n_opp * 0.10)
        factor = 1.0 + weight * (ratio - 1.0)
        return float(np.clip(factor, 0.88, 1.12)), n_opp
    except Exception:
        return 1.0, 0
# ──────────────────────────────────────────────
# [UPGRADE 5] PLAYER REGIME — HOT / COLD / AVERAGE
# ──────────────────────────────────────────────
def compute_player_regime_hot_cold(stat_series, n_recent=5):
    """Tag player as Hot / Cold / Average based on last N-game z-score vs season."""
    try:
        arr = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
        if len(arr) < n_recent + 3:
            return "Average", 0.0
        season_mu = arr.mean(); season_sigma = arr.std(ddof=1)
        if season_sigma < 1e-6:
            return "Average", 0.0
        recent_mu = arr[:n_recent].mean()  # array is newest-first
        z = (recent_mu - season_mu) / season_sigma
        if z > 0.8:  return "Hot",  float(z)
        if z < -0.8: return "Cold", float(z)
        return "Average", float(z)
    except Exception:
        return "Average", 0.0
# ──────────────────────────────────────────────
# [UPGRADE NEW] L3 / L5 / L10 TREND CONVERGENCE + MOMENTUM
# Sharp bettors look for alignment across multiple lookback windows.
# When L3 > L5 > L10 on an Over (or L3 < L5 < L10 on Under), it's a
# strong confirmation signal. Linear regression slope gives momentum.
# ──────────────────────────────────────────────
def compute_trend_convergence(stat_series, line, side="Over"):
    """
    Returns (convergence_score, trend_label, slope_per_game, l3_avg, l5_avg, l10_avg).
    convergence_score: -1.0 to +1.0 (positive = favors side, negative = fades side)
    [AUDIT IMPROVEMENT] Added L20 window. Research (Deshpande & Jensen 2016; EVAnalytics
    NBA prop modeling docs) shows 20-game rolling window outperforms 10-game as a baseline
    because it reduces game-to-game noise while preserving recent form signal.
    L20 receives a small weight (0.10) — primarily a baseline anchor to detect regression.
    """
    try:
        arr = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
        if len(arr) < 5:
            return 0.0, "Insufficient", 0.0, None, None, None
        # Newest first — align with game_log order
        l3  = float(np.mean(arr[:3]))  if len(arr) >= 3  else None
        l5  = float(np.mean(arr[:5]))  if len(arr) >= 5  else None
        l10 = float(np.mean(arr[:10])) if len(arr) >= 10 else float(np.mean(arr))
        # [AUDIT IMPROVEMENT] L20 window: larger sample anchor to detect genuine trend vs noise
        l20 = float(np.mean(arr[:20])) if len(arr) >= 20 else None
        is_over = "over" in str(side).lower()
        # Linear regression slope on last 10 games (positive = trending up, negative = trending down)
        x = np.arange(len(arr[:10]))
        y = arr[:10]
        if len(x) >= 3:
            slope = float(np.polyfit(x, y[::-1], 1)[0])  # reverse: oldest→newest
        else:
            slope = 0.0
        fav_line = float(line)
        # Convergence: each rolling window above/below line contributes
        # [AUDIT IMPROVEMENT] L20 added as low-weight baseline anchor (10% weight)
        # redistributed from L10 (0.20→0.10) so total still sums to 1.0
        score = 0.0
        weight_total = 0.0
        for avg, w in [(l3, 0.50), (l5, 0.30), (l10, 0.10), (l20, 0.10)]:
            if avg is None:
                continue
            diff_norm = float(np.clip((avg - fav_line) / max(abs(fav_line), 1.0), -0.5, 0.5))
            score += (diff_norm if is_over else -diff_norm) * w
            weight_total += w
        if weight_total > 0:
            score = score / weight_total * 2.0  # normalize to [-1, 1]
        score = float(np.clip(score, -1.0, 1.0))
        # Alignment label
        if score > 0.30 and slope > 0:
            label = "Strong Bull"
        elif score > 0.15:
            label = "Bull"
        elif score < -0.30 and slope < 0:
            label = "Strong Bear"
        elif score < -0.15:
            label = "Bear"
        else:
            label = "Neutral"
        return score, label, slope, l3, l5, l10
    except Exception:
        return 0.0, "Neutral", 0.0, None, None, None
# ──────────────────────────────────────────────
# [v4.0 UPGRADE] CONSECUTIVE STREAK DETECTOR
# Research: 5+ consecutive over/under is a strong signal professional bettors
# use to detect line inefficiency. Books adjust slowly — consecutive streaks
# create a window of 1-2 games where the line hasn't fully adjusted.
# L3 consecutive = strong signal; L5 consecutive = very strong signal.
# ──────────────────────────────────────────────
def compute_consecutive_streak(stat_series, line, side="Over"):
    """
    Detect consecutive over/under streaks in recent games.
    Returns (streak_count, streak_label, streak_signal).
    streak_count: int — number of consecutive games on same side (negative = opposite side)
    streak_label: str — human-readable label
    streak_signal: float — probability nudge (-0.04 to +0.03)
    """
    try:
        arr = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
        if len(arr) < 3:
            return 0, "Insufficient", 0.0
        fav = float(line)
        is_over = "over" in str(side).lower()
        # Array is newest-first; build over/under per game
        results = np.where(arr > fav, 1, -1)  # 1=over, -1=under (push ignored as 0)
        results = np.where(arr == fav, 0, results)
        # Count consecutive streak from most recent game
        streak = 0
        direction = results[0]
        if direction == 0:
            return 0, "Push Streak", 0.0
        for r in results:
            if r == direction:
                streak += 1
            else:
                break
        # If betting the same direction as streak, signal depends on streak length
        signal_dir = 1 if direction == 1 else -1
        match_bet = (is_over and direction == 1) or (not is_over and direction == -1)
        if streak >= 5:
            label = f"{'Over' if direction==1 else 'Under'} Streak L{streak}"
            # 5+ streak: books likely still under-adjusted → +ev for continuation, but also regression risk
            nudge = +0.02 if match_bet else -0.02   # mild continuation bias (books adjust slowly)
        elif streak >= 3:
            label = f"{'Over' if direction==1 else 'Under'} Streak L{streak}"
            nudge = +0.015 if match_bet else -0.015
        else:
            label = "No Streak"
            nudge = 0.0
        return int(streak) * signal_dir, label, float(nudge)
    except Exception:
        return 0, "N/A", 0.0
# ──────────────────────────────────────────────
# [RESEARCH UPGRADE] ROLLING 3-GAME MINUTES AVERAGE
# CMU study: 3-game rolling minutes is the most predictive fatigue signal.
# A player who logged 40+ min in last 2 games faces real physical degradation
# even if today is not technically a B2B. This adjusts both the projection
# and the fatigue multiplier beyond the schedule-calendar approach.
# ──────────────────────────────────────────────
def compute_rolling_minutes_fatigue(game_log_df, n_recent=3):
    """
    Returns (rolling_avg_minutes, minutes_fatigue_mult, minutes_fatigue_label).
    High recent minutes → penalty; low minutes → neutral.
    Thresholds calibrated from NBA average ~32 min/game for starters.
    """
    if game_log_df is None or game_log_df.empty or "MIN" not in game_log_df.columns:
        return None, 1.0, "Normal"
    try:
        df = game_log_df.head(n_recent).copy()
        mins = df["MIN"].apply(lambda v:
            float(str(v).split(":")[0]) if isinstance(v, str) and ":" in str(v)
            else safe_float(v, default=0.0))
        active = mins[mins >= 5]
        if active.empty:
            return None, 1.0, "Normal"
        avg = float(active.mean())
        # 36+ min/game over L3 → meaningful fatigue; 40+ → severe
        if avg >= 40.0:
            return avg, 0.93, "High-Load"    # ~7% penalty
        if avg >= 37.0:
            return avg, 0.96, "Heavy"        # ~4% penalty
        if avg >= 34.0:
            return avg, 0.985, "Moderate"    # ~1.5% penalty
        return avg, 1.0, "Normal"
    except Exception:
        return None, 1.0, "Normal"
# ──────────────────────────────────────────────
# [AUDIT v5.1] PLAYER SHOOTING LUCK REGRESSION
# ──────────────────────────────────────────────
# When a player's recent TS% significantly deviates from their season baseline,
# they are likely to regress. This is one of the top-ranked signals in NBA prop
# ML models (SHAP importance studies at squared2020.com, inpredictable.com).
#
# Research: 3-game TS% spike of +8%+ above season avg → P(regression) ~70-75%.
# For scoring props, this is a strong fade/lean signal on hot-shooting streaks.
#
# Implementation:
#   - Season avg TS% = total FGA + 0.44*FTA (last 20 games)
#   - Recent TS% = last 5 games
#   - If |delta| >= 0.06: apply regression nudge (damped to avoid over-correction)
#   - Applies only to Points, PRA, PA, FGM, FTM markets (shooting-dependent)
# ──────────────────────────────────────────────
def compute_shooting_luck_regression(game_log_df, market, n_recent=5, n_season=20):
    """
    Returns (shooting_luck_mult, shooting_luck_label).
    >1.0 = player shooting below average → expect mean reversion upward
    <1.0 = player shooting above average → expect mean reversion downward
    =1.0 = within normal range or not applicable
    """
    _applicable = {"Points", "PRA", "PA", "FGM", "FTM", "H1 Points", "H2 Points",
                   "3PM", "3PA", "H1 3PM"}  # 3P% mean-reversion is one of the strongest in NBA research
    if market not in _applicable:
        return 1.0, "N/A"
    if game_log_df is None or game_log_df.empty:
        return 1.0, "N/A"
    try:
        # For 3PM/3PA markets: use 3P% mean reversion instead of TS%
        _is_3p_market = market in ("3PM", "3PA", "H1 3PM")
        if _is_3p_market:
            needed = {"FG3M", "FG3A"}
        else:
            needed = {"PTS", "FGA", "FTA", "FGM", "FTM"}
        df = game_log_df.copy()
        for col in needed:
            if col not in df.columns:
                return 1.0, "N/A"
        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=list(needed))
        if len(df) < max(n_recent + 1, 6):
            return 1.0, "N/A"
        if _is_3p_market:
            # 3P% = FG3M / FG3A — regress to mean when very hot or cold
            def _3p_pct(sub):
                attempts = sub["FG3A"].sum()
                if attempts < 3:
                    return None
                return float(sub["FG3M"].sum() / attempts)
            season_df = df.head(n_season)
            recent_df = df.head(n_recent)
            ts_season = _3p_pct(season_df)
            ts_recent = _3p_pct(recent_df)
        else:
            def _ts_pct(sub):
                total_pts = sub["PTS"].sum()
                tsa = sub["FGA"].sum() + 0.44 * sub["FTA"].sum()
                if tsa < 1.0:
                    return None
                return float(total_pts / (2.0 * tsa))
            season_df = df.head(n_season)
            recent_df = df.head(n_recent)
            ts_season = _ts_pct(season_df)
            ts_recent = _ts_pct(recent_df)
        if ts_season is None or ts_recent is None or ts_season <= 0:
            return 1.0, "N/A"
        delta = ts_recent - ts_season  # positive = hot streak above average
        # Thresholds (validated against empirical NBA prop hit rates):
        # |delta| < 0.04: noise — no signal
        # 0.04-0.07: mild deviation — small nudge (~1.5%)
        # 0.07-0.12: moderate — moderate nudge (~3%)
        # >0.12: extreme — capped nudge (~5%)
        # Direction: regression TOWARD mean → mult OPPOSITE to delta
        abs_d = abs(delta)
        if abs_d < 0.04:
            return 1.0, "TS% Normal"
        sign = -1.0 if delta > 0 else 1.0   # hot streak → expect downward regression
        if abs_d < 0.07:
            nudge = 0.015
            label_prefix = "Mild"
        elif abs_d < 0.12:
            nudge = 0.030
            label_prefix = "Moderate"
        else:
            nudge = 0.050
            label_prefix = "Strong"
        mult = float(np.clip(1.0 + sign * nudge, 0.90, 1.10))
        direction = "hot🔥→expect regression" if delta > 0 else "cold❄️→expect bounce"
        label = f"{label_prefix} TS% {direction} (Δ{delta:+.1%})"
        return mult, label
    except Exception:
        return 1.0, "N/A"
# ──────────────────────────────────────────────
# [RESEARCH UPGRADE] BOTH-TEAMS-B2B SUPPRESSOR
# When both teams are on a B2B, combined scoring drops 6-10 pts,
# pace slows, and turnovers increase. Per research, this is the
# single most impactful fatigue scenario — worse than one team B2B alone.
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*4, show_spinner=False)
def get_team_b2b_flag(team_abbr, game_date):
    """Return True if team_abbr is playing on the 2nd night of a B2B."""
    try:
        from nba_api.stats.endpoints import TeamGameLogs
        # Resolve team abbreviation → team_id using nba_teams
        abbr_upper = str(team_abbr).upper()
        team_list = nba_teams.get_teams()
        team_id = None
        for t in team_list:
            if t["abbreviation"].upper() == abbr_upper:
                team_id = t["id"]
                break
        if not team_id:
            return False
        logs = TeamGameLogs(
            team_id_nullable=str(team_id),
            season_nullable=get_season_string(),
            league_id_nullable="00",
            timeout=15,
        ).get_data_frames()[0]
        if logs is None or logs.empty:
            return False
        dates = pd.to_datetime(logs["GAME_DATE"], errors="coerce").dropna()
        sorted_dates = sorted([d.date() for d in dates], reverse=True)
        # Check if yesterday is in the log
        yesterday = (pd.Timestamp(game_date) - pd.Timedelta(days=1)).date()
        return yesterday in sorted_dates
    except Exception:
        return False
def compute_both_teams_b2b(team_abbr, opp_abbr, player_b2b, game_date):
    """
    Returns (both_b2b_flag, both_b2b_mult).
    both_b2b_mult: applied to volume props when both teams are fatigued.
    Per research: combined scoring -6-10 pts → ~3-4% vol prop penalty.
    Opp B2B when player is NOT on B2B: defensive boost for the player.
    """
    try:
        opp_is_b2b = get_team_b2b_flag(opp_abbr, game_date)
        if player_b2b and opp_is_b2b:
            # Both teams tired: pace slows, scoring suppressed globally
            return True, 0.965   # ~3.5% penalty for both-tired games
        return False, 1.0
    except Exception:
        return False, 1.0
# ──────────────────────────────────────────────
# [RESEARCH UPGRADE] OVER-RATE L10 / MEAN REVERSION FLAG
# When a player goes OVER their line ≥8/10 games, books under-adjust
# (recency bias overcorrection). Regression to mean is a real edge.
# Conversely, ≤2/10 (cold streak) → recovery probability is higher.
# This is only a signal when the line has NOT already moved to reflect it.
# ──────────────────────────────────────────────
def compute_over_rate_and_mean_reversion(stat_series, line):
    """
    Returns (over_rate_l10, reversion_signal, reversion_label).
    over_rate_l10: float 0.0-1.0
    reversion_signal: float -0.05 to +0.05 (EV nudge for mean reversion)
    reversion_label: str
    """
    try:
        arr = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
        if len(arr) < 5:
            return None, 0.0, "Insufficient"
        window = arr[:10]
        fav = float(line)
        over_rate = float(np.mean(window > fav))
        # [BUG FIX v4.0] Ordered from most extreme to least extreme so each threshold is reachable.
        # Previous code checked >= 0.80 before >= 0.90, making "Strong Regression" dead code.
        # Extreme hot streak: clear regression signal (must check BEFORE the 0.80 threshold)
        if over_rate >= 0.90:
            return over_rate, -0.04, "Strong Regression"
        # Strong hot streak: player probably going back to mean → slight Under lean
        if over_rate >= 0.80:
            return over_rate, -0.025, "Regression Risk"
        # Extreme cold streak: recovery probability (check BEFORE the 0.20 threshold)
        if over_rate <= 0.10:
            return over_rate, +0.035, "Strong Recovery"
        # Cold streak: recovery probability → slight Over lean
        if over_rate <= 0.20:
            return over_rate, +0.020, "Recovery Likely"
        return over_rate, 0.0, "Normal"
    except Exception:
        return None, 0.0, "Normal"
# ──────────────────────────────────────────────
# [AUDIT NEW] ALTITUDE ADJUSTMENT
# Denver (Ball Arena, ~5,280 ft) is the only significant high-altitude NBA venue.
# Research: visiting teams at altitude see ~2-3% scoring decline in 4th quarters
# due to reduced aerobic capacity, especially on B2B games (compounding fatigue).
# Denver at sea level vs. altitude-adapted teams: slight visiting advantage.
# Sources: Sports Medicine studies on altitude sports performance; NBA tracking data.
# ──────────────────────────────────────────────
_HIGH_ALTITUDE_TEAMS = {"DEN"}  # Ball Arena: 5,280 ft
_ALTITUDE_PENALTY_VISITOR = 0.975   # ~2.5% decline for visiting team at altitude
_ALTITUDE_BONUS_VISITING_DEN = 1.01  # ~1% benefit when Denver visits sea-level (altitude-adapted lungs)
def compute_altitude_factor(team_abbr, opp_abbr, is_home, market):
    """
    Returns altitude adjustment multiplier (float only).
    - Visiting team at Denver: 0.977 (aerobic penalty ~2.3%)
    - Denver playing away: 1.01 (altitude-adapted advantage ~1%)
    - All other venues: 1.0
    Applied to all volume/endurance markets.
    """
    _relevant = {"Points","PRA","PR","PA","RA","Rebounds","Assists","Fantasy Score",
                 "3PM","FGM","H1 Points","H2 Points","Steals","Blocks","Stocks"}
    if market not in _relevant:
        return 1.0
    try:
        t = str(team_abbr or "").upper().strip()
        o = str(opp_abbr or "").upper().strip()
        # Player's team is visiting Denver
        if o in _HIGH_ALTITUDE_TEAMS and is_home is False:
            return _ALTITUDE_PENALTY_VISITOR
        # Player is on Denver, playing away (altitude-adapted advantage)
        if t in _HIGH_ALTITUDE_TEAMS and is_home is False:
            return _ALTITUDE_BONUS_VISITING_DEN
        return 1.0
    except Exception:
        return 1.0
# ──────────────────────────────────────────────
# [RESEARCH UPGRADE] TRAVEL FATIGUE
# Cross-country travel + direction effect: east travel worse than west.
# Research: teams traveling eastward win 44.51% vs 40.83% westward.
# Cross-country B2B: ~5-7% performance decline vs average 3-5%.
# ──────────────────────────────────────────────
# Approximate longitude for each NBA team city (used for travel direction)
_NBA_TEAM_LONGITUDES = {
    "ATL": -84.4, "BOS": -71.1, "BKN": -74.0, "CHA": -80.8,
    "CHI": -87.6, "CLE": -81.7, "DAL": -96.8, "DEN": -104.9,
    "DET": -83.0, "GSW": -122.2, "HOU": -95.4, "IND": -86.2,
    "LAC": -118.2, "LAL": -118.2, "MEM": -90.0, "MIA": -80.2,
    "MIL": -87.9, "MIN": -93.3, "NOP": -90.1, "NYK": -74.0,
    "OKC": -97.5, "ORL": -81.4, "PHI": -75.2, "PHX": -112.1,
    "POR": -122.7, "SAC": -121.5, "SAS": -98.5, "TOR": -79.4,
    "UTA": -111.9, "WAS": -77.0,
}
def compute_travel_fatigue(team_abbr, game_log_df, game_date, b2b_flag):
    """
    Estimate travel fatigue from last game location.
    Returns (travel_mult, travel_label).
    Only applied on B2B games (otherwise rest negates travel effect).
    """
    if not b2b_flag:
        return 1.0, "Normal"
    try:
        t_key = str(team_abbr).upper()
        home_lon = _NBA_TEAM_LONGITUDES.get(t_key)
        if home_lon is None or game_log_df is None or game_log_df.empty:
            return 1.0, "Normal"
        # Infer last opponent from game log
        last_game = game_log_df.iloc[0]
        matchup = str(last_game.get("MATCHUP", ""))
        # MATCHUP format: "TOR vs. BOS" or "TOR @ BOS"
        parts = matchup.replace("vs.", "vs").replace("@", "at").split()
        away = "@" in str(last_game.get("MATCHUP","")) or " @ " in matchup
        # Extract opponent abbreviation
        opp_from_matchup = parts[-1].strip().upper() if parts else ""
        # If player was away last game, they traveled TO opp; now they travel back or to next city
        prev_lon = _NBA_TEAM_LONGITUDES.get(opp_from_matchup, home_lon)
        lon_delta = home_lon - prev_lon  # positive = traveled west→east (eastward travel)
        miles_approx = abs(lon_delta) * 53.0  # rough miles per degree longitude at NBA latitudes
        direction = "East" if lon_delta > 5 else ("West" if lon_delta < -5 else "Local")
        if miles_approx >= 1800 and direction == "East":
            return 0.945, "Cross-Country East"  # Worst case: ~5.5% penalty
        if miles_approx >= 1800:
            return 0.960, "Cross-Country West"  # ~4% penalty
        if miles_approx >= 900:
            return 0.975, "Long Haul"           # ~2.5% penalty
        return 1.0, "Normal"
    except Exception:
        return 1.0, "Normal"
# ──────────────────────────────────────────────
# [RESEARCH UPGRADE] L10 ROLLING DvP
# Season-average DvP misses recency: a team that traded away their
# best perimeter defender mid-season looks fine in full-season stats
# but is now exploitable. Rolling L10 DvP catches this degradation.
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*4, show_spinner=False)
def get_dvp_rolling_l10(opp_abbr, pos_bucket, market):
    """
    Pull opponent's season positional defense data (LeagueDashPtDefend).
    Returns (dvp_mult, dvp_label) — mult relative to league avg for that position.
    [AUDIT FIX] NOTE: Despite the function name "l10", LeagueDashPtDefend does NOT
    support a last-N-games filter — it always returns season-to-date aggregates.
    The label is therefore "Season DvP" not "L10 DvP". Attempting to pass
    last_n_games param causes API 400 errors. True rolling L10 DvP would require
    fetching per-game opponent boxscores and aggregating ourselves (not done here
    to avoid rate-limiting; season-avg DvP is still a valid signal).
    """
    try:
        from nba_api.stats.endpoints import LeagueDashPtDefend
        pos_map = {"Guard": "G", "Wing": "F", "Big": "C", "Unknown": "G"}
        pt_pos = pos_map.get(pos_bucket, "G")
        raw = LeagueDashPtDefend(
            league_id="00",
            per_mode_simple="PerGame",
            defense_category=pt_pos,
            season=get_season_string(),
            timeout=20,
        ).get_data_frames()[0]
        if raw is None or raw.empty:
            return 1.0, "Avg"
        opp_key = str(opp_abbr).upper()
        # Match team abbreviation
        row = raw[raw["TEAM_ABBREVIATION"].str.upper() == opp_key]
        if row.empty:
            return 1.0, "Avg"
        # Map market to the relevant defensive stat column available in LeagueDashPtDefend.
        # [AUDIT UPGRADE] Added DFGA (field goals attempted against) as proxy for
        # rebound/assist opportunity volume when direct reb/ast cols aren't available.
        stat_col = {
            "Points":   "D_FG_PCT",    # opponent FG% allowed (lower = elite D)
            "PRA":      "D_FG_PCT",
            "PA":       "D_FG_PCT",
            "PR":       "D_FG_PCT",
            "3PM":      "D_FG3_PCT",   # opponent 3P% allowed
            "Rebounds": "D_FGA",       # volume proxy: more FGA allowed → more rebounds
            "RA":       "D_FGA",
            "Assists":  "D_FGA",       # more FGA allowed → more open looks → more assists
        }.get(market)
        if stat_col is None or stat_col not in row.columns:
            return 1.0, "Avg"
        league_avg = float(raw[stat_col].mean())
        opp_val = float(row[stat_col].values[0])
        if league_avg <= 0:
            return 1.0, "Avg"
        # Ratio: >1.0 means opponent allows more → favorable for Over
        ratio = opp_val / league_avg
        if ratio >= 1.08:
            return float(np.clip(ratio, 1.0, 1.12)), "Soft Defense"
        if ratio <= 0.92:
            return float(np.clip(ratio, 0.88, 1.0)), "Elite Defense"
        return 1.0, "Avg"
    except Exception:
        return 1.0, "Avg"
# ──────────────────────────────────────────────
# [AUDIT UPGRADE] PER-MINUTE PRODUCTION RATIO
# Research (Unabated, ETR 2024-25): "The heart of any NBA projection is projected minutes.
# Per-minute efficiency is the most stable predictor when role/minutes change."
# Use pts/min, reb/min, ast/min to compute how efficiently the player produces.
# When projected minutes > recent average (injury absorption), this scales the projection up.
# When minutes are being restricted, it scales down.
# ──────────────────────────────────────────────
def compute_per_minute_production(game_log_df, market, n_games=10):
    """
    Returns (pts_per_min, prod_mult) where prod_mult adjusts projection when
    projected minutes deviate from recent average.
    prod_mult = proj_minutes / recent_avg_minutes (capped ±20%).
    This is the single strongest signal for injury-replacement props.
    """
    if game_log_df is None or game_log_df.empty:
        return None, 1.0
    try:
        df = game_log_df.head(n_games).copy()
        mins = df["MIN"].apply(lambda v:
            float(str(v).split(":")[0]) if isinstance(v, str) and ":" in str(v)
            else safe_float(v, default=0.0))
        active_mask = mins >= 5
        active_mins = mins[active_mask]
        if active_mins.empty or active_mins.mean() < 1:
            return None, 1.0
        stat_col = STAT_FIELDS.get(market)
        if stat_col is None:
            return None, 1.0
        if isinstance(stat_col, tuple):
            stat_vals = sum(pd.to_numeric(df.get(c, pd.Series()), errors="coerce").fillna(0) for c in stat_col)
        else:
            stat_vals = pd.to_numeric(df.get(stat_col, pd.Series()), errors="coerce").fillna(0)
        active_stat = stat_vals[active_mask]
        if len(active_stat) < 3:
            return None, 1.0
        avg_min  = float(active_mins.mean())
        avg_stat = float(active_stat.mean())
        if avg_min <= 0:
            return None, 1.0
        per_min = avg_stat / avg_min
        return float(per_min), 1.0   # prod_mult=1.0; caller applies when proj_minutes known
    except Exception:
        return None, 1.0
# ──────────────────────────────────────────────
# [UPGRADE 2] PROJECTED MINUTES
# ──────────────────────────────────────────────
def compute_projected_minutes(game_log_df, n_games=10):
    """Return rolling average of minutes played (DNP-aware) and a DNP-risk flag."""
    if game_log_df is None or game_log_df.empty or "MIN" not in game_log_df.columns:
        return None, False
    try:
        df = game_log_df.head(n_games).copy()
        mins = df["MIN"].apply(lambda v:
            float(str(v).split(":")[0]) if isinstance(v, str) and ":" in str(v)
            else safe_float(v, default=0.0))
        active = mins[mins >= 5]
        if active.empty:
            return None, True
        avg_min = float(active.mean())
        # DNP risk: player was actually held out (0-4 min) in 50%+ of recent games,
        # or averaged <8 min active (deep rotation / G-League shuttle).
        # 19 min/game is NOT dnp risk — that's a normal rotation player.
        true_dnps = (mins <= 4).sum()
        dnp_risk = avg_min < 8.0 or (true_dnps >= len(df) * 0.50)
        return avg_min, bool(dnp_risk)
    except Exception:
        return None, False
# ──────────────────────────────────────────────
# [UPGRADE NEW] OPPONENT FATIGUE FACTOR
# When the opposing team is on a B2B or 3-in-4 stretch,
# their defense degrades — this gives us a BOOST for the player's stats.
# Sharp bettors specifically target opponents in compressed schedules.
# NBA research shows defensive rating drops ~2-3 pts on B2B for the defense.
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*2, show_spinner=False)
def get_opponent_schedule_fatigue(opp_abbr, game_date):
    """
    Look up the opponent team's recent schedule to detect fatigue.
    Returns (opp_fatigue_mult, opp_fatigue_label).
    mult > 1.0 means opponent is tired → our player benefits.
    """
    if not opp_abbr:
        return 1.0, "Unknown"
    try:
        # Find an opponent player to use as a proxy for their schedule
        # We use the team's own game log via scoreboard history
        teams = nba_teams.get_teams()
        opp_team = next((t for t in teams
                         if t.get("abbreviation","").upper() == str(opp_abbr).upper()), None)
        if not opp_team:
            return 1.0, "Unknown"
        # Use scoreboardv2 to get recent game dates for the opponent team
        from nba_api.stats.endpoints import LeagueGameLog
        bulk = _fetch_bulk_gamelogs()
        if bulk is None:
            return 1.0, "Unknown"
        # Get the team's players from bulk logs; find any active player
        # Use team abbreviation from MATCHUP column
        team_abbr_upper = str(opp_abbr).upper()
        team_games = bulk[bulk["MATCHUP"].str.upper().str.contains(team_abbr_upper, na=False)]
        if team_games.empty:
            return 1.0, "Unknown"
        dates = pd.to_datetime(team_games["GAME_DATE"], errors="coerce").dropna().dt.date.unique()
        if len(dates) == 0:
            return 1.0, "Unknown"
        all_dates = sorted(dates.tolist() + [game_date])
        recent = [d for d in all_dates if (game_date - d).days <= 4]
        n_in_4 = len(recent)
        recent6 = [d for d in all_dates if (game_date - d).days <= 6]
        n_in_6 = len(recent6)
        # Check if opp is on B2B
        prev_games = [d for d in dates if (game_date - d).days >= 1]
        on_b2b = any((game_date - d).days == 1 for d in prev_games)
        if n_in_6 >= 4:
            return 1.04, "Opp 4-in-6"
        if n_in_4 >= 3:
            return 1.025, "Opp 3-in-4"
        if on_b2b:
            return 1.015, "Opp B2B"
        return 1.0, "Opp Rested"
    except Exception:
        return 1.0, "Unknown"
def volatility_penalty_factor(cv):
    """
    Smooth, continuous EV penalty factor for high-CV stats.
    [AUDIT FIX] Previous version had hard cliff-steps (0.85→0.65→0.45 at fixed CV thresholds),
    creating discontinuities where CV=0.249 got 0.85 but CV=0.251 got 0.65 (-24% jump).
    New formula uses linear interpolation across four anchor points derived from
    the same empirical calibration — same overall shape, no artificial cliffs.
    """
    if cv is None: return 0.0
    v = float(cv)
    if v <= 0.0:   return 1.0
    if v >= 0.35:  return 0.0
    # Four empirical anchor points (cv, penalty_factor):
    # (0.00, 1.00) — perfect consistency
    # (0.20, 1.00) — low volatility: no penalty
    # (0.25, 0.85) — moderate: 15% penalty
    # (0.30, 0.65) — high: 35% penalty
    # (0.35, 0.00) — too volatile: no bet
    anchors = [(0.0, 1.0), (0.20, 1.0), (0.25, 0.85), (0.30, 0.65), (0.35, 0.0)]
    for i in range(len(anchors) - 1):
        cv0, p0 = anchors[i]
        cv1, p1 = anchors[i + 1]
        if cv0 <= v <= cv1:
            t = (v - cv0) / (cv1 - cv0)
            return float(p0 + t * (p1 - p0))
    return 0.0
# [FIX 5] Skewness-adjusted volatility gate
def passes_volatility_gate(cv, ev_raw, skew=None, bet_type="Over"):
    """
    [AUDIT UPGRADE] EV-rescue override for CV 0.35–0.42 range.
    Research (OddsJam, Unabated 2024): High-variance stats (Blocks, Steals, 3PM)
    frequently have CV > 0.35 yet show genuine edge. A hard 0.35 cutoff discards
    these. Allow CV up to 0.42 when EV >= 12% (strong model consensus).
    """
    if cv is None:
        return False, "no stat history (CV unavailable)"
    v = float(cv)
    ev_f = float(ev_raw) if ev_raw is not None else None
    # [v6.0 ROI BOOST] Hard cutoff tightened from 0.42 → 0.38: high-variance props
    # destroy ROI even with large edges. Research: CV>0.38 has negative CLV at scale.
    if v > 0.38:
        return False, "CV>0.38 (too volatile — variance overwhelms edge)"
    # 0.32–0.38 range: only pass with elite EV (≥15%)
    if v > 0.32:
        if ev_f is None or ev_f < 0.15:
            return False, f"CV>{v:.2f} needs EV>=15% (high-variance stat)"
    # [v6.0] Raised from 6% → 8% for CV 0.25-0.32 range
    if v > 0.25 and (ev_f is None or ev_f < 0.08):
        return False, "CV>0.25 needs EV>=8%"
    # [v6.0] Raised from 2% → 5% minimum EV for medium-CV bets
    if v > 0.15 and (ev_f is None or ev_f < 0.05):
        return False, "CV>0.15 needs EV>=5%"
    # [v6.0] Even low-CV bets need meaningful edge
    if ev_f is not None and ev_f < 0.03:
        return False, "EV<3% (below noise floor — not worth staking)"
    # [FIX 5] Skewness-adjusted threshold
    # [v6.0] Tightened skewness gates for ROI protection
    if skew is not None and v > 0.18:
        is_over = "over" in str(bet_type).lower()
        # Negative skew + Over bet = tail risk of low games
        if float(skew) < -0.5 and is_over:
            tightened = 0.25
            if v > tightened and (ev_f is None or ev_f < 0.10):
                return False, f"CV>{tightened:.2f} (neg-skew+Over tightened, needs EV>=10%)"
        # Positive skew + Under bet = tail risk of blow-up games
        elif float(skew) > 0.5 and not is_over:
            tightened = 0.25
            if v > tightened and (ev_f is None or ev_f < 0.10):
                return False, f"CV>{tightened:.2f} (pos-skew+Under tightened, needs EV>=10%)"
    return True, ""
# ──────────────────────────────────────────────
# BOOTSTRAP WITH PER-PLAYER NOISE
# ──────────────────────────────────────────────
def bootstrap_prob_over(stat_series, line, n_sims=20000, cv_override=None, market="default"):
    x = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
    if x.size < 4:
        mu = float(np.nanmean(x)) if x.size else None
        sigma = float(np.nanstd(x, ddof=1)) if x.size > 1 else None
        return None, mu, sigma
    # [UPGRADE 4] Principled exponential decay (stat-specific λ, not arbitrary linear)
    if x.size >= 6:
        lam = LAMBDA_DECAY_BY_STAT.get(market, LAMBDA_DECAY_BY_STAT["default"])
        w = np.array([lam ** i for i in range(x.size)], dtype=float)
        w = w / w.sum()
    else:
        w = None
    # [AUDIT FIX] Per-call unique seed: fixed seed 42 gave identical noise patterns
    # across all players. Use data-derived seed so each player's bootstrap is independent.
    _seed = int(abs(hash(tuple(x.round(3).tolist()))) % (2**31))
    rng = np.random.default_rng(_seed)
    sims = rng.choice(x, size=int(n_sims), replace=True, p=w)
    cv = cv_override or (float(x.std(ddof=1) / x.mean()) if x.mean() != 0 else 0.20)
    # [v6.0] Reduced noise scale from 0.40 → 0.30 for tighter probability estimates
    noise_scale = max(0.03, min(cv * 0.30, 0.18))
    noise = rng.normal(0, float(x.std(ddof=1) * noise_scale), int(n_sims))
    sims_noisy = np.clip(sims + noise, 0, None)
    p_over = float((sims_noisy > float(line)).mean())
    mu_w = float(np.average(x, weights=w) if w is not None else x.mean())
    sigma_w = float(np.sqrt(np.average((x - mu_w)**2, weights=w)) if w is not None else x.std(ddof=1))
    return float(np.clip(p_over, 1e-4, 1-1e-4)), mu_w, max(1e-9, sigma_w)
# ──────────────────────────────────────────────
# [v5.0] NEGATIVE BINOMIAL PROBABILITY ESTIMATOR
# For count stats (3PM, Assists, Rebounds, Blocks, Steals, etc.)
# NegBin handles overdispersion (σ² > μ) which all NBA counting stats exhibit.
# Research: BinomialBasketball.com, squared2020.com — NegBin beats Poisson/bootstrap
# for integer-valued stats by correctly modeling per-player consistency variance.
# scipy.stats.nbinom parameterization: n=r (dispersion), p=r/(r+μ)
# Returns (p_over, mu_weighted, sigma_weighted) matching bootstrap_prob_over signature.
# ──────────────────────────────────────────────
def negbinom_prob_over(stat_series, line, market="default", min_n=6):
    """
    Negative Binomial probability estimate P(X > line) for count stats.
    Falls back to None if insufficient data or conditions not met (σ² ≤ μ).
    """
    from scipy.stats import nbinom
    x = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
    if x.size < min_n:
        return None, None, None
    # Exponential decay weights (same λ as bootstrap)
    lam = LAMBDA_DECAY_BY_STAT.get(market, LAMBDA_DECAY_BY_STAT["default"])
    if x.size >= 6:
        w = np.array([lam ** i for i in range(x.size)], dtype=float)
        w = w / w.sum()
    else:
        w = np.ones(x.size) / x.size
    mu_w = float(np.average(x, weights=w))
    # Weighted variance
    var_w = float(np.average((x - mu_w) ** 2, weights=w))
    if mu_w <= 0 or var_w <= 0.90 * mu_w:
        # Variance significantly < mean → underdispersed: NegBin invalid; return None.
        # [AUDIT FIX] Threshold relaxed from strict mu → 0.90*mu:
        # variance slightly below mean is often noise; only reject clearly underdispersed cases.
        return None, mu_w, max(1e-9, np.sqrt(var_w))
    # Method of moments: r = μ² / (σ² - μ)
    r = (mu_w ** 2) / max(var_w - mu_w, 1e-6)
    r = float(np.clip(r, 0.5, 50.0))   # Clip: very low r = extreme overdispersion (e.g. blocks)
    p = r / (r + mu_w)
    # P(X > line) = 1 - P(X <= floor(line)) = survival function at floor(line)
    # nbinom.sf(k, n, p) = P(X > k)
    p_over = float(nbinom.sf(int(np.floor(line)), r, p))
    sigma_w = float(np.sqrt(var_w))
    return float(np.clip(p_over, 1e-4, 1 - 1e-4)), mu_w, max(1e-9, sigma_w)
# ──────────────────────────────────────────────
# EMPIRICAL CORRELATION  [FIX 1: Spearman]
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60, show_spinner=False)
def empirical_leg_correlation(pid1, pid2, mkt1, mkt2, n_games=20):
    try:
        gl1, _ = fetch_player_gamelog(pid1, max_games=n_games)
        gl2, _ = fetch_player_gamelog(pid2, max_games=n_games)
        if gl1.empty or gl2.empty:
            return None
        s1 = compute_stat_from_gamelog(gl1, mkt1).rename("s1")
        s2 = compute_stat_from_gamelog(gl2, mkt2).rename("s2")
        # [AUDIT FIX] Normalize GAME_DATE to ISO date before merge — NBA API returns
        # dates in varying formats ("Mar 05, 2025" vs "2025-03-05") depending on
        # bulk-cache vs per-player path; mismatches silently produce empty merges.
        _d1 = pd.to_datetime(gl1["GAME_DATE"], errors="coerce").dt.date.reset_index(drop=True).rename("GAME_DATE")
        _d2 = pd.to_datetime(gl2["GAME_DATE"], errors="coerce").dt.date.reset_index(drop=True).rename("GAME_DATE")
        df1 = pd.concat([_d1, s1.reset_index(drop=True)], axis=1)
        df2 = pd.concat([_d2, s2.reset_index(drop=True)], axis=1)
        merged = df1.merge(df2, on="GAME_DATE", how="inner")
        if len(merged) < 6:
            return None
        # [FIX 1] Spearman rank correlation for count data
        corr = float(merged["s1"].corr(merged["s2"], method="spearman"))
        return float(np.clip(corr, -0.50, 0.70)) if not np.isnan(corr) else None
    except Exception:
        return None
def estimate_player_correlation(leg1, leg2):
    pid1 = leg1.get("player_id")
    pid2 = leg2.get("player_id")
    # Same player + same market = identical bet (perfect correlation)
    if pid1 and pid2 and int(pid1) == int(pid2) and leg1.get("market") == leg2.get("market"):
        return 1.0
    # Same player + different markets — correlated via shared minutes/game script
    if pid1 and pid2 and int(pid1) == int(pid2):
        m1l = leg1.get("market") or ""
        m2l = leg2.get("market") or ""
        _pts_grp = {"Points", "PRA", "PA", "Alt Points"}
        _reb_grp = {"Rebounds", "PR", "RA"}
        _ast_grp = {"Assists", "PA", "RA"}
        _same_family = (
            (m1l in _pts_grp and m2l in _pts_grp) or
            (m1l in _reb_grp and m2l in _reb_grp) or
            (m1l in _ast_grp and m2l in _ast_grp)
        )
        return 0.70 if _same_family else 0.45  # different families: correlated via minutes
    if pid1 and pid2:
        emp = empirical_leg_correlation(
            int(pid1), int(pid2), leg1.get("market","Points"), leg2.get("market","Points")
        )
        if emp is not None:
            return float(emp)
    corr = 0.0
    m1, m2 = leg1.get("market"), leg2.get("market")
    same_team = bool(leg1.get("team") and leg2.get("team") and leg1["team"] == leg2["team"])
    if same_team:
        # [AUDIT FIX] Market-specific same-team correlation — flat +0.15 was too coarse:
        # Same stat category on same team shares pace/game-script strongly
        # Cross-category (Points vs Rebounds) shares less — driven by minutes, not plays
        if m1 == m2:
            corr += 0.20   # same market + same team: share touches and game script directly
        else:
            _pts_grp = {"Points", "PRA", "PA", "Alt Points"}
            _reb_grp = {"Rebounds", "PR", "RA"}
            _ast_grp = {"Assists", "PA", "RA"}
            _same_grp = (
                (m1 in _pts_grp and m2 in _pts_grp) or
                (m1 in _reb_grp and m2 in _reb_grp) or
                (m1 in _ast_grp and m2 in _ast_grp)
            )
            corr += 0.12 if _same_grp else 0.06  # cross-category: weaker same-team link
    if m1 == m2 and not same_team: corr += 0.10
    if set([m1,m2]) == {"Points","PRA"}: corr += 0.14
    if m1 in ["Rebounds","RA"] and m2 in ["Rebounds","RA"]: corr += 0.06
    if m1 in ["Assists","RA"] and m2 in ["Assists","RA"]: corr += 0.05
    ctx1, ctx2 = float(leg1.get("context_mult",1.0)), float(leg2.get("context_mult",1.0))
    if ctx1>1.03 and ctx2>1.03: corr += 0.04
    if ctx1<0.97 and ctx2<0.97: corr += 0.03
    if (ctx1>1.03 and ctx2<0.97) or (ctx1<0.97 and ctx2>1.03): corr -= 0.05
    return float(np.clip(corr, -0.25, 0.45))
# ──────────────────────────────────────────────
# LINE MOVEMENT ALERT  [FIX 10: dedup]
# ──────────────────────────────────────────────
def get_line_movement_signal(player_norm, market_key, current_line, side="Over"):
    # [AUDIT FIX] Use file-based immutable opening line (survives page refreshes)
    # Session state resets on refresh, allowing false "opening" re-records mid-day
    _file_open, _ = get_opening_line(player_norm, str(market_key), side)
    if _file_open is None:
        save_opening_line(player_norm, str(market_key), side, current_line, None)
        _file_open = float(current_line)
    opening = _file_open
    delta = float(current_line) - float(opening)
    is_over = "over" in str(side).lower()
    steam = (delta > 0.5 and is_over) or (delta < -0.5 and not is_over)
    fade  = (delta < -0.5 and is_over) or (delta > 0.5 and not is_over)
    msg = ""
    if abs(delta) >= 0.5:
        direction = "UP" if delta > 0 else "DOWN"
        msg = f"Line moved {direction} {abs(delta):.1f} pts from open ({opening:.1f} -> {current_line:.1f})"
        # [FIX 10] Alert deduplication
        alert_hash = f"{player_norm}_{market_key}_{side}_{direction}_{round(abs(delta),1)}"
        issued = st.session_state.get("_issued_mv_alerts", set())
        if alert_hash not in issued:
            if steam: msg += " STEAM (confirms your side)"
            if fade:  msg += " FADE (sharps vs your side)"
            issued.add(alert_hash)
            st.session_state["_issued_mv_alerts"] = issued
        else:
            msg = ""  # suppress duplicate alert
    return {
        "direction": "UP" if delta > 0 else ("DOWN" if delta < 0 else "FLAT"),
        "pips": float(delta),
        "steam": steam,
        "fade": fade,
        "msg": msg,
        "opening": float(opening),
    }
# ──────────────────────────────────────────────
# REGIME CLASSIFIER
# ──────────────────────────────────────────────
def classify_regime(cv, blowout_prob, ctx_mult):
    try: v = float(cv) if cv is not None else None
    except: v = None
    try: b = float(blowout_prob) if blowout_prob is not None else 0.10
    except: b = 0.10
    try: c = float(ctx_mult) if ctx_mult is not None else 1.0
    except: c = 1.0
    v_score = 0.0 if v is None else float(np.clip((v-0.15)/0.25, 0.0, 1.0))
    b_score = float(np.clip((b-0.08)/0.20, 0.0, 1.0))
    c_score = float(np.clip(abs(c-1.0)/0.20, 0.0, 1.0))
    score = float(np.clip(0.55*v_score + 0.30*b_score + 0.15*c_score, 0.0, 1.0))
    if score >= 0.65: return "Chaotic", score
    if score >= 0.40: return "Mixed", score
    return "Stable", score
# ──────────────────────────────────────────────
# MARKET PRICING  [FIX 6: remove_vig] [FIX 8: neg-edge guard]
# ──────────────────────────────────────────────
def implied_prob_from_decimal(price):
    if price is None: return None
    try: return float(np.clip(1.0/float(price), 1e-6, 1.0-1e-6))
    except: return None
def ev_per_dollar(p_win, price):
    if p_win is None or price is None: return None
    try:
        p, o = float(p_win), float(price)
        if o <= 1.0: return None
        return float(p*(o-1.0) - (1.0-p))
    except: return None
# [FIX 6] No-vig price calculation
def remove_vig(price_over, price_under):
    """Return no-vig (fair) decimal prices for a two-sided market."""
    try:
        ip_o = 1.0 / max(float(price_over), 1.0001)
        ip_u = 1.0 / max(float(price_under), 1.0001)
        overround = ip_o + ip_u
        # [AUDIT FIX] Reject corrupted markets: overround should be 1.02-1.10 for typical juice
        if overround <= 0.99 or overround > 1.15:
            return float(price_over), float(price_under)
        return float(1.0 / (ip_o / overround)), float(1.0 / (ip_u / overround))
    except Exception:
        return float(price_over), float(price_under)
def classify_edge(ev):
    """[v6.0] Raised thresholds: only Strong Edge bets are worth taking for 200%+ ROI."""
    if ev is None: return None
    e = float(ev)
    if e <= 0.0:   return "No Edge"
    if e < 0.05:   return "Lean Edge"
    if e < 0.10:   return "Solid Edge"
    return "Strong Edge"
# ──────────────────────────────────────────────
# [UPGRADE NEW] COMPOSITE SHARPNESS SCORE (0–100)
# Combines: model edge + CLV direction + steam move + sharp alignment +
#           trend convergence + hot/cold signal + regime stability.
# This is the single most actionable signal for sharp bettors:
# Score ≥ 70 = elite bet (fire max Kelly)
# Score 55–69 = solid play (standard Kelly)
# Score 40–54 = lean play (half Kelly)
# Score < 40 = skip (below threshold)
# ──────────────────────────────────────────────
def compute_composite_sharpness(
    ev_adj,           # float: EV after volatility penalty (e.g. 0.08 = 8%)
    p_cal,            # float: calibrated win probability
    p_implied,        # float: market implied probability
    hot_cold,         # str: "Hot" / "Cold" / "Average"
    mv_signal,        # dict: line_movement signal
    sharp_div,        # dict: sharp book divergence
    regime,           # str: "Stable" / "Mixed" / "Chaotic"
    trend_score,      # float: -1 to +1 from compute_trend_convergence
    vol_cv,           # float: coefficient of variation
    dnp_risk,         # bool
    b2b,              # bool
    fatigue_label,    # str: "3-in-4" / "4-in-6" / "Normal"
    game_total,       # float or None: game O/U total
    # [v3.0] New signal inputs
    clutch_label=None,   # str: "Clutch Elite" / "Clutch+" / "Clutch-"
    playoff_label=None,  # str: "Tanking" / "Play-In Bubble" / etc.
    wl_factor=1.0,       # float: win/loss split factor
    dnp_prob=0.05,       # float: quantified DNP probability 0-1
):
    """Returns (composite_score 0–100, score_components dict)."""
    if p_cal is None or ev_adj is None:
        return 0, {}
    score = 0.0
    components = {}
    # 1. Model EV (max 35 pts)  — [v6.0] increased weight & steeper scale for high-EV
    # Nonlinear: sqrt scaling rewards large edges disproportionately
    _ev_f = float(ev_adj)
    ev_pts = float(np.clip(math.sqrt(max(0, _ev_f) / 0.15) * 35.0, 0.0, 35.0))
    score += ev_pts
    components["ev"] = round(ev_pts, 1)
    # 2. Model advantage over market (max 20 pts)
    adv = (float(p_cal) - float(p_implied)) if p_implied is not None else 0.0
    adv_pts = float(np.clip(adv / 0.15 * 20.0, -10.0, 20.0))
    score += adv_pts
    components["advantage"] = round(adv_pts, 1)
    # 3. Steam move / line movement (max 15 pts)
    mv_pts = 0.0
    if isinstance(mv_signal, dict):
        steam = mv_signal.get("steam", False)
        pips  = float(mv_signal.get("pips", 0.0) or 0.0)
        if steam and abs(pips) >= 0.5:
            mv_pts = min(15.0, 8.0 + abs(pips) * 4.0)
        elif not steam and abs(pips) >= 0.5:
            mv_pts = -8.0   # fade signal: sharps on other side
    score += mv_pts
    components["steam"] = round(mv_pts, 1)
    # 4. Sharp book divergence (max 10 pts)
    sh_pts = 0.0
    if isinstance(sharp_div, dict) and sharp_div:
        if sharp_div.get("confirm"):
            sh_pts = 10.0
        elif sharp_div.get("fade_model"):
            sh_pts = -10.0
    score += sh_pts
    components["sharp_confirm"] = round(sh_pts, 1)
    # 5. Trend convergence (max 15 pts)
    tr_pts = float(np.clip(float(trend_score or 0.0) * 15.0, -10.0, 15.0))
    score += tr_pts
    components["trend"] = round(tr_pts, 1)
    # 6. Hot/cold signal (max 8 pts)
    hc_pts = 8.0 if hot_cold == "Hot" else (-5.0 if hot_cold == "Cold" else 0.0)
    score += hc_pts
    components["hot_cold"] = round(hc_pts, 1)
    # 7. Regime penalty (max -20 pts) [v6.0] increased penalties
    reg_pts = 0.0 if regime == "Stable" else (-10.0 if regime == "Mixed" else -20.0)
    score += reg_pts
    components["regime"] = round(reg_pts, 1)
    # 8. Volatility (max -10 pts for high CV)
    cv_pts = 0.0
    if vol_cv is not None:
        v = float(vol_cv)
        cv_pts = float(np.clip(-(v - 0.15) / 0.25 * 10.0, -10.0, 0.0))
    score += cv_pts
    components["volatility"] = round(cv_pts, 1)
    # 9. Risk flags (DNP, B2B, fatigue)
    risk_pts = 0.0
    if dnp_risk: risk_pts -= 8.0
    if b2b:      risk_pts -= 4.0
    if fatigue_label == "3-in-4":  risk_pts -= 5.0
    elif fatigue_label == "4-in-6": risk_pts -= 8.0
    score += risk_pts
    components["risk_flags"] = round(risk_pts, 1)
    # 10. Game total environment bonus (max 5 pts)
    gt_pts = 0.0
    if game_total is not None:
        t = float(game_total)
        if t >= 225:   gt_pts = 5.0
        elif t >= 218: gt_pts = 2.0
        elif t <= 208: gt_pts = -3.0
    score += gt_pts
    components["game_total"] = round(gt_pts, 1)
    # 11. [v3.0] Clutch performance signal (max +5 / -5)
    clutch_pts = 0.0
    if clutch_label:
        if "Elite" in str(clutch_label):   clutch_pts = 5.0
        elif "Clutch+" in str(clutch_label): clutch_pts = 3.0
        elif "Clutch-" in str(clutch_label): clutch_pts = -4.0
    score += clutch_pts
    components["clutch"] = round(clutch_pts, 1)
    # 12. [v3.0] Playoff/Tanking context (max +4 / -5)
    playoff_pts = 0.0
    if playoff_label:
        if "Tanking" in str(playoff_label):          playoff_pts = -5.0
        elif "Load Mgmt" in str(playoff_label):       playoff_pts = -3.0
        elif "Play-In Bubble" in str(playoff_label):  playoff_pts = 4.0
        elif "Out of Race" in str(playoff_label):     playoff_pts = -2.0
    score += playoff_pts
    components["playoff_context"] = round(playoff_pts, 1)
    # 13. [v3.0] Win/Loss split factor deviation (max +3 / -3)
    wl_pts = float(np.clip((float(wl_factor or 1.0) - 1.0) * 30.0, -3.0, 3.0))
    score += wl_pts
    components["wl_split"] = round(wl_pts, 1)
    # 14. [v3.0] Quantified DNP risk penalty (max -8)
    dnp_q_pts = float(np.clip(-float(dnp_prob or 0) * 12.0, -8.0, 0.0))
    score += dnp_q_pts
    components["dnp_prob"] = round(dnp_q_pts, 1)
    final = float(np.clip(score, 0.0, 100.0))
    components["total"] = round(final, 1)
    return round(final, 1), components
def sharpness_tier(score):
    """[v6.0] Raised thresholds for elite-only betting. SKIP below 50."""
    if score is None: return "SKIP", "#4A607A"
    s = float(score)
    if s >= 75: return "ELITE",  "#00FFB2"
    if s >= 62: return "SOLID",  "#00AAFF"
    if s >= 50: return "LEAN",   "#FFB800"
    return "SKIP", "#FF3358"
def kelly_fraction(p, price):
    try:
        p, o = float(p), float(price)
        if o<=1.0 or p<=0 or p>=1: return 0.0
        b=o-1.0; q=1.0-p
        if b < 1e-9: return 0.0   # guard against floating-point near-zero denominator
        return float(max(0.0, (b*p-q)/b))
    except: return 0.0
# [FIX 8] Hard negative-edge guard
# [v5.0] Sharpness-aware Kelly: dynamically scales Kelly fraction based on composite
# sharpness score (0–100). Elite bets (≥70) get full frac_kelly; SKIP (<40) get 0.
# Research: Fractional Kelly prevents ruin from estimation errors; dynamic scaling
# captures that higher-signal bets deserve proportionally more allocation.
def recommended_stake(bankroll, p, price_decimal, frac_kelly, cap_frac=0.05,
                      sharpness_score=None):
    try: br=float(bankroll)
    except: br=0.0
    if br<=0 or p is None or price_decimal is None: return 0.0, 0.0, "bankroll<=0"
    k = kelly_fraction(float(p), float(price_decimal))
    if k <= 0:
        return 0.0, 0.0, "negative edge - hard blocked"
    # [v6.0] Aggressive Kelly multiplier — concentrate capital on elite plays
    _base_frac = max(0.0, min(1.0, float(frac_kelly)))
    if sharpness_score is not None:
        _s = float(sharpness_score)
        if _s >= 75:
            _sharp_mult = 1.60    # Elite bet: fire heavy (60% above base Kelly)
        elif _s >= 62:
            _sharp_mult = 1.15    # Solid bet: slight boost
        elif _s >= 50:
            _sharp_mult = 0.60    # Lean bet: reduced sizing
        else:
            _sharp_mult = 0.0     # SKIP: zero allocation (don't waste capital on noise)
        _base_frac = _base_frac * _sharp_mult
    f = _base_frac * k
    f = min(f, float(cap_frac))
    stake = br * f
    if stake <= 0: return 0.0, 0.0, "kelly<=0"
    return float(stake), float(f), "ok"
# ──────────────────────────────────────────────
# TEAM CONTEXT
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*3, show_spinner=False)
def get_team_context():
    try:
        ss = get_season_string()
        adv_raw = LeagueDashTeamStats(season=ss, measure_type_detailed="Advanced",
                                      per_mode_detailed="PerGame").get_data_frames()[0]
        _adv_cols = ["TEAM_ID","TEAM_ABBREVIATION","PACE","REB_PCT","AST_PCT"]
        # L-2 audit fix: also fetch NET_RATING for blowout-risk mismatch calculation
        if "NET_RATING" in adv_raw.columns:
            _adv_cols.append("NET_RATING")
        adv = adv_raw[_adv_cols]
        try:
            defn = LeagueDashTeamStats(season=ss, measure_type_detailed_defense="Defense",
                                       per_mode_detailed="PerGame").get_data_frames()[0][
                   ["TEAM_ID","TEAM_ABBREVIATION","DEF_RATING"]]
            df = adv.merge(defn, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")
        except Exception:
            df = adv.copy()
            df["DEF_RATING"] = 113.0
        if "NET_RATING" not in df.columns:
            df["NET_RATING"] = 0.0
        league_avg = {c: df[c].mean() for c in ["PACE","DEF_RATING","REB_PCT","AST_PCT"]}
        ctx = {}
        for _, r in df.iterrows():
            ctx[str(r["TEAM_ABBREVIATION"]).upper()] = {
                "PACE": float(r.get("PACE",0)),
                "DEF_RATING": float(r.get("DEF_RATING",113)),
                "NET_RATING": float(r.get("NET_RATING",0)),
                "REB_PCT": float(r.get("REB_PCT",0)),
                "AST_PCT": float(r.get("AST_PCT",0)),
            }
        return ctx, league_avg
    except Exception:
        return {}, {}
TEAM_CTX, LEAGUE_CTX = get_team_context()
def get_context_multiplier(opp, market, position):
    def _hash_fallback(o):
        # [AUDIT FIX] Removed ASCII-hash pseudo-randomness (deterministic noise, not signal).
        # Fall back to position-based priors only when team context is unavailable.
        base = 1.0
        bucket = get_position_bucket(position or "")
        if bucket=="Guard" and market in ["Assists","PA","RA"]: base *= 1.02
        elif bucket=="Big" and market in ["Rebounds","PR","RA"]: base *= 1.03
        return float(np.clip(base, 0.92, 1.08))
    if not LEAGUE_CTX or not TEAM_CTX or not opp:
        return _hash_fallback(opp)
    ok = str(opp).upper()
    if ok not in TEAM_CTX:
        return _hash_fallback(opp)
    o = TEAM_CTX[ok]
    lg = LEAGUE_CTX
    pace_f = o["PACE"] / (lg.get("PACE",100) or 1)
    def_f  = (lg.get("DEF_RATING",113) or 1) / (o["DEF_RATING"] or 1)
    reb_adj = (lg.get("REB_PCT",0.5) or 1) / (o.get("REB_PCT",0.5) or 1)
    ast_adj = (lg.get("AST_PCT",0.6) or 1) / (o.get("AST_PCT",0.6) or 1)
    bucket  = get_position_bucket(position or "")
    if bucket=="Guard":   pos_f = 0.5*ast_adj + 0.5*pace_f
    elif bucket=="Wing":  pos_f = 0.5*def_f + 0.5*pace_f
    elif bucket=="Big":   pos_f = 0.6*reb_adj + 0.4*def_f
    else:                 pos_f = pace_f
    if market=="Rebounds":  mult = 0.30*pace_f+0.25*def_f+0.30*reb_adj+0.15*pos_f
    elif market=="Assists": mult = 0.30*pace_f+0.25*def_f+0.30*ast_adj+0.15*pos_f
    elif market in ("RA","PA","PR"): mult = 0.25*pace_f+0.20*def_f+0.25*reb_adj+0.20*ast_adj+0.10*pos_f
    else:                   mult = 0.45*pace_f+0.40*def_f+0.15*pos_f
    return float(np.clip(mult, 0.80, 1.30))
def advanced_context_multiplier(player_name, market, opp, teammate_out):
    pos = get_player_position(player_name) or ""
    base = get_context_multiplier(opp, market, pos)
    if teammate_out: base *= 1.05
    return float(base)
def estimate_blowout_risk(team_abbr, opp_abbr, spread_abs=None):
    # [Research-upgraded] Logistic sigmoid model calibrated from historical NBA blowout data.
    # Research finding: p_blowout = 1 / (1 + exp(-0.25 * (|spread| - 11)))
    # Empirical: ~5% at spread=5, ~25% at spread=11, ~50% at spread=16
    if spread_abs is not None:
        s = abs(float(spread_abs))
        p = float(1.0 / (1.0 + math.exp(-0.25 * (s - 11.0))))
        # Clip to reasonable range: games rarely blow out at <5 spread
        return float(np.clip(p, 0.04, 0.55))
    if TEAM_CTX and LEAGUE_CTX and team_abbr and opp_abbr:
        tk, ok = str(team_abbr).upper(), str(opp_abbr).upper()
        if tk in TEAM_CTX and ok in TEAM_CTX:
            # L-2 audit fix: use NET_RATING differential as strength-of-mismatch proxy.
            # DEF_RATING gap conflated poor defense (high-scoring) with game closeness.
            net_mismatch = abs(TEAM_CTX[tk].get("NET_RATING",0) - TEAM_CTX[ok].get("NET_RATING",0))
            # ~2 pts net mismatch ≈ 1 pt spread; calibrate to logistic sigmoid
            implied_spread = net_mismatch * 0.9
            p = float(1.0 / (1.0 + math.exp(-0.25 * (implied_spread - 11.0))))
            return float(np.clip(p, 0.04, 0.45))
    return 0.10
# ──────────────────────────────────────────────
# [UPGRADE NEW] GAME TOTAL + SPREAD FETCHER
# Fetch game O/U total and spread from Odds API — used for:
#   1) Accurate blowout risk (spread replaces DEF_RATING heuristic)
#   2) Game-script context multiplier (high total → more scoring props)
#   3) Pace environment signal
# Sharp bettors consistently use game totals as the primary filter
# for prop bet environments. A game with total O/U 225+ is a much
# better scoring environment than one with total O/U 205.
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_game_total_and_spread(event_id):
    """
    Returns (total, spread, home_ml, away_ml) from Odds API for a given event.
    total: the over/under game total (e.g. 218.5)
    spread: the home team point spread (e.g. -4.5 means home favored by 4.5)
    Returns (None, None, None, None) on failure.
    """
    key = odds_api_key()
    if not key or not event_id:
        return None, None, None, None
    try:
        data, err = http_get_json(
            f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events/{event_id}/odds",
            {"apiKey": key, "regions": "us", "markets": "totals,spreads,h2h",
             "oddsFormat": "decimal", "dateFormat": "iso"}
        )
        if err or not data:
            return None, None, None, None
        total, spread, home_ml, away_ml = None, None, None, None
        for book in data.get("bookmakers", []):
            bk = book.get("key", "")
            for mkt in book.get("markets", []):
                mk = mkt.get("key", "")
                if mk == "totals" and total is None:
                    for out in mkt.get("outcomes", []):
                        if str(out.get("name", "")).lower() == "over":
                            total = safe_float(out.get("point"))
                            break
                elif mk == "spreads" and spread is None:
                    # H-7 audit fix: Odds API does not guarantee outcome ordering.
                    # Filter by home team name to always get the home spread.
                    home_name = data.get("home_team", "")
                    home_out = next(
                        (o for o in mkt.get("outcomes", [])
                         if normalize_name(o.get("name","")) == normalize_name(home_name)
                         and o.get("point") is not None),
                        None
                    )
                    if home_out is None:
                        # Fallback: take first non-None point if team name match fails
                        home_out = next((o for o in mkt.get("outcomes", []) if o.get("point") is not None), None)
                    if home_out is not None:
                        spread = safe_float(home_out.get("point"))
                elif mk == "h2h":
                    outs = mkt.get("outcomes", [])
                    home_name = data.get("home_team", "")
                    if len(outs) >= 2:
                        for out in outs:
                            if out.get("name", "") == home_name:
                                home_ml = safe_float(out.get("price"))
                            else:
                                away_ml = safe_float(out.get("price"))
            # Prefer sharp books for these — stop early if we have all data from sharp source
            if total is not None and spread is not None and bk in ("pinnacle", "circa", "bookmaker"):
                break
        return total, spread, home_ml, away_ml
    except Exception:
        return None, None, None, None
def compute_implied_team_total(game_total, spread, is_home):
    """
    [v4.0] Compute implied team total from game O/U total and spread.
    Sharp bettors use implied team totals (ITT) as the primary per-team scoring signal.
    Formula: ITT = (total / 2) + (spread / 2) for the team being analyzed.
    Positive spread = underdog; negative spread = favorite.
    Example: Total=226, spread=-4.5 for home team → ITT_home = 226/2 + 4.5/2 = 115.25
    Returns (itt_favorite, itt_underdog) or (None, None) on failure.
    """
    if game_total is None or spread is None:
        return None, None
    try:
        t = float(game_total)
        s = float(spread)   # home team spread (negative = home favored)
        # Home team is favored when spread < 0
        home_favored = s < 0
        abs_spread = abs(s)
        itt_home = (t / 2.0) + (abs_spread / 2.0 if home_favored else -abs_spread / 2.0)
        itt_away = t - itt_home
        return float(itt_home), float(itt_away)
    except Exception:
        return None, None
def compute_game_script_mult(game_total, spread_abs, market, team_abbr, opp_abbr,
                              is_home=None, game_spread=None):
    """
    Compute a game-script context multiplier based on the game total and spread.
    [v4.0] Now uses IMPLIED TEAM TOTAL (ITT) for the player's team when available.
    ITT is more accurate than raw game total as it accounts for point spread.
    High totals boost scoring, rebounding, and assists props.
    Blowout risk (large spread) penalizes starters via early garbage time.
    League avg total ~220, avg ITT ~110. Values calibrated from empirical NBA data.
    """
    try:
        mult = 1.0
        # [v4.0] Use implied team total when spread info is available
        _effective_total = float(game_total) if game_total is not None else None
        if game_total is not None and game_spread is not None and is_home is not None:
            _itt_home, _itt_away = compute_implied_team_total(game_total, game_spread, is_home)
            if _itt_home is not None:
                # Use player's team ITT × 2 as effective total (so league avg ~220 still anchors)
                _player_itt = _itt_home if is_home else _itt_away
                if _player_itt is not None:
                    _effective_total = float(_player_itt) * 2.0  # rescale to game-total units
        if _effective_total is not None:
            t = _effective_total
            # League avg ~220. Each 5 points above/below adjusts by ~1.5%
            delta = (t - 220.0) / 5.0
            if market in ("Points", "PRA", "PA", "PR", "Alt Points", "H1 Points", "H2 Points"):
                mult *= float(np.clip(1.0 + delta * 0.015, 0.88, 1.12))
            elif market in ("Assists", "RA"):
                mult *= float(np.clip(1.0 + delta * 0.012, 0.90, 1.10))
            elif market in ("Rebounds", "PR"):
                # More possessions = more rebounding opportunities but also more FGM
                mult *= float(np.clip(1.0 + delta * 0.008, 0.92, 1.08))
            elif market in ("3PM", "Alt 3PM", "H1 3PM"):
                # High-total games correlate with open 3s
                mult *= float(np.clip(1.0 + delta * 0.010, 0.90, 1.10))
            # Low total games (< 210): harder for any scoring props
            if t < 210 and market in ("Points", "PRA", "PA", "PR"):
                mult *= float(np.clip(1.0 - (210 - t) / 200.0, 0.87, 1.0))
        return float(np.clip(mult, 0.80, 1.20))
    except Exception:
        return 1.0
# ──────────────────────────────────────────────
# ODDS API  [FIX 7: retry on 429]
# ──────────────────────────────────────────────
def odds_api_key():
    return (st.secrets.get("ODDS_API_KEY","") if hasattr(st,"secrets") else "") or os.getenv("ODDS_API_KEY","")
# [FIX 7] Retry with exponential backoff on 429
def http_get_json(url, params, timeout=25):
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            rem = r.headers.get("x-requests-remaining")
            used = r.headers.get("x-requests-used")
            st.session_state["_odds_headers_last"] = {"remaining":rem,"used":used,"ts":_now_iso()}
            if r.status_code == 429:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                return None, "Rate limited (429) - quota exhausted"
            r.raise_for_status()
            return r.json(), None
        except requests.exceptions.HTTPError as e:
            detail = ""
            try: detail = e.response.text[:2000]
            except: pass
            return None, f"HTTP {getattr(e.response,'status_code',None)}: {detail}"
        except requests.exceptions.ConnectionError:
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            return None, "Connection failed after retries"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    return None, "All retries failed"
def _utc_to_et_date(utc_iso: str) -> str:
    """Convert UTC ISO timestamp to US Eastern date string (YYYY-MM-DD).
    NBA games tip off in Eastern time; Odds API returns UTC. Late games
    (e.g. 10 PM ET = 3 AM UTC next day) must be mapped to their ET date
    so today's slate is shown correctly when using date.today().
    """
    try:
        dt_str = utc_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        # [AUDIT FIX] Precise DST boundaries instead of month approximation.
        # Previous code used "3 < month < 11" which was wrong on transition Sundays
        # (e.g. March 8, 2026 = DST start → should be EST UTC-5 until 2 AM local).
        # DST: 2nd Sunday in March at 2 AM → 1st Sunday in November at 2 AM.
        # Compute 2nd Sunday in March and 1st Sunday in November for dt.year.
        def _nth_sunday(year, month, n):
            """Return date of nth Sunday of given month/year (1-indexed)."""
            d = date(year, month, 1)
            # Day of week: Monday=0, Sunday=6
            first_sunday = d + timedelta(days=(6 - d.weekday()) % 7)
            return first_sunday + timedelta(weeks=n - 1)
        dst_start = _nth_sunday(dt.year, 3, 2)   # 2nd Sunday March
        dst_end   = _nth_sunday(dt.year, 11, 1)  # 1st Sunday November
        dt_date = dt.date()
        in_edt = dst_start <= dt_date < dst_end
        offset = -4 if in_edt else -5
        et_dt = dt + timedelta(hours=offset)
        return et_dt.date().isoformat()
    except Exception:
        return utc_iso[:10]  # fallback to UTC date
# Lines cache TTL: 2 hours — reduces Odds API usage significantly.
# Use the Force Refresh button to bypass when needed.
_LINES_CACHE_TTL = 60 * 60 * 2
@st.cache_data(ttl=_LINES_CACHE_TTL, show_spinner=False)
def odds_get_events(date_iso=None):
    key = odds_api_key()
    if not key: return [], "Missing ODDS_API_KEY"
    data, err = http_get_json(f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events", {"apiKey":key})
    if err or not isinstance(data, list): return [], err or "Unexpected events response"
    if date_iso:
        # Convert UTC commence_time → Eastern date to handle late games (10 PM ET = next UTC day)
        return [ev for ev in data if _utc_to_et_date(ev.get("commence_time") or "") == date_iso], None
    return data, None
# [FIX 14] Week-ahead: fetch events for a date range
@st.cache_data(ttl=_LINES_CACHE_TTL, show_spinner=False)
def odds_get_events_range(start_iso, end_iso):
    key = odds_api_key()
    if not key: return [], "Missing ODDS_API_KEY"
    data, err = http_get_json(f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events", {"apiKey":key})
    if err or not isinstance(data, list): return [], err or "Unexpected events response"
    filtered = []
    for ev in data:
        ct = _utc_to_et_date(ev.get("commence_time") or "")
        if start_iso <= ct <= end_iso:
            filtered.append(ev)
    return filtered, None
@st.cache_data(ttl=_LINES_CACHE_TTL, show_spinner=False)
def odds_get_event_odds(event_id, market_keys, regions=REGION_US):
    key = odds_api_key()
    if not key: return None, "Missing ODDS_API_KEY"
    data, err = http_get_json(
        f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events/{event_id}/odds",
        {"apiKey":key,"regions":regions,"markets":",".join(market_keys),
         "oddsFormat":"decimal","dateFormat":"iso"}
    )
    return data, err
@st.cache_data(ttl=60*60*24, show_spinner=False)
def lookup_player_id(name):
    if not name: return None
    nm = normalize_name(name)
    plist = nba_players.get_players()
    for p in plist:
        if normalize_name(p.get("full_name","")) == nm:
            return p.get("id")
    names = [p.get("full_name","") for p in plist]
    cand = difflib.get_close_matches(name, names, n=1, cutoff=0.75)
    if cand:
        for p in plist:
            if p.get("full_name") == cand[0]: return p.get("id")
    return None
def nba_headshot_url(pid):
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png" if pid else None
@st.cache_data(ttl=60*60*24, show_spinner=False)
def get_team_maps():
    teams = nba_teams.get_teams()
    by_name = {}
    for t in teams:
        full = t.get("full_name","")
        by_name[normalize_name(full)] = {"abbr":t.get("abbreviation",""),"id":t.get("id"),"full_name":full}
    for a,tgt in [("la clippers","los angeles clippers"),("la lakers","los angeles lakers")]:
        if normalize_name(tgt) in by_name: by_name[normalize_name(a)] = by_name[normalize_name(tgt)]
    return by_name
def map_team_name_to_abbr(name):
    m = get_team_maps()
    rec = m.get(normalize_name(name))
    return rec["abbr"] if rec else None
@st.cache_data(ttl=60*60*24, show_spinner=False)
def team_id_to_abbr_map():
    return {int(t["id"]): t["abbreviation"] for t in nba_teams.get_teams()}
@st.cache_data(ttl=60*10, show_spinner=False)
def nba_scoreboard_games(game_date):
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=game_date.strftime("%m/%d/%Y"))
        df = sb.get_data_frames()[0]
        return [{"game_id":str(r.get("GAME_ID")),"home_team_id":int(r.get("HOME_TEAM_ID")),
                 "away_team_id":int(r.get("VISITOR_TEAM_ID"))} for _,r in df.iterrows()], None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"
def opponent_from_team_abbr(team_abbr, game_date):
    games, _ = nba_scoreboard_games(game_date)
    tid_map = team_id_to_abbr_map()
    for g in (games or []):
        ha = tid_map.get(g["home_team_id"])
        aa = tid_map.get(g["away_team_id"])
        if ha == team_abbr: return aa, True
        if aa == team_abbr: return ha, False
    return None, None
def _parse_player_prop_outcomes(event_odds, market_key, book_filter=None):
    if not event_odds: return [], None
    eid = event_odds.get("id"); home = event_odds.get("home_team"); away = event_odds.get("away_team")
    ct = event_odds.get("commence_time"); books = event_odds.get("bookmakers",[]) or []
    rows = []
    for b in books:
        bkey = b.get("key")
        if book_filter and book_filter not in ("consensus","all") and bkey != book_filter: continue
        for mk in b.get("markets",[]) or []:
            if mk.get("key") != market_key: continue
            for out in mk.get("outcomes",[]) or []:
                player = out.get("description") or out.get("name")
                point_val = out.get("point")
                side_val  = out.get("name") or ""
                # Handle binary markets (DD/TD: Yes/No) which have no numeric point.
                # First Basket is a binary win/lose market (player name = outcome); routing
                # it through the numeric prop engine (P(pts > 0.5) ≈ 1.0) produces garbage.
                # Skip First Basket entirely — it needs a dedicated binary model.
                if market_key == "player_first_basket":
                    continue
                if point_val is None and player:
                    side_lwr = side_val.strip().lower()
                    if side_lwr == "no":
                        continue          # skip No side — engine models Yes probability
                    # "Yes" outcomes → map to "Over"
                    side_val  = "Over"
                    point_val = 0.5
                if player and point_val is not None:
                    rows.append({"player":player,"player_norm":normalize_name(player),
                                 "line":float(point_val),"price":out.get("price"),
                                 "book":(bkey or ""),"side":side_val,
                                 "market_key":market_key,"event_id":eid,
                                 "home_team":home,"away_team":away,"commence_time":ct})
    if book_filter == "consensus" and rows:
        df = pd.DataFrame(rows)
        df = df[pd.to_numeric(df["line"],errors="coerce").notna()].copy()
        if df.empty: return [], None
        df["w"] = df["book"].apply(lambda k: book_sharpness(str(k)))
        name_map = df.groupby("player_norm")["player"].agg(lambda x: x.value_counts().index[0]).to_dict()
        out_rows = []
        for (pn, side), sub in df.groupby(["player_norm","side"], dropna=False):
            sub = sub.copy().sort_values("line")
            if float(sub["w"].sum() or 0)>0:
                cw = sub["w"].cumsum(); cutoff=0.5*float(sub["w"].sum())
                line_med = float(sub.loc[cw>=cutoff,"line"].iloc[0])
            else:
                line_med = float(sub["line"].median())
            price_syn = None
            try:
                v = sub[pd.to_numeric(sub["price"],errors="coerce").notna()].copy()
                if not v.empty:
                    v["pf"] = pd.to_numeric(v["price"],errors="coerce").astype(float)
                    v = v[v["pf"]>1.0001]
                    if not v.empty:
                        v["pi"] = 1.0/v["pf"]; ws=float(v["w"].sum() or 0)
                        ps = float((v["pi"]*v["w"]).sum()/max(ws,1e-9))
                        ps = float(np.clip(ps,1e-6,1-1e-6)); price_syn=float(1.0/ps)
            except Exception: price_syn=None
            out_rows.append({"player":name_map.get(pn,pn),"player_norm":pn,"line":float(line_med),
                             "price":price_syn,"book":"consensus","side":side,"market_key":market_key,
                             "event_id":eid,"home_team":home,"away_team":away,"commence_time":ct})
        return out_rows, None
    return rows, None
def find_player_line_from_events(player_name, market_key, date_iso, book_choice):
    # [FIX] Skip unsupported markets early to avoid HTTP 422 from Odds API
    if market_key in _UNSUPPORTED_API_KEYS:
        return None, None, f"Market {market_key} is unsupported by Odds API — use PP/UD/Sleeper"
    evs, err = odds_get_events(date_iso)
    if err: return None, None, err
    if not evs: return None, None, "No events for that date"
    target = normalize_name(player_name)
    # [FIX H1/H2/ALT] Use broader regions for specialty markets
    regions = "us,us2,eu,uk" if market_key in SPECIALTY_MARKET_KEYS else REGION_US
    for ev in evs:
        eid = ev.get("id")
        if not eid: continue
        odds, oerr = odds_get_event_odds(eid, (market_key,), regions=regions)
        if oerr or not odds: continue
        rows, _ = _parse_player_prop_outcomes(odds, market_key, book_filter=book_choice)
        for r in rows:
            if r.get("player_norm") == target: return float(r["line"]), r, None
        norms = [r.get("player_norm","") for r in rows]
        close = difflib.get_close_matches(target, norms, n=1, cutoff=0.88)
        if close:
            rr = next((x for x in rows if x.get("player_norm")==close[0]), None)
            if rr: return float(rr["line"]), rr, None
    return None, None, "Player/market not found in Odds API props"
# Known major sportsbooks — shown as fallback when Odds API has no data for the date
_FALLBACK_BOOKS = [
    "draftkings", "fanduel", "betmgm", "caesars", "pointsbet",
    "bet365", "betrivers", "unibet", "wynnbet", "barstool",
]
@st.cache_data(ttl=60*30, show_spinner=False)
def get_sportsbook_choices(date_iso):
    evs, err = odds_get_events(date_iso)
    if err:
        # API key missing or quota exceeded — return fallback with error
        return ["consensus"] + _FALLBACK_BOOKS, err
    if not evs:
        # No games on this date — return fallback silently
        return ["consensus"] + _FALLBACK_BOOKS, None
    # Try multiple market keys so we find bookmakers even if one market isn't offered
    _probe_markets = [
        ODDS_MARKETS["Points"], ODDS_MARKETS["Rebounds"], ODDS_MARKETS["Assists"],
        ODDS_MARKETS["PRA"],
    ]
    for ev in evs[:15]:
        eid = ev.get("id")
        if not eid: continue
        for mk in _probe_markets:
            odds, oerr = odds_get_event_odds(eid, (mk,))
            if odds and not oerr:
                books = sorted(list(dict.fromkeys(
                    b.get("key") for b in odds.get("bookmakers", []) if b.get("key"))))
                if books:
                    return ["consensus"] + books, None
    # Events exist but none returned bookmaker data — use fallback
    return ["consensus"] + _FALLBACK_BOOKS, None
# ──────────────────────────────────────────────
# SHARP BOOK DIVERGENCE ALERT
# ──────────────────────────────────────────────
def sharp_divergence_alert(event_id, market_key, player_norm, side, model_side="Over"):
    try:
        sharp_books = ["pinnacle","circa","bookmaker"]
        soft_odds, _ = odds_get_event_odds(str(event_id), (str(market_key),))
        if not soft_odds: return {}
        sharp_lines, soft_lines = [], []
        for b in soft_odds.get("bookmakers",[]) or []:
            bk = (b.get("key") or "").lower()
            for mk in b.get("markets",[]) or []:
                if mk.get("key") != market_key: continue
                for out in mk.get("outcomes",[]) or []:
                    pn = normalize_name(out.get("description") or out.get("name") or "")
                    if pn == player_norm and (out.get("name") or "").lower() == side.lower():
                        line = out.get("point")
                        if line is not None:
                            if bk in sharp_books: sharp_lines.append(float(line))
                            else: soft_lines.append(float(line))
        if not sharp_lines or not soft_lines: return {}
        sl = np.mean(sharp_lines); softl = np.mean(soft_lines)
        diff = sl - softl  # positive = sharp line higher than soft line
        is_over = "over" in str(model_side).lower()
        # [AUDIT FIX] Previous "confirm": abs(diff) < 0.3 just meant lines agree — not a
        # confirmation of your bet direction. True sharp confirmation means sharps pushed the
        # line in YOUR direction (higher for Over, lower for Under = books limiting Under bets).
        # "confirm": sharp line > soft line AND betting Over, OR sharp line < soft line AND Under.
        confirm = (diff > 0.25 and is_over) or (diff < -0.25 and not is_over)
        # Fade: sharps on opposite side — line moved against your direction
        fade_model = (diff < -0.50 and is_over) or (diff > 0.50 and not is_over)
        # Lines agree: no divergence signal either way
        no_signal = abs(diff) < 0.25
        return {"sharp_line": sl, "soft_line": softl, "diff": diff,
                "confirm": confirm, "fade_model": fade_model, "no_signal": no_signal}
    except Exception:
        return {}
# ──────────────────────────────────────────────
# CLV TRACKING  [FIX 6: no-vig CLV]
# ──────────────────────────────────────────────
def fetch_latest_market_for_leg(leg):
    try:
        eid = leg.get("event_id"); mk = leg.get("market_key")
        if not eid or not mk: return None, None, None, "missing event_id/market_key"
        pn = leg.get("player_norm") or normalize_name(leg.get("player",""))
        side = (leg.get("side") or "Over").strip()
        for bf in [(leg.get("book") or "consensus").strip().lower(), "consensus"]:
            odds, oerr = odds_get_event_odds(str(eid), (str(mk),))
            if oerr or not odds: return None, None, None, oerr or "fetch failed"
            rows, _ = _parse_player_prop_outcomes(odds, str(mk), book_filter=(bf if bf!="all" else None))
            m = next((r for r in rows if r.get("player_norm")==pn and str(r.get("side","")).strip()==side), None)
            if m:
                ln = safe_float(m.get("line")); pr = m.get("price")
                try: pr = float(pr) if pr is not None else None
                except: pr = None
                return ln, pr, (m.get("book") or bf), None
        return None, None, None, "player/side not found"
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"
def apply_clv_update_to_legs(legs):
    errs, out = [], []
    for leg in legs:
        leg2 = dict(leg)
        line0 = safe_float(leg2.get("line")); price0 = safe_float(leg2.get("price_decimal"))
        line1, price1, book_used, err = fetch_latest_market_for_leg(leg2)
        leg2["close_ts"]=_now_iso(); leg2["line_close"]=line1
        leg2["price_close"]=price1; leg2["book_close"]=book_used
        side = (leg2.get("side") or "Over").strip().lower()
        if line0 is not None and line1 is not None:
            leg2["clv_line"]=float(line1-line0)
            leg2["clv_line_fav"]=bool(line1<line0 if "under" not in side else line1>line0)
        else:
            leg2["clv_line"]=None; leg2["clv_line_fav"]=None
        # M-2 audit fix: the complement approximation (1/(1-1/p)) yields overround=1.0,
        # so remove_vig returns the raw price unchanged — no vig is actually removed.
        # Fix: fetch the other side's closing price for proper two-sided vig removal.
        # For the open price, use a standard ~4.5% player-prop overround assumption.
        if price0 is not None and price1 is not None and price0 > 1 and price1 > 1:
            _OVR_ASSUMED = 1.045  # typical player-prop overround; makes nv ~ price * OVR
            # Open side: no historical other side available; apply standard overround
            imp0 = 1.0 / price0
            nv0 = float(np.clip(1.0 / (imp0 / _OVR_ASSUMED), 1.001, 50.0))
            # Close side: try to fetch live other-side price for true two-sided removal
            opp_side = "Under" if "over" in (leg2.get("side") or "Over").lower() else "Over"
            leg_opp = dict(leg2); leg_opp["side"] = opp_side
            _, price_opp, _, _ = fetch_latest_market_for_leg(leg_opp)
            if price_opp is not None and float(price_opp) > 1:
                nv1, _ = remove_vig(price1, float(price_opp))
            else:
                imp1 = 1.0 / price1
                nv1 = float(np.clip(1.0 / (imp1 / _OVR_ASSUMED), 1.001, 50.0))
            leg2["clv_price"]=float(nv1-nv0)
            leg2["clv_price_fav"]=bool(nv1>nv0)
            leg2["clv_price_novig_open"]=float(nv0)
            leg2["clv_price_novig_close"]=float(nv1)
        else:
            leg2["clv_price"]=None; leg2["clv_price_fav"]=None
        if err: errs.append(f"{leg2.get('player')} {leg2.get('market')}: {err}")
        out.append(leg2)
    return out, errs
# ──────────────────────────────────────────────
# USAGE RATE FROM BOX SCORE
# ──────────────────────────────────────────────
def compute_usage_rate(game_log_df, n_games=10):
    """Approximate possession usage: FGA + 0.44*FTA + TOV per game."""
    if game_log_df is None or game_log_df.empty:
        return None
    df = game_log_df.head(n_games).copy()
    if not all(c in df.columns for c in ["FGA","FTA","TOV"]):
        return None
    try:
        fga = pd.to_numeric(df["FGA"], errors="coerce").fillna(0)
        fta = pd.to_numeric(df["FTA"], errors="coerce").fillna(0)
        tov = pd.to_numeric(df["TOV"], errors="coerce").fillna(0)
        return float((fga + 0.44*fta + tov).mean())
    except Exception:
        return None
# ──────────────────────────────────────────────
# [AUDIT IMPROVEMENT] USAGE TREND DETECTOR (WALLY PIPP EFFECT)
# Compares L5 usage vs L20 usage to detect rising/falling role in team offense.
# A player whose L5 usage is significantly above their L20 baseline is likely
# receiving more offensive responsibility (injury to teammate, coach adjustment,
# or emerging role). L5 spike = leading indicator for stat line improvement.
# Research: Haralabob Voulgaris identifies usage trend as a top-3 signal for
# same-week NBA props. EVAnalytics (2024) found L5 vs L15 usage delta had 0.31
# Spearman correlation with Points over/under outcomes.
# ──────────────────────────────────────────────
def compute_usage_trend(game_log_df, n_recent=5, n_baseline=20, market="Points"):
    """
    [AUDIT IMPROVEMENT] Detect rising/falling usage trend (Wally Pipp effect).
    Returns (usage_trend_mult, usage_trend_label, l5_usage, l20_usage).
    usage_trend_mult: float multiplier for scoring/volume props
      >1.0 = usage trending up (positive signal for Points/FGM/PRA)
      <1.0 = usage declining (negative signal)
    Only meaningful for volume-dependent markets (Points, FGM, PRA, PA, Assists).
    """
    _VOLUME_MARKETS = {"Points", "FGM", "FGA", "PRA", "PA", "Assists", "Fantasy Score",
                       "H1 Points", "H2 Points", "Q1 Points"}
    if market not in _VOLUME_MARKETS:
        return 1.0, "N/A", None, None
    if game_log_df is None or game_log_df.empty:
        return 1.0, "Insufficient", None, None
    df = game_log_df.copy()
    if not all(c in df.columns for c in ["FGA", "FTA", "TOV"]):
        return 1.0, "Insufficient", None, None
    try:
        fga = pd.to_numeric(df["FGA"], errors="coerce").fillna(0)
        fta = pd.to_numeric(df["FTA"], errors="coerce").fillna(0)
        tov = pd.to_numeric(df["TOV"], errors="coerce").fillna(0)
        poss = fga + 0.44 * fta + tov  # proxy possession usage
        if len(poss) < n_recent + 3:
            return 1.0, "Insufficient", None, None
        l5_usage  = float(poss.iloc[:n_recent].mean())
        l20_usage = float(poss.iloc[:n_baseline].mean()) if len(poss) >= n_baseline else float(poss.mean())
        if l20_usage <= 0.5:
            return 1.0, "Avg", l5_usage, l20_usage
        delta_pct = (l5_usage - l20_usage) / l20_usage  # fractional change
        # Significant usage spike/drop: ±15% threshold (research-validated)
        # Weight: 0.50 max contribution — usage is directional, not causal alone
        if delta_pct >= 0.20:
            mult = float(np.clip(1.0 + delta_pct * 0.50, 1.0, 1.10))
            label = f"Usage Spike +{delta_pct*100:.0f}% L5vsL20"
        elif delta_pct >= 0.10:
            mult = float(np.clip(1.0 + delta_pct * 0.40, 1.0, 1.05))
            label = f"Usage Rising +{delta_pct*100:.0f}% L5vsL20"
        elif delta_pct <= -0.20:
            mult = float(np.clip(1.0 + delta_pct * 0.50, 0.90, 1.0))
            label = f"Usage Drop {delta_pct*100:.0f}% L5vsL20"
        elif delta_pct <= -0.10:
            mult = float(np.clip(1.0 + delta_pct * 0.40, 0.95, 1.0))
            label = f"Usage Fading {delta_pct*100:.0f}% L5vsL20"
        else:
            mult = 1.0
            label = f"Usage Stable ({delta_pct*100:+.0f}% L5vsL20)"
        return mult, label, l5_usage, l20_usage
    except Exception:
        return 1.0, "Avg", None, None
# ──────────────────────────────────────────────
# [v4.0] ON/OFF LINEUP USAGE BOOST
# When a star teammate is OUT, the player's usage increases by their
# "absorbed" share of possessions. Research: ~60-70% of a star's
# possessions are redistributed proportionally by usage rate.
# Guards absorb more (faster pace, ball handling); Bigs absorb less.
# This is the most precise injury replacement model available without
# full lineup data — uses game-log filtered samples as a proxy.
# ──────────────────────────────────────────────
def compute_lineup_injury_boost(player_name, team_abbr, out_player_name,
                                market, position_bucket, usage_rate, n_games=10):
    """
    [v4.0] Estimate usage boost when a key teammate is out.
    Returns (lineup_boost_mult, lineup_label).
    Improvement over flat 1.05: scales by injured player's likely usage and
    market-specific absorption (Points/Assists benefit most; Rebounds less so).
    """
    if not out_player_name or not team_abbr:
        return 1.0, "No Lineup Change"
    try:
        # Estimate the injured player's usage tier from their name
        # Use position and team as proxies when game log isn't available
        out_pid = lookup_player_id(str(out_player_name).title())
        out_usage = None
        if out_pid:
            try:
                out_gl, _ = fetch_player_gamelog(out_pid, max_games=n_games)
                if not out_gl.empty:
                    out_usage = compute_usage_rate(out_gl, n_games)
            except Exception:
                pass
        # Fallback usage estimate based on injured player role
        if out_usage is None:
            out_usage = 18.0  # league avg starter usage
        # How much of injured player's usage does OUR player absorb?
        # Research: ~35-45% goes to closest positional match,
        # ~20-30% distributed across other players.
        # Market-specific absorption: PG/SG absorb assists/scoring, PF/C absorb rebounds
        _absorption = {
            "Points": 0.35, "PRA": 0.30, "PA": 0.30, "PR": 0.25,
            "Assists": 0.40,  # ball-handling redistributed more to next PG/SG
            "Rebounds": 0.20,  # rebounds redistributed to same-position player
            "RA": 0.25, "3PM": 0.30, "FGM": 0.30, "FGA": 0.30,
        }
        # Our player's current usage as share of team
        player_usage = usage_rate if (usage_rate and usage_rate > 0) else 15.0
        _abs_rate = _absorption.get(market, 0.28)
        # Position proximity bonus: Guard absorbs guard's usage better
        _pos_bonus = 1.15 if position_bucket in ("Guard",) else 1.0
        absorbed_usage = out_usage * _abs_rate * _pos_bonus
        # Convert to multiplier: how much does our player's stat increase?
        if player_usage <= 0:
            return 1.05, "Lineup Boost (Low Usage)"
        boost_ratio = (player_usage + absorbed_usage) / player_usage
        boost_mult = float(np.clip(boost_ratio, 1.0, 1.20))  # cap at 20% boost
        label = f"Lineup Boost +{(boost_mult-1)*100:.0f}% ({out_player_name.split()[-1] if out_player_name else 'teammate'} OUT)"
        return boost_mult, label
    except Exception:
        return 1.05, "Lineup Boost (Fallback)"
# ──────────────────────────────────────────────
# PACE-ADJUSTED STAT SERIES
# ──────────────────────────────────────────────
def compute_pace_adjusted_series(stat_series, opp_team):
    """Scale stat series by opponent-vs-league pace ratio (dampened 50%)."""
    if stat_series is None or len(stat_series.dropna()) < 4:
        return stat_series
    if not LEAGUE_CTX or not TEAM_CTX:
        return stat_series
    opp_key = str(opp_team or "").upper()
    if opp_key not in TEAM_CTX:
        return stat_series
    try:
        opp_pace = TEAM_CTX[opp_key].get("PACE", 100)
        league_pace = LEAGUE_CTX.get("PACE", 100) or 1.0
        pace_adj = opp_pace / league_pace
        adj_factor = float(np.clip(1.0 + 0.5*(pace_adj - 1.0), 0.88, 1.12))
        return stat_series * adj_factor
    except Exception:
        return stat_series
# [v5.0] Game-specific pace multiplier from O/U game total.
# Voulgaris method: game total is the single best environmental signal for scoring props.
# O/U 230+ = high-pace game → props hit more. O/U 205 = grind = props hit less.
# Uses league-average implied total (~214 pts in 2024-25) as baseline reference.
# Returns a multiplier 0.92–1.08 for use as an additional projection factor.
_LEAGUE_AVG_GAME_TOTAL = 226.5  # 2024-25 league average O/U total (empirical; update each season)
def compute_game_total_pace_mult(game_total, market):
    """Compute pace/environment multiplier from game O/U total vs league average."""
    if game_total is None:
        return 1.0
    # Only applies to scoring/volume markets; not rate stats (shooting %, DD/TD)
    _pace_sensitive = {"Points", "Rebounds", "Assists", "PRA", "PR", "PA", "RA",
                       "3PM", "Fantasy Score", "FTA", "FTM", "Steals", "Blocks", "Stocks"}
    if market not in _pace_sensitive:
        return 1.0
    try:
        gt = float(game_total)
        # Linear: each 10 pts above/below avg = ~2.5% shift (dampened 50% for props)
        raw_adj = (gt - _LEAGUE_AVG_GAME_TOTAL) / 10.0 * 0.025
        return float(np.clip(1.0 + raw_adj, 0.92, 1.08))
    except Exception:
        return 1.0
# ──────────────────────────────────────────────
# PER-POSITION DEFENSIVE GRADES  (one call, all teams)
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*6, show_spinner=False)
def _fetch_positional_def_full(position_bucket):
    """One LeagueDashPtDefend call → {TEAM_ABBR: pts_allowed} + league_avg."""
    try:
        from nba_api.stats.endpoints import LeagueDashPtDefend
        cat_map = {"Guard": "Guards", "Wing": "Forwards", "Big": "Centers", "Unknown": "Overall"}
        category = cat_map.get(position_bucket, "Overall")
        df = LeagueDashPtDefend(
            league_id="00", per_mode_simple="PerGame",
            defense_category=category, season=get_season_string(),
        ).get_data_frames()[0]
        if df.empty:
            return {}, None
        league_avg = float(df["PTS_ALLOWED"].mean())
        team_map = {str(r["TEAM_ABBREVIATION"]).upper(): float(r["PTS_ALLOWED"])
                    for _, r in df.iterrows()}
        return team_map, league_avg
    except Exception:
        return {}, None
def get_opp_positional_pts_allowed(opp_abbr, position_bucket):
    """Returns (opp_pts_allowed, league_avg) using cached bulk table."""
    team_map, league_avg = _fetch_positional_def_full(position_bucket)
    opp = str(opp_abbr or "").upper()
    return team_map.get(opp), league_avg
def positional_def_multiplier(opp_abbr, position_bucket, market):
    """Return a multiplier based on opponent's positional defensive strength.
    [AUDIT UPGRADE] Extended to Rebounds, Assists, Blocks, Steals, Stocks:
    - Scoring: full PTS_ALLOWED ratio impact (caps ±18%)
    - Rebounding: bigs dominate, so positional reb defense matters (caps ±10%)
    - Assists/playmaking: guard DvP affects ball-handler ast rate (caps ±8%)
    - Blocks/Steals: low-volume, high-variance; apply light positional signal (caps ±6%)
    Market → sensitivity mapping calibrated from Cleaning the Glass DvP data.
    """
    # Market sensitivity: how strongly positional defense impacts each market
    _MARKET_SENSITIVITY = {
        "Points":     (1.00, 0.82, 1.18),   # (scale, min_clip, max_clip)
        "PRA":        (0.90, 0.84, 1.16),
        "PR":         (0.85, 0.85, 1.15),
        "PA":         (0.80, 0.86, 1.14),
        "H1 Points":  (0.90, 0.84, 1.16),
        "H2 Points":  (0.90, 0.84, 1.16),
        "Rebounds":   (0.60, 0.90, 1.10),   # Rebounding less tied to scoring DvP
        "RA":         (0.55, 0.91, 1.09),
        "Assists":    (0.50, 0.92, 1.08),   # Assists: guard positional DvP moderately useful
        "Blocks":     (0.35, 0.94, 1.06),   # Low-volume, use light signal
        "Steals":     (0.35, 0.94, 1.06),
        "Stocks":     (0.35, 0.94, 1.06),
        "3PM":        (0.45, 0.93, 1.07),   # 3PM: opponent perimeter defense relevant
    }
    params = _MARKET_SENSITIVITY.get(market)
    if params is None:
        return 1.0
    scale, lo, hi = params
    try:
        opp_pts, league_avg = get_opp_positional_pts_allowed(opp_abbr, position_bucket)
        if opp_pts is None or league_avg is None or league_avg == 0:
            return 1.0
        raw_ratio = opp_pts / league_avg          # >1 = soft defense, <1 = elite
        scaled_ratio = 1.0 + (raw_ratio - 1.0) * scale
        return float(np.clip(scaled_ratio, lo, hi))
    except Exception:
        return 1.0
# ──────────────────────────────────────────────
# INJURY REPORT  (ESPN public API — works every day, not just game days)
# ──────────────────────────────────────────────
_ESPN_INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_injury_report():
    """Fetch NBA injury report from ESPN (primary) with NBA API fallback.
    Returns:
        dict: {team_abbr_upper: [{"player": str, "status": str, "reason": str}, ...]}
        AND sets st.session_state["injury_team_map"] = {team_abbr: [player_name, ...]} for OUT/DOUBTFUL only.
    """
    out = {}
    try:
        r = requests.get(
            _ESPN_INJURY_URL,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=15,
        )
        if r.ok:
            data = r.json()
            for entry in data.get("injuries", []):
                team_info = entry.get("team", {})
                abbr = str(team_info.get("abbreviation", "")).upper()
                for inj in entry.get("injuries", []):
                    athlete = inj.get("athlete", {})
                    pname = athlete.get("fullName", "") or athlete.get("displayName", "")
                    status_raw = str(inj.get("status", "")).upper()
                    status = status_raw if status_raw in ("OUT","DOUBTFUL","QUESTIONABLE") else status_raw
                    reason_detail = inj.get("details", {}) or {}
                    reason = reason_detail.get("type","") or reason_detail.get("returnDate","")
                    if pname and status:
                        out.setdefault(abbr, []).append({
                            "player": pname, "status": status, "reason": reason,
                        })
    except Exception:
        pass
    # NBA API fallback (only on game days)
    if not out:
        try:
            from nba_api.stats.endpoints import InjuryReport as NBAInjuryReport
            df = NBAInjuryReport(game_date=date.today().strftime("%m/%d/%Y")).get_data_frames()[0]
            for _, r in df.iterrows():
                team = str(r.get("TEAM_TRICODE","")).upper()
                status = str(r.get("PLAYER_STATUS","")).upper()
                if status in ("OUT","DOUBTFUL","QUESTIONABLE"):
                    out.setdefault(team, []).append({
                        "player": r.get("PLAYER_NAME",""),
                        "status": status,
                        "reason": r.get("RETURN_FROM_INJURY",""),
                    })
        except Exception:
            pass
    return out
# ──────────────────────────────────────────────
# [UPGRADE 9] ROTOWIRE NEWS SCRAPER
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*5, show_spinner=False)
def fetch_rotowire_news():
    """Scrape Rotowire NBA injury page. Returns (rows, error)."""
    try:
        url = "https://www.rotowire.com/basketball/injury-report.php"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=15)
        if not r.ok:
            return [], f"HTTP {r.status_code}"
        text = r.text
        rows = []
        # Each player row contains: player link, team, position, status, return date, injury detail
        player_blocks = re.findall(
            r'<td[^>]*class="[^"]*player[^"]*"[^>]*>.*?<a[^>]+>([^<]+)</a>.*?</td>'
            r'.*?<td[^>]*>([A-Z]{2,4})</td>'      # team abbr
            r'.*?<td[^>]*>([A-Z]+)</td>'            # position
            r'.*?<td[^>]*>([^<]{2,30})</td>'        # status
            r'.*?<td[^>]*>([^<]{0,30})</td>',       # return date
            text, re.DOTALL
        )
        for m in player_blocks:
            pname = m[0].strip(); team = m[1].strip()
            pos = m[2].strip(); status = m[3].strip(); ret = m[4].strip()
            if pname and status:
                rows.append({"player": pname, "team": team, "pos": pos,
                             "status": status, "return": ret})
        # Fallback: simpler pattern if the above misses things
        if not rows:
            simple = re.findall(
                r'href="/basketball/player-profile[^"]*">([^<]+)</a>.*?'
                r'class="[^"]*status[^"]*"[^>]*>([^<]+)<',
                text, re.DOTALL
            )
            for pname, status in simple[:60]:
                rows.append({"player": pname.strip(), "status": status.strip(),
                             "team": "", "pos": "", "return": ""})
        return rows[:80], None
    except Exception as ex:
        return [], f"{type(ex).__name__}: {ex}"
def build_injury_team_map(injury_dict):
    """Build {team_abbr: [player_name_lower]} for OUT/DOUBTFUL players only — used for auto key_teammate_out."""
    result = {}
    for team, players in (injury_dict or {}).items():
        out_players = [p["player"].lower() for p in players
                       if str(p.get("status","")).upper() in ("OUT","DOUBTFUL")]
        if out_players:
            result[str(team).upper()] = out_players
    return result
# ──────────────────────────────────────────────
# DD / TD PROBABILITY
# ──────────────────────────────────────────────
def compute_dd_prob(game_log_df, n_games=10):
    """Historical frequency of double-doubles from game log."""
    if game_log_df is None or game_log_df.empty:
        return None
    df = game_log_df.head(n_games).copy()
    try:
        dd = sum(
            1 for _, row in df.iterrows()
            if sum(1 for c in ["PTS","REB","AST","BLK","STL"]
                   if safe_float(row.get(c)) >= 10) >= 2
        )
        return float(dd / len(df)) if len(df) > 0 else None
    except Exception:
        return None
def compute_td_prob(game_log_df, n_games=10):
    """Historical frequency of triple-doubles from game log."""
    if game_log_df is None or game_log_df.empty:
        return None
    df = game_log_df.head(n_games).copy()
    try:
        td = sum(
            1 for _, row in df.iterrows()
            if sum(1 for c in ["PTS","REB","AST","BLK","STL"]
                   if safe_float(row.get(c)) >= 10) >= 3
        )
        return float(td / len(df)) if len(df) > 0 else None
    except Exception:
        return None
# ──────────────────────────────────────────────
# PRIZEPICKS INGESTION
# ──────────────────────────────────────────────
PRIZEPICKS_API = "https://api.prizepicks.com/projections"
_PP_NBA_LEAGUE_PREFIXES = ("NBA",)  # matches "NBA", "NBA 1Q", "NBA 1H", "NBA 2H", "NBA_1Q", etc.

def _pp_league_is_nba(league_str: str) -> bool:
    """Return True if the league string is any NBA variant from PrizePicks API."""
    s = re.sub(r"[\s_\-\(\)]+", " ", str(league_str or "").upper()).strip()
    return any(s == p or s.startswith(p + " ") or s.startswith(p + "_") for p in _PP_NBA_LEAGUE_PREFIXES)

def _parse_pp_response(data, league_filter=("NBA", "NBA 1Q", "NBA 1H", "NBA 2H")):
    """Parse PrizePicks JSON response into list of prop dicts.
    Pass league_filter=None to accept all leagues (e.g. when user pastes full-site JSON).
    Uses fuzzy NBA league matching so "NBA (1H)", "NBA_1Q", "NBA2H" etc. all pass.
    """
    _PP_VALID_TYPES = {"projection", "new_player_projection", "boardprojection"}
    included = {item["id"]: item for item in data.get("included", []) if isinstance(item, dict) and "id" in item}
    rows = []
    for proj in data.get("data", []):
        if not isinstance(proj, dict):
            continue
        _type = str(proj.get("type", "")).lower()
        attrs = proj.get("attributes", {}) or {}
        # Accept if type matches OR if it has required fields (future-proof against type renames)
        _has_fields = bool(attrs.get("stat_type") and attrs.get("line_score") is not None)
        if _type not in _PP_VALID_TYPES and not _has_fields:
            continue
        if league_filter:
            league = str(attrs.get("league", "") or "")
            if league and not _pp_league_is_nba(league):
                continue
        rels = proj.get("relationships", {}) or {}
        player_id = (rels.get("new_player", {}).get("data", {}) or {}).get("id")
        if not player_id:
            player_id = (rels.get("player", {}).get("data", {}) or {}).get("id")
        player_attrs = included.get(player_id, {}).get("attributes", {}) if player_id else {}
        player_name = player_attrs.get("name", "") or attrs.get("name", "") or attrs.get("display_name", "")
        stat_type = attrs.get("stat_type", "")
        line_score = attrs.get("line_score")
        # odds_type: "standard" | "goblin" (low-mult) | "demon" (high-mult)
        # rank: 1=goblin, 2=standard, 3=demon (numeric fallback)
        odds_type = str(attrs.get("odds_type", "") or "").lower().strip()
        rank_val   = attrs.get("rank", None)
        if not odds_type and rank_val is not None:
            odds_type = {1: "goblin", 2: "standard", 3: "demon"}.get(int(rank_val), "")
        if player_name and stat_type and line_score is not None:
            try:
                rows.append({
                    "player": player_name,
                    "stat_type": stat_type,
                    "line": float(line_score),
                    "start_time": attrs.get("start_time", ""),
                    "source": "prizepicks",
                    "odds_type": odds_type or "standard",
                })
            except (TypeError, ValueError):
                pass
    return rows
def _parse_pp_response_all(data):
    """Parse PP JSON accepting all leagues — used when user pastes full-site JSON."""
    return _parse_pp_response(data, league_filter=None)
def _pp_request(per_page=500, cookies_str="", single_stat="true"):
    """Make one PrizePicks API request.
    Uses multi-browser curl_cffi impersonation (matching GOLFAPP's proven approach),
    then ScraperAPI proxy, then direct request.
    Returns (response_object_or_None, error_str_or_None).
    single_stat='true'  → standard stats (Points, Rebounds, etc.)
    single_stat='false' → combo/specialty stats (PRA, Pts+Reb, Fantasy Score, etc.)
    """
    url = PRIZEPICKS_API
    params = {"per_page": str(per_page),
              "single_stat": single_stat, "in_play": "false"}
    headers = {
        "Accept": "application/vnd.api+json",
        "Referer": "https://app.prizepicks.com/",
        "Origin": "https://app.prizepicks.com",
    }
    data = None
    resp_obj = None

    # Parse optional cookie string from user settings
    cookie_dict = {}
    if cookies_str and not cookies_str.strip().startswith("{"):
        for part in cookies_str.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip().encode("latin-1", errors="ignore").decode("latin-1")
                v = v.strip().encode("latin-1", errors="ignore").decode("latin-1")
                if k:
                    cookie_dict[k] = v

    # ── Method 1: curl_cffi with multi-browser impersonation (proven in GOLFAPP) ──
    try:
        from curl_cffi import requests as cffi_requests
        for browser in ("edge101", "safari17_0", "chrome124", "chrome120"):
            try:
                r = cffi_requests.get(
                    url, params=params, headers=headers,
                    cookies=cookie_dict or None,
                    impersonate=browser,
                    timeout=25,
                )
                if r.ok:
                    resp_obj = r
                    break
            except Exception:
                continue
        if resp_obj is not None:
            return resp_obj, None
    except ImportError:
        pass

    # ── Method 2: ScraperAPI proxy (bypasses PerimeterX on cloud IPs) ──
    scraper_key = ""
    try:
        scraper_key = st.secrets.get("SCRAPER_API_KEY", "") or os.environ.get("SCRAPER_API_KEY", "")
    except Exception:
        scraper_key = os.environ.get("SCRAPER_API_KEY", "")
    if scraper_key:
        try:
            full_url = f"{url}?per_page={per_page}&single_stat={single_stat}&in_play=false"
            r = requests.get(
                "https://api.scraperapi.com",
                params={"api_key": scraper_key, "url": full_url, "render": "false"},
                timeout=30,
            )
            if r.ok:
                return r, None
        except Exception:
            pass

    # ── Method 3: cloudscraper (handles Cloudflare JS challenges) ──
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        r = scraper.get(url, params=params, headers={
            **headers,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }, cookies=cookie_dict or None, timeout=25)
        if r.ok:
            return r, None
    except ImportError:
        pass
    except Exception:
        pass

    # ── Method 4: Direct request (works from residential IPs) ──
    try:
        r = requests.get(url, params=params, headers={
            **headers,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }, cookies=cookie_dict or None, timeout=20)
        if r.ok:
            return r, None
        return None, f"HTTP {r.status_code}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
def _pp_fetch_one(per_page, cookies_str, single_stat):
    """Fetch one PP request with retry. Returns (rows, error)."""
    for attempt in range(3):
        r, err = _pp_request(per_page=per_page, cookies_str=cookies_str, single_stat=single_stat)
        if err:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            return [], err
        if r is None:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            return [], "No response from PrizePicks"
        try:
            rows = _parse_pp_response(r.json())
        except Exception as e:
            return [], f"Parse error: {e}"
        return rows, None
    return [], None
@st.cache_data(ttl=60*10, show_spinner=False)
def _fetch_prizepicks_lines_cached(cookies_str=""):
    """Fetch both single-stat AND combo/specialty markets and merge."""
    all_rows = []
    seen = set()
    last_err = None
    for single_stat in ("true", "false"):  # true=standard, false=combo/specialty
        for per_page in (500, 250):
            rows, err = _pp_fetch_one(per_page, cookies_str, single_stat)
            if err:
                last_err = err
                break  # try next single_stat value
            for row in rows:
                key = (row["player"], row["stat_type"])
                if key not in seen:
                    seen.add(key)
                    all_rows.append(row)
            if rows:
                break  # got results for this single_stat, no need to try smaller per_page
    if all_rows:
        return all_rows, None
    return [], last_err or "No NBA props found — slate may not be posted yet"
def fetch_prizepicks_lines():
    cookies_str = st.session_state.get("pp_cookies", "")
    # ── 0. Check relay URL first (local relay script or ngrok tunnel) ──
    relay_url = st.session_state.get("pp_relay_url", "").strip()
    if relay_url:
        try:
            r = requests.get(relay_url, timeout=10)
            if r.ok:
                data = r.json()
                rows = data if isinstance(data, list) else data.get("rows", [])
                if rows:
                    return rows, None
        except Exception:
            pass  # relay unavailable — fall through to direct API
    # ── 1. Check background auto-fetcher state (recent in-memory result) ──
    _auto_rows, _auto_age, _ = get_pp_auto_lines()
    if _auto_rows and _auto_age is not None and _auto_age < 900:
        return _auto_rows, None
    # ── 2. Detect if user pasted a full PP JSON response into the cookies/JSON field ──
    _stripped = cookies_str.strip() if cookies_str else ""
    if _stripped.startswith("{") and '"data"' in _stripped:
        try:
            data = json.loads(_stripped)
            rows = _parse_pp_response_all(data)
            if rows:
                return rows, None
        except Exception as _je:
            return [], f"Stored JSON parse error: {_je}"
    # ── 3. Direct API call with cache ──
    if st.session_state.get("_pp_last_cookies_used") != cookies_str:
        _fetch_prizepicks_lines_cached.clear()
        st.session_state["_pp_last_cookies_used"] = cookies_str
    direct_rows, direct_err = _fetch_prizepicks_lines_cached(cookies_str=cookies_str)
    if direct_rows:
        return direct_rows, None
    # ── 4. Scraper service DB fallback (no cookies needed) ──
    db_rows, db_age, _ = _load_pp_from_scraper_db()
    if db_rows:
        _age_str = f" ({db_age // 60}m ago)" if db_age else ""
        return db_rows, None
    # ── 5. Disk cache fallback ──
    disk_rows, disk_age = _load_pp_disk_cache(max_age_sec=3600)
    if disk_rows:
        return disk_rows, None
    return [], direct_err
# ──────────────────────────────────────────────
# PP AUTO-FETCH BACKGROUND ENGINE
# Runs a daemon thread that fetches PP lines on a configurable interval.
# Works from a local/residential IP; for Streamlit Cloud use pp_relay.py + ngrok.
# ──────────────────────────────────────────────
_PP_DISK_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pp_lines_cache.json")
def _save_pp_disk_cache(rows: list):
    try:
        with open(_PP_DISK_CACHE, "w") as f:
            json.dump({"ts": time.time(), "rows": rows}, f)
    except Exception:
        pass
def _load_pp_disk_cache(max_age_sec: int = 1800):
    try:
        with open(_PP_DISK_CACHE) as f:
            d = json.load(f)
        age = int(time.time() - d.get("ts", 0))
        if age < max_age_sec:
            return d.get("rows", []), age
    except Exception:
        pass
    return None, None
# [AUDIT FIX] Module-level singleton for PP background fetcher state.
# Previously _pp_auto_state() returned a new dict every call, breaking singleton semantics:
# thread tracking, lock sharing, and stop_evt coordination all failed silently,
# causing unbounded thread spawning and no actual shared state between reruns.
_PP_AUTO_STATE: dict = {
    "rows": [],
    "ts": 0.0,
    "err": None,
    "cookies": "",
    "relay_url": "",
    "enabled": False,
    "interval": 600,
    "lock": threading.Lock(),
    "thread": None,
    "stop_evt": threading.Event(),
}

def _pp_auto_state() -> dict:
    """Return the module-level singleton state dict for the PP background fetcher."""
    return _PP_AUTO_STATE
def _pp_auto_loop(state: dict):
    """Background daemon: fetch PP lines every `interval` seconds."""
    while True:
        state["stop_evt"].wait(state.get("interval", 600))
        state["stop_evt"].clear()
        if not state.get("enabled"):
            continue
        try:
            relay_url = state.get("relay_url", "").strip()
            rows: list = []
            if relay_url:
                try:
                    r = requests.get(relay_url, timeout=10)
                    if r.ok:
                        data = r.json()
                        rows = data if isinstance(data, list) else data.get("rows", [])
                except Exception:
                    pass
            if not rows:
                cookies_str = state.get("cookies", "")
                rows_s, err_s = _pp_fetch_one(500, cookies_str, "true")
                rows_c, _     = _pp_fetch_one(500, cookies_str, "false")
                rows = (rows_s or []) + (rows_c or [])
                err = err_s if not rows else None
            else:
                err = None
            with state["lock"]:
                if rows:
                    seen: set = set()
                    deduped = []
                    for row in rows:
                        k = (row.get("player"), row.get("stat_type"))
                        if k not in seen:
                            seen.add(k)
                            deduped.append(row)
                    state["rows"] = deduped
                    state["ts"] = time.time()
                    state["err"] = None
                    _save_pp_disk_cache(deduped)
                else:
                    state["err"] = err
        except Exception as e:
            with state["lock"]:
                state["err"] = str(e)
def _ensure_pp_auto_thread():
    state = _pp_auto_state()
    if state.get("thread") and state["thread"].is_alive():
        return
    t = threading.Thread(target=_pp_auto_loop, args=(state,), daemon=True, name="pp_auto_fetcher")
    state["thread"] = t
    t.start()
def set_pp_auto_fetch(enabled: bool, interval_sec: int = 600, cookies: str = "", relay_url: str = ""):
    """Enable/disable the background fetcher and trigger an immediate fetch."""
    state = _pp_auto_state()
    with state["lock"]:
        state["enabled"] = enabled
        state["interval"] = max(60, interval_sec)
        state["cookies"] = cookies
        state["relay_url"] = relay_url
    if enabled:
        _ensure_pp_auto_thread()
        state["stop_evt"].set()  # poke thread to fetch immediately
def _load_pp_from_scraper_db():
    """Load latest lines from the PrizePicks scraper service SQLite database."""
    try:
        _db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nba_prizepicks.db")
        if not os.path.exists(_db_path):
            return None, None, None
        import sqlite3
        conn = sqlite3.connect(_db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT player_name, stat_type, line_score, start_time, odds_type "
                    "FROM nba_prizepicks_lines WHERE is_latest = 1")
        db_rows = cur.fetchall()
        if not db_rows:
            conn.close()
            return None, None, None
        # Get last pull time
        cur.execute("SELECT last_success FROM scraper_status WHERE scraper_name='nba_prizepicks'")
        status_row = cur.fetchone()
        last_success = status_row["last_success"] if status_row else None
        conn.close()
        rows = []
        for r in db_rows:
            rows.append({
                "player": r["player_name"],
                "stat_type": r["stat_type"],
                "line": r["line_score"],
                "start_time": r["start_time"] or "",
                "source": "prizepicks",
                "odds_type": r["odds_type"] or "standard",
            })
        age = None
        if last_success:
            from datetime import datetime as _dt
            try:
                _ts = _dt.strptime(last_success, "%Y-%m-%d %H:%M:%S.%f")
                age = int(time.time() - _ts.timestamp())
            except Exception:
                try:
                    _ts = _dt.strptime(last_success, "%Y-%m-%d %H:%M:%S")
                    age = int(time.time() - _ts.timestamp())
                except Exception:
                    pass
        return rows, age, last_success
    except Exception:
        return None, None, None

def get_pp_auto_lines():
    """Return (rows, age_sec, err) from background state, disk cache, or scraper DB."""
    state = _pp_auto_state()
    with state["lock"]:
        rows = list(state.get("rows", []))
        ts   = state.get("ts", 0.0)
        err  = state.get("err")
    if rows:
        return rows, int(time.time() - ts), err
    disk_rows, disk_age = _load_pp_disk_cache(max_age_sec=3600)
    if disk_rows:
        return disk_rows, disk_age, err
    # Fallback: check the scraper service's SQLite database
    db_rows, db_age, _ = _load_pp_from_scraper_db()
    if db_rows:
        return db_rows, db_age, err
    return [], None, err
# ──────────────────────────────────────────────
# UNDERDOG INGESTION
# ──────────────────────────────────────────────
# Try v3 then v4 endpoint
_UNDERDOG_ENDPOINTS = [
    "https://api.underdogfantasy.com/v3/over_under_lines",
    "https://api.underdogfantasy.com/v4/over_under_lines",
    "https://api.underdogfantasy.com/v2/over_under_lines",
]
_UD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://underdogfantasy.com/",
    "Origin": "https://underdogfantasy.com",
}
# Basketball sport IDs that Underdog may use (numeric or string)
_UD_BASKETBALL_SPORT_IDS = {"nba", "basketball", "5", "4", "nba_basketball", "basketball_nba", ""}
@st.cache_data(ttl=60*60*2, show_spinner=False)
def _fetch_underdog_lines_cached(cookies_str=""):
    """Inner cached fetch — call via fetch_underdog_lines() which clears cache first."""
    last_err = "No Underdog props found — slate may not be posted yet"
    # Parse optional cookie string
    cookie_dict = {}
    if cookies_str:
        for part in cookies_str.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                cookie_dict[k.strip()] = v.strip()
    for url in _UNDERDOG_ENDPOINTS:
        try:
            # Try curl_cffi first (Chrome impersonation bypasses some bot blocks)
            try:
                from curl_cffi import requests as cffi_requests
                r = cffi_requests.get(url, headers=_UD_HEADERS,
                                      cookies=cookie_dict or None,
                                      impersonate="chrome120", timeout=20)
            except ImportError:
                r = requests.get(url, headers=_UD_HEADERS,
                                 cookies=cookie_dict or None, timeout=20)
            if r.status_code == 429:
                return [], "Underdog rate-limited (429) — try again in 30s"
            if r.status_code == 403:
                last_err = "Underdog HTTP 403 — API blocked on this server. Paste your Underdog browser cookies above or use Manual Import to paste JSON."
                continue
            if not r.ok:
                last_err = f"Underdog HTTP {r.status_code} from {url}"
                continue
            data = r.json()
            # Build lookup maps — handle both v3 and v4 response shapes
            appearances = {}
            for a in data.get("appearances", []):
                appearances[str(a.get("id", ""))] = a
            players_map = {}
            for p in data.get("players", []):
                players_map[str(p.get("id", ""))] = p
            rows = []
            # v3/v4: over_under_lines list
            lines_list = data.get("over_under_lines", data.get("lines", []))
            for line in lines_list:
                try:
                    ou = line.get("over_under", line)  # v3 nests under "over_under"
                    app_stat = ou.get("appearance_stat", {})
                    app_id = str(app_stat.get("appearance_id", ""))
                    app = appearances.get(app_id, {})
                    # Sport filter: allow NBA/basketball, or empty/unknown (pass-through)
                    sport = str(app.get("sport_id", app.get("sport", ""))).lower().strip()
                    if sport and sport not in _UD_BASKETBALL_SPORT_IDS:
                        continue
                    player_id = str(app.get("player_id", ""))
                    player = players_map.get(player_id, {})
                    player_name = (
                        f"{player.get('first_name','')} {player.get('last_name','')}".strip()
                        or player.get("name", "")
                    )
                    stat_type = app_stat.get("display_stat", app_stat.get("stat", ""))
                    stat_value = line.get("stat_value", ou.get("stat_value"))
                    if player_name and stat_type and stat_value is not None:
                        rows.append({
                            "player": player_name,
                            "stat_type": stat_type,
                            "line": float(stat_value),
                            "source": "underdog",
                        })
                except Exception:
                    continue
            if rows:
                return rows, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return [], last_err
def fetch_underdog_lines():
    """Clear cache then fetch fresh Underdog lines using any saved cookies."""
    cookies_str = st.session_state.get("ud_cookies", "")
    if st.session_state.get("_ud_last_cookies_used") != cookies_str:
        _fetch_underdog_lines_cached.clear()
        st.session_state["_ud_last_cookies_used"] = cookies_str
    return _fetch_underdog_lines_cached(cookies_str=cookies_str)
def map_platform_stat_to_market(stat_type):
    """Map PrizePicks/Underdog stat label to internal market name.
    Handles all known PP API stat_type strings including combo, specialty,
    half/quarter, shooting volume, and fantasy markets.
    """
    mapping = {
        # ── Standard ──────────────────────────────────────────────────────
        "Points": "Points", "Pts": "Points", "PTS": "Points",
        "Rebounds": "Rebounds", "Reb": "Rebounds", "REB": "Rebounds",
        "Total Rebounds": "Rebounds", "Rebs": "Rebounds",
        "Assists": "Assists", "Ast": "Assists", "AST": "Assists",
        "Assts": "Assists",
        # 3PM — all PrizePicks API variants
        "3-Pointers Made": "3PM", "3 Pointers Made": "3PM", "3PM": "3PM",
        "3-PT Made": "3PM", "3PT Made": "3PM", "3-Pt Made": "3PM",
        "3 PT Made": "3PM", "Three Pointers Made": "3PM",
        "3-Point Field Goals Made": "3PM",
        # ── Combo ─────────────────────────────────────────────────────────
        "Pts+Reb+Ast": "PRA", "Points+Rebounds+Assists": "PRA", "PRA": "PRA",
        "Pts + Reb + Ast": "PRA", "Points + Rebounds + Assists": "PRA",
        "Pts+Reb+Ast (PRA)": "PRA",
        "Pts+Reb": "PR", "Points+Rebounds": "PR", "PR": "PR",
        "Pts + Reb": "PR", "Points + Rebounds": "PR",
        "Pts+Ast": "PA", "Points+Assists": "PA", "PA": "PA",
        "Pts + Ast": "PA", "Points + Assists": "PA",
        "Reb+Ast": "RA", "Rebounds+Assists": "RA", "RA": "RA",
        "Reb + Ast": "RA", "Rebounds + Assists": "RA",
        # ── Defense ───────────────────────────────────────────────────────
        "Blocked Shots": "Blocks", "Blocks": "Blocks", "Blk": "Blocks",
        "BLK": "Blocks", "Blks": "Blocks", "Block": "Blocks",
        "Blocked": "Blocks",
        "Steals": "Steals", "Stl": "Steals", "STL": "Steals",
        "Steal": "Steals", "Stls": "Steals",
        "Turnovers": "Turnovers", "Tov": "Turnovers", "TOV": "Turnovers",
        "Turnover": "Turnovers", "TOs": "Turnovers",
        # Stocks — all variants
        "Blks+Stls": "Stocks", "Stocks": "Stocks", "Blk+Stl": "Stocks",
        "Blocks+Steals": "Stocks", "Blks + Stls": "Stocks",
        "Blocks + Steals": "Stocks", "Stls+Blks": "Stocks",
        "Steals+Blocks": "Stocks", "Stl+Blk": "Stocks",
        # ── Shooting volume (PP specialty) ────────────────────────────────
        "Field Goals Made": "FGM", "FGM": "FGM", "FG Made": "FGM",
        "Field Goals Attempted": "FGA", "FGA": "FGA", "FG Attempted": "FGA",
        "FG Attempts": "FGA",
        "3-Pt Attempts": "3PA", "3PA": "3PA", "3-Point Attempts": "3PA",
        "3 Pt Attempts": "3PA", "3PT Attempts": "3PA",
        "Three Point Attempts": "3PA", "3-Point Field Goals Attempted": "3PA",
        "Free Throws Made": "FTM", "FTM": "FTM", "FT Made": "FTM",
        "Free Throws Attempted": "FTA", "FTA": "FTA", "FT Attempted": "FTA",
        "FT Attempts": "FTA",
        # ── Fantasy / DFS ─────────────────────────────────────────────────
        "Fantasy Score": "Fantasy Score", "Fantasy Points": "Fantasy Score",
        "DFS Points": "Fantasy Score", "FP": "Fantasy Score",
        # ── Binary / special ──────────────────────────────────────────────
        "Double Double": "Double Double", "Double-Double": "Double Double",
        "Dbl Dbl": "Double Double", "DD": "Double Double",
        "Triple Double": "Triple Double", "Triple-Double": "Triple Double",
        "Trip Dbl": "Triple Double", "TD": "Triple Double",
        # ── 1st Half ──────────────────────────────────────────────────────
        "H1 Points": "H1 Points", "1H Points": "H1 Points",
        "1st Half Points": "H1 Points", "First Half Points": "H1 Points",
        "H1 Pts": "H1 Points", "1H Pts": "H1 Points",
        "H1 Rebounds": "H1 Rebounds", "1H Rebounds": "H1 Rebounds",
        "1st Half Rebounds": "H1 Rebounds", "H1 Reb": "H1 Rebounds",
        "H1 Assists": "H1 Assists", "1H Assists": "H1 Assists",
        "1st Half Assists": "H1 Assists", "H1 Ast": "H1 Assists",
        "H1 3PM": "H1 3PM", "1H 3PM": "H1 3PM",
        "1st Half 3-Pointers Made": "H1 3PM", "H1 3-Pointers Made": "H1 3PM",
        "H1 PRA": "H1 PRA", "1H PRA": "H1 PRA",
        "1st Half PRA": "H1 PRA", "H1 Pts+Reb+Ast": "H1 PRA",
        # ── 2nd Half ──────────────────────────────────────────────────────
        "H2 Points": "H2 Points", "2H Points": "H2 Points",
        "2nd Half Points": "H2 Points", "Second Half Points": "H2 Points",
        "H2 Pts": "H2 Points", "2H Pts": "H2 Points",
        "H2 Rebounds": "H2 Rebounds", "2H Rebounds": "H2 Rebounds",
        "2nd Half Rebounds": "H2 Rebounds", "H2 Reb": "H2 Rebounds",
        "H2 Assists": "H2 Assists", "2H Assists": "H2 Assists",
        "2nd Half Assists": "H2 Assists", "H2 Ast": "H2 Assists",
        "H2 PRA": "H2 PRA", "2H PRA": "H2 PRA",
        "2nd Half PRA": "H2 PRA", "H2 Pts+Reb+Ast": "H2 PRA",
        # ── 1st Quarter ───────────────────────────────────────────────────
        "Q1 Points": "Q1 Points", "1Q Points": "Q1 Points",
        "1st Quarter Points": "Q1 Points", "Q1 Pts": "Q1 Points",
        "Q1 Rebounds": "Q1 Rebounds", "1Q Rebounds": "Q1 Rebounds",
        "1st Quarter Rebounds": "Q1 Rebounds", "Q1 Reb": "Q1 Rebounds",
        "Q1 Assists": "Q1 Assists", "1Q Assists": "Q1 Assists",
        "1st Quarter Assists": "Q1 Assists", "Q1 Ast": "Q1 Assists",
    }
    s = str(stat_type).strip()
    s_lower = s.lower()
    for k, v in mapping.items():
        if k.lower() == s_lower:
            return v
    # Fuzzy fallback: normalize whitespace/separators and retry
    s_norm = re.sub(r"[\s\-\+]+", " ", s_lower).strip()
    for k, v in mapping.items():
        k_norm = re.sub(r"[\s\-\+]+", " ", k.lower()).strip()
        if k_norm == s_norm:
            return v
    return None
# ──────────────────────────────────────────────
# LINE SHOPPING — BEST AVAILABLE PRICE
# ──────────────────────────────────────────────
def get_best_available_price(event_id, market_key, player_norm, side):
    """Return (best_price, best_book) across all books for a given player/market/side."""
    try:
        odds, err = odds_get_event_odds(str(event_id), (str(market_key),))
        if err or not odds:
            return None, None
        best_price, best_book = None, None
        for b in odds.get("bookmakers", []) or []:
            bkey = b.get("key", "")
            for mk in b.get("markets", []) or []:
                if mk.get("key") != market_key:
                    continue
                for out in mk.get("outcomes", []) or []:
                    pn = normalize_name(out.get("description") or out.get("name") or "")
                    if pn == player_norm and (out.get("name") or "").lower() == str(side).lower():
                        p = safe_float(out.get("price"))
                        if p > 1.0 and (best_price is None or p > best_price):
                            best_price = p
                            best_book = bkey
        return best_price, best_book
    except Exception:
        return None, None
# ──────────────────────────────────────────────
# PROP LINE HISTORY DATABASE
# ──────────────────────────────────────────────
PROP_HISTORY_PATH = "prop_line_history.jsonl"
def save_prop_line(player, market, line, price, book, event_id=None):
    """Append a prop line snapshot to JSONL history file."""
    try:
        with open(PROP_HISTORY_PATH, "a") as f:
            f.write(json.dumps({
                "ts": _now_iso(), "player": player, "market": market,
                "line": float(line) if line is not None else None,
                "price": float(price) if price is not None else None,
                "book": book, "event_id": event_id,
            }) + "\n")
    except Exception:
        pass
# ──────────────────────────────────────────────
# [UPGRADE 10] OPENING LINE CAPTURE
# ──────────────────────────────────────────────
def save_opening_line(player_norm, market_key, side, line, price):
    """Persist the first-seen line for a player/market/side as the 'opening' line."""
    try:
        data = {}
        if os.path.exists(OPENING_LINES_PATH):
            with open(OPENING_LINES_PATH) as f:
                data = json.load(f)
        # Include date in key so each day gets its own independent opening line
        today = date.today().isoformat()
        key = f"{player_norm}|{market_key}|{side}|{today}"
        if key not in data:  # Only write once per key (true opening)
            data[key] = {"line": float(line), "price": price, "ts": _now_iso(), "date": today}
            with open(OPENING_LINES_PATH, "w") as f:
                json.dump(data, f)
    except Exception as ex:
        logging.warning(f"save_opening_line: {ex}")
def get_opening_line(player_norm, market_key, side):
    """Return (opening_line, opening_price) or (None, None) if not recorded."""
    try:
        if not os.path.exists(OPENING_LINES_PATH):
            return None, None
        with open(OPENING_LINES_PATH) as f:
            data = json.load(f)
        today = date.today().isoformat()
        key = f"{player_norm}|{market_key}|{side}|{today}"
        rec = data.get(key)
        if rec:
            return rec.get("line"), rec.get("price")
    except Exception as ex:
        logging.warning(f"get_opening_line: {ex}")
    return None, None
def clear_opening_lines():
    try:
        if os.path.exists(OPENING_LINES_PATH):
            os.remove(OPENING_LINES_PATH)
    except Exception:
        pass
# ──────────────────────────────────────────────
# WATCHLIST PERSISTENCE
# ──────────────────────────────────────────────
def load_watchlist(uid):
    path = WATCHLIST_PATH_TPL.format(uid=re.sub(r"[^a-zA-Z0-9_-]", "_", uid or "default"))
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f) or []
    except Exception:
        pass
    return []
def save_watchlist(uid, players):
    path = WATCHLIST_PATH_TPL.format(uid=re.sub(r"[^a-zA-Z0-9_-]", "_", uid or "default"))
    try:
        with open(path, "w") as f:
            json.dump(list(players), f)
    except Exception:
        pass
def load_prop_line_history(player=None, market=None, limit=500):
    """Load prop line history filtered by player/market."""
    try:
        if not os.path.exists(PROP_HISTORY_PATH):
            return pd.DataFrame()
        rows = []
        with open(PROP_HISTORY_PATH) as f:
            for raw in f:
                try:
                    r = json.loads(raw.strip())
                    if player and normalize_name(r.get("player","")) != normalize_name(player):
                        continue
                    if market and r.get("market","").lower() != market.lower():
                        continue
                    rows.append(r)
                except Exception:
                    continue
        return pd.DataFrame(rows[-limit:]) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
# ──────────────────────────────────────────────
# DISCORD / TELEGRAM ALERTS
# ──────────────────────────────────────────────
def send_discord_alert(webhook_url, message):
    if not webhook_url:
        return False, "No webhook URL"
    try:
        r = requests.post(webhook_url, json={"content": message, "username": "NBA Quant Engine"}, timeout=10)
        r.raise_for_status()
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return False, "Token/chat_id missing"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
def format_edge_alert(leg):
    p_cal = leg.get("p_cal") or 0
    ev = (leg.get("ev_adj_pct") or (leg.get("ev_adj", 0) * 100 if leg.get("ev_adj") else 0))
    proj = leg.get("proj")
    return (
        f"**{leg.get('player','?')}** — {leg.get('market','?')} O{leg.get('line','?')}\n"
        f"Proj: {proj:.1f} | P: {p_cal*100:.1f}% | EV: {ev:.1f}%\n"
        f"Book: {leg.get('book','?')} | {leg.get('edge_cat','')}"
    ) if proj else (
        f"**{leg.get('player','?')}** — {leg.get('market','?')} O{leg.get('line','?')}\n"
        f"P: {p_cal*100:.1f}% | EV: {ev:.1f}% | {leg.get('edge_cat','')}"
    )
# ──────────────────────────────────────────────
# [UPGRADE 23] ALERT DIGEST FORMATTER
# ──────────────────────────────────────────────
def format_digest_message(edges, as_of=None):
    """Format a ranked daily digest of top edges for Discord/Telegram."""
    ts = (as_of or datetime.utcnow()).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"**NBA QUANT ENGINE — DAILY DIGEST** ({ts})\n"]
    for i, leg in enumerate(edges[:15], 1):
        ev = leg.get("ev_adj_pct") or (float(leg.get("ev_adj", 0) or 0) * 100)
        p  = float(leg.get("p_cal") or 0) * 100
        proj = leg.get("proj")
        proj_str = f"{proj:.1f}" if proj is not None else "--"
        lines.append(
            f"{i}. **{leg.get('player','?')}** {leg.get('market','?')} "
            f"O{leg.get('line','?')} | Proj: {proj_str} | "
            f"P: {p:.0f}% | EV: {ev:+.1f}% | {leg.get('edge_cat','')}"
        )
    return "\n".join(lines)
# ──────────────────────────────────────────────
# [UPGRADE 31] CLV LEADERBOARD
# ──────────────────────────────────────────────
def compute_clv_leaderboard(history_df, top_n=20):
    """Return top bets ranked by no-vig CLV price improvement."""
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in history_df.iterrows():
        try:
            legs = json.loads(r.get("legs", "[]")) if isinstance(r.get("legs"), str) else []
        except Exception:
            legs = []
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            clv_p = leg.get("clv_price")
            clv_l = leg.get("clv_line")
            if clv_p is None and clv_l is None:
                continue
            rows.append({
                "ts":     r.get("ts", ""),
                "player": leg.get("player", "?"),
                "market": leg.get("market", "?"),
                "line":   leg.get("line"),
                "side":   leg.get("side", "Over"),
                "result": r.get("result", "Pending"),
                "clv_line":  safe_round(clv_l, 2),
                "clv_price": safe_round(clv_p, 4),
                "clv_line_fav":  bool(leg.get("clv_line_fav")),
                "clv_price_fav": bool(leg.get("clv_price_fav")),
                "novig_open":  safe_round(leg.get("clv_price_novig_open"), 4),
                "novig_close": safe_round(leg.get("clv_price_novig_close"), 4),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("clv_price", ascending=False).head(top_n)
    return df.reset_index(drop=True)
# ──────────────────────────────────────────────
# [UPGRADE 32] PER-BOOK MARKET EFFICIENCY SCORE
# ──────────────────────────────────────────────
def compute_book_efficiency(history_df):
    """Track per-book win rate and CLV rate to rank market efficiency."""
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in history_df.iterrows():
        res = r.get("result", "Pending")
        if res not in ("HIT", "MISS"):
            continue
        y = 1 if res == "HIT" else 0
        try:
            legs = json.loads(r.get("legs", "[]")) if isinstance(r.get("legs"), str) else []
        except Exception:
            legs = []
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            book = leg.get("book") or "unknown"
            rows.append({
                "book":    book,
                "y":       y,
                "clv_fav": int(bool(leg.get("clv_price_fav"))),
                "ev_adj":  safe_float(leg.get("ev_adj"), 0.0),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    result = df.groupby("book").agg(
        bets=("y", "size"),
        hit_rate=("y", "mean"),
        clv_fav_rate=("clv_fav", "mean"),
        avg_ev=("ev_adj", "mean"),
    ).reset_index()
    result["hit_rate_%"] = (result["hit_rate"] * 100).round(1)
    result["clv_fav_%"]  = (result["clv_fav_rate"] * 100).round(1)
    result["avg_ev_%"]   = (result["avg_ev"] * 100).round(2)
    result = result[result["bets"] >= 3].sort_values("hit_rate_%", ascending=False)
    return result[["book", "bets", "hit_rate_%", "clv_fav_%", "avg_ev_%"]].reset_index(drop=True)
# ──────────────────────────────────────────────
# [v6.0] MONTE CARLO BACKTEST ENGINE
# Simulates N bets using the quant engine's edge detection and Kelly sizing
# to estimate ROI, Sharpe ratio, max drawdown, and win rate.
# ──────────────────────────────────────────────
def run_monte_carlo_backtest(n_bets=1000, bankroll_start=1000.0,
                              min_ev=0.05, min_sharpness=50,
                              frac_kelly=0.25, max_cap=0.05,
                              edge_mean=0.08, edge_std=0.03,
                              win_prob_mean=0.58, win_prob_std=0.06,
                              price_decimal=1.909, seed=42):
    """
    Monte Carlo backtest: simulates betting with the v6.0 quant engine parameters.
    Uses realistic edge distributions derived from historical NBA prop market data.
    Returns dict with: roi, sharpe, max_drawdown, win_rate, profit, bets_placed,
                       bankroll_curve, per_bet_results
    """
    rng = np.random.default_rng(seed)
    bankroll = float(bankroll_start)
    total_staked = 0.0
    total_profit = 0.0
    wins = 0
    losses = 0
    bets_placed = 0
    bankroll_curve = [bankroll]
    per_bet = []
    max_bankroll = bankroll
    max_drawdown = 0.0
    for i in range(n_bets * 3):
        if bets_placed >= n_bets or bankroll <= 10:
            break
        _ev = float(rng.normal(edge_mean, edge_std))
        _prob = float(np.clip(rng.normal(win_prob_mean, win_prob_std), 0.40, 0.85))
        _cv = float(rng.uniform(0.08, 0.35))
        _ev_pts = float(np.clip(math.sqrt(max(0, _ev) / 0.15) * 35.0, 0.0, 35.0))
        _adv_pts = float(np.clip((_prob - 0.524) / 0.15 * 20.0, -10.0, 20.0))
        _cv_pts = float(np.clip(-(_cv - 0.15) / 0.25 * 10.0, -10.0, 0.0))
        _bonus = float(rng.normal(10, 8))
        _sharpness = float(np.clip(_ev_pts + _adv_pts + _cv_pts + _bonus, 0, 100))
        if _ev < min_ev:
            continue
        if _sharpness < min_sharpness:
            continue
        if _cv > 0.38:
            continue
        if _cv > 0.32 and _ev < 0.15:
            continue
        if _cv > 0.25 and _ev < 0.08:
            continue
        b = price_decimal - 1.0
        q = 1.0 - _prob
        k_full = max(0, (b * _prob - q) / b)
        if k_full <= 0:
            continue
        if _sharpness >= 75:
            _mult = 1.60
        elif _sharpness >= 62:
            _mult = 1.15
        elif _sharpness >= 50:
            _mult = 0.60
        else:
            _mult = 0.0
        if _mult == 0:
            continue
        f = min(frac_kelly * _mult * k_full, max_cap)
        stake = bankroll * f
        if stake < 1.0:
            continue
        won = rng.random() < _prob
        if won:
            profit = stake * (price_decimal - 1.0)
            wins += 1
        else:
            profit = -stake
            losses += 1
        bankroll += profit
        total_staked += stake
        total_profit += profit
        bets_placed += 1
        max_bankroll = max(max_bankroll, bankroll)
        _dd = (max_bankroll - bankroll) / max_bankroll if max_bankroll > 0 else 0
        max_drawdown = max(max_drawdown, _dd)
        bankroll_curve.append(bankroll)
        per_bet.append({
            "bet_num": bets_placed, "ev": round(_ev, 4), "prob": round(_prob, 4),
            "sharpness": round(_sharpness, 1), "stake": round(stake, 2),
            "won": won, "profit": round(profit, 2), "bankroll": round(bankroll, 2),
        })
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0.0
    if per_bet:
        returns = [b["profit"] / b["stake"] if b["stake"] > 0 else 0 for b in per_bet]
        _mean_r = float(np.mean(returns))
        _std_r = float(np.std(returns)) if len(returns) > 1 else 1.0
        sharpe = (_mean_r / _std_r * math.sqrt(900)) if _std_r > 0 else 0.0
    else:
        sharpe = 0.0
    return {
        "roi_pct": round(roi, 2), "total_profit": round(total_profit, 2),
        "total_staked": round(total_staked, 2), "bets_placed": bets_placed,
        "bets_filtered": i + 1 - bets_placed,
        "win_rate_pct": round(win_rate, 2), "wins": wins, "losses": losses,
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "final_bankroll": round(bankroll, 2),
        "bankroll_curve": bankroll_curve,
    }
# ──────────────────────────────────────────────
# [UPGRADE 34] BAYESIAN PRIOR UPDATE FROM HISTORY
# ──────────────────────────────────────────────
def compute_history_based_priors(legs_df, position_bucket):
    """Blend positional priors with personal hit/miss data by market."""
    base = dict(POSITIONAL_PRIORS.get(position_bucket, POSITIONAL_PRIORS["Unknown"]))
    if legs_df is None or legs_df.empty:
        return base
    settled = legs_df[legs_df["y"].notna() & legs_df["market"].notna()].copy()
    if len(settled) < 20:
        return base
    mkt_stats = settled.groupby("market").agg(
        hit_rate=("y", "mean"), n=("y", "size")
    ).reset_index()
    for _, row in mkt_stats.iterrows():
        mkt = row["market"]; n = row["n"]
        if n < 8 or mkt not in base:
            continue
        # Bayesian blend: personal history weight grows with sample size (max 35%)
        w_personal = min(n / (n + 30), 0.35)
        # Positive hit rate → market is offering value → effectively lower the prior (easier line)
        adj = 1.0 + w_personal * (row["hit_rate"] - 0.50) * 0.25
        base[mkt] = base[mkt] * float(np.clip(adj, 0.88, 1.12))
    return base
# ──────────────────────────────────────────────
# KELLY PARLAY OPTIMIZER
# ──────────────────────────────────────────────
def kelly_parlay_optimizer(legs, payout_mult, max_legs=4, bankroll=1000.0, frac_kelly=0.25):
    """Find best 2-N leg combos using PSD covariance matrix + Gaussian copula MC simulation."""
    from itertools import combinations
    import scipy.stats as _sc
    N_SIMS_PARLAY = 3000
    valid = [l for l in legs if l.get("gate_ok") and float(l.get("p_cal") or 0) > 0.50]
    if len(valid) < 2:
        return []
    # Pre-build full N×N correlation matrix once (avoid redundant pairwise calls in inner loop)
    nv = len(valid)
    full_corr = np.eye(nv)
    for _i in range(nv):
        for _j in range(_i + 1, nv):
            c = float(estimate_player_correlation(valid[_i], valid[_j]) or 0.0)
            full_corr[_i, _j] = full_corr[_j, _i] = c
    # H-5 audit fix: derive seed from leg identities so different combos get different draws.
    # Hard-coded 42 caused identical noise patterns across all parlay simulations.
    _seed = hash(tuple(sorted(l.get("player","") + l.get("market","") for l in valid))) % (2**31)
    rng = np.random.default_rng(_seed)
    results = []
    for n in range(2, min(max_legs + 1, nv + 1)):
        for combo in combinations(range(nv), n):
            combo_legs = [valid[i] for i in combo]
            probs = np.array([float(l["p_cal"]) for l in combo_legs])
            naive_joint = float(np.prod(probs))
            # Extract n×n sub-matrix and make PSD via eigenvalue clipping
            sub_corr = full_corr[np.ix_(list(combo), list(combo))]
            evals, evecs = np.linalg.eigh(sub_corr)
            evals = np.clip(evals, 1e-6, None)
            corr_psd = evecs @ np.diag(evals) @ evecs.T
            # Renormalize diagonal to 1.0 (clipping can perturb it) and enforce symmetry
            _d = np.sqrt(np.maximum(np.diag(corr_psd), 1e-12))
            corr_psd = corr_psd / np.outer(_d, _d)
            np.fill_diagonal(corr_psd, 1.0)
            corr_psd = (corr_psd + corr_psd.T) / 2.0
            # Gaussian copula MC
            z = rng.multivariate_normal(np.zeros(n), corr_psd, N_SIMS_PARLAY)
            u = _sc.norm.cdf(z)
            hits = u < probs  # shape (N_SIMS_PARLAY, n)
            joint = float(hits.all(axis=1).mean())
            joint = float(np.clip(joint, 1e-6, 1.0))
            ev = payout_mult * joint - 1.0
            kelly_f = max(0.0, ev / (payout_mult - 1.0)) if payout_mult > 1 else 0.0
            stake = min(bankroll * frac_kelly * kelly_f, bankroll * 0.05)
            results.append({
                "combo": " + ".join(f"{l['player']} {l['market']}" for l in combo_legs),
                "n_legs": n,
                "joint_prob_%": round(joint * 100, 1),
                "naive_prob_%": round(naive_joint * 100, 1),
                "ev_%": round(ev * 100, 1),
                "payout_x": payout_mult,
                "kelly_stake_$": round(stake, 2),
            })
    return sorted(results, key=lambda x: x["ev_%"], reverse=True)[:25]
# ──────────────────────────────────────────────
# MONTE CARLO GAME SIMULATION
# ──────────────────────────────────────────────
def monte_carlo_game_sim(legs, n_sims=20000, payout_mult=3.0):
    """Correlated MC simulation across all legs."""
    try:
        import scipy.stats as _sc
        valid = [l for l in legs if l.get("p_cal")]
        if not valid:
            return None
        n = len(valid)
        probs = np.array([float(l["p_cal"]) for l in valid])
        corr_mat = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                c = estimate_player_correlation(valid[i], valid[j])
                corr_mat[i,j] = corr_mat[j,i] = float(c or 0.0)
        evals, evecs = np.linalg.eigh(corr_mat)
        evals = np.clip(evals, 1e-6, None)
        corr_psd = evecs @ np.diag(evals) @ evecs.T
        # Renormalize diagonal to 1.0 and enforce symmetry
        _d = np.sqrt(np.maximum(np.diag(corr_psd), 1e-12))
        corr_psd = corr_psd / np.outer(_d, _d)
        np.fill_diagonal(corr_psd, 1.0)
        corr_psd = (corr_psd + corr_psd.T) / 2.0
        # [BUG FIX v4.0] Use dynamic seed derived from leg data, not fixed 42.
        # [AUDIT FIX] Include player_id + market in seed so different player combos
        # with identical probabilities get different random sequences. Previous sorted()
        # call lost ordering information, making two different combos with same
        # probability set indistinguishable.
        _mc_seed = int(abs(hash(tuple(
            (int(float(l.get("p_cal", 0.5)) * 1e6),
             int(l.get("player_id") or 0),
             str(l.get("market", ""))[:4])
            for l in valid
        ))) % (2**31))
        rng = np.random.default_rng(_mc_seed)
        z = rng.multivariate_normal(np.zeros(n), corr_psd, n_sims)
        u = _sc.norm.cdf(z)
        hits = u < probs
        joint_hits = hits.all(axis=1)
        joint_prob = float(joint_hits.mean())
        ev = payout_mult * joint_prob - 1.0
        return {
            "joint_prob_%": round(joint_prob * 100, 2),
            "naive_joint_%": round(float(np.prod(probs)) * 100, 2),
            "ev_%": round(ev * 100, 2),
            "per_leg_sim_%": [round(float(hits[:,i].mean()) * 100, 1) for i in range(n)],
            "n_sims": n_sims,
        }
    except Exception as e:
        return {"error": str(e)}
# ──────────────────────────────────────────────
# ROLLING BRIER SCORE
# ──────────────────────────────────────────────
def compute_rolling_brier(legs_df, windows=(25, 50, 100)):
    """Compute Brier scores over trailing windows and a rolling series."""
    if legs_df is None or legs_df.empty:
        return {}
    d = legs_df[legs_df["y"].notna()].copy().reset_index(drop=True)
    if len(d) < 10:
        return {}
    result = {}
    for w in windows:
        if len(d) >= w:
            tail = d.tail(w)
            result[f"last_{w}"] = float(np.mean((tail["p_raw"].values.astype(float) - tail["y"].values.astype(float))**2))
    if len(d) >= 10:
        series = []
        for i in range(9, len(d)):
            window_d = d.iloc[max(0, i-24):i+1]
            series.append(float(np.mean((window_d["p_raw"].values.astype(float) - window_d["y"].values.astype(float))**2)))
        result["rolling_series"] = series
    return result
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# CACHE PRE-WARMER  (eliminates NBA API I/O from threads)
# ──────────────────────────────────────────────
def pre_warm_scanner_caches(candidates, n_games):
    """
    Pre-fetch player IDs, game logs, and positions for all unique players in
    parallel (4 workers) before the main scan threads start.  Uses a session-
    state set to skip players already warmed this session.
    """
    unique_names = list(dict.fromkeys(pname for pname, *_ in candidates))
    already_warm = st.session_state.get("_prewarm_done", set())
    to_warm = [n for n in unique_names if n not in already_warm]
    if not to_warm:
        return set(unique_names)   # everything already cached
    # Bulk game logs + bulk position map (each = ONE API call for all players)
    _fetch_bulk_gamelogs()
    _ensure_pid_position_map()
    def _warm_one(name):
        try:
            pid = lookup_player_id(name)
            if pid:
                fetch_player_gamelog(player_id=pid, max_games=max(6, n_games + 5))
                get_player_position(name)
                return name
        except Exception:
            pass
        return None
    resolved = set()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for result in ex.map(_warm_one, to_warm):
            if result:
                resolved.add(result)
    already_warm.update(resolved)
    st.session_state["_prewarm_done"] = already_warm
    return resolved
# ══════════════════════════════════════════════
# v3.0 DEEP AUDIT UPGRADES — ADVANCED SIGNALS
# Research-backed features targeting professional-grade edge:
# 1. Win/Loss performance split (game context factor)
# 2. Clutch performance factor (late-game usage)
# 3. DNP probability score (quantified, not binary)
# 4. Projection confidence interval (80% CI)
# 5. FTA rate vs opponent (foul-drawing props)
# 6. Playoff implications / tanking factor
# 7. Alt line EV comparison
# 8. Middle opportunity detection
# ══════════════════════════════════════════════
# ──────────────────────────────────────────────
# [v3.0] WIN/LOSS PERFORMANCE SPLIT
# Sharp bettors know: stars play differently in W vs L.
# In wins, stars can rest early in blowouts -> lower counting stats.
# In losses, stars play max minutes trying to claw back -> higher.
# Research: avg win/loss stat differential for starters ~8-12%.
# ──────────────────────────────────────────────
def compute_win_loss_split(game_log_df, market, expected_win_prob=0.5):
    """
    Compare player's stats in team wins vs losses.
    Returns (wl_factor, wl_label, w_avg, l_avg).
    expected_win_prob: from moneyline (0.0-1.0). 0.5 = neutral.
    """
    if game_log_df is None or game_log_df.empty:
        return 1.0, "N/A", None, None
    try:
        df = game_log_df.copy()
        wl_col = next((c for c in ["WL", "W_L"] if c in df.columns), None)
        if wl_col is None:
            return 1.0, "N/A", None, None
        stat_col = STAT_FIELDS.get(market)
        if stat_col is None:
            return 1.0, "N/A", None, None
        if isinstance(stat_col, tuple):
            df["_stat"] = sum(pd.to_numeric(df.get(c), errors="coerce").fillna(0) for c in stat_col)
        else:
            df["_stat"] = pd.to_numeric(df.get(stat_col), errors="coerce")
        w_games = df[df[wl_col] == "W"]
        l_games = df[df[wl_col] == "L"]
        if len(w_games) < 3 or len(l_games) < 3:
            return 1.0, "Insufficient", None, None
        w_avg = float(w_games["_stat"].mean())
        l_avg = float(l_games["_stat"].mean())
        # M-5 audit fix: drop guard that silently discards extreme splits (e.g. player only
        # blocks in wins → l_avg=0, which is itself a strong signal and should not be dropped.
        if pd.isna(w_avg) or pd.isna(l_avg):
            return 1.0, "N/A", None, None
        win_ratio = (w_avg / l_avg) if l_avg > 1e-9 else (2.0 if w_avg > 1e-9 else 1.0)
        # Weighted blend: expected stat = win_prob * w_avg + (1-win_prob) * l_avg
        win_prob = float(np.clip(expected_win_prob or 0.5, 0.2, 0.8))
        expected_avg = win_prob * w_avg + (1 - win_prob) * l_avg
        season_avg = df["_stat"].mean()
        if pd.isna(season_avg) or season_avg <= 1e-6:
            return 1.0, "N/A", w_avg, l_avg
        factor = float(np.clip(expected_avg / season_avg, 0.90, 1.12))
        if win_ratio > 1.08:
            label = f"W-Heavy ({w_avg:.1f}W/{l_avg:.1f}L)"
        elif win_ratio < 0.92:
            label = f"L-Heavy ({w_avg:.1f}W/{l_avg:.1f}L)"
        else:
            label = f"Even ({w_avg:.1f}W/{l_avg:.1f}L)"
        return factor, label, w_avg, l_avg
    except Exception:
        return 1.0, "N/A", None, None
# ──────────────────────────────────────────────
# [v3.0] CLUTCH PERFORMANCE FACTOR
# Players with elite clutch stats see usage spike +20-30% in close games.
# Role players get buried. Key insight: props set from season averages
# don't account for game-state-dependent usage.
# Only applied when game is expected to be close (spread < 6).
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*6, show_spinner=False)
def get_clutch_performance_factor(player_id, market, spread_abs=None):
    """
    Fetch clutch splits (last 5 min, within 5 pts) vs overall season avg.
    Returns (clutch_factor, clutch_label).
    Only meaningful when game spread < 6 pts (close game likely).
    """
    if spread_abs is not None and float(spread_abs) >= 6.5:
        return 1.0, "Not Close"
    try:
        from nba_api.stats.endpoints import PlayerDashboardByClutch
        season = get_season_string()
        clutch_data = PlayerDashboardByClutch(
            player_id=player_id,
            season=season,
            per_mode_simple="PerGame",
            timeout=15,
        ).get_data_frames()
        if not clutch_data or len(clutch_data) < 2:
            return 1.0, "N/A"
        overall_df = clutch_data[0]
        clutch_df  = clutch_data[1]
        if overall_df.empty or clutch_df.empty:
            return 1.0, "N/A"
        clutch_gp = safe_float(clutch_df.iloc[0].get("GP", 0))
        if clutch_gp < 5:
            return 1.0, "Insufficient Clutch"
        stat_col_map = {
            "Points": "PTS", "Rebounds": "REB", "Assists": "AST",
            "3PM": "FG3M", "Steals": "STL", "Blocks": "BLK",
            "PRA": "PTS",
        }
        sc = stat_col_map.get(market)
        if sc is None or sc not in clutch_df.columns:
            return 1.0, "N/A"
        clutch_val  = safe_float(clutch_df.iloc[0].get(sc, 0))
        overall_val = safe_float(overall_df.iloc[0].get(sc, 0))
        if overall_val <= 1e-6:
            return 1.0, "N/A"
        ratio = clutch_val / overall_val
        # Clutch situations are ~10-15% of game time, so apply 15% weight
        factor = float(np.clip(1.0 + 0.15 * (ratio - 1.0), 0.94, 1.08))
        if ratio >= 1.20:
            label = f"Clutch Elite ({ratio:.2f}x)"
        elif ratio >= 1.08:
            label = f"Clutch+ ({ratio:.2f}x)"
        elif ratio <= 0.80:
            label = f"Clutch- ({ratio:.2f}x)"
        else:
            label = f"Clutch Neutral ({ratio:.2f}x)"
        return factor, label
    except Exception:
        return 1.0, "N/A"
# ──────────────────────────────────────────────
# [v3.0] QUANTIFIED DNP PROBABILITY SCORE
# Converts binary DNP flag to a continuous probability (0.0-1.0).
# Inputs: recent DNP frequency, injury status, minutes trend.
# This allows proportional stake scaling rather than binary half/zero.
# ──────────────────────────────────────────────
def compute_dnp_probability(game_log_df, injury_status=None, n_games=10):
    """
    Returns (dnp_prob 0.0-1.0, risk_label).
    """
    if game_log_df is None or game_log_df.empty:
        return 0.08, "Unknown"
    try:
        df = game_log_df.head(n_games).copy()
        if "MIN" not in df.columns:
            return 0.05, "Minimal Risk"
        mins = df["MIN"].apply(lambda v:
            float(str(v).split(":")[0]) if isinstance(v, str) and ":" in str(v)
            else safe_float(v, default=0.0))
        n_dnps = int((mins <= 4).sum())
        dnp_rate = n_dnps / max(len(df), 1)
        active = mins[mins >= 5]
        min_trend_factor = 0.0
        if len(active) >= 5:
            slope = float(np.polyfit(np.arange(len(active)), active.values[::-1], 1)[0])
            if slope < -0.5:
                min_trend_factor = min(0.20, abs(slope) * 0.04)
        # [Research-validated] Play-through rates per WSJ/EVAnalytics historical data:
        # Probable: ~90% play, Questionable: ~55%, Doubtful: ~3%, Out: 0%
        status_dnp_prob = {
            "OUT": 0.97, "DOUBTFUL": 0.97, "QUESTIONABLE": 0.45,
            "PROBABLE": 0.10, "ACTIVE": 0.02, "": 0.0
        }
        inj_risk = status_dnp_prob.get(str(injury_status or "").upper(), 0.0)
        avg_min = float(active.mean()) if not active.empty else 0.0
        low_min_risk = max(0.0, (15.0 - avg_min) / 100.0) if avg_min < 15 else 0.0
        if inj_risk >= 0.70:
            dnp_prob = float(np.clip(inj_risk * 0.80 + dnp_rate * 0.20, 0.0, 0.95))
        else:
            dnp_prob = float(np.clip(
                dnp_rate * 0.50 + min_trend_factor * 0.20 +
                inj_risk * 0.20 + low_min_risk * 0.10,
                0.0, 0.90
            ))
        if dnp_prob >= 0.65:   label = "Critical Risk"
        elif dnp_prob >= 0.40: label = "High Risk"
        elif dnp_prob >= 0.20: label = "Moderate Risk"
        elif dnp_prob >= 0.08: label = "Low Risk"
        else:                  label = "Minimal Risk"
        return float(dnp_prob), label
    except Exception:
        return 0.05, "Minimal Risk"
# ──────────────────────────────────────────────
# [v3.0] PROJECTION CONFIDENCE INTERVAL
# 80% CI gives bettors the realistic range of outcomes.
# Narrow CI = high-confidence bet. Wide CI = volatile / fade.
# ──────────────────────────────────────────────
def compute_projection_ci(mu, sigma, half_factor=1.0):
    """
    Returns (lower_80, upper_80) — the 80% confidence interval for the projection.
    Uses normal approximation (z=1.28 for 80% CI).
    """
    if mu is None or sigma is None:
        return None, None
    try:
        z = 1.28
        hf = float(half_factor) if half_factor else 1.0
        mu_h = float(mu) * hf
        sig_h = float(sigma) * hf
        lower = max(0.0, mu_h - z * sig_h)
        upper = mu_h + z * sig_h
        return float(lower), float(upper)
    except Exception:
        return None, None
# ──────────────────────────────────────────────
# [v3.0] FTA ALLOWED BY OPPONENT (FOUL-DRAWING FACTOR)
# For FTA, FTM, Stocks props: opponent foul rate is critical.
# Teams that allow many FTA create foul-drawing opportunities.
# Research: 20%+ spread between most/least foul-heavy teams.
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*4, show_spinner=False)
def get_opponent_fta_rate_factor(opp_abbr, market):
    """
    Returns (fta_factor, fta_label) for FTA/FTM/Stocks/Points props.
    >1.0 = opponent commits more fouls (favorable for FTA/FTM).
    """
    if market not in ("FTA", "FTM", "Stocks", "Steals", "Points", "PRA"):
        return 1.0, "N/A"
    try:
        from nba_api.stats.endpoints import LeagueDashTeamStats
        ss = get_season_string()
        opp_stats = LeagueDashTeamStats(
            season=ss,
            measure_type_detailed_defense="Opponent",
            per_mode_detailed="PerGame",
            timeout=20,
        ).get_data_frames()[0]
        if opp_stats.empty:
            return 1.0, "Avg"
        opp_key = str(opp_abbr).upper()
        row = opp_stats[opp_stats["TEAM_ABBREVIATION"].str.upper() == opp_key]
        if row.empty:
            return 1.0, "Avg"
        fta_col = next((c for c in ["OPP_FTA", "FTA"] if c in row.columns), None)
        if fta_col is None:
            return 1.0, "Avg"
        opp_fta = float(row[fta_col].values[0])
        league_avg = float(opp_stats[fta_col].mean())
        if league_avg <= 0:
            return 1.0, "Avg"
        ratio = opp_fta / league_avg
        market_weight = {"FTA": 1.0, "FTM": 1.0, "Stocks": 0.3, "Steals": 0.4, "Points": 0.25, "PRA": 0.15}
        w = market_weight.get(market, 0.2)
        factor = float(np.clip(1.0 + (ratio - 1.0) * w, 0.93, 1.07))
        if ratio >= 1.12:
            label = f"Foul-Heavy ({ratio:.2f}x)"
        elif ratio <= 0.88:
            label = f"Foul-Light ({ratio:.2f}x)"
        else:
            label = "Avg Foul Rate"
        return factor, label
    except Exception:
        return 1.0, "Avg"
# ──────────────────────────────────────────────
# [AUDIT IMPROVEMENT] OPPONENT TRUE SHOOTING % FACTOR
# True Shooting % (TS%) = PTS / (2 * (FGA + 0.44*FTA)) measures scoring efficiency.
# SHAP analysis of NBA prop models identifies opponent TS% allowed as the top
# efficiency signal for Points props (Deshpande & Jensen 2016; EVAnalytics 2024).
# Teams allowing high TS% have weak interior + perimeter defense → scoring boost.
# Teams with elite defensive TS% suppression → discount on scoring props.
# Applied only to Points/FGM/PRA/PA markets (not counting/rebounding stats).
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*4, show_spinner=False)
def get_opponent_ts_pct_factor(opp_abbr, market):
    """
    Returns (ts_factor, ts_label) for scoring-efficiency props.
    >1.0 = opponent allows above-average TS% (favorable for scoring props).
    <1.0 = opponent suppresses TS% (discount for scoring props).
    Only applied to: Points, FGM, PRA, PA, 3PM (partial).
    """
    _SCORING_MARKETS = {"Points", "FGM", "PRA", "PA", "3PM", "H1 Points", "H2 Points", "Q1 Points"}
    if market not in _SCORING_MARKETS:
        return 1.0, "N/A"
    try:
        from nba_api.stats.endpoints import LeagueDashTeamStats
        ss = get_season_string()
        # Opponent stats: get pts/fga/fta allowed per game
        opp_stats = LeagueDashTeamStats(
            season=ss,
            measure_type_detailed_defense="Opponent",
            per_mode_detailed="PerGame",
            timeout=20,
        ).get_data_frames()[0]
        if opp_stats is None or opp_stats.empty:
            return 1.0, "Avg"
        opp_key = str(opp_abbr).upper()
        row = opp_stats[opp_stats["TEAM_ABBREVIATION"].str.upper() == opp_key]
        if row.empty:
            return 1.0, "Avg"
        # Compute opponent-allowed TS%: PTS_allowed / (2*(FGA_allowed + 0.44*FTA_allowed))
        # Columns vary by API version; try common names
        pts_col = next((c for c in ["OPP_PTS", "PTS"] if c in row.columns), None)
        fga_col = next((c for c in ["OPP_FGA", "FGA"] if c in row.columns), None)
        fta_col = next((c for c in ["OPP_FTA", "FTA"] if c in row.columns), None)
        if not all([pts_col, fga_col, fta_col]):
            return 1.0, "Avg"
        r = row.iloc[0]
        opp_pts = float(r[pts_col])
        opp_fga = float(r[fga_col])
        opp_fta = float(r[fta_col])
        denom = 2.0 * (opp_fga + 0.44 * opp_fta)
        if denom <= 0:
            return 1.0, "Avg"
        opp_ts = opp_pts / denom
        # League average TS% allowed
        all_pts  = opp_stats[pts_col].astype(float)
        all_fga  = opp_stats[fga_col].astype(float)
        all_fta  = opp_stats[fta_col].astype(float)
        all_denom = 2.0 * (all_fga + 0.44 * all_fta)
        all_denom = all_denom.replace(0, np.nan)
        league_ts = float((all_pts / all_denom).mean())
        if league_ts <= 0:
            return 1.0, "Avg"
        ratio = opp_ts / league_ts
        # Market-specific weight: Points/FGM most sensitive to TS%, 3PM partial
        market_weight = {
            "Points": 0.80, "FGM": 0.70, "PRA": 0.50, "PA": 0.45,
            "3PM": 0.40, "H1 Points": 0.80, "H2 Points": 0.80, "Q1 Points": 0.80,
        }
        w = market_weight.get(market, 0.5)
        # Factor capped at ±8% to prevent single signal overwhelming composite
        factor = float(np.clip(1.0 + (ratio - 1.0) * w, 0.92, 1.08))
        if ratio >= 1.06:
            label = f"Soft D (TS%={opp_ts:.3f})"
        elif ratio <= 0.94:
            label = f"Elite D (TS%={opp_ts:.3f})"
        else:
            label = f"Avg D (TS%={opp_ts:.3f})"
        return factor, label
    except Exception:
        return 1.0, "Avg"
# ──────────────────────────────────────────────
# [RESEARCH IMPROVEMENT] OPPONENT eFG% FACTOR FOR 3PM/FGM PROPS
# Effective Field Goal % (eFG%) = (FGM + 0.5*FG3M) / FGA weights 3-pointers
# appropriately. While TS% (already implemented) captures scoring efficiency broadly,
# eFG% is the more direct signal for pure field goal volume props (3PM, FGM, FGA):
# - TS% includes free throws → noise for 3PM props where FTA is irrelevant
# - eFG% isolates field-goal shooting quality, penalizing low-3P% defenses directly
# Research (DataJocks NBA shooting analysis 2024): eFG% allowed by team vs position
# has 0.34 Spearman correlation with opponent 3PM outcomes over season.
# Applied separately from TS% to avoid double-counting: TS% → scoring/FTA markets,
# eFG% → pure shooting volume markets (3PM, FGM, FGA).
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*4, show_spinner=False)
def get_opponent_efg_factor(opp_abbr, market):
    """
    Returns (efg_factor, efg_label) for shooting volume props.
    >1.0 = opponent allows above-average eFG% (favorable for 3PM/FGM).
    <1.0 = opponent suppresses eFG% (discount for shooting props).
    Distinct from TS%: focused on field goal efficiency, not overall scoring efficiency.
    Applied to: 3PM, FGM, FGA (NOT Points — TS% handles that).
    """
    _EFG_MARKETS = {"3PM", "FGM", "FGA", "3PA", "H1 3PM"}
    if market not in _EFG_MARKETS:
        return 1.0, "N/A"
    try:
        from nba_api.stats.endpoints import LeagueDashTeamStats
        ss = get_season_string()
        opp_stats = LeagueDashTeamStats(
            season=ss,
            measure_type_detailed_defense="Opponent",
            per_mode_detailed="PerGame",
            timeout=20,
        ).get_data_frames()[0]
        if opp_stats is None or opp_stats.empty:
            return 1.0, "Avg"
        opp_key = str(opp_abbr).upper()
        row = opp_stats[opp_stats["TEAM_ABBREVIATION"].str.upper() == opp_key]
        if row.empty:
            return 1.0, "Avg"
        # eFG% = (FGM + 0.5*FG3M) / FGA; look for opponent-allowed columns
        fgm_col = next((c for c in ["OPP_FGM", "FGM"] if c in row.columns), None)
        fg3m_col = next((c for c in ["OPP_FG3M", "FG3M"] if c in row.columns), None)
        fga_col  = next((c for c in ["OPP_FGA", "FGA"]  if c in row.columns), None)
        if not all([fgm_col, fg3m_col, fga_col]):
            return 1.0, "Avg"
        r = row.iloc[0]
        opp_fgm  = float(r[fgm_col])
        opp_fg3m = float(r[fg3m_col])
        opp_fga  = float(r[fga_col])
        if opp_fga <= 0:
            return 1.0, "Avg"
        opp_efg = (opp_fgm + 0.5 * opp_fg3m) / opp_fga
        # League average eFG% allowed
        all_fgm  = opp_stats[fgm_col].astype(float)
        all_fg3m = opp_stats[fg3m_col].astype(float)
        all_fga  = opp_stats[fga_col].astype(float)
        all_fga  = all_fga.replace(0, np.nan)
        league_efg = float(((all_fgm + 0.5 * all_fg3m) / all_fga).mean())
        if league_efg <= 0:
            return 1.0, "Avg"
        ratio = opp_efg / league_efg
        # 3PM is most sensitive to eFG% (3P% defense), FGM/FGA slightly less
        market_weight = {"3PM": 0.85, "3PA": 0.75, "FGM": 0.60, "FGA": 0.60, "H1 3PM": 0.85}
        w = market_weight.get(market, 0.6)
        factor = float(np.clip(1.0 + (ratio - 1.0) * w, 0.93, 1.07))
        if ratio >= 1.06:
            label = f"Soft FG-D (eFG%={opp_efg:.3f})"
        elif ratio <= 0.94:
            label = f"Elite FG-D (eFG%={opp_efg:.3f})"
        else:
            label = f"Avg FG-D (eFG%={opp_efg:.3f})"
        return factor, label
    except Exception:
        return 1.0, "Avg"
# ──────────────────────────────────────────────
# [AUDIT IMPROVEMENT] PLAYER FREE THROW RATE (FTr) TREND FACTOR
# FTr = FTA / FGA measures how aggressively a player attacks the rim.
# A rising FTr in recent games → player drawing more contact → scoring boost.
# Particularly powerful for Points/FTA props: foul-drawing is sticky in short runs.
# Research: Basketball-Reference / Cleaning the Glass identify FTr as a leading
# indicator for point total performance (correlation ~0.28 over L5 vs L20 windows).
# ──────────────────────────────────────────────
def compute_player_ftr_factor(game_log_df, market, n_recent=5, n_baseline=20):
    """
    [AUDIT IMPROVEMENT] Player's recent free throw rate vs baseline.
    Returns (ftr_factor, ftr_label).
    >1.0 = player attacking the rim more than usual (scoring boost)
    <1.0 = player avoiding contact / fewer FT opportunities (discount)
    Applied to: Points, FTA, FTM, PRA, PA markets.
    """
    _FTR_MARKETS = {"Points", "FTA", "FTM", "PRA", "PA", "H1 Points", "H2 Points"}
    if market not in _FTR_MARKETS:
        return 1.0, "N/A"
    if game_log_df is None or game_log_df.empty:
        return 1.0, "Insufficient"
    if not all(c in game_log_df.columns for c in ["FTA", "FGA"]):
        return 1.0, "Insufficient"
    try:
        fta = pd.to_numeric(game_log_df["FTA"], errors="coerce").fillna(0)
        fga = pd.to_numeric(game_log_df["FGA"], errors="coerce").fillna(0)
        # Only include games where player actually attempted field goals (avoid DNP noise)
        active = fga > 0
        if active.sum() < n_recent + 2:
            return 1.0, "Insufficient"
        ftr = fta / fga.replace(0, np.nan)
        l5_ftr  = float(ftr[active].iloc[:n_recent].mean())
        l20_ftr = float(ftr[active].iloc[:n_baseline].mean()) if active.sum() >= n_baseline else float(ftr[active].mean())
        if l20_ftr <= 0.01:
            return 1.0, "Avg FTr"
        delta_pct = (l5_ftr - l20_ftr) / l20_ftr
        # Market weight: FTA/FTM most sensitive; Points partial
        market_weight = {"FTA": 1.0, "FTM": 1.0, "Points": 0.35, "PRA": 0.25, "PA": 0.30,
                         "H1 Points": 0.35, "H2 Points": 0.35}
        w = market_weight.get(market, 0.3)
        factor = float(np.clip(1.0 + delta_pct * w * 0.5, 0.96, 1.06))
        if delta_pct >= 0.20:
            label = f"FTr Spike +{delta_pct*100:.0f}% (L5={l5_ftr:.2f})"
        elif delta_pct <= -0.20:
            label = f"FTr Drop {delta_pct*100:.0f}% (L5={l5_ftr:.2f})"
        else:
            label = f"FTr Normal ({l5_ftr:.2f})"
        return factor, label
    except Exception:
        return 1.0, "Avg"
# ──────────────────────────────────────────────
# [v3.0] PLAYOFF IMPLICATIONS / TANKING FACTOR
# Late-season context is one of the most underpriced signals in props.
# Tanking teams bench stars -> UNDER on minutes-dependent props.
# Bubble teams push hard -> OVER on all props for starters.
# Research: stars on eliminated teams average 3-5 fewer minutes in April.
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*8, show_spinner=False)
def get_playoff_implications_factor(team_abbr, game_date, market):
    """
    Returns (playoff_factor, playoff_label) based on team standings.
    Applied only in March-April.
    """
    if not team_abbr or game_date.month not in (3, 4):
        return 1.0, "Regular"
    try:
        from nba_api.stats.endpoints import LeagueStandingsV3
        standings = LeagueStandingsV3(
            league_id="00",
            season=get_season_string(),
            season_type="Regular Season",
            timeout=20,
        ).get_data_frames()[0]
        if standings.empty:
            return 1.0, "Regular"
        team_upper = str(team_abbr).upper()
        abbr_col = next((c for c in standings.columns
                         if "Abbreviation" in c or "ABBREVIATION" in c), None)
        if abbr_col is None:
            return 1.0, "Regular"
        row = standings[standings[abbr_col].str.upper() == team_upper]
        if row.empty:
            return 1.0, "Regular"
        conf_rank = safe_float(row.iloc[0].get("ConferenceRank", 8))
        games_back_col = next((c for c in row.columns
                                if "GamesBack" in c or "ConferenceGamesBack" in c), None)
        games_back = safe_float(row.iloc[0].get(games_back_col, 5)) if games_back_col else 5.0
        if conf_rank <= 3 and game_date.month == 4:
            factor, label = 0.975, "Clinched (Load Mgmt Risk)"
        elif conf_rank <= 6:
            factor, label = 1.0, "Playoff Push"
        elif conf_rank <= 10:
            factor, label = 1.02, "Play-In Bubble"
        elif games_back >= 8 and game_date.month == 4:
            volume_markets = {"Points", "PRA", "PA", "PR", "RA", "Assists", "Rebounds", "3PM", "FTM", "FTA"}
            factor = 0.96 if market in volume_markets else 0.985
            label = "Tanking"
        elif games_back >= 5:
            factor, label = 0.99, "Out of Race"
        else:
            factor, label = 1.0, "Regular"
        return float(factor), label
    except Exception:
        return 1.0, "Regular"
# ──────────────────────────────────────────────
# [v3.0] ALT LINE EV COMPARISON
# Professional bettors always shop alt lines.
# Shows EV at +/-0.5 from current line — helps identify optimal entry.
# ──────────────────────────────────────────────
def compute_alt_line_ev(p_cal, price_decimal, line, step=0.5, n_steps=3, sigma=None):
    """
    Compute EV at alternative lines around the current line.
    Returns list of dicts: [{line, delta, p_est, ev, edge_cat}].
    sigma: actual bootstrap sigma from compute_leg_projection — much more accurate than line/4 heuristic.
    """
    if p_cal is None or price_decimal is None:
        return []
    try:
        from scipy.stats import norm
        results = []
        orig_line = float(line)
        orig_p    = float(p_cal)
        z_orig = norm.ppf(1 - orig_p)
        # [FIX v4.0] Use actual bootstrap sigma when available; line/4 is a poor fallback
        # for low-line markets (3PM, Steals) where sigma << line/4.
        sigma_est = max(0.5, float(sigma)) if (sigma is not None and float(sigma) > 0) else max(1.0, orig_line / 4.0)
        for i in range(-n_steps, n_steps + 1):
            if i == 0:
                continue
            new_line = orig_line + i * step
            if new_line <= 0:
                continue
            z_new = z_orig + (new_line - orig_line) / sigma_est
            p_new = float(np.clip(1 - norm.cdf(z_new), 0.01, 0.99))
            ev_new = ev_per_dollar(p_new, price_decimal)
            results.append({
                "line": round(new_line, 1),
                "delta": round(i * step, 1),
                "p_est": round(p_new, 3),
                "ev": round(ev_new, 4) if ev_new is not None else None,
                "edge_cat": classify_edge(ev_new) if ev_new is not None else "N/A",
            })
        return results
    except Exception:
        return []
# ──────────────────────────────────────────────
# [v3.0] MIDDLE / LINE SHOPPING DETECTOR
# When books disagree on a player's line by >=1 unit,
# there's an opportunity to bet both sides and capture the middle.
# ──────────────────────────────────────────────
def detect_middle_opportunity(player_name, market_key, event_id):
    """
    Check all available books for line discrepancies on the same player.
    Returns (middle_exists, low_line, high_line, middle_prob_est, book_details).
    """
    if not event_id or not market_key:
        return False, None, None, None, []
    try:
        regions = "us,us2,eu,uk"
        odds, err = odds_get_event_odds(event_id, [market_key], regions=regions)
        if err or not odds:
            return False, None, None, None, []
        rows, _ = _parse_player_prop_outcomes(odds, market_key, book_filter=None)
        player_norm = normalize_name(player_name)
        over_rows = [r for r in rows
                     if r.get("player_norm") == player_norm and
                     "over" in str(r.get("side", "")).lower() and
                     r.get("line") is not None]
        if len(over_rows) < 2:
            return False, None, None, None, []
        lines = sorted(set(float(r["line"]) for r in over_rows))
        min_line = min(lines)
        max_line = max(lines)
        spread = max_line - min_line
        if spread < 0.5:
            return False, min_line, max_line, None, []
        from scipy.stats import norm
        # [BUG FIX v4.0] Previous formula was norm.cdf(line/sigma) which is zero-centered.
        # Correct formula: P(low < X < high) where X ~ N(mean, sigma).
        # Use midpoint of spread as estimate of mean, line/4 as sigma (same heuristic as before
        # but now properly centered). This estimates prob of landing in the middle window.
        mean_est = (min_line + max_line) / 2.0
        sigma_est = max(1.0, mean_est / 4.0)
        mid_prob = float(norm.cdf((max_line - mean_est) / sigma_est) -
                         norm.cdf((min_line - mean_est) / sigma_est))
        mid_prob = float(np.clip(mid_prob, 0.01, 0.50))
        book_details = [{"book": r["book"], "line": float(r["line"])} for r in over_rows]
        return spread >= 1.0, min_line, max_line, round(mid_prob, 3), book_details
    except Exception:
        return False, None, None, None, []
# ──────────────────────────────────────────────
# [v7.0] POSSESSION-LEVEL SIMULATION P(OVER) — 60/40 SIM/BOOTSTRAP ENSEMBLE
# ──────────────────────────────────────────────
_SIM_N_SIMS = 3000       # simulation count (raised from 500 to 3000 for tighter distributions)
_SIM_BLEND_WEIGHT = 0.60  # simulation weight in ensemble (60% sim, 40% bootstrap)
_BOOTSTRAP_BLEND_WEIGHT = 0.40  # bootstrap weight in ensemble

@st.cache_data(ttl=60*30, show_spinner=False)
def _run_player_simulation(
    player_name: str,
    player_id: str,
    team_abbr: str,
    opp_abbr: str,
    is_home: bool,
    gamelog_json: str,       # JSON-serialized gamelog for caching (Streamlit cache requires hashable)
    position: str,
    rest_days: int,
    n_games: int,
    game_spread: float,
) -> dict:
    """Run the possession-level simulator and return P(over) for all stats.

    Returns a dict: {stat_key: {"p_over_at_line": {line_str: p}, "mean": float, "std": float}}
    Returns empty dict on failure.
    """
    if not _SIM_AVAILABLE:
        return {}
    try:
        gamelog_df = pd.read_json(gamelog_json) if gamelog_json else pd.DataFrame()
    except Exception:
        gamelog_df = pd.DataFrame()
    if gamelog_df.empty:
        return {}
    try:
        loader = SimulationDataLoader()
        # Build profile for the target player
        target_profile = loader.build_player_profile(
            player_name=player_name,
            player_id=str(player_id),
            gamelog_df=gamelog_df,
            position=position,
            is_starter=True,
            rotation_order=0,
            rest_days=rest_days,
            n_games=n_games,
        )
        # Build a default roster around the player (we only care about this player's distribution)
        home_profiles = [target_profile] if is_home else GameEngine._default_roster(True)
        away_profiles = GameEngine._default_roster(False) if is_home else [target_profile]
        if is_home:
            # Fill remaining 11 slots with defaults
            home_profiles = [target_profile] + GameEngine._default_roster(True)[1:]
        else:
            away_profiles = [target_profile] + GameEngine._default_roster(False)[1:]
        # Fix player IDs so target player isn't overwritten by defaults
        for i, p in enumerate(home_profiles[1:] if is_home else away_profiles[1:], 1):
            p.player_id = f"{'h' if is_home else 'a'}_{i}"

        # Get team pace from TEAM_CTX
        home_pace = float(TEAM_CTX.get(str(team_abbr if is_home else opp_abbr).upper(), {}).get("PACE", 100.0))
        away_pace = float(TEAM_CTX.get(str(opp_abbr if is_home else team_abbr).upper(), {}).get("PACE", 100.0))

        engine = GameEngine(
            config=None,
            home_profiles=home_profiles,
            away_profiles=away_profiles,
            home_name=team_abbr if is_home else opp_abbr,
            away_name=opp_abbr if is_home else team_abbr,
            home_pace=home_pace,
            away_pace=away_pace,
            pre_game_spread=float(game_spread) if game_spread is not None else 0.0,
        )
        output = engine.run_simulation(n=_SIM_N_SIMS, seed=None)
        # Extract distributions for the target player
        pid_str = str(player_id)
        if pid_str not in output.distributions:
            return {}
        result = {}
        for stat_key, dist in output.distributions[pid_str].items():
            result[stat_key] = {
                "mean": float(dist.mean),
                "std": float(dist.std),
                "values": dist.values.tolist(),
            }
        return result
    except Exception:
        return {}


def _get_sim_prob_over(sim_result: dict, stat_key: str, line: float) -> float | None:
    """Extract P(over line) from cached simulation result for a given stat key."""
    if not sim_result or stat_key not in sim_result:
        return None
    try:
        values = np.array(sim_result[stat_key]["values"])
        return float(np.mean(values > line))
    except Exception:
        return None


# MAIN PROJECTION ENGINE  [FIX 3: minutes filter]
# ──────────────────────────────────────────────
def compute_leg_projection(
    player_name, market_name, line, meta,
    n_games, key_teammate_out,
    bankroll=0.0, frac_kelly=0.25, max_risk_frac=0.05,
    market_prior_weight=0.65, exclude_chaotic=True,
    game_date=None, is_home=None,
    injury_team_map=None,   # {team_abbr_upper: [player_name_lower, ...]} for OUT/DOUBTFUL players
):
    errors = []
    game_date = game_date or date.today()
    player_id = lookup_player_id(player_name)
    if not player_id:
        errors.append("Could not resolve NBA player id.")
        return {"player":player_name,"market":market_name,"line":float(line),
                "proj":None,"p_over":None,"p_cal":None,"edge":None,
                "team":None,"opp":None,"headshot":None,"errors":errors,
                "player_id":None,"gate_ok":False,"gate_reason":"no player id"}
    gldf, gl_errs = fetch_player_gamelog(player_id=player_id, max_games=max(6, n_games+5))
    if gl_errs: errors.extend([f"NBA API: {m}" for m in gl_errs])
    gldf_n = gldf.head(n_games) if not gldf.empty else gldf
    # [FIX 3] Filter out DNP/garbage-time games (MIN < threshold)
    if not gldf_n.empty and "MIN" in gldf_n.columns:
        try:
            min_vals = gldf_n["MIN"].apply(lambda v:
                float(str(v).split(":")[0]) if isinstance(v, str) and ":" in str(v)
                else safe_float(v, default=0.0))
            mask = min_vals >= MIN_MINUTES_THRESHOLD
            n_excluded = int((~mask).sum())
            gldf_filtered = gldf_n[mask]
            if len(gldf_filtered) >= 4:
                gldf_n = gldf_filtered
                if n_excluded > 0:
                    errors.append(f"Excluded {n_excluded} low-minute games (<{MIN_MINUTES_THRESHOLD} min)")
        except Exception:
            pass
    # ── Detect special market types ──────────────────────────────
    half_factor = HALF_FACTOR.get(market_name, 1.0)
    is_half_market = market_name in HALF_FACTOR
    is_dd_td = market_name in DD_TD_MARKETS
    is_fantasy = market_name in FANTASY_MARKETS
    # For half/Q1 markets, find effective full-game stat field
    base_market = market_name
    if is_half_market:
        base_market = (market_name.replace("H1 ","").replace("H2 ","").replace("Q1 ",""))
    stat_series = compute_stat_from_gamelog(gldf_n, base_market) if not gldf_n.empty else pd.Series([], dtype=float)
    # [AUDIT FIX] For half/Q1 markets: try real per-period boxscore data.
    # If successful, stat_series is replaced with actual H1/H2/Q1 values so
    # CV, skewness, and bootstrap all operate on real half-game distributions.
    _orig_half_factor = half_factor   # save for Bayesian prior scaling below
    if is_half_market and player_id:
        _hg_series = fetch_player_halfgame_log(player_id, gldf_n, market_name, n_games=n_games)
        if _hg_series is not None and len(_hg_series.dropna()) >= 3:
            stat_series = _hg_series
            half_factor = 1.0   # Real half-game data; no scaling needed anywhere
            errors.append(f"Real {market_name} split: {len(_hg_series.dropna())} games from boxscores")
    vol_cv, vol_label = compute_volatility(stat_series)
    stat_skew = compute_skewness(stat_series)
    # [AUDIT FIX] Skewness heuristic when n<4: market-type prior so gate isn't fully bypassed
    if stat_skew is None and not stat_series.dropna().empty:
        if base_market in ("3PM", "Blocks", "Steals", "Stocks", "FTM", "FTA"):
            stat_skew = 0.70   # Count/rate stats dominated by zeros: strong right skew
        elif base_market in ("Rebounds", "RA", "PR"):
            stat_skew = 0.25   # Mildly right-skewed (occasional bigs blowups)
    rest_mult, rest_days = compute_rest_factor(gldf, game_date)
    # [UPGRADE 2] Projected minutes + DNP risk
    proj_minutes, dnp_risk = compute_projected_minutes(gldf_n, n_games=n_games)
    # [AUDIT UPGRADE] Per-minute production ratio — strongest signal for injury-replacement props.
    # Compute baseline production/min; prod_mult applied post-shrinkage if proj_minutes diverges.
    _per_min_rate, _ = compute_per_minute_production(gldf_n, base_market if not is_dd_td else "Points", n_games=n_games)
    # Compute historical avg_minutes from gldf_n for minutes-based projection scaling below
    _hist_avg_minutes = None
    if not gldf_n.empty and "MIN" in gldf_n.columns:
        try:
            _hist_mins = gldf_n["MIN"].apply(lambda v:
                float(str(v).split(":")[0]) if isinstance(v, str) and ":" in str(v)
                else safe_float(v, default=0.0))
            _hist_active = _hist_mins[_hist_mins >= 5]
            if len(_hist_active) >= 3:
                _hist_avg_minutes = float(_hist_active.mean())
        except Exception:
            pass
    # Usage rate signal
    usage_rate = compute_usage_rate(gldf_n, n_games=n_games)
    # [AUDIT FIX] B2B flag from rest_days already computed above — remove redundant
    # second call to compute_rest_factor which made the same API call twice.
    b2b_flag = (rest_days == 0)
    # Resolve team/opponent
    team_abbr, opp_abbr, is_home_resolved = None, None, is_home
    if not gldf_n.empty:
        try:
            matchup = str(gldf_n.iloc[0].get("MATCHUP","")).strip()
            if " vs " in matchup.lower().replace("vs.","vs"):
                pts = re.split(r'\s+vs\.?\s+', matchup, flags=re.IGNORECASE)
                if len(pts)==2: team_abbr, opp_abbr = pts[0].strip(), pts[1].strip()
            elif " @ " in matchup:
                pts = matchup.split(" @ ")
                if len(pts)==2: team_abbr, opp_abbr = pts[0].strip(), pts[1].strip()
        except Exception: pass
    if team_abbr and not opp_abbr:
        try:
            opp_abbr, is_home_resolved = opponent_from_team_abbr(team_abbr, game_date)
        except Exception: pass
    if meta:
        try:
            home_abbr = map_team_name_to_abbr(meta.get("home_team","") or "")
            away_abbr = map_team_name_to_abbr(meta.get("away_team","") or "")
            if team_abbr and home_abbr and away_abbr:
                if team_abbr == home_abbr: opp_abbr, is_home_resolved = away_abbr, True
                elif team_abbr == away_abbr: opp_abbr, is_home_resolved = home_abbr, False
        except Exception: pass
    # ── Auto key_teammate_out from injury map ─────────────────────
    auto_inj_triggered = False
    auto_inj_player = None
    if not key_teammate_out and injury_team_map and team_abbr:
        team_key = str(team_abbr).upper()
        out_players = injury_team_map.get(team_key, [])
        player_lower = (player_name or "").lower()
        # Any OUT/DOUBTFUL teammate (exclude the player themselves)
        for op in out_players:
            if op and op != player_lower:
                key_teammate_out = True
                auto_inj_triggered = True
                auto_inj_player = op
                break
    ha_mult = compute_home_away_factor(gldf_n, base_market, is_home_resolved)
    # [v4.0] Resolve position early so lineup boost can use pos_bucket
    _pos_str_early = get_player_position(player_name) or ""
    _pos_bucket_early = get_position_bucket(_pos_str_early)
    # [v4.0 UPGRADE] Lineup injury boost: now uses compute_lineup_injury_boost for precision.
    # Replaces flat 1.03-1.08 heuristic with usage-rate-scaled absorption model.
    _usage_boost = 1.0
    _lineup_label = "Normal"
    if key_teammate_out:
        _auto_out_name = auto_inj_player or ""
        # Market-specific direction: Assists/PA/RA penalized when playmaker is out
        _assist_heavy = base_market in ("Assists", "PA", "RA", "PRA")
        if _assist_heavy:
            _usage_boost = 0.97  # assists/playmaking drops with star PG out
            _lineup_label = "Assist Drag (Star PG Out)"
        else:
            # Use the precise lineup model
            _lineup_mult, _lineup_label = compute_lineup_injury_boost(
                player_name, team_abbr, _auto_out_name,
                base_market, _pos_bucket_early, usage_rate, n_games=n_games
            )
            _usage_boost = _lineup_mult
    ctx_mult = advanced_context_multiplier(player_name, base_market, opp_abbr, False)
    if key_teammate_out:
        ctx_mult *= _usage_boost
    blowout_prob = estimate_blowout_risk(team_abbr, opp_abbr, spread_abs=None)
    # [UPGRADE NEW] Fetch game total + spread for accurate blowout risk and game-script context
    _game_total, _game_spread, _home_ml, _away_ml = None, None, None, None
    if meta and meta.get("event_id"):
        try:
            _game_total, _game_spread, _home_ml, _away_ml = fetch_game_total_and_spread(meta["event_id"])
        except Exception:
            pass
    if _game_spread is not None:
        try:
            blowout_prob = estimate_blowout_risk(team_abbr, opp_abbr, spread_abs=abs(float(_game_spread)))
        except Exception:
            pass
    # [UPGRADE NEW] Schedule fatigue (3-in-4, 4-in-6)
    _fatigue_mult, _fatigue_label = compute_schedule_fatigue(gldf, game_date)
    # [RESEARCH UPGRADE] Rolling 3-game minutes fatigue (CMU: most predictive signal)
    _rolling_min_avg, _rolling_min_mult, _rolling_min_label = compute_rolling_minutes_fatigue(gldf, n_recent=3)
    # Combine with schedule fatigue — take the more severe penalty
    if _rolling_min_mult < _fatigue_mult:
        _fatigue_mult = _rolling_min_mult
        _fatigue_label = _rolling_min_label if _fatigue_label == "Normal" else f"{_fatigue_label}+{_rolling_min_label}"
    # [RESEARCH UPGRADE] Both-teams-B2B suppressor
    _both_b2b, _both_b2b_mult = compute_both_teams_b2b(team_abbr, opp_abbr, b2b_flag, game_date)
    # [UPGRADE NEW] Opponent fatigue → defensive boost
    _opp_fatigue_mult, _opp_fatigue_label = get_opponent_schedule_fatigue(opp_abbr, game_date)
    # [UPGRADE NEW] Game-script multiplier from game total
    _game_script_mult = compute_game_script_mult(_game_total, _game_spread, base_market, team_abbr, opp_abbr,
                                                  is_home=is_home_resolved, game_spread=_game_spread)
    # [RESEARCH UPGRADE] Travel fatigue (cross-country B2B is worst case: ~5-7% penalty)
    _travel_mult, _travel_label = compute_travel_fatigue(team_abbr, gldf, game_date, b2b_flag)
    # [RESEARCH UPGRADE] L10 rolling DvP — computed after pos_bucket is resolved below
    _dvp_l10_mult, _dvp_l10_label = 1.0, "Avg"  # placeholder; overwritten after pos_bucket computed
    # Blowout risk trims expected minutes: high-blowout games see starters sit early.
    # Cap at 12% reduction (blowout_prob=0.80 → ×0.88) to avoid over-penalizing.
    if proj_minutes is not None and blowout_prob is not None and blowout_prob > 0.20:
        blowout_min_adj = 1.0 - min(0.12, (float(blowout_prob) - 0.20) * 0.20)
        proj_minutes = proj_minutes * blowout_min_adj
    pos_str = get_player_position(player_name) or ""
    pos_bucket = get_position_bucket(pos_str)
    # [AUDIT FIX] Apply position-specific half-game factor when real split data isn't available
    # [v5.0] Pass spread_abs for Voulgaris close-game H2 boost
    if is_half_market and half_factor != 1.0:
        # [AUDIT FIX] Default spread_abs to league avg 4.5 when None (manual entries, no spread data)
        # Without this, Voulgaris H2 close-game boost never applies to manual lines.
        _spread_for_half = abs(float(_game_spread)) if _game_spread is not None else 4.5
        half_factor = get_half_factor(market_name, pos_bucket, spread_abs=_spread_for_half,
                                      game_total=_game_total)
        _orig_half_factor = half_factor
    # Positional defensive grade multiplier
    pos_def_mult = positional_def_multiplier(opp_abbr, pos_bucket, base_market)
    # Pace-adjusted stat series
    pace_adj_series = compute_pace_adjusted_series(stat_series, opp_abbr)
    # [UPGRADE 3] Opponent-specific historical factor
    opp_specific_factor, n_vs_opp = compute_opp_specific_factor(gldf_n, opp_abbr, base_market)
    # [UPGRADE 5] Hot / cold regime (uses full season log for broader z-score)
    hot_cold_label, hot_cold_z = compute_player_regime_hot_cold(stat_series)
    # [RESEARCH UPGRADE] L10 rolling DvP — resolve pos_bucket now that it's computed
    _dvp_l10_mult, _dvp_l10_label = get_dvp_rolling_l10(opp_abbr, pos_bucket, base_market)
    # [RESEARCH UPGRADE] Over-rate L10 + mean reversion signal
    _over_rate_l10, _reversion_signal, _reversion_label = compute_over_rate_and_mean_reversion(
        stat_series, float(line))
    # [v3.0] Win/Loss performance split
    _expected_win_prob = 0.5
    if _home_ml is not None and _away_ml is not None:
        try:
            _home_imp = 1.0 / float(_home_ml) if float(_home_ml) > 0 else 0.5
            _away_imp = 1.0 / float(_away_ml) if float(_away_ml) > 0 else 0.5
            _total_imp = _home_imp + _away_imp
            if _total_imp > 0:
                _home_win_prob = _home_imp / _total_imp
                _expected_win_prob = _home_win_prob if is_home_resolved else (1.0 - _home_win_prob)
        except Exception:
            pass
    _wl_factor, _wl_label, _w_avg, _l_avg = compute_win_loss_split(
        gldf_n, base_market, expected_win_prob=_expected_win_prob)
    # [v3.0] Clutch performance factor (only for close games)
    _spread_abs_val = abs(float(_game_spread)) if _game_spread is not None else None
    _clutch_factor, _clutch_label = get_clutch_performance_factor(
        player_id, base_market, spread_abs=_spread_abs_val)
    # [v3.0] FTA opponent foul rate (for FTA/FTM/Stocks/Points props)
    _fta_factor, _fta_label = get_opponent_fta_rate_factor(opp_abbr, base_market)
    # [v3.0] Playoff implications / tanking factor (March-April only)
    _playoff_factor, _playoff_label = get_playoff_implications_factor(team_abbr, game_date, base_market)
    # [v3.0] Referee crew foul tendency (FTA/FTM/Stocks/Points — crew_chief passed as None;
    # callers may inject crew_chief_name via meta dict when schedule data is available)
    _crew_chief = None
    if meta:
        _crew_chief = meta.get("crew_chief") or meta.get("referee")
    _ref_factor, _ref_label, _ref_tier = get_referee_foul_factor(_crew_chief, base_market)
    # [v3.0] Team-specific HCA factor (VSiN-sourced margins override league avg 3.0)
    _hca_factor, _hca_label = get_team_hca_factor(team_abbr, is_home_resolved, base_market)
    # [AUDIT v5.1] Player shooting luck regression (SHAP top-3 signal for scoring props)
    # When TS% L5 significantly deviates from season baseline → regression expected.
    # Uses full gamelog (20 game baseline, 5 game recent window).
    _shoot_luck_mult, _shoot_luck_label = compute_shooting_luck_regression(
        gldf, base_market, n_recent=5, n_season=20)
    # [v3.0] Position-specific B2B extra penalty (PubMed 2020: Guards -0.8% extra, Bigs minimal)
    _pos_b2b_mult = 1.0
    if b2b_flag and pos_bucket:
        _pos_b2b_extra = POSITION_B2B_EXTRA_PENALTY.get(pos_bucket, POSITION_B2B_EXTRA_PENALTY["Unknown"])
        _pos_b2b_mult = float(np.clip(1.0 - _pos_b2b_extra, 0.97, 1.0))
    # [v3.0] DNP probability score (quantified, replaces binary flag for stake scaling)
    # Get player's injury status from injury map if available
    _inj_status_player = None
    if injury_team_map and team_abbr:
        _full_inj_data = fetch_injury_report()
        _team_inj_list = _full_inj_data.get(str(team_abbr).upper(), [])
        _player_lower = (player_name or "").lower()
        for _inj_entry in _team_inj_list:
            if normalize_name(_inj_entry.get("player","")) == normalize_name(player_name):
                _inj_status_player = _inj_entry.get("status","")
                break
    _dnp_prob_score, _dnp_prob_label = compute_dnp_probability(
        gldf_n, injury_status=_inj_status_player, n_games=n_games)
    # DD / TD: short-circuit to frequency-based probability
    if is_dd_td:
        prob_fn = compute_dd_prob if market_name == "Double Double" else compute_td_prob
        dd_prob = prob_fn(gldf_n, n_games=n_games)
        p_over_raw = dd_prob
        mu_raw = dd_prob
        sigma = None
        if p_over_raw is None:
            errors.append("Insufficient history for DD/TD probability.")
    else:
        # For half markets: convert line to equivalent full-game threshold
        effective_line = float(line) / half_factor if is_half_market and half_factor > 0 else float(line)
        # [UPGRADE 4] Pass market for stat-specific λ decay
        p_over_raw, mu_raw, sigma = bootstrap_prob_over(
            pace_adj_series, effective_line, cv_override=vol_cv, market=base_market
        )
        # [v5.0] Negative Binomial blend for count stats (overdispersed integer distributions)
        # NegBin is theoretically superior for 3PM, Assists, Rebounds, Blocks, Steals.
        # We blend 65% NegBin + 35% bootstrap when conditions are met (n>=6, overdispersed).
        # This gives us better probability estimates for sharp-edge identification on count props.
        if base_market in NEGBINOM_MARKETS and len(pace_adj_series.dropna()) >= 6:
            try:
                _nb_p, _nb_mu, _nb_sigma = negbinom_prob_over(
                    pace_adj_series, effective_line, market=base_market)
                if _nb_p is not None and p_over_raw is not None:
                    # Blend: weight NegBin more heavily for larger samples (better r estimate).
                    # [AUDIT FIX] Raised cap from 0.70 → 0.82; increased slope.
                    # At n=6 (just enough), NegBin r-estimate is noisy → 50% blend.
                    # At n=15+, r-estimate is stable → up to 82% NegBin (clearly superior for counts).
                    _n_valid_nb = len(pace_adj_series.dropna())
                    # [v6.0] Raised NegBin cap from 0.82 → 0.92: NegBin dominates for count stats
                    _nb_weight = float(np.clip(0.50 + (_n_valid_nb - 6) * 0.06, 0.50, 0.92))
                    p_over_raw = float(_nb_weight * _nb_p + (1.0 - _nb_weight) * p_over_raw)
                    if _nb_mu is not None:
                        mu_raw = float(0.60 * _nb_mu + 0.40 * mu_raw)
                    if _nb_sigma is not None and sigma is not None:
                        sigma = float(0.60 * _nb_sigma + 0.40 * sigma)
                    errors.append(f"NegBin blend ({_nb_weight:.0%} NegBin | overdispersed count stat)")
                elif _nb_p is not None and p_over_raw is None:
                    p_over_raw, mu_raw, sigma = _nb_p, _nb_mu, _nb_sigma
            except Exception:
                pass
        if p_over_raw is None:
            errors.append(f"Insufficient history (need >=4 games, have {len(stat_series.dropna())})")
    # ── [v7.0] Possession-Level Simulation Ensemble Blend ─────────────
    # Run the simulator to get a full-game distribution, then blend
    # sim P(over) with bootstrap P(over) at 60%/40% (sim/bootstrap).
    # For H1/H2 markets, use the simulator's native h1_/h2_ distributions
    # which model actual rotation patterns, blowout dynamics, foul trouble,
    # and game script — far more accurate than a flat 52/48 split.
    _sim_p_over = None
    _sim_mu = None
    _sim_used = False
    if _SIM_AVAILABLE and team_abbr and opp_abbr and p_over_raw is not None:
        try:
            # Determine simulation stat key: half markets map directly to h1_/h2_ keys
            if is_half_market and _SIM_AVAILABLE:
                _sim_stat_key = SimulationDataLoader.market_to_sim_key(market_name)
            else:
                _sim_stat_key = SimulationDataLoader.market_to_sim_key(base_market)
            if _sim_stat_key is not None:
                # Serialize gamelog for Streamlit caching (requires hashable args)
                _gl_json = gldf_n.to_json() if not gldf_n.empty else ""
                _sim_spread = float(_game_spread) if _game_spread is not None else 0.0
                _sim_result = _run_player_simulation(
                    player_name=player_name,
                    player_id=str(player_id),
                    team_abbr=str(team_abbr).upper(),
                    opp_abbr=str(opp_abbr).upper(),
                    is_home=bool(is_home_resolved) if is_home_resolved is not None else True,
                    gamelog_json=_gl_json,
                    position=pos_str or "",
                    rest_days=int(rest_days),
                    n_games=n_games,
                    game_spread=_sim_spread,
                )
                # For half markets, use the sim's native half-game stat key directly
                # (models rotation patterns, blowout rest, foul trouble, game script)
                _sim_line = float(line)  # use raw line for half markets (sim outputs half-game stats)
                _sim_p = _get_sim_prob_over(_sim_result, _sim_stat_key, _sim_line)
                if _sim_p is not None:
                    _sim_p_over = _sim_p
                    _sim_mu = _sim_result.get(_sim_stat_key, {}).get("mean")
                    # Blend: 60% simulation + 40% bootstrap (sim models game dynamics;
                    # bootstrap captures recent form and stat-specific variance)
                    _bootstrap_p = p_over_raw
                    p_over_raw = float(
                        _SIM_BLEND_WEIGHT * _sim_p_over
                        + _BOOTSTRAP_BLEND_WEIGHT * _bootstrap_p
                    )
                    # Also blend the projection mean if sim mean available
                    if _sim_mu is not None and mu_raw is not None:
                        mu_raw = float(
                            _SIM_BLEND_WEIGHT * _sim_mu
                            + _BOOTSTRAP_BLEND_WEIGHT * mu_raw
                        )
                    _sim_used = True
                    errors.append(
                        f"Sim ensemble: {_SIM_BLEND_WEIGHT:.0%} sim ({_SIM_N_SIMS} runs) "
                        f"+ {_BOOTSTRAP_BLEND_WEIGHT:.0%} bootstrap"
                        + (f" [native {_sim_stat_key}]" if is_half_market else "")
                    )
        except Exception:
            pass  # Simulation failure is non-fatal; fall back to bootstrap-only
    n_valid = int(stat_series.dropna().count())
    # [AUDIT FIX] Half-market with real boxscore data: scale positional prior by _orig_half_factor
    # so shrinkage operates in half-game units (not full-game units)
    if is_half_market and _orig_half_factor != half_factor and mu_raw is not None:
        orig_prior_val = POSITIONAL_PRIORS.get(pos_bucket, POSITIONAL_PRIORS["Unknown"]).get(base_market)
        if orig_prior_val is not None:
            k = max(2.0, 8.0 / (1.0 + math.log1p(max(n_valid, 1) / 5.0)))
            w_p = k / (k + max(n_valid, 1))
            mu_shrunk = float(w_p * orig_prior_val * _orig_half_factor + (1.0 - w_p) * mu_raw)
        else:
            mu_shrunk = mu_raw  # No prior for this market: use observed directly
    else:
        # [v5.0] Pass custom priors if user has calibrated personal priors from history
        _custom_priors_session = st.session_state.get("_custom_priors")
        mu_shrunk = bayesian_shrink(mu_raw, n_valid, base_market, pos_bucket,
                                    custom_priors=_custom_priors_session) if mu_raw is not None else None
    # [AUDIT IMPROVEMENT] Opponent True Shooting % efficiency factor (SHAP top-signal for Points props)
    _ts_factor, _ts_label = get_opponent_ts_pct_factor(opp_abbr, base_market)
    # [RESEARCH IMPROVEMENT] Opponent eFG% factor for 3PM/FGM props (distinct from TS%)
    # TS% → scoring/FTA markets; eFG% → pure shooting volume markets (no double-counting)
    _efg_factor, _efg_label = get_opponent_efg_factor(opp_abbr, base_market)
    # [AUDIT IMPROVEMENT] Usage trend: L5 vs L20 usage (Wally Pipp / role-change signal)
    _usage_trend_mult, _usage_trend_label, _l5_usage, _l20_usage = compute_usage_trend(
        gldf_n, n_recent=5, n_baseline=20, market=base_market)
    # [AUDIT IMPROVEMENT] Player FTr trend: foul-drawing rate vs baseline (L5 vs L20)
    _ftr_factor, _ftr_label = compute_player_ftr_factor(gldf_n, base_market, n_recent=5, n_baseline=20)
    # [AUDIT v5.1] Per-minute production scaling: if proj_minutes significantly differs
    # from historical average, scale projection proportionally via per-minute rate.
    # This is the primary signal for injury-replacement props (CMU/EVAnalytics research).
    # Only applied when: per_min_rate is valid, proj_minutes known, minutes deviate >5%.
    _minutes_prod_mult = 1.0
    _minutes_prod_label = "Normal"
    if (_per_min_rate is not None and _per_min_rate > 0
            and proj_minutes is not None and _hist_avg_minutes is not None
            and _hist_avg_minutes > 0 and not is_dd_td):
        # Ratio of projected to historical avg minutes
        _min_ratio = float(proj_minutes) / float(_hist_avg_minutes)
        # Only adjust if deviation is meaningful (>5% change in minutes)
        if abs(_min_ratio - 1.0) > 0.05:
            # Dampen: full minutes change → 80% production change (not 100%)
            # because per-minute efficiency also fluctuates when minutes change
            _dampened_ratio = 1.0 + (_min_ratio - 1.0) * 0.80
            _minutes_prod_mult = float(np.clip(_dampened_ratio, 0.70, 1.30))
            _dir = "+" if _min_ratio > 1.0 else ""
            _minutes_prod_label = f"Min Adj: proj {proj_minutes:.0f} vs hist {_hist_avg_minutes:.0f} ({_dir}{(_min_ratio-1)*100:.0f}%)"
    # Apply half factor and positional D to projection — include opp-specific factor + all new multipliers
    # [v3.0] Added: wl_factor, clutch_factor, fta_factor, playoff_factor, ref_factor, hca_factor, pos_b2b_mult
    # [AUDIT] Added: ts_factor (opp TS%), efg_factor (opp eFG%), usage_trend_mult (Wally Pipp), ftr_factor
    # [v5.0] Game-total pace multiplier (Voulgaris method: game O/U is primary scoring env signal)
    _game_total_pace_mult = compute_game_total_pace_mult(_game_total, base_market)
    # [AUDIT] Altitude factor: Denver Ball Arena (5,280 ft) visiting-team penalty / DEN away boost
    _altitude_is_home = bool(is_home_resolved) if is_home_resolved is not None else True
    _altitude_mult = compute_altitude_factor(team_abbr, opp_abbr, _altitude_is_home, base_market)
    # [AUDIT UPGRADE] Log-additive dampening for 20+ multiplicative adjustments.
    # Pure multiplicative compounding with 20 factors can produce extreme projections
    # (e.g., 5 factors each at +8% → 1.08^5 = 1.47x, a 47% uplift on the base).
    # Fix: compute combined multiplier in log-space then hard-cap at ±30%.
    # This preserves the direction and relative ranking of all signals while
    # preventing any single or joint compounding from dominating the projection.
    if mu_shrunk is not None:
        _all_mults = [
            ctx_mult, rest_mult, ha_mult, pos_def_mult, opp_specific_factor,
            _fatigue_mult, _opp_fatigue_mult, _game_script_mult,
            _both_b2b_mult, _travel_mult, _dvp_l10_mult,
            _wl_factor, _clutch_factor, _fta_factor, _playoff_factor,
            _ref_factor, _hca_factor, _pos_b2b_mult,
            _ts_factor, _efg_factor, _usage_trend_mult, _ftr_factor,
            _game_total_pace_mult, _altitude_mult, _shoot_luck_mult,
            _minutes_prod_mult,
        ]
        # Sum log-adjustments (log(m) ≈ m-1 for small m; exact for any m)
        _log_combined = sum(math.log(max(float(m), 1e-6)) for m in _all_mults)
        # Cap total combined adjustment at ±30% (research-validated: individual player
        # context factors rarely exceed 25-30% total across all signals)
        # [v6.0] Tightened cap from ±30%/35% → ±25% to prevent projection overfit
        _log_combined = float(np.clip(_log_combined, math.log(0.75), math.log(1.25)))
        _combined_mult = math.exp(_log_combined)
        proj_full = float(mu_shrunk * _combined_mult)
    else:
        proj_full = None
    proj = proj_full * half_factor if (proj_full is not None and is_half_market) else proj_full
    # [v3.0] Confidence interval for the projection
    _ci_lower, _ci_upper = compute_projection_ci(mu_shrunk, sigma, half_factor=half_factor)
    regime_label, regime_score = classify_regime(vol_cv, blowout_prob, ctx_mult)
    price_decimal = None
    try:
        if meta and meta.get("price") is not None:
            price_decimal = float(meta.get("price"))
        elif not meta:
            price_decimal = 1.909  # Manual line: assume standard -110 so EV/gate compute properly
    except Exception: pass
    p_implied = implied_prob_from_decimal(price_decimal)
    # Determine side early for skewness gate
    side_str = (meta.get("side") if meta else "Over") or "Over"
    # [UPGRADE 6] Over/Under asymmetry: model the correct side independently
    _is_under = "under" in str(side_str).lower()
    p_model = (1.0 - p_over_raw if _is_under and p_over_raw is not None else p_over_raw)
    sharp = book_sharpness(meta.get("book") if meta else None)
    w_model = float(market_prior_weight)
    w_eff = float(np.clip(w_model*(1.0-0.60*sharp)+0.15, 0.10, 0.95))
    if p_model is not None and p_implied is not None:
        p_raw = float(np.clip(w_eff*p_model + (1.0-w_eff)*p_implied, 1e-4, 1-1e-4))
    else:
        p_raw = p_model
    p_cal = p_raw
    # [AUDIT FIX] Mean reversion nudge — stored but NOT yet applied here.
    # The streak signal (applied later at line ~5475) also uses recent-game divergence.
    # Applying BOTH would double-count the regression signal (±0.06 instead of ±0.04).
    # Fix: take the STRONGER of the two signals, applied in one place (streak block below).
    # _reversion_signal is preserved for the output dict; streak block does the combined apply.
    _combined_streak_reversion_nudge = 0.0
    if p_cal is not None:
        _is_over_side = "under" not in str(side_str).lower()
        _rev_nudge = (_reversion_signal if _is_over_side else -_reversion_signal) if _reversion_signal != 0.0 else 0.0
        _combined_streak_reversion_nudge = _rev_nudge   # will be combined with streak nudge below
    ev_raw = ev_per_dollar(p_cal, price_decimal) if (p_cal is not None and price_decimal is not None) else None
    pen = volatility_penalty_factor(vol_cv)
    # [AUDIT FIX] Asymmetric vol penalty: adjust based on skew-side alignment
    # When skew favors our bet direction (tail points our way), volatility is less harmful → soften penalty.
    # When skew opposes us (tail points against us), volatility is more harmful → tighten penalty.
    if stat_skew is not None and vol_cv is not None and float(vol_cv) > 0.20:
        _sk = float(stat_skew)
        _is_under_pen = "under" in str(side_str).lower()
        _skew_helps = (_sk < -0.4 and _is_under_pen) or (_sk > 0.4 and not _is_under_pen)
        _skew_hurts = (_sk < -0.4 and not _is_under_pen) or (_sk > 0.4 and _is_under_pen)
        if _skew_helps:
            pen = min(1.0, pen * 1.12)   # Skew aligned with our side: soften penalty by up to 12%
        elif _skew_hurts:
            pen = pen * 0.88             # Skew opposed to our side: tighten penalty by 12%
    ev_adj = float(ev_raw * pen) if ev_raw is not None else None
    # [FIX 5] Pass skewness to volatility gate
    gate_ok, gate_reason = passes_volatility_gate(vol_cv, ev_raw, skew=stat_skew, bet_type=side_str)
    if exclude_chaotic and regime_label=="Chaotic":
        # [AUDIT FIX] Preserve original gate_reason alongside chaotic reason for diagnostic clarity
        _prev_reason = gate_reason if not gate_ok else ""
        _chaotic_reason = "chaotic regime (high volatility + blowout risk)"
        gate_reason = f"{_prev_reason} + {_chaotic_reason}" if _prev_reason else _chaotic_reason
        gate_ok = False
    if not gate_ok: ev_adj = None
    stake_dollars, stake_frac, stake_reason = 0.0, 0.0, "gated"
    # [AUDIT FIX] Minimum EV threshold: noise below 2% is not worth staking
    # [AUDIT FIX 2] Enforce daily/weekly loss stops before computing stake
    _uid = st.session_state.get("_auth_user", "")
    _stop_hit, _stop_msg = is_loss_stop_active(_uid, bankroll)
    if _stop_hit:
        gate_ok = False
        gate_reason = _stop_msg
    # [v6.0] Raised minimum EV from 2% → 5%: only stake on bets with clear edge
    if gate_ok and p_cal is not None and price_decimal is not None and ev_adj is not None and ev_adj >= 0.05:
        stake_dollars, stake_frac, stake_reason = recommended_stake(
            bankroll, float(p_cal), float(price_decimal), frac_kelly, max_risk_frac)
        # [v3.0] DNP risk: proportional scaling using probability score instead of binary half/zero
        if stake_dollars > 0 and _dnp_prob_score > 0.05:
            if _dnp_prob_score >= 0.65:
                stake_dollars = 0.0
                stake_frac    = 0.0
                stake_reason  = "dnp_critical_risk"
            elif _dnp_prob_score >= 0.40:
                _dnp_scale = 1.0 - _dnp_prob_score  # e.g. 0.60 prob → scale to 0.40
                stake_dollars *= _dnp_scale
                stake_frac    *= _dnp_scale
                stake_reason  = f"dnp_prob_{_dnp_prob_score:.0%}"
            elif _dnp_prob_score >= 0.20 or (dnp_risk and float(proj_minutes or 30) < 5.0):
                stake_dollars *= 0.50
                stake_frac    *= 0.50
                stake_reason  = "dnp_risk_half"
    mk_key = meta.get("market_key") if meta else ODDS_MARKETS.get(market_name,"")
    player_norm = normalize_name(player_name)
    mv_signal = get_line_movement_signal(player_norm, str(mk_key), float(line), side_str)
    sharp_div = {}
    if meta and meta.get("event_id"):
        try:
            sharp_div = sharp_divergence_alert(meta["event_id"], mk_key, player_norm, side_str, side_str) or {}
        except Exception: sharp_div = {}
    # [UPGRADE NEW] L3/L5/L10 trend convergence
    _trend_score, _trend_label, _trend_slope, _l3, _l5, _l10 = compute_trend_convergence(
        stat_series, float(line), side=side_str)
    # [v4.0 UPGRADE] Consecutive streak signal — combined with reversion nudge in single apply.
    # [AUDIT FIX] Using combined nudge prevents double-counting from reversion + streak signals.
    # Only the STRONGER of the two signals contributes beyond the base; capped at ±0.045 total.
    _streak_count, _streak_label, _streak_signal = compute_consecutive_streak(
        stat_series, float(line), side=side_str)
    # Combine: streak and reversion directional agreement → additive up to cap
    #           streak and reversion in opposition → take the stronger one
    _str_same_dir = (_streak_signal >= 0) == (_combined_streak_reversion_nudge >= 0) or _streak_signal == 0
    if _str_same_dir:
        # Same direction: cap additive combination at ±0.045
        _final_nudge = float(np.clip(_combined_streak_reversion_nudge + _streak_signal * 0.40, -0.045, 0.045))
    else:
        # Opposite directions: use the stronger signal only
        _final_nudge = _combined_streak_reversion_nudge if abs(_combined_streak_reversion_nudge) >= abs(_streak_signal) else _streak_signal
    if _final_nudge != 0.0 and p_cal is not None:
        p_cal = float(np.clip(float(p_cal) + _final_nudge, 1e-4, 1 - 1e-4))
        # Recompute EV after combined nudge — this ev_adj is used for stake computation
        ev_raw = ev_per_dollar(p_cal, price_decimal) if (p_cal is not None and price_decimal is not None) else None
        ev_adj = float(ev_raw * pen) if (ev_raw is not None and gate_ok) else None
    # [UPGRADE NEW] Composite sharpness score
    _sharpness, _sharpness_components = compute_composite_sharpness(
        ev_adj=ev_adj, p_cal=p_cal, p_implied=p_implied,
        hot_cold=hot_cold_label, mv_signal=mv_signal,
        sharp_div=sharp_div, regime=regime_label,
        trend_score=_trend_score, vol_cv=vol_cv,
        dnp_risk=bool(dnp_risk), b2b=b2b_flag,
        fatigue_label=_fatigue_label, game_total=_game_total,
        # [v3.0] New signals fed into sharpness
        clutch_label=_clutch_label, playoff_label=_playoff_label,
        wl_factor=_wl_factor, dnp_prob=_dnp_prob_score,
    )
    _sharpness_tier_label, _sharpness_tier_color = sharpness_tier(_sharpness)
    # [v5.0] Recompute stake with sharpness-aware Kelly now that _sharpness is known.
    # This replaces the initial stake (which had no sharpness info) with a more precise allocation.
    # Elite bets get up to 20% more; low-signal bets get 40%-75% of base Kelly fraction.
    # [v6.0] Raised minimum EV from 2% → 5%: only stake on bets with clear edge
    if gate_ok and p_cal is not None and price_decimal is not None and ev_adj is not None and ev_adj >= 0.05:
        stake_dollars, stake_frac, stake_reason = recommended_stake(
            bankroll, float(p_cal), float(price_decimal), frac_kelly, max_risk_frac,
            sharpness_score=_sharpness)
        if stake_dollars > 0 and _dnp_prob_score > 0.05:
            if _dnp_prob_score >= 0.65:
                stake_dollars, stake_frac, stake_reason = 0.0, 0.0, "dnp_critical_risk"
            elif _dnp_prob_score >= 0.40:
                _dnp_scale = 1.0 - _dnp_prob_score
                stake_dollars *= _dnp_scale; stake_frac *= _dnp_scale
                stake_reason = f"dnp_prob_{_dnp_prob_score:.0%}"
            elif _dnp_prob_score >= 0.20 or (dnp_risk and float(proj_minutes or 30) < 5.0):
                stake_dollars *= 0.50; stake_frac *= 0.50
                stake_reason = "dnp_risk_half"
    # [v3.0] Alt line EV table and middle detection
    _alt_line_evs = compute_alt_line_ev(p_cal, price_decimal, line, sigma=sigma)
    _middle_exists, _middle_low, _middle_high, _middle_prob, _middle_books = (
        detect_middle_opportunity(player_name, mk_key, meta.get("event_id") if meta else None)
        if meta and meta.get("event_id") else (False, None, None, None, [])
    )
    return {
        "player":           player_name,
        "player_norm":      player_norm,
        "player_id":        player_id,
        "market":           market_name,
        "line":             float(line),
        "proj":             float(proj) if proj is not None else None,
        "proj_vs_line":     float(proj - line) if proj is not None else None,
        "p_over":           float(p_raw) if p_raw is not None else None,
        "p_raw":            float(p_raw) if p_raw is not None else None,
        "p_model":          float(p_model) if p_model is not None else None,
        "p_cal":            float(p_cal) if p_cal is not None else None,
        "p_implied":        float(p_implied) if p_implied is not None else None,
        "advantage":        float(p_cal - p_implied) if (p_cal and p_implied) else None,
        "price_decimal":    float(price_decimal) if price_decimal is not None else None,
        "book":             meta.get("book") if meta else None,
        "event_id":         meta.get("event_id") if meta else None,
        "market_key":       meta.get("market_key") if meta else None,
        "side":             side_str,
        "commence_time":    meta.get("commence_time") if meta else None,
        "regime":           regime_label,
        "regime_score":     float(regime_score),
        "ev_raw":           float(ev_raw) if ev_raw is not None else None,
        "ev_adj":           float(ev_adj) if ev_adj is not None else None,
        "ev_pct":           float(ev_adj*100) if ev_adj is not None else None,
        "stake":            float(stake_dollars),
        "stake_frac":       float(stake_frac),
        "stake_reason":     stake_reason,
        "vol_penalty":      float(pen),
        "gate_ok":          bool(gate_ok),
        "gate_reason":      gate_reason,
        "edge":             float(ev_adj) if ev_adj is not None else None,
        "edge_cat":         classify_edge(ev_adj),
        "team":             team_abbr,
        "opp":              opp_abbr,
        "is_home":          is_home_resolved,
        "headshot":         nba_headshot_url(player_id),
        "blowout_prob":     float(blowout_prob),
        "context_mult":     float(ctx_mult),
        "rest_mult":        float(rest_mult),
        "rest_days":        int(rest_days),
        "ha_mult":          float(ha_mult),
        "volatility_cv":    vol_cv,
        "volatility_label": vol_label,
        "stat_skewness":    stat_skew,
        "position":         pos_str,
        "position_bucket":  pos_bucket,
        "n_games_used":     n_valid,
        "mu_raw":           float(mu_raw) if mu_raw is not None else None,
        "mu_shrunk":        float(mu_shrunk) if mu_shrunk is not None else None,
        "sigma":            float(sigma) if sigma is not None else None,
        "negbinom_used":    base_market in NEGBINOM_MARKETS and len(pace_adj_series.dropna()) >= 6,
        "line_movement":    mv_signal,
        "sharp_div":        sharp_div,
        "usage_rate":       float(usage_rate) if usage_rate is not None else None,
        "pos_def_mult":     float(pos_def_mult),
        "half_factor":      float(half_factor),
        "pace_adj":         True if opp_abbr and TEAM_CTX.get(str(opp_abbr).upper()) else False,
        "auto_inj":          auto_inj_triggered,
        "auto_inj_player":   auto_inj_player,
        "lineup_boost":      float(_usage_boost),
        "lineup_label":      _lineup_label,
        "key_teammate_out":  key_teammate_out,
        # [UPGRADE 1] Explicit B2B flag
        "b2b":               b2b_flag,
        # [UPGRADE 2] Projected minutes
        "proj_minutes":      float(proj_minutes) if proj_minutes is not None else None,
        "dnp_risk":          bool(dnp_risk),
        # [UPGRADE 3] Opponent-specific factor
        "opp_specific_factor": float(opp_specific_factor),
        "n_vs_opp":          int(n_vs_opp),
        # [UPGRADE 5] Hot/cold regime
        "hot_cold":          hot_cold_label,
        "hot_cold_z":        float(hot_cold_z),
        # [UPGRADE NEW] Schedule fatigue
        "schedule_fatigue":  _fatigue_mult,
        "fatigue_label":     _fatigue_label,
        # [UPGRADE NEW] Opponent fatigue
        "opp_fatigue_mult":  _opp_fatigue_mult,
        "opp_fatigue_label": _opp_fatigue_label,
        # [UPGRADE NEW] Game total / spread / game-script
        "game_total":        float(_game_total) if _game_total is not None else None,
        "game_spread":       float(_game_spread) if _game_spread is not None else None,
        "game_script_mult":  float(_game_script_mult),
        # [UPGRADE NEW] Trend convergence
        "trend_score":       float(_trend_score),
        "trend_label":       _trend_label,
        "trend_slope":       float(_trend_slope),
        "l3_avg":            float(_l3) if _l3 is not None else None,
        "l5_avg":            float(_l5) if _l5 is not None else None,
        "l10_avg":           float(_l10) if _l10 is not None else None,
        # [UPGRADE NEW] Composite sharpness score
        "sharpness_score":   float(_sharpness),
        "sharpness_tier":    _sharpness_tier_label,
        "sharpness_color":   _sharpness_tier_color,
        "sharpness_components": _sharpness_components,
        # [RESEARCH UPGRADE] Rolling minutes fatigue (CMU most-predictive signal)
        "rolling_min_avg":   float(_rolling_min_avg) if _rolling_min_avg is not None else None,
        "rolling_min_label": _rolling_min_label,
        # [RESEARCH UPGRADE] Both-teams-B2B
        "both_b2b":          bool(_both_b2b),
        "both_b2b_mult":     float(_both_b2b_mult),
        # [RESEARCH UPGRADE] Travel fatigue
        "travel_mult":       float(_travel_mult),
        "travel_label":      _travel_label,
        # [RESEARCH UPGRADE] L10 rolling DvP
        "dvp_l10_mult":      float(_dvp_l10_mult),
        "dvp_l10_label":     _dvp_l10_label,
        # [RESEARCH UPGRADE] Over-rate L10 + mean reversion
        "over_rate_l10":     float(_over_rate_l10) if _over_rate_l10 is not None else None,
        "reversion_label":   _reversion_label,
        # [v4.0] Consecutive streak signal
        "streak_count":      int(_streak_count),
        "streak_label":      _streak_label,
        "streak_signal":     float(_streak_signal),
        # [v3.0] Win/Loss split
        "wl_factor":         float(_wl_factor),
        "wl_label":          _wl_label,
        "w_avg":             float(_w_avg) if _w_avg is not None else None,
        "l_avg":             float(_l_avg) if _l_avg is not None else None,
        "expected_win_prob": float(_expected_win_prob),
        # [v3.0] Clutch performance
        "clutch_factor":     float(_clutch_factor),
        "clutch_label":      _clutch_label,
        # [v3.0] FTA opponent foul rate
        "fta_factor":        float(_fta_factor),
        "fta_label":         _fta_label,
        # [AUDIT IMPROVEMENT] Opponent TS% efficiency factor
        "ts_factor":         float(_ts_factor),
        "ts_label":          _ts_label,
        # [RESEARCH IMPROVEMENT] Opponent eFG% factor for 3PM/FGM props
        "efg_factor":        float(_efg_factor),
        "efg_label":         _efg_label,
        # [AUDIT IMPROVEMENT] Usage trend (Wally Pipp effect: L5 vs L20 usage)
        "usage_trend_mult":  float(_usage_trend_mult),
        "usage_trend_label": _usage_trend_label,
        "l5_usage":          float(_l5_usage) if _l5_usage is not None else None,
        "l20_usage":         float(_l20_usage) if _l20_usage is not None else None,
        # [AUDIT IMPROVEMENT] Player FTr trend (foul-drawing rate vs baseline)
        "ftr_factor":        float(_ftr_factor),
        "ftr_label":         _ftr_label,
        # [v3.0] Playoff implications
        "playoff_factor":    float(_playoff_factor),
        "playoff_label":     _playoff_label,
        # [v3.0] Quantified DNP probability
        "dnp_prob_score":    float(_dnp_prob_score),
        "dnp_prob_label":    _dnp_prob_label,
        # [v3.0] Projection confidence interval (80% CI)
        "ci_lower_80":       _ci_lower,
        "ci_upper_80":       _ci_upper,
        # [v3.0] Alt line EV table
        "alt_line_evs":      _alt_line_evs,
        # [v3.0] Middle opportunity
        "middle_exists":     bool(_middle_exists),
        "middle_low":        float(_middle_low) if _middle_low is not None else None,
        "middle_high":       float(_middle_high) if _middle_high is not None else None,
        "middle_prob":       float(_middle_prob) if _middle_prob is not None else None,
        "middle_books":      _middle_books,
        # [v3.0] Referee / HCA / position B2B factors
        "ref_factor":        float(_ref_factor),
        "ref_label":         _ref_label,
        "ref_tier":          _ref_tier,
        "hca_factor":        float(_hca_factor),
        "hca_label":         _hca_label,
        "pos_b2b_mult":      float(_pos_b2b_mult),
        # [AUDIT] Altitude adjustment (Denver Ball Arena 5,280 ft)
        "altitude_mult":     float(_altitude_mult),
        # [AUDIT v5.1] Player shooting luck regression (TS% L5 vs season)
        "shoot_luck_mult":   float(_shoot_luck_mult),
        "shoot_luck_label":  _shoot_luck_label,
        # [AUDIT v5.1] Minutes-based production scaling (injury-replacement primary signal)
        "minutes_prod_mult": float(_minutes_prod_mult),
        "minutes_prod_label": _minutes_prod_label,
        "hist_avg_minutes":  float(_hist_avg_minutes) if _hist_avg_minutes is not None else None,
        # [v7.0] Possession-level simulation ensemble
        "sim_used":          _sim_used,
        "sim_p_over":        float(_sim_p_over) if _sim_p_over is not None else None,
        "sim_mu":            float(_sim_mu) if _sim_mu is not None else None,
        "sim_blend_weight":  float(_SIM_BLEND_WEIGHT) if _sim_used else None,
        "sim_n_sims":        int(_SIM_N_SIMS) if _sim_used else None,
        "errors":            errors,
    }
# ──────────────────────────────────────────────
# CALIBRATION ENGINE  [FIX 9: training range + OOD]
# ──────────────────────────────────────────────
def _expand_history_legs(history_df):
    if history_df is None or history_df.empty: return pd.DataFrame()
    rows = []
    for _, r in history_df.iterrows():
        bet_res = str(r.get("result","Pending"))
        # [FIX 11] Include PASS decisions for calibration analysis (no outcome)
        if bet_res not in ("HIT","MISS","PUSH","SKIP"): continue
        try:
            legs = json.loads(r.get("legs","[]")) if isinstance(r.get("legs"),str) else (r.get("legs") or [])
        except: legs = []
        # Per-leg results: if stored use them; otherwise fall back to parent result only for single-leg bets
        try:
            leg_results_list = json.loads(r.get("leg_results","[]")) if isinstance(r.get("leg_results"),str) else []
            if not isinstance(leg_results_list, list): leg_results_list = []
        except: leg_results_list = []
        n_legs = len(legs)
        for i, leg in enumerate(legs):
            if not isinstance(leg,dict): continue
            # Determine this leg's individual result
            if i < len(leg_results_list) and leg_results_list[i] in ("HIT","MISS","PUSH"):
                leg_res = leg_results_list[i]
            elif n_legs == 1 and bet_res in ("HIT","MISS","PUSH"):
                # Single-leg bet: parent result is the leg result
                leg_res = bet_res
            else:
                # Multi-leg without individual results logged — skip for calibration
                # to avoid incorrectly assigning the parlay outcome to all legs
                leg_res = "UNKNOWN"
            row = {
                "ts":r.get("ts"),"market":leg.get("market"),"player":leg.get("player"),
                "p_raw":safe_float(leg.get("p_raw") or leg.get("p_over"), default=np.nan),
                "price_decimal":safe_float(leg.get("price_decimal"), default=np.nan),
                "cv":safe_float(leg.get("volatility_cv"), default=np.nan),
                "ev_adj":safe_float(leg.get("ev_adj"), default=np.nan),
                "result":leg_res,
                "bet_result":bet_res,
                "decision":str(r.get("decision","BET")),
                "clv_line_fav":leg.get("clv_line_fav"),
                "clv_price_fav":leg.get("clv_price_fav"),
            }
            if leg_res in ("HIT","MISS","PUSH"):
                row["y"] = 1.0 if leg_res=="HIT" else 0.0
            elif bet_res == "SKIP":
                row["y"] = np.nan  # PASS has no outcome
            else:
                row["y"] = np.nan  # multi-leg leg with no individual result
            rows.append(row)
    df = pd.DataFrame(rows)
    return df[pd.to_numeric(df["p_raw"],errors="coerce").notna()].copy() if not df.empty else df
# [FIX 9] Store training range in calibrator
def fit_monotone_calibrator(df_legs, n_bins=12):
    if df_legs is None or df_legs.empty: return None
    # Only use settled legs (with outcomes) for fitting
    d = df_legs[df_legs["y"].notna()].copy()
    d = d[(d["p_raw"]>=0.01)&(d["p_raw"]<=0.99)]
    # [FIX v4.0] Reduced minimum from 80→40: with ≥40 samples and isotonic regression,
    # calibration is statistically meaningful. 80 required ~2+ months of active betting,
    # delaying calibration benefits too long. Use fewer bins for small samples.
    if len(d) < 40: return None
    _actual_bins = n_bins if len(d) >= 80 else max(4, n_bins // 2)  # fewer bins for small N
    d["bin"] = pd.cut(d["p_raw"], bins=_actual_bins, labels=False, include_lowest=True)
    _min_n_per_bin = 3 if len(d) < 80 else 5   # fewer required samples per bin for small N
    g = d.groupby("bin",dropna=True).agg(p_mid=("p_raw","mean"),win=("y","mean"),n=("y","size")).reset_index()
    g = g[g["n"]>=_min_n_per_bin].sort_values("p_mid")
    if g.empty or len(g)<3: return None
    # [AUDIT FIX] Use sklearn IsotonicRegression (PAVA) instead of manual maximum.accumulate.
    # maximum.accumulate creates plateau artifacts (e.g., p_raw=0.4→win=0.45, p_raw=0.5→win=0.45).
    # Isotonic regression (PAVA) is smoother and statistically principled.
    try:
        from sklearn.isotonic import IsotonicRegression
        _ir = IsotonicRegression(out_of_bounds="clip")
        win_mono = _ir.fit_transform(g["p_mid"].values.astype(float), g["win"].values.astype(float))
    except Exception:
        win_mono = np.maximum.accumulate(g["win"].values.astype(float))
    win_mono = np.clip(win_mono, 0.01, 0.99)
    return {
        "x":g["p_mid"].values.astype(float).tolist(),
        "y":win_mono.tolist(),
        "n":int(len(d)),
        "training_min": float(d["p_raw"].min()),
        "training_max": float(d["p_raw"].max()),
    }
# [FIX 9] OOD detection in calibrator
def apply_calibrator(p_raw, calib):
    if p_raw is None: return None
    try: p = float(p_raw)
    except: return None
    if calib is None: return float(np.clip(p, 0.0, 1.0))
    xs = calib.get("x") or []; ys = calib.get("y") or []
    if len(xs)<2 or len(xs)!=len(ys): return float(np.clip(p, 0.0, 1.0))
    try:
        t_min = calib.get("training_min", 0.0)
        t_max = calib.get("training_max", 1.0)
        # [AUDIT FIX] OOD: return raw p (no extrapolation) instead of silently extrapolating
        # [v6.0] Tightened OOD range from 85-115% → 92-108% for conservative calibration
        if p < t_min * 0.92 or p > t_max * 1.08:
            return float(np.clip(p, 0.0, 1.0))
        return float(np.clip(np.interp(p, xs, ys), 0.0, 1.0))
    except: return float(np.clip(p, 0.0, 1.0))
def recompute_pricing_fields(leg, calib):
    p_raw = leg.get("p_raw")
    p_cal = apply_calibrator(p_raw, calib)
    leg["p_cal"] = p_cal
    price = leg.get("price_decimal")
    p_imp = implied_prob_from_decimal(price) if price is not None else None
    leg["p_implied"] = p_imp
    leg["advantage"] = float(p_cal-p_imp) if (p_cal and p_imp) else None
    ev_raw = ev_per_dollar(p_cal, price) if (p_cal and price) else None
    leg["ev_raw"] = ev_raw
    pen = volatility_penalty_factor(leg.get("volatility_cv"))
    leg["vol_penalty"] = pen
    side_str = leg.get("side", "Over") or "Over"
    gate_ok, gate_reason = passes_volatility_gate(
        leg.get("volatility_cv"), ev_raw,
        skew=leg.get("stat_skewness"), bet_type=side_str)
    regime_label, regime_score = classify_regime(leg.get("volatility_cv"),leg.get("blowout_prob"),leg.get("context_mult"))
    leg["regime"]=regime_label; leg["regime_score"]=float(regime_score)
    if bool(st.session_state.get("exclude_chaotic",True)) and regime_label=="Chaotic":
        gate_ok, gate_reason = False, "chaotic regime"
    leg["gate_ok"]=gate_ok; leg["gate_reason"]=gate_reason
    leg["ev_adj"] = (ev_raw*pen) if (ev_raw is not None and gate_ok) else None
    leg["ev_pct"] = float(leg["ev_adj"]*100) if leg["ev_adj"] is not None else None
    leg["edge"] = leg["ev_adj"]
    leg["edge_cat"] = classify_edge(leg["ev_adj"])
    bankroll = float(st.session_state.get("bankroll",0.0) or 0)
    frac_k = float(st.session_state.get("frac_kelly",0.25) or 0.25)
    cap_frac = float(st.session_state.get("max_risk_per_bet",5.0) or 5.0)/100.0
    uid = st.session_state.get("_auth_user", "")
    _stop_active, _stop_reason = is_loss_stop_active(uid, bankroll)
    if _stop_active:
        gate_ok = False
        gate_reason = _stop_reason
        leg["gate_ok"] = False
        leg["gate_reason"] = _stop_reason
    if gate_ok and p_cal and price and leg.get("ev_adj") and float(leg["ev_adj"])>0 and bankroll>0:
        sd, sf, sr = recommended_stake(bankroll, float(p_cal), float(price), frac_k, cap_frac)
        leg["stake"]=float(sd); leg["stake_frac"]=float(sf); leg["stake_reason"]=sr
    else:
        leg["stake"]=float(leg.get("stake",0) or 0)
        leg["stake_frac"]=float(leg.get("stake_frac",0) or 0)
        leg["stake_reason"]=leg.get("stake_reason") or "gated"
    return leg
# ──────────────────────────────────────────────
# HISTORY PERSISTENCE  [FIX 15: no expiry]
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# USER AUTHENTICATION SYSTEM
# ──────────────────────────────────────────────
AUTH_DB_PATH = "users_auth.json"
def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.strip().encode()).hexdigest()
def _load_auth_db() -> dict:
    try:
        if os.path.exists(AUTH_DB_PATH):
            with open(AUTH_DB_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}
def _save_auth_db(db: dict):
    try:
        with open(AUTH_DB_PATH, "w") as f:
            json.dump(db, f, indent=2)
    except Exception:
        pass
def _register_user(username: str, password: str, email: str = "") -> tuple:
    username = username.strip().lower()
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r'^[a-z0-9_]+$', username):
        return False, "Username: only letters, numbers, underscores allowed."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."
    db = _load_auth_db()
    if username in db:
        return False, "Username already taken. Please choose another."
    db[username] = {
        "pw_hash": _hash_pw(password),
        "email": email.strip(),
        "created": datetime.utcnow().isoformat() + "Z"
    }
    _save_auth_db(db)
    return True, "Account created!"
def _authenticate(username: str, password: str) -> tuple:
    username = username.strip().lower()
    db = _load_auth_db()
    if username not in db:
        return False, "Invalid username or password."
    if db[username]["pw_hash"] != _hash_pw(password):
        return False, "Invalid username or password."
    return True, "Login successful!"
def _get_user_email(username: str) -> str:
    db = _load_auth_db()
    return db.get(username, {}).get("email", "")
def user_state_path(uid): return f"user_state_{re.sub(r'[^a-zA-Z0-9_-]','_',uid or 'default')}.json"
def history_path(uid):    return f"history_{re.sub(r'[^a-zA-Z0-9_-]','_',uid or 'default')}.csv"
# [FIX 15] No TTL on user data - persists forever on disk
def load_user_state(uid):
    fp = user_state_path(uid)
    try:
        if os.path.exists(fp):
            with open(fp) as f: return json.load(f) or {}
    except Exception: pass
    return {}
def save_user_state(uid, state):
    try:
        with open(user_state_path(uid),"w") as f: json.dump(state or {}, f)
    except Exception: pass
def load_history(uid):
    fp = history_path(uid)
    try:
        if os.path.exists(fp): return pd.read_csv(fp)
    except Exception: pass
    return pd.DataFrame()
def compute_period_loss_pct(uid, bankroll, days=1):
    """Return net loss as a fraction of bankroll over the last `days` calendar days.
    Positive return = net loss. Returns 0.0 on any error."""
    try:
        if bankroll <= 0: return 0.0
        h = load_history(uid)
        if h.empty or "date" not in h.columns: return 0.0
        cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
        recent = h[h["date"] >= cutoff] if "date" in h.columns else pd.DataFrame()
        if recent.empty: return 0.0
        pnl = 0.0
        for _, row in recent.iterrows():
            outcome = str(row.get("outcome","")).lower()
            stake = float(row.get("stake", 0) or 0)
            price = float(row.get("price_decimal", 2.0) or 2.0)
            if outcome == "win":
                pnl += stake * (price - 1.0)
            elif outcome == "loss":
                pnl -= stake
        # Return loss as positive fraction (loss_pct = 0.10 means 10% of bankroll lost)
        return max(0.0, -pnl / bankroll)
    except Exception:
        return 0.0
def is_loss_stop_active(uid, bankroll):
    """Check daily and weekly loss stops. Returns (blocked: bool, reason: str)."""
    try:
        max_daily  = float(st.session_state.get("max_daily_loss",  15) or 15) / 100.0
        max_weekly = float(st.session_state.get("max_weekly_loss", 25) or 25) / 100.0
        if max_daily > 0:
            daily_loss = compute_period_loss_pct(uid, bankroll, days=1)
            if daily_loss >= max_daily:
                return True, f"daily loss stop hit ({daily_loss*100:.1f}% >= {max_daily*100:.0f}%)"
        if max_weekly > 0:
            weekly_loss = compute_period_loss_pct(uid, bankroll, days=7)
            if weekly_loss >= max_weekly:
                return True, f"weekly loss stop hit ({weekly_loss*100:.1f}% >= {max_weekly*100:.0f}%)"
    except Exception:
        pass
    return False, ""
def append_history(uid, row):
    df = load_history(uid)
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(history_path(uid), index=False)
# ============================================================
# STREAMLIT UI
# ============================================================
