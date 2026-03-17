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
from urllib.parse import quote as _url_quote
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import requests
import streamlit as st
from streamlit_cookies_controller import CookieController
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
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
            model="claude-sonnet-4-6",
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
            model="claude-sonnet-4-6",
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
            model="claude-sonnet-4-6",
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
            model="claude-sonnet-4-6",
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
    # Minutes (PP specialty — no Odds API key; PP/UD/Sleeper source only)
    "Minutes":         "player_minutes",
}
# Markets with no confirmed Odds API key — available via PP/UD/Sleeper only.
# These will be skipped during Odds API fetches and the user will be warned.
ODDS_API_UNSUPPORTED_MARKETS = {"FGA", "3PA", "FTM", "FTA", "Minutes"}
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
    # Minutes (PP specialty — requires special parse in compute_stat_from_gamelog)
    "Minutes":         "MIN",
    # Half markets map to full-game fields (adjusted via HALF_FACTOR)
    "H1 Points":       "PTS",
    "H1 Rebounds":     "REB",
    "H1 Assists":      "AST",
    "H1 3PM":          "FG3M",
    "H1 PRA":          ("PTS","REB","AST"),
    "H1 FGM":          "FGM",
    "H1 FGA":          "FGA",
    "H1 FTM":          "FTM",
    "H1 FTA":          "FTA",
    "H2 Points":       "PTS",
    "H2 Rebounds":     "REB",
    "H2 Assists":      "AST",
    "H2 PRA":          ("PTS","REB","AST"),
    "H2 FGM":          "FGM",
    "H2 FGA":          "FGA",
    "H2 FTM":          "FTM",
    "H2 FTA":          "FTA",
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
    "betonlineag":0.45,"bovada":0.40,"mybookieag":0.30,
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
LAMBDA_DECAY_BY_STAT = {
    "Points": 0.88, "Rebounds": 0.85, "Assists": 0.88,
    "3PM": 0.84, "PRA": 0.87, "PR": 0.86, "PA": 0.87, "RA": 0.85,
    "Blocks": 0.83, "Steals": 0.84, "Turnovers": 0.86, "Stocks": 0.83,
    "H1 Points": 0.88, "H1 Rebounds": 0.85, "H1 Assists": 0.88,
    "H2 Points": 0.88, "H2 Rebounds": 0.85, "H2 Assists": 0.88,
    "Q1 Points": 0.87, "Q1 Rebounds": 0.84, "Q1 Assists": 0.87,
    "Alt Points": 0.88, "Alt Rebounds": 0.85, "Alt Assists": 0.88, "Alt 3PM": 0.84,
    "Fantasy Score": 0.88,
    "default": 0.88,
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
    # Minutes: parse "MM:SS" strings to decimal minutes
    if market == "Minutes":
        try:
            def _parse_min(v):
                v = str(v)
                if ":" in v:
                    parts = v.split(":")
                    return float(parts[0]) + float(parts[1]) / 60.0
                return safe_float(v, default=0.0)
            return df["MIN"].apply(_parse_min) if "MIN" in df.columns else pd.Series([], dtype=float)
        except Exception:
            return pd.Series([], dtype=float)
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
    # Hard cutoff at 0.42: beyond this, variance overwhelms any edge
    if v > 0.42:
        return False, "CV>0.42 (too volatile — variance overwhelms edge)"
    # 0.35–0.42 range: only pass with very strong EV (≥12%)
    if v > 0.35:
        if ev_f is None or ev_f < 0.12:
            return False, f"CV>{v:.2f} needs EV>=12% (high-variance stat)"
    if v > 0.25 and (ev_f is None or ev_f < 0.06):
        return False, "CV>0.25 needs EV>=6%"
    # [AUDIT FIX] Low-CV bets still need minimum EV — no edge means no bet
    if v > 0.15 and (ev_f is None or ev_f < 0.02):
        return False, "CV>0.15 needs EV>=2%"
    # [FIX 5] Skewness-adjusted threshold
    if skew is not None and v > 0.20:
        is_over = "over" in str(bet_type).lower()
        # Negative skew + Over bet = tail risk of low games
        if float(skew) < -0.5 and is_over:
            tightened = 0.30
            if v > tightened and (ev_f is None or ev_f < 0.08):
                return False, f"CV>{tightened:.2f} (neg-skew+Over tightened, needs EV>=8%)"
        # Positive skew + Under bet = tail risk of blow-up games
        elif float(skew) > 0.5 and not is_over:
            tightened = 0.30
            if v > tightened and (ev_f is None or ev_f < 0.08):
                return False, f"CV>{tightened:.2f} (pos-skew+Under tightened, needs EV>=8%)"
    return True, ""
# ──────────────────────────────────────────────
# FANO FACTOR VOLATILITY GATE (v5.0)
# Uses variance/mean (Fano factor) for count stats rather than CV.
# CV fails for count stats because σ² ∝ μ for Poisson/NegBin distributions.
# ──────────────────────────────────────────────
def passes_volatility_gate_v2(stat_series, ev_raw, market, skew=None, bet_type="Over"):
    """Fano-factor gate for count stats; standard CV gate for continuous stats."""
    arr = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
    if len(arr) < 4:
        return False, "insufficient data"
    if market in NEGBINOM_MARKETS:
        mu = arr.mean()
        if mu < 0.3:
            return False, f"mean too low ({mu:.1f})"
        var = arr.var(ddof=1) if len(arr) > 1 else 0.0
        fano = var / max(mu, 0.01)
        ev_f = float(ev_raw) if ev_raw is not None else None
        if fano > 3.5 and (ev_f is None or ev_f < 0.10):
            return False, f"Fano={fano:.1f} extreme, needs EV>=10%"
        if fano > 2.5 and (ev_f is None or ev_f < 0.06):
            return False, f"Fano={fano:.1f} high, needs EV>=6%"
        return True, ""
    else:
        cv = float(arr.std(ddof=1) / arr.mean()) if arr.mean() != 0 else None
        return passes_volatility_gate(cv, ev_raw, skew=skew, bet_type=bet_type)
# ──────────────────────────────────────────────
# KDE PROBABILITY ESTIMATOR (v5.0)
# Gaussian KDE for continuous stats (Points, PRA, Fantasy Score).
# Outperforms bootstrap for large samples on continuous distributions.
# ──────────────────────────────────────────────
KDE_MARKETS = frozenset({
    "Points", "PRA", "PR", "PA", "Fantasy Score",
    "H1 Points", "H2 Points", "Q1 Points",
})
def kde_prob_over(stat_series, line, market="default", min_n=8):
    """Gaussian KDE probability estimate with exponential recency weighting."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return None, None, None
    x = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
    if x.size < min_n:
        return None, None, None
    lam = LAMBDA_DECAY_BY_STAT.get(market, 0.88)
    w = np.array([lam ** i for i in range(x.size)], dtype=float)
    w /= w.sum()
    mu_w = float(np.average(x, weights=w))
    var_w = float(np.average((x - mu_w) ** 2, weights=w))
    sigma_w = max(1e-9, np.sqrt(var_w))
    n_eff = 1.0 / np.sum(w ** 2)
    bw = 1.06 * sigma_w * max(n_eff, 1.0) ** (-1 / 5)
    reps = np.maximum(np.round(w * 1000).astype(int), 1)
    x_rep = np.repeat(x, reps)
    if len(x_rep) < 5:
        return None, mu_w, sigma_w
    try:
        kde = gaussian_kde(x_rep, bw_method=max(bw / max(x_rep.std(ddof=1), 1e-6), 0.05))
        p_over = float(kde.integrate_box_1d(float(line), float(line) + 10 * sigma_w))
        return float(np.clip(p_over, 1e-4, 1 - 1e-4)), mu_w, sigma_w
    except Exception:
        return None, mu_w, sigma_w
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
    noise_scale = max(0.05, min(cv * 0.40, 0.25))
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
    if ev is None: return None
    e = float(ev)
    if e <= 0.0:   return "No Edge"
    if e < 0.04:   return "Lean Edge"
    if e < 0.08:   return "Solid Edge"
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
    # 1. Model EV (max 30 pts)  — linear from 0% → 15% EV maps to 0→30
    ev_pts = float(np.clip(float(ev_adj) / 0.15 * 30.0, 0.0, 30.0))
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
    # 7. Regime penalty (max -15 pts)
    reg_pts = 0.0 if regime == "Stable" else (-7.0 if regime == "Mixed" else -15.0)
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
    if score is None: return "SKIP", "#4A607A"
    s = float(score)
    if s >= 70: return "ELITE",  "#00FFB2"
    if s >= 55: return "SOLID",  "#00AAFF"
    if s >= 40: return "LEAN",   "#FFB800"
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
                      sharpness_score=None, ci_lower=None, ci_upper=None, line=None):
    try: br=float(bankroll)
    except: br=0.0
    if br<=0 or p is None or price_decimal is None: return 0.0, 0.0, "bankroll<=0"
    k = kelly_fraction(float(p), float(price_decimal))
    if k <= 0:
        return 0.0, 0.0, "negative edge - hard blocked"
    # [v5.0] Dynamic Kelly multiplier from sharpness score
    _base_frac = max(0.0, min(1.0, float(frac_kelly)))
    if sharpness_score is not None:
        _s = float(sharpness_score)
        if _s >= 70:
            _sharp_mult = 1.20
        elif _s >= 55:
            _sharp_mult = 1.00
        elif _s >= 40:
            _sharp_mult = 0.75
        else:
            _sharp_mult = 0.40
        _base_frac = _base_frac * _sharp_mult
    # [v5.0] Uncertainty-adjusted Kelly: wider CI → bet less
    if ci_lower is not None and ci_upper is not None and line is not None:
        try:
            ci_width = float(ci_upper) - float(ci_lower)
            _ln = max(float(line), 0.5)
            uncertainty_discount = float(np.clip(1.0 - ci_width / _ln * 0.4, 0.20, 1.0))
            _base_frac = _base_frac * uncertainty_discount
        except Exception:
            pass
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
    # 1. session state override (entered in Settings)
    _ss_key = st.session_state.get("_odds_api_key_override","")
    if _ss_key: return _ss_key
    # 2. saved settings file
    _saved = load_pp_settings().get("odds_api_key","")
    if _saved: return _saved
    # 3. Streamlit secrets / env
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
            league = str(attrs.get("league", "") or "").strip()
            # Reject non-NBA leagues; also reject rows with NO league when filter is active
            # (empty league = non-NBA sport that omitted the field)
            if not league or not _pp_league_is_nba(league):
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
    Tries curl_cffi (Chrome TLS impersonation) → cloudscraper → plain requests.
    Returns (response_object_or_None, error_str_or_None).
    single_stat='true'  → standard stats (Points, Rebounds, etc.)
    single_stat='false' → combo/specialty stats (PRA, Pts+Reb, Fantasy Score, etc.)
    """
    url = PRIZEPICKS_API
    params = {"per_page": str(per_page),
              "single_stat": single_stat, "in_play": "false"}
    # Full Chrome 120 headers — reduces bot-detection fingerprint
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/vnd.api+json",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://app.prizepicks.com/",
        "Origin": "https://app.prizepicks.com",
        "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Connection": "keep-alive",
        "DNT": "1",
    }
    # Parse optional cookie string from user settings
    # Guard: if the stored value is actually a JSON response (user pasted JSON into cookies field),
    # skip it here — fetch_prizepicks_lines() handles that case before calling _pp_request.
    cookie_dict = {}
    if cookies_str and not cookies_str.strip().startswith("{"):
        for part in cookies_str.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                # Sanitize: HTTP cookie headers must be latin-1 safe
                k = k.strip().encode("latin-1", errors="ignore").decode("latin-1")
                v = v.strip().encode("latin-1", errors="ignore").decode("latin-1")
                if k:
                    cookie_dict[k] = v
    # ── Attempt 1: curl_cffi Chrome TLS impersonation (bypasses PerimeterX) ──
    try:
        from curl_cffi import requests as cffi_requests
        r = cffi_requests.get(
            url, params=params, headers=headers,
            cookies=cookie_dict or None,
            impersonate="chrome120",
            timeout=25,
        )
        if r.status_code not in (403, 429):
            return r, None
    except ImportError:
        pass
    except Exception:
        pass
    # ── Attempt 2: cloudscraper (handles Cloudflare JS challenges) ──
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        r = scraper.get(url, params=params, headers=headers,
                        cookies=cookie_dict or None, timeout=25)
        if r.status_code not in (403, 429):
            return r, None
    except ImportError:
        pass
    except Exception:
        pass
    # ── Attempt 3: plain requests (works locally, often blocked on cloud IPs) ──
    try:
        r = requests.get(url, params=params, headers=headers,
                         cookies=cookie_dict or None, timeout=20)
        return r, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
def _pp_fetch_one(per_page, cookies_str, single_stat):
    """Fetch one PP request with retry on 429. Returns (rows, error)."""
    for attempt in range(3):
        r, err = _pp_request(per_page=per_page, cookies_str=cookies_str, single_stat=single_stat)
        if err:
            return [], err
        if r is None:
            return [], "No response from PrizePicks"
        if r.status_code == 429:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            return [], (
                "PrizePicks rate-limited (429) — Streamlit Cloud IP is throttled. "
                "Wait 60s and retry, or run the app locally."
            )
        if r.status_code == 403:
            return [], (
                "Direct API blocked (403 — expected on cloud servers). "
                "Set up a **proxy** in Settings → PP Connection → Method ① "
                "(free ScraperAPI or ScrapingBee). Or use **Manual Import**."
            )
        if not r.ok:
            return [], f"HTTP {r.status_code}: {r.text[:300]}"
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
                # Hard errors (403/429/network) — abort entirely
                if any(x in err for x in ("403", "429", "rate-limited", "PerimeterX")):
                    return [], err
                break  # soft error — try next single_stat value
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
def _save_pp_opening_lines(rows):
    """Save opening lines for PP/UD props to enable steam detection."""
    try:
        for row in rows:
            pn = normalize_name(row.get("player", ""))
            mkt = map_platform_stat_to_market(row.get("stat_type", ""))
            if mkt:
                mk_key = ODDS_MARKETS.get(mkt, "")
                if mk_key:
                    save_opening_line(pn, mk_key, "Over", row.get("line", 0), None)
    except Exception:
        pass

def _save_pp_opening_lines_safe(rows):
    """Save opening lines for PP props to enable steam detection. Silent failure."""
    try:
        _save_pp_opening_lines(rows)
    except Exception:
        pass


def fetch_prizepicks_lines():
    """Unified PP fetch — cascading fallback:
    1. Scraper DB  2. Proxy (ScraperAPI/ScrapingBee)  3. Playwright
    4. Browser-side fetch  5. Relay URL  6. Auto-fetcher cache
    7. Stored JSON  8. Direct API (403 on cloud — last resort)
    """
    errors = []
    cookies_str = st.session_state.get("pp_cookies", "")

    # ── 1. Scraper DB ──
    db_rows, db_age, _ = _load_pp_from_scraper_db()
    if db_rows and db_age is not None and db_age < 1200:
        _save_pp_opening_lines_safe(db_rows)
        return db_rows, None

    # ── 2. Scraping proxy (MOST LIKELY TO WORK on Streamlit Cloud) ──
    _psvc = st.session_state.get("pp_proxy_service", "").strip()
    _pkey = st.session_state.get("pp_proxy_key", "").strip()
    if _psvc and _pkey:
        try:
            rows, err = _fetch_pp_via_proxy(_psvc, _pkey)
            if rows:
                _save_pp_opening_lines_safe(rows)
                return rows, None
            if err:
                errors.append(f"Proxy ({_psvc}): {err}")
        except Exception as e:
            errors.append(f"Proxy ({_psvc}): {type(e).__name__}: {e}")
    else:
        errors.append("Proxy: not configured (Settings → PP Connection → Method ①)")

    # ── 3. Playwright headless fetch ──
    try:
        from pp_scraper import run_once, save_to_disk as _pp_save_disk
        _pw_rows = run_once(headless=True)
        if _pw_rows:
            _pp_save_disk(_pw_rows)
            _save_pp_opening_lines_safe(_pw_rows)
            return _pw_rows, None
    except ImportError:
        pass
    except Exception as _pw_e:
        errors.append(f"Playwright: {_pw_e}")

    # ── 4. Browser-side fetch ──
    try:
        rows, err = _fetch_pp_via_browser()
        if rows:
            _save_pp_opening_lines_safe(rows)
            return rows, None
        if err:
            errors.append(f"Browser: {err}")
    except Exception as e:
        errors.append(f"Browser: {e}")

    # ── 5. Relay URL ──
    relay_url = st.session_state.get("pp_relay_url", "").strip()
    if relay_url:
        try:
            r = requests.get(relay_url, timeout=10)
            if r.ok:
                data = r.json()
                rows = data if isinstance(data, list) else data.get("rows", [])
                if rows:
                    _save_pp_opening_lines_safe(rows)
                    return rows, None
        except Exception as e:
            errors.append(f"Relay: {e}")

    # ── 6. Background auto-fetcher cache ──
    _auto_rows, _auto_age, _ = get_pp_auto_lines()
    if _auto_rows and _auto_age is not None and _auto_age < 900:
        return _auto_rows, None

    # ── 7. Stored JSON (user pasted full PP response) ──
    _stripped = cookies_str.strip() if cookies_str else ""
    if _stripped.startswith("{") and '"data"' in _stripped:
        try:
            data = json.loads(_stripped)
            rows = _parse_pp_response_all(data)
            if rows:
                _save_pp_opening_lines_safe(rows)
                return rows, None
        except Exception as _je:
            errors.append(f"Stored JSON: {_je}")

    # ── 8. Direct API (ONLY works from residential IPs — 403 on cloud) ──
    if st.session_state.get("_pp_last_cookies_used") != cookies_str:
        _fetch_prizepicks_lines_cached.clear()
        st.session_state["_pp_last_cookies_used"] = cookies_str
    cached_rows, cached_err = _fetch_prizepicks_lines_cached(cookies_str=cookies_str)
    if cached_rows:
        _save_pp_opening_lines_safe(cached_rows)
        return cached_rows, None
    if cached_err:
        if "403" in str(cached_err) and _pkey:
            errors.append("Direct API: 403 (expected on cloud — proxy should handle this)")
        else:
            errors.append(f"Direct: {cached_err}")

    # ── All methods failed — show ALL errors for diagnosis ──
    if errors:
        return [], " | ".join(errors[:4])
    return [], "All PrizePicks fetch methods failed"
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
_PP_SCRAPER_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nba_prizepicks.db")
def _load_pp_from_scraper_db(max_age_sec: int = 1800):
    """Load latest PP lines from pp_scraper.py SQLite DB. Returns (rows, age_sec, err)."""
    try:
        import sqlite3 as _sqlite3
        if not os.path.exists(_PP_SCRAPER_DB):
            return [], None, "scraper DB not found"
        conn = _sqlite3.connect(_PP_SCRAPER_DB)
        cur = conn.execute(
            "SELECT player_name, stat_type, line_score, start_time, odds_type, fetched_at "
            "FROM nba_prizepicks_lines WHERE is_latest = 1"
        )
        rows_raw = cur.fetchall()
        ts_row = conn.execute(
            "SELECT last_success FROM scraper_status WHERE scraper_name = 'nba_prizepicks'"
        ).fetchone()
        conn.close()
        if not rows_raw:
            return [], None, "no rows in scraper DB"
        if ts_row:
            from datetime import datetime as _dt, timezone as _tz
            try:
                _ts = _dt.strptime(ts_row[0], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=_tz.utc)
                age = int(time.time() - _ts.timestamp())
            except Exception:
                age = 0
        else:
            age = 0
        rows = [
            {"player": r[0], "stat_type": r[1], "line": r[2],
             "start_time": r[3], "odds_type": r[4] or "standard"}
            for r in rows_raw
        ]
        return rows, age, None
    except Exception as _e:
        return [], None, str(_e)
# ── SCRAPING PROXY CONFIG ──────────────────────────────────────────────────
_PROXY_SERVICES = {
    "scraperapi": {
        "url_tpl": "http://api.scraperapi.com?api_key={key}&url={url}&render=false&premium=true",
        "signup": "https://www.scraperapi.com/signup",
        "free_tier": "5,000 credits/month (premium=10 credits/req → ~250 fetches)",
        "credits_per_req": 10,
    },
    "scrapingbee": {
        "url_tpl": "https://app.scrapingbee.com/api/v1/?api_key={key}&url={url}&render_js=false&forward_headers=true",
        "signup": "https://www.scrapingbee.com/",
        "free_tier": "1,000 req/month",
        "credits_per_req": 1,
    },
    "zenrows": {
        "url_tpl": "https://api.zenrows.com/v1/?apikey={key}&url={url}&premium_proxy=true",
        "signup": "https://www.zenrows.com/",
        "free_tier": "1,000 req/month",
        "credits_per_req": 1,
    },
}
@st.cache_data(ttl=60*5, show_spinner=False)
def _fetch_pp_via_proxy(proxy_service="scraperapi", proxy_key=""):
    """Fetch PrizePicks via residential proxy. Bypasses PerimeterX IP block.
    URL-encodes the target so proxy services don't mangle query params."""
    if not proxy_key:
        return [], "No proxy API key configured"
    svc = _PROXY_SERVICES.get(proxy_service)
    if not svc:
        return [], f"Unknown proxy service: {proxy_service}"
    all_rows, last_err = [], None
    for single_stat in ("true", "false"):
        target = f"https://api.prizepicks.com/projections?per_page=500&single_stat={single_stat}&in_play=false"
        # URL-encode the ENTIRE target so ? and & aren't parsed as proxy params
        encoded_target = _url_quote(target, safe="")
        proxy_url = svc["url_tpl"].format(key=proxy_key, url=encoded_target)
        try:
            r = requests.get(proxy_url, timeout=60, headers={
                "Accept": "application/vnd.api+json",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://app.prizepicks.com/",
                "Origin": "https://app.prizepicks.com",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            })
            if r.status_code == 401:
                return [], f"{proxy_service} API key invalid — check Settings → PP Connection"
            if r.status_code == 429:
                return [], f"{proxy_service} rate limited — free tier may be exhausted this month"
            # ScraperAPI 500 = protected domain → retry with ultra_premium
            if r.status_code == 500 and proxy_service == "scraperapi":
                ultra_url = proxy_url.replace("premium=true", "ultra_premium=true")
                r = requests.get(ultra_url, timeout=60, headers={
                    "Accept": "application/vnd.api+json",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://app.prizepicks.com/",
                    "Origin": "https://app.prizepicks.com",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                })
                if not r.ok:
                    last_err = (
                        f"{proxy_service} HTTP {r.status_code} (tried premium + ultra_premium). "
                        "Try switching to ScrapingBee in Settings."
                    )
                    continue
            elif not r.ok:
                last_err = f"{proxy_service} HTTP {r.status_code}: {r.text[:300]}"
                continue
            # Parse response — check we got JSON, not an HTML error page
            try:
                data = r.json()
            except Exception:
                last_err = f"{proxy_service}: response is not JSON (likely HTML error page). Check API key."
                continue
            rows = _parse_pp_response(data)
            all_rows.extend(rows)
        except Exception as e:
            last_err = f"{proxy_service}: {type(e).__name__}: {e}"
    if all_rows:
        seen = set()
        deduped = [r for r in all_rows if (r["player"], r["stat_type"]) not in seen and not seen.add((r["player"], r["stat_type"]))]
        _save_pp_disk_cache(deduped)
        return deduped, None
    return [], last_err or "No NBA props found via proxy"
def _fetch_pp_via_browser():
    """Execute fetch() in the user's browser (residential IP — bypasses datacenter blocks)."""
    try:
        from streamlit_js_eval import streamlit_js_eval
    except ImportError:
        return [], "streamlit-js-eval not installed"
    js_code = """
await (async () => {
  try {
    const h = {'Accept':'application/vnd.api+json','Referer':'https://app.prizepicks.com/'};
    const [r1,r2] = await Promise.all([
      fetch('https://api.prizepicks.com/projections?per_page=500&single_stat=true&in_play=false', {headers:h}),
      fetch('https://api.prizepicks.com/projections?per_page=500&single_stat=false&in_play=false', {headers:h})
    ]);
    if(!r1.ok&&!r2.ok) return JSON.stringify({error:'HTTP '+r1.status});
    const [d1,d2] = await Promise.all([r1.ok?r1.json():{data:[],included:[]}, r2.ok?r2.json():{data:[],included:[]}]);
    return JSON.stringify({data:[...(d1.data||[]),...(d2.data||[])],included:[...(d1.included||[]),...(d2.included||[])]});
  } catch(e) { return JSON.stringify({error:e.message}); }
})()
"""
    try:
        raw = streamlit_js_eval(js_expressions=js_code, key="pp_browser_fetch")
        if raw is None:
            return [], None  # first render — JS hasn't executed yet
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict) and "error" in data:
            return [], f"Browser: {data['error']}"
        if isinstance(data, dict) and "data" in data:
            rows = _parse_pp_response(data)
            if rows:
                _save_pp_disk_cache(rows)
            return rows, None
        return [], "Browser returned 0 NBA props"
    except Exception as e:
        return [], f"Browser fetch: {e}"
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
    "proxy_service": "",
    "proxy_key": "",
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
            rows: list = []
            err = None
            # Try Playwright first (real browser, bypasses PerimeterX)
            try:
                from pp_scraper import run_once as _pw_run, save_to_disk as _pw_save
                rows = _pw_run(headless=True)
                if rows:
                    _pw_save(rows)
            except Exception:
                pass
            # Fallback to relay URL
            if not rows:
                relay_url = state.get("relay_url", "").strip()
                if relay_url:
                    try:
                        r = requests.get(relay_url, timeout=10)
                        if r.ok:
                            data = r.json()
                            rows = data if isinstance(data, list) else data.get("rows", [])
                    except Exception:
                        pass
            # Proxy fallback (residential IPs — bypasses server IP block)
            if not rows:
                _psvc = state.get("proxy_service", "")
                _pkey = state.get("proxy_key", "")
                if _psvc and _pkey:
                    try:
                        rows, _ = _fetch_pp_via_proxy(_psvc, _pkey)
                    except Exception:
                        pass
            # Fallback to direct API (works from residential IPs)
            if not rows:
                cookies_str = state.get("cookies", "")
                rows_s, err_s = _pp_fetch_one(500, cookies_str, "true")
                rows_c, _     = _pp_fetch_one(500, cookies_str, "false")
                rows = (rows_s or []) + (rows_c or [])
                err = err_s if not rows else None
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
def set_pp_auto_fetch(enabled: bool, interval_sec: int = 600, cookies: str = "", relay_url: str = "", proxy_service: str = "", proxy_key: str = ""):
    """Enable/disable the background fetcher and trigger an immediate fetch."""
    state = _pp_auto_state()
    with state["lock"]:
        state["enabled"] = enabled
        state["interval"] = max(60, interval_sec)
        state["cookies"] = cookies
        state["relay_url"] = relay_url
        state["proxy_service"] = proxy_service
        state["proxy_key"] = proxy_key
    if enabled:
        _ensure_pp_auto_thread()
        state["stop_evt"].set()  # poke thread to fetch immediately
def get_pp_auto_lines():
    """Return (rows, age_sec, err) from background state or disk cache."""
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
    rows, err = _fetch_underdog_lines_cached(cookies_str=cookies_str)
    if rows and not err:
        _save_pp_opening_lines(rows)
    return rows, err
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
        "Made Baskets": "FGM", "Field Goals": "FGM", "Baskets Made": "FGM",
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
        "1st Half Pts": "H1 Points", "First Half Pts": "H1 Points",
        "Pts 1st Half": "H1 Points", "Points 1st Half": "H1 Points",
        "NBA 1H Points": "H1 Points",
        "H1 Rebounds": "H1 Rebounds", "1H Rebounds": "H1 Rebounds",
        "1st Half Rebounds": "H1 Rebounds", "H1 Reb": "H1 Rebounds",
        "1st Half Reb": "H1 Rebounds", "First Half Reb": "H1 Rebounds",
        "Reb 1st Half": "H1 Rebounds", "Rebounds 1st Half": "H1 Rebounds",
        "First Half Rebounds": "H1 Rebounds",
        "H1 Assists": "H1 Assists", "1H Assists": "H1 Assists",
        "1st Half Assists": "H1 Assists", "H1 Ast": "H1 Assists",
        "1st Half Ast": "H1 Assists", "First Half Ast": "H1 Assists",
        "Ast 1st Half": "H1 Assists", "Assists 1st Half": "H1 Assists",
        "First Half Assists": "H1 Assists",
        "H1 3PM": "H1 3PM", "1H 3PM": "H1 3PM",
        "1st Half 3-Pointers Made": "H1 3PM", "H1 3-Pointers Made": "H1 3PM",
        "1st Half 3PM": "H1 3PM", "First Half 3PM": "H1 3PM",
        "1st Half 3-PT Made": "H1 3PM", "3PM 1st Half": "H1 3PM",
        "1st Half Three Pointers Made": "H1 3PM", "First Half 3-Pointers Made": "H1 3PM",
        "H1 PRA": "H1 PRA", "1H PRA": "H1 PRA",
        "1st Half PRA": "H1 PRA", "H1 Pts+Reb+Ast": "H1 PRA",
        "1st Half Pts+Reb+Ast": "H1 PRA", "First Half PRA": "H1 PRA",
        "1H Pts+Reb+Ast": "H1 PRA",
        # ── 2nd Half ──────────────────────────────────────────────────────
        "H2 Points": "H2 Points", "2H Points": "H2 Points",
        "2nd Half Points": "H2 Points", "Second Half Points": "H2 Points",
        "H2 Pts": "H2 Points", "2H Pts": "H2 Points",
        "2nd Half Pts": "H2 Points", "Second Half Pts": "H2 Points",
        "Pts 2nd Half": "H2 Points", "Points 2nd Half": "H2 Points",
        "NBA 2H Points": "H2 Points",
        "H2 Rebounds": "H2 Rebounds", "2H Rebounds": "H2 Rebounds",
        "2nd Half Rebounds": "H2 Rebounds", "H2 Reb": "H2 Rebounds",
        "2nd Half Reb": "H2 Rebounds", "Second Half Reb": "H2 Rebounds",
        "Reb 2nd Half": "H2 Rebounds", "Rebounds 2nd Half": "H2 Rebounds",
        "Second Half Rebounds": "H2 Rebounds",
        "H2 Assists": "H2 Assists", "2H Assists": "H2 Assists",
        "2nd Half Assists": "H2 Assists", "H2 Ast": "H2 Assists",
        "2nd Half Ast": "H2 Assists", "Second Half Ast": "H2 Assists",
        "Ast 2nd Half": "H2 Assists", "Assists 2nd Half": "H2 Assists",
        "Second Half Assists": "H2 Assists",
        "H2 3PM": "H2 3PM", "2H 3PM": "H2 3PM",
        "2nd Half 3PM": "H2 3PM", "Second Half 3PM": "H2 3PM",
        "2nd Half 3-Pointers Made": "H2 3PM",
        "H2 PRA": "H2 PRA", "2H PRA": "H2 PRA",
        "2nd Half PRA": "H2 PRA", "H2 Pts+Reb+Ast": "H2 PRA",
        "2nd Half Pts+Reb+Ast": "H2 PRA", "Second Half PRA": "H2 PRA",
        "2H Pts+Reb+Ast": "H2 PRA",
        # H1/H2 shooting volume
        "H1 FGM": "H1 FGM", "1H FGM": "H1 FGM", "1st Half FGM": "H1 FGM",
        "1st Half Field Goals Made": "H1 FGM",
        "H1 FGA": "H1 FGA", "1H FGA": "H1 FGA", "1st Half FGA": "H1 FGA",
        "1st Half Field Goals Attempted": "H1 FGA",
        "H1 FTM": "H1 FTM", "1H FTM": "H1 FTM", "1st Half FTM": "H1 FTM",
        "1st Half Free Throws Made": "H1 FTM",
        "H1 FTA": "H1 FTA", "1H FTA": "H1 FTA", "1st Half FTA": "H1 FTA",
        "1st Half Free Throws Attempted": "H1 FTA",
        "H2 FGM": "H2 FGM", "2H FGM": "H2 FGM", "2nd Half FGM": "H2 FGM",
        "2nd Half Field Goals Made": "H2 FGM",
        "H2 FGA": "H2 FGA", "2H FGA": "H2 FGA", "2nd Half FGA": "H2 FGA",
        "2nd Half Field Goals Attempted": "H2 FGA",
        "H2 FTM": "H2 FTM", "2H FTM": "H2 FTM", "2nd Half FTM": "H2 FTM",
        "2nd Half Free Throws Made": "H2 FTM",
        "H2 FTA": "H2 FTA", "2H FTA": "H2 FTA", "2nd Half FTA": "H2 FTA",
        "2nd Half Free Throws Attempted": "H2 FTA",
        # ── 1st Quarter ───────────────────────────────────────────────────
        "Q1 Points": "Q1 Points", "1Q Points": "Q1 Points",
        "1st Quarter Points": "Q1 Points", "Q1 Pts": "Q1 Points",
        "1st Quarter Pts": "Q1 Points", "Pts 1st Quarter": "Q1 Points",
        "Points 1st Quarter": "Q1 Points",
        "Q1 Rebounds": "Q1 Rebounds", "1Q Rebounds": "Q1 Rebounds",
        "1st Quarter Rebounds": "Q1 Rebounds", "Q1 Reb": "Q1 Rebounds",
        "1st Quarter Reb": "Q1 Rebounds", "Rebounds 1st Quarter": "Q1 Rebounds",
        "Q1 Assists": "Q1 Assists", "1Q Assists": "Q1 Assists",
        "1st Quarter Assists": "Q1 Assists", "Q1 Ast": "Q1 Assists",
        "1st Quarter Ast": "Q1 Assists", "Assists 1st Quarter": "Q1 Assists",
        # ── Minutes ───────────────────────────────────────────────────────
        "Minutes": "Minutes", "Min": "Minutes",
        "Minutes Played": "Minutes", "Mins": "Minutes",
        "Player Minutes": "Minutes", "Total Minutes": "Minutes",
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
    # ── Regex fallback for half/quarter variants ────────────────────────
    # Catches any remaining PP/UD format: "1st Half X", "X 1st Half", "1H X", etc.
    _s_upper = s.upper().strip()
    _HALF_PREFIXES = [
        (r'(?:1ST\s*HALF|FIRST\s*HALF|1H|H1)\s+(.+)',  "H1"),
        (r'(.+?)\s+(?:1ST\s*HALF|FIRST\s*HALF|1H)',    "H1"),
        (r'(?:2ND\s*HALF|SECOND\s*HALF|2H|H2)\s+(.+)', "H2"),
        (r'(.+?)\s+(?:2ND\s*HALF|SECOND\s*HALF|2H)',   "H2"),
        (r'(?:1ST\s*QUARTER|FIRST\s*QUARTER|1Q|Q1)\s+(.+)', "Q1"),
        (r'(.+?)\s+(?:1ST\s*QUARTER|FIRST\s*QUARTER|1Q)', "Q1"),
    ]
    # Base stat → canonical market map (full-game only)
    _BASE_MAP = {
        "POINTS": "Points", "PTS": "Points",
        "REBOUNDS": "Rebounds", "REB": "Rebounds", "REBS": "Rebounds",
        "ASSISTS": "Assists", "AST": "Assists",
        "3PM": "3PM", "3-PT MADE": "3PM", "3PT MADE": "3PM",
        "THREE POINTERS MADE": "3PM", "3 POINTERS MADE": "3PM",
        "PRA": "PRA", "PTS+REB+AST": "PRA", "POINTS+REBOUNDS+ASSISTS": "PRA",
    }
    for _pat, _prefix in _HALF_PREFIXES:
        _m = re.match(_pat, _s_upper)
        if _m:
            _inner = _m.group(1).strip()
            _base_mkt = _BASE_MAP.get(_inner)
            if _base_mkt:
                _result = f"{_prefix} {_base_mkt}"
                if _result in STAT_FIELDS:
                    logging.info(f"Regex-matched '{stat_type}' → '{_result}'")
                    return _result
    logging.info(f"Unmapped platform stat_type: '{stat_type}'")
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
def compute_projection_ci(mu, sigma, half_factor=1.0, stat_series=None):
    """
    Returns (lower_80, upper_80) confidence interval.
    [v5.0] Uses leave-one-out conformal prediction when stat_series is provided
    (distribution-free, exact coverage). Falls back to normal approximation.
    """
    if mu is None or sigma is None:
        return None, None
    try:
        hf = float(half_factor) if half_factor else 1.0
        mu_h = float(mu) * hf
        sig_h = float(sigma) * hf
        if stat_series is not None:
            arr = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
            if len(arr) >= 5:
                # Leave-one-out conformal: nonconformity score = |y_i - mu_LOO|
                n = len(arr)
                scores = np.abs(arr * hf - mu_h)
                # 80% coverage: use 80th percentile of scores as radius
                q80 = float(np.percentile(scores, 80))
                lower = max(0.0, mu_h - q80)
                upper = mu_h + q80
                return float(lower), float(upper)
        # Normal approximation fallback
        z = 1.28
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
# MAIN PROJECTION ENGINE  [FIX 3: minutes filter]
# ──────────────────────────────────────────────
def compute_leg_projection(
    player_name, market_name, line, meta,
    n_games, key_teammate_out,
    bankroll=0.0, frac_kelly=0.25, max_risk_frac=0.05,
    market_prior_weight=0.65, exclude_chaotic=True,
    game_date=None, is_home=None,
    injury_team_map=None,   # {team_abbr_upper: [player_name_lower, ...]} for OUT/DOUBTFUL players
    skip_halfgame_boxscores=False,   # True for scanner (saves ~2s per candidate)
    skip_expensive_signals=False,    # True for scanner (saves ~7 API calls per candidate)
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
    if is_half_market and player_id and not skip_halfgame_boxscores:
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
    if skip_expensive_signals:
        _clutch_factor, _clutch_label = 1.0, "Skipped"
    else:
        _clutch_factor, _clutch_label = get_clutch_performance_factor(
            player_id, base_market, spread_abs=_spread_abs_val)
    # [v3.0] FTA opponent foul rate (for FTA/FTM/Stocks/Points props)
    if skip_expensive_signals:
        _fta_factor, _fta_label = 1.0, "Skipped"
    else:
        _fta_factor, _fta_label = get_opponent_fta_rate_factor(opp_abbr, base_market)
    # [v3.0] Playoff implications / tanking factor (March-April only)
    if skip_expensive_signals:
        _playoff_factor, _playoff_label = 1.0, "Skipped"
    else:
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
        # [v5.0 FIX] For half markets WITHOUT real boxscore data: scale stat SERIES DOWN
        # by half_factor, then bootstrap against the ACTUAL line (not line/half_factor).
        # The old approach (effective_line = line/0.52 = 24.04) made bootstrap vs full-game
        # series return ~50% for any half-game prop — systematically killing H1/H2 edges.
        if is_half_market and half_factor != 1.0:
            _bootstrap_series = pace_adj_series * half_factor
            _bootstrap_line = float(line)
            # NOTE: _orig_half_factor still holds original value for Bayesian prior scaling
            half_factor = 1.0   # proj_full is now in half-game units; no re-scale needed
            errors.append(
                f"[H-FIX] {market_name}: series_mean={float(_bootstrap_series.mean()):.1f}, "
                f"line={_bootstrap_line:.1f}, hf_used={_orig_half_factor:.3f}"
            )
        else:
            _bootstrap_series = pace_adj_series
            _bootstrap_line = float(line)
        # [UPGRADE 4] Pass market for stat-specific λ decay
        p_over_raw, mu_raw, sigma = bootstrap_prob_over(
            _bootstrap_series, _bootstrap_line, cv_override=vol_cv, market=base_market
        )
        # [v5.0] Negative Binomial blend for count stats (overdispersed integer distributions)
        # NegBin is theoretically superior for 3PM, Assists, Rebounds, Blocks, Steals.
        # We blend 65% NegBin + 35% bootstrap when conditions are met (n>=6, overdispersed).
        # This gives us better probability estimates for sharp-edge identification on count props.
        if base_market in NEGBINOM_MARKETS and len(_bootstrap_series.dropna()) >= 6:
            try:
                _nb_p, _nb_mu, _nb_sigma = negbinom_prob_over(
                    _bootstrap_series, _bootstrap_line, market=base_market)
                if _nb_p is not None and p_over_raw is not None:
                    # Blend: weight NegBin more heavily for larger samples (better r estimate).
                    # [AUDIT FIX] Raised cap from 0.70 → 0.82; increased slope.
                    # At n=6 (just enough), NegBin r-estimate is noisy → 50% blend.
                    # At n=15+, r-estimate is stable → up to 82% NegBin (clearly superior for counts).
                    _n_valid_nb = len(_bootstrap_series.dropna())
                    _nb_weight = float(np.clip(0.45 + (_n_valid_nb - 6) * 0.05, 0.45, 0.82))
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
        # [v5.0] KDE blend for continuous stats (Points, PRA, etc.) — blends at 50% weight
        if base_market in KDE_MARKETS and len(_bootstrap_series.dropna()) >= 8:
            try:
                _kde_p, _kde_mu, _kde_sigma = kde_prob_over(_bootstrap_series, _bootstrap_line, market=base_market)
                if _kde_p is not None and p_over_raw is not None:
                    p_over_raw = float(0.50 * _kde_p + 0.50 * p_over_raw)
                    if _kde_mu is not None and mu_raw is not None:
                        mu_raw = float(0.50 * _kde_mu + 0.50 * mu_raw)
            except Exception:
                pass
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
    if skip_expensive_signals:
        _ts_factor, _ts_label = 1.0, "Skipped"
        _efg_factor, _efg_label = 1.0, "Skipped"
    else:
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
        _log_combined = float(np.clip(_log_combined, math.log(0.70), math.log(1.35)))
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
    # [v5.0] Use Fano-factor gate for count stats; CV gate for continuous stats
    gate_ok, gate_reason = passes_volatility_gate_v2(stat_series, ev_raw, base_market, skew=stat_skew, bet_type=side_str)
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
    if gate_ok and p_cal is not None and price_decimal is not None and ev_adj is not None and ev_adj >= 0.02:
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
    if not skip_expensive_signals and meta and meta.get("event_id"):
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
    if gate_ok and p_cal is not None and price_decimal is not None and ev_adj is not None and ev_adj >= 0.02:
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
def fit_platt_calibrator(p_raws, y_labels):
    """
    [v5.0] Fit a Platt (logistic) calibrator for n < 200 samples,
    isotonic regression for n >= 200.
    Returns calib dict compatible with apply_calibrator.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        import numpy as _np2
        p = _np2.array(p_raws, dtype=float)
        y = _np2.array(y_labels, dtype=float)
        valid = ~(_np2.isnan(p) | _np2.isnan(y))
        p, y = p[valid], y[valid]
        n = len(p)
        if n < 20:
            return None
        p_sorted = _np2.sort(p)
        if n < 200:
            # Platt scaling
            lr = LogisticRegression(C=1.0, max_iter=500)
            lr.fit(p.reshape(-1, 1), y)
            y_cal = lr.predict_proba(p_sorted.reshape(-1, 1))[:, 1]
            calib_type = "platt"
        else:
            # Isotonic regression
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p, y)
            y_cal = ir.predict(p_sorted)
            calib_type = "isotonic"
        return {
            "type": calib_type,
            "x": p_sorted.tolist(),
            "y": y_cal.tolist(),
            "training_min": float(p.min()),
            "training_max": float(p.max()),
            "n": n,
        }
    except Exception as e:
        logging.warning(f"fit_platt_calibrator: {e}")
        return None
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
        if p < t_min * 0.85 or p > t_max * 1.15:
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
st.set_page_config(
    page_title="NBA ALPHA ENGINE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto"
)
# ─── FONTS + GLOBAL PREMIUM STYLES ───────────────────────────
st.html("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Fira+Code:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<div style="display:none"><style>
/* ---------------------------------------------------
   DESIGN TOKENS
--------------------------------------------------- */
:root {
  --bg:       #030810;
  --bg2:      #060D18;
  --bg3:      #080F1C;
  --bg4:      #0A1220;
  --panel:    #0C1524;
  --panel2:   #0E1828;
  --border:   #112030;
  --border2:  #1A3050;
  --green:    #00FFB2;
  --green2:   #00D494;
  --green-lo: #00FFB210;
  --blue:     #00AAFF;
  --blue2:    #0088DD;
  --blue-lo:  #00AAFF10;
  --red:      #FF3358;
  --red-lo:   #FF335810;
  --amber:    #FFB800;
  --amber-lo: #FFB80010;
  --purple:   #A855F7;
  --muted:    #3A5570;
  --muted2:   #2A4060;
  --text:     #B8CCE0;
  --text-hi:  #E8F0FA;
  --text-lo:  #4A607A;
  --font-head: 'Chakra Petch', monospace;
  --font-mono: 'Fira Code', monospace;
  --font-body: 'Space Grotesk', sans-serif;
  --radius:   3px;
  --radius2:  6px;
}
/* ---------------------------------------------------
   BASE / APP SHELL
--------------------------------------------------- */
.stApp {
  background: var(--bg) !important;
  font-family: var(--font-mono) !important;
  color: var(--text) !important;
}
/* Subtle grid overlay */
.stApp::before {
  content: "";
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image:
    linear-gradient(rgba(0,170,255,0.012) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,170,255,0.012) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
}
/* CRT scanline texture */
.stApp::after {
  content: "";
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 3px,
    rgba(0,0,0,0.04) 3px,
    rgba(0,0,0,0.04) 4px
  );
  pointer-events: none;
  z-index: 9998;
}
.block-container {
  padding-top: 0.8rem !important;
  padding-bottom: 2rem !important;
  max-width: 1500px !important;
}
/* ---------------------------------------------------
   TAB BAR
--------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg2) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
  padding: 0 0.5rem !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--font-head) !important;
  font-size: 0.68rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  color: var(--muted) !important;
  padding: 0.65rem 1.1rem !important;
  border-bottom: 2px solid transparent !important;
  text-transform: uppercase !important;
  transition: color 0.15s, border-color 0.15s !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text) !important;
  background: rgba(0,170,255,0.04) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--green) !important;
  border-bottom: 2px solid var(--green) !important;
  background: transparent !important;
  text-shadow: 0 0 12px rgba(0,255,178,0.4) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding-top: 1.2rem !important;
  background: transparent !important;
}
/* ---------------------------------------------------
   BUTTONS
--------------------------------------------------- */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--green) !important;
  color: var(--green) !important;
  font-family: var(--font-head) !important;
  font-size: 0.70rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  padding: 0.5rem 1.4rem !important;
  border-radius: var(--radius) !important;
  transition: all 0.18s ease !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button:hover {
  background: var(--green) !important;
  color: #030810 !important;
  box-shadow: 0 0 24px rgba(0,255,178,0.40), 0 0 48px rgba(0,255,178,0.15) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active {
  transform: translateY(0px) !important;
}
/* ---------------------------------------------------
   INPUTS
--------------------------------------------------- */
.stTextInput input,
.stNumberInput input,
.stSelectbox > div > div,
.stTextArea textarea {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-hi) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  border-radius: var(--radius) !important;
  transition: border-color 0.15s !important;
}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
  border-color: rgba(0,255,178,0.4) !important;
  box-shadow: 0 0 0 2px rgba(0,255,178,0.06) !important;
  outline: none !important;
}
.stTextInput label p,
.stNumberInput label p,
.stSelectbox label p,
.stTextArea label p {
  font-family: var(--font-head) !important;
  font-size: 0.60rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  margin-bottom: 3px !important;
}
/* ---------------------------------------------------
   SLIDERS
--------------------------------------------------- */
[data-testid="stSlider"] label p {
  font-family: var(--font-head) !important;
  font-size: 0.60rem !important;
  color: var(--muted) !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div {
  background: var(--border) !important;
}
/* ---------------------------------------------------
   DATAFRAMES / TABLES
--------------------------------------------------- */
.stDataFrame {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius2) !important;
  overflow: hidden !important;
}
.stDataFrame [data-testid="stDataFrameResizable"] {
  border-radius: var(--radius2) !important;
}
.stDataFrame thead tr th {
  background: var(--bg3) !important;
  color: var(--green) !important;
  font-family: var(--font-head) !important;
  font-size: 0.62rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.10em !important;
  border-bottom: 1px solid var(--border2) !important;
  padding: 0.5rem 0.8rem !important;
}
.stDataFrame tbody tr td {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  color: var(--text) !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0.4rem 0.8rem !important;
}
.stDataFrame tbody tr:hover td {
  background: rgba(0,170,255,0.04) !important;
}
.stDataFrame tbody tr:nth-child(even) td {
  background: rgba(0,0,0,0.15) !important;
}
/* ---------------------------------------------------
   METRICS
--------------------------------------------------- */
[data-testid="stMetric"] {
  background: linear-gradient(135deg, var(--panel) 0%, var(--panel2) 100%) !important;
  border: 1px solid var(--border) !important;
  border-top: 2px solid var(--green) !important;
  padding: 0.9rem 1.1rem !important;
  border-radius: var(--radius2) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
  position: relative !important;
  overflow: hidden !important;
}
[data-testid="stMetric"]::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--green), transparent);
  opacity: 0.5;
}
[data-testid="stMetric"]:hover {
  border-color: var(--border2) !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}
[data-testid="stMetricLabel"] > div {
  font-family: var(--font-head) !important;
  font-size: 0.60rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] > div {
  font-family: var(--font-mono) !important;
  font-size: 1.45rem !important;
  font-weight: 700 !important;
  color: var(--text-hi) !important;
  letter-spacing: -0.01em !important;
}
[data-testid="stMetricDelta"] > div {
  font-family: var(--font-mono) !important;
  font-size: 0.68rem !important;
}
/* ---------------------------------------------------
   ALERTS / NOTIFICATIONS
--------------------------------------------------- */
[data-testid="stAlert"] {
  background: var(--panel) !important;
  border-radius: var(--radius) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.73rem !important;
}
.stSuccess {
  border-left: 3px solid var(--green) !important;
  background: var(--green-lo) !important;
}
.stWarning {
  border-left: 3px solid var(--amber) !important;
  background: var(--amber-lo) !important;
}
.stError {
  border-left: 3px solid var(--red) !important;
  background: var(--red-lo) !important;
}
.stInfo {
  border-left: 3px solid var(--blue) !important;
  background: var(--blue-lo) !important;
}
/* ---------------------------------------------------
   EXPANDERS
--------------------------------------------------- */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg2) !important;
  margin-bottom: 0.4rem !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--font-head) !important;
  font-size: 0.68rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  color: var(--text-lo) !important;
  padding: 0.5rem 0.8rem !important;
}
[data-testid="stExpander"] summary:hover {
  color: var(--text) !important;
  background: rgba(255,255,255,0.02) !important;
}
/* ---------------------------------------------------
   CHECKBOXES & RADIO
--------------------------------------------------- */
[data-testid="stCheckbox"] label p,
[data-testid="stRadio"] label p {
  font-family: var(--font-body) !important;
  font-size: 0.73rem !important;
  color: var(--text) !important;
  letter-spacing: 0.01em !important;
}
[data-testid="stCheckbox"] [data-testid="stCheckboxWidget"],
[data-testid="stRadio"] input {
  accent-color: var(--green) !important;
}
/* ---------------------------------------------------
   SPINNER
--------------------------------------------------- */
[data-testid="stSpinner"] {
  font-family: var(--font-head) !important;
  font-size: 0.68rem !important;
  color: var(--green) !important;
  letter-spacing: 0.1em !important;
}
/* ---------------------------------------------------
   CAPTIONS & MARKDOWN TEXT
--------------------------------------------------- */
.stMarkdown p {
  font-family: var(--font-body) !important;
  color: var(--text) !important;
  font-size: 0.82rem !important;
  line-height: 1.6 !important;
}
[data-testid="stCaptionContainer"] p {
  font-family: var(--font-mono) !important;
  font-size: 0.63rem !important;
  color: var(--text-lo) !important;
  letter-spacing: 0.04em !important;
}
h1, h2, h3 {
  font-family: var(--font-head) !important;
  color: var(--text-hi) !important;
  letter-spacing: 0.04em !important;
}
/* ---------------------------------------------------
   SELECT SLIDER (RADIO BUTTONS)
--------------------------------------------------- */
[data-testid="stRadio"] > div {
  gap: 0.4rem !important;
}
[data-testid="stRadio"] label {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.3rem 0.7rem !important;
  transition: all 0.15s !important;
}
[data-testid="stRadio"] label:has(input:checked) {
  border-color: var(--green) !important;
  background: var(--green-lo) !important;
  color: var(--green) !important;
}
/* ---------------------------------------------------
   SCROLLBARS
--------------------------------------------------- */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
/* ---------------------------------------------------
   ANIMATIONS
--------------------------------------------------- */
@keyframes pulse-green {
  0%, 100% { box-shadow: 0 0 4px #00FFB2; opacity: 1; }
  50%       { box-shadow: 0 0 10px #00FFB2, 0 0 20px #00FFB240; opacity: 0.8; }
}
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position: 200% center; }
}
@keyframes fade-in {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}
.pulse-dot {
  animation: pulse-green 2s ease-in-out infinite;
}
.fade-in {
  animation: fade-in 0.3s ease-out;
}
/* ---------------------------------------------------
   MULTISELECT
--------------------------------------------------- */
[data-testid="stMultiSelect"] > div > div {
  background: var(--bg3) !important;
  border-color: var(--border) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.75rem !important;
}
[data-baseweb="tag"] {
  background: var(--border2) !important;
  font-family: var(--font-mono) !important;
}
/* ---------------------------------------------------
   DATE INPUT
--------------------------------------------------- */
[data-testid="stDateInput"] input {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-hi) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.75rem !important;
  border-radius: var(--radius) !important;
}
/* ---------------------------------------------------
   TOP NAV BAR HIDE / STREAMLIT CHROME REMOVAL
--------------------------------------------------- */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
/* Hide header content but NOT the sidebar toggle */
header    { visibility: hidden; }
header > * { visibility: hidden; }
/* Keep the mobile sidebar toggle button visible and tappable */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] button {
  visibility: visible !important;
  display: flex !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}
[data-testid="stSidebarCollapsedControl"] {
  position: fixed !important;
  top: 0.75rem !important;
  left: 0.75rem !important;
  z-index: 999999 !important;
  background: rgba(0,255,178,0.12) !important;
  border: 1px solid rgba(0,255,178,0.3) !important;
  border-radius: 6px !important;
  padding: 0.3rem !important;
  min-width: 2.5rem !important;
  min-height: 2.5rem !important;
  align-items: center !important;
  justify-content: center !important;
}
[data-testid="stSidebarCollapsedControl"]:active {
  background: rgba(0,255,178,0.25) !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
  visibility: visible !important;
  display: block !important;
  fill: #00FFB2 !important;
  width: 1.2rem !important;
  height: 1.2rem !important;
}
/* ---------------------------------------------------
   DIVIDERS
--------------------------------------------------- */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 0.8rem 0 !important;
}
/* ---------------------------------------------------
   SIDEBAR OVERRIDE
--------------------------------------------------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #04080F 0%, #060D18 100%) !important;
  border-right: 1px solid #0A1828 !important;
}
</style></div>
""")
# ─── AUTH GATE ────────────────────────────────────────────────
# ── Cookie-based session persistence ──────────────────────────
_cookie_ctrl = CookieController(key="auth_cookie_ctrl")
if not st.session_state.get("_auth_user"):
    _cookie_user = _cookie_ctrl.get("auth_user")
    if _cookie_user and isinstance(_cookie_user, str):
        _db = _load_auth_db()
        if _cookie_user in _db:
            st.session_state["_auth_user"] = _cookie_user
            st.session_state["user_id"] = _cookie_user
            _saved = load_user_state(_cookie_user)
            for _sk in ["bankroll", "market_prior_weight", "n_games",
                        "frac_kelly", "payout_multi", "max_risk_per_bet",
                        "max_daily_loss", "max_weekly_loss",
                        "exclude_chaotic", "show_unders", "max_req_day"]:
                if _sk in _saved:
                    st.session_state[_sk] = _saved[_sk]
            st.session_state["_active_user_id"] = _cookie_user
            st.rerun()
if not st.session_state.get("_auth_user"):
    st.html("""
<style>
.auth-wrap {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; min-height:80vh; padding:2rem 1rem;
}
.auth-box {
    background:linear-gradient(135deg,#060E1C,#07101E);
    border:1px solid #0E2040; border-top:2px solid #00FFB2;
    border-radius:8px; padding:2rem 2.4rem; width:100%;
    max-width:420px; box-shadow:0 0 40px rgba(0,255,178,0.08);
}
.auth-title {
    font-family:'Chakra Petch',monospace; font-size:1.4rem;
    font-weight:700; color:#EEF4FF; letter-spacing:0.06em;
    margin-bottom:0.2rem;
}
.auth-subtitle {
    font-family:'Fira Code',monospace; font-size:0.60rem;
    color:#2A6080; letter-spacing:0.12em; text-transform:uppercase;
    margin-bottom:1.6rem;
}
</style>
<div class="auth-wrap">
<div class="auth-box">
<div class="auth-title">NBA <span style="color:#00FFB2;">ALPHA</span> ENGINE</div>
<div class="auth-subtitle">Quantitative Sports Analytics · v2.2</div>
</div>
</div>
""")
    _auth_tab = st.radio("", ["Sign In", "Create Account"], horizontal=True,
                         label_visibility="collapsed",
                         key="_auth_tab_radio")
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if _auth_tab == "Sign In":
            with st.form("signin_form", clear_on_submit=False):
                st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;"
                            "color:#2A6080;letter-spacing:0.16em;text-transform:uppercase;"
                            "margin-bottom:0.8rem;'>▸ SIGN IN TO YOUR ACCOUNT</div>",
                            unsafe_allow_html=True)
                _si_user = st.text_input("Username", placeholder="your_username",
                                         key="_si_username")
                _si_pass = st.text_input("Password", type="password",
                                         placeholder="••••••••", key="_si_password")
                _si_btn = st.form_submit_button("SIGN IN", use_container_width=True)
                if _si_btn:
                    if not _si_user.strip():
                        st.error("Please enter your username.")
                    else:
                        _ok, _msg = _authenticate(_si_user, _si_pass)
                        if _ok:
                            _uid = _si_user.strip().lower()
                            st.session_state["_auth_user"] = _uid
                            st.session_state["user_id"] = _uid
                            # Load saved user state
                            _saved = load_user_state(_uid)
                            for _sk in ["bankroll", "market_prior_weight", "n_games",
                                        "frac_kelly", "payout_multi", "max_risk_per_bet",
                                        "max_daily_loss", "max_weekly_loss",
                                        "exclude_chaotic", "show_unders", "max_req_day"]:
                                if _sk in _saved:
                                    st.session_state[_sk] = _saved[_sk]
                            st.session_state["_active_user_id"] = _uid
                            _cookie_ctrl.set("auth_user", _uid, max_age=30 * 24 * 3600)
                            st.rerun()
                        else:
                            st.error(_msg)
        else:
            with st.form("signup_form", clear_on_submit=False):
                st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;"
                            "color:#2A6080;letter-spacing:0.16em;text-transform:uppercase;"
                            "margin-bottom:0.8rem;'>▸ CREATE FREE ACCOUNT</div>",
                            unsafe_allow_html=True)
                _su_user = st.text_input("Username", placeholder="choose_a_username",
                                         key="_su_username",
                                         help="3+ characters, letters/numbers/underscores only")
                _su_email = st.text_input("Email (optional)", placeholder="you@email.com",
                                          key="_su_email")
                _su_pass = st.text_input("Password", type="password",
                                         placeholder="••••••••", key="_su_password",
                                         help="Minimum 6 characters")
                _su_pass2 = st.text_input("Confirm Password", type="password",
                                          placeholder="••••••••", key="_su_password2")
                _su_btn = st.form_submit_button("CREATE ACCOUNT", use_container_width=True)
                if _su_btn:
                    if _su_pass != _su_pass2:
                        st.error("Passwords do not match.")
                    else:
                        _ok, _msg = _register_user(_su_user, _su_pass, _su_email)
                        if _ok:
                            _uid = _su_user.strip().lower()
                            st.session_state["_auth_user"] = _uid
                            st.session_state["user_id"] = _uid
                            st.session_state["bankroll"] = 1000.0
                            st.session_state["_active_user_id"] = _uid
                            _cookie_ctrl.set("auth_user", _uid, max_age=30 * 24 * 3600)
                            st.success(_msg + " Welcome!")
                            st.rerun()
                        else:
                            st.error(_msg)
    st.stop()
# ── Authenticated: set user_id from auth session ──────────────
_auth_username = st.session_state["_auth_user"]
if st.session_state.get("user_id") != _auth_username:
    st.session_state["user_id"] = _auth_username
    _saved_state = load_user_state(_auth_username)
    # Restore all saved settings for this user
    for _sk in ["bankroll", "market_prior_weight", "n_games", "frac_kelly",
                "payout_multi", "max_risk_per_bet", "max_daily_loss",
                "max_weekly_loss", "exclude_chaotic", "show_unders", "max_req_day"]:
        if _sk in _saved_state:
            st.session_state[_sk] = _saved_state[_sk]
    st.session_state["_active_user_id"] = _auth_username
# ─── MAIN HEADER ─────────────────────────────────────────────
_now_str = datetime.utcnow().strftime("%b %d %Y  %H:%M UTC")
_hdr = (
    "<div style='background:linear-gradient(135deg,#060E1C,#07101E,#050C18);"
    "border:1px solid #0E2040;border-top:2px solid #00FFB2;border-radius:6px;"
    "padding:1.1rem 1.6rem 1rem 1.6rem;margin-bottom:1.4rem;overflow:hidden;'>"
    "<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;flex-wrap:wrap;'>"
    "<div>"
    "<div style='font-family:Chakra Petch,monospace;font-size:0.55rem;color:#2A5070;"
    "letter-spacing:0.30em;text-transform:uppercase;margin-bottom:0.2rem;'>QUANTITATIVE SPORTS ANALYTICS</div>"
    "<div style='font-family:Chakra Petch,monospace;font-size:1.75rem;font-weight:700;"
    "color:#EEF4FF;letter-spacing:0.05em;line-height:1.05;'>NBA "
    "<span style='color:#00FFB2;'>ALPHA</span> ENGINE "
    "<span style='font-size:0.60rem;color:#2A5070;vertical-align:middle;"
    "margin-left:0.6rem;font-weight:400;letter-spacing:0.12em;'>v3.0</span></div>"
    "<div style='display:flex;align-items:center;gap:1.2rem;margin-top:0.4rem;flex-wrap:wrap;'>"
    "<div style='display:flex;align-items:center;gap:0.4rem;'>"
    "<div style='width:6px;height:6px;border-radius:50%;background:#00FFB2;"
    "box-shadow:0 0 6px #00FFB2;flex-shrink:0;'></div>"
    "<span style='font-family:Fira Code,monospace;font-size:0.58rem;"
    "color:#00FFB2;letter-spacing:0.08em;'>LIVE ODDS</span></div>"
    "<span style='font-family:Fira Code,monospace;font-size:0.55rem;color:#1A3050;'>|</span>"
    "<span style='font-family:Fira Code,monospace;font-size:0.55rem;color:#2A4060;"
    "letter-spacing:0.06em;'>BOOTSTRAP · BAYESIAN · KELLY</span>"
    "<span style='font-family:Fira Code,monospace;font-size:0.55rem;color:#1A3050;'>|</span>"
    "<span style='font-family:Fira Code,monospace;font-size:0.55rem;color:#2A4060;"
    "letter-spacing:0.06em;'>-110 VIG REMOVED</span>"
    "</div></div>"
    "<div style='display:flex;gap:1rem;align-items:flex-start;flex-wrap:wrap;'>"
    "<div style='background:#04080F;border:1px solid #0A1828;border-radius:4px;"
    "padding:0.5rem 0.8rem;min-width:90px;text-align:center;'>"
    "<div style='font-family:Fira Code,monospace;font-size:0.50rem;color:#2A4060;"
    "letter-spacing:0.12em;margin-bottom:2px;'>BREAKEVEN</div>"
    "<div style='font-family:Fira Code,monospace;font-size:0.90rem;font-weight:700;"
    "color:#FFB800;letter-spacing:0.04em;'>52.4%</div>"
    "<div style='font-family:Fira Code,monospace;font-size:0.47rem;color:#1A3050;"
    "margin-top:1px;'>AT -110</div></div>"
    "<div style='background:#04080F;border:1px solid #0A1828;border-radius:4px;"
    "padding:0.5rem 0.8rem;min-width:90px;text-align:center;'>"
    "<div style='font-family:Fira Code,monospace;font-size:0.50rem;color:#2A4060;"
    "letter-spacing:0.12em;margin-bottom:2px;'>PP FLOOR</div>"
    "<div style='font-family:Fira Code,monospace;font-size:0.90rem;font-weight:700;"
    "color:#00AAFF;letter-spacing:0.04em;'>50.0%</div>"
    "<div style='font-family:Fira Code,monospace;font-size:0.47rem;color:#1A3050;"
    "margin-top:1px;'>PRIZEPICKS</div></div>"
    f"<div style='background:#04080F;border:1px solid #0A1828;border-radius:4px;"
    f"padding:0.5rem 0.8rem;min-width:110px;text-align:center;'>"
    f"<div style='font-family:Fira Code,monospace;font-size:0.50rem;color:#2A4060;"
    f"letter-spacing:0.12em;margin-bottom:2px;'>UTC</div>"
    f"<div style='font-family:Fira Code,monospace;font-size:0.70rem;font-weight:600;"
    f"color:#8BA8BF;letter-spacing:0.02em;'>{_now_str}</div>"
    f"<div style='font-family:Fira Code,monospace;font-size:0.47rem;color:#1A3050;"
    f"margin-top:1px;'>MARKET CLOCK</div></div>"
    "</div></div>"
    "<div style='margin-top:0.8rem;height:1px;"
    "background:linear-gradient(90deg,#00FFB230,#00AAFF20,transparent);'></div>"
    "</div>"
)
st.html(_hdr)
# ─── CARD & UI HELPERS ────────────────────────────────────────
def make_card(content_html, border_color="#112030", glow=False, accent_top=None):
    glow_css = f"box-shadow:0 0 28px {border_color}60,0 4px 16px rgba(0,0,0,0.4);" if glow else "box-shadow:0 2px 12px rgba(0,0,0,0.3);"
    top_border = f"border-top:2px solid {accent_top};" if accent_top else ""
    return (
        f"<div style=\"background:linear-gradient(135deg,#0C1524 0%,#0A1220 100%);"
        f"border:1px solid {border_color};{top_border}border-radius:6px;"
        f"padding:1.1rem 1.3rem;margin-bottom:0.9rem;{glow_css}"
        f"font-family:'Fira Code',monospace;transition:all 0.2s;\">"
        f"{content_html}</div>"
    )
def section_header(title, color="#3A5570", icon=""):
    return (
        f"<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;"
        f"font-weight:700;color:{color};letter-spacing:0.20em;text-transform:uppercase;"
        f"margin:1.2rem 0 0.6rem 0;display:flex;align-items:center;gap:0.5rem;'>"
        f"<div style='width:3px;height:12px;background:{color};border-radius:1px;opacity:0.7;'></div>"
        f"{icon}{title}"
        f"<div style='flex:1;height:1px;background:linear-gradient(90deg,{color}40,transparent);margin-left:0.5rem;'></div>"
        f"</div>"
    )
def color_for_edge(cat):
    if cat == "Strong Edge": return "#00FFB2"
    if cat == "Solid Edge":  return "#00AAFF"
    if cat == "Lean Edge":   return "#FFB800"
    return "#3A5570"
def prob_bar_html(p, line_pct=0.50, label=""):
    if p is None: return "<span style='color:#3A5570;font-size:0.72rem;'>—</span>"
    pct = int(round(p * 100))
    color = "#00FFB2" if p > 0.57 else ("#FFB800" if p > 0.52 else "#FF3358")
    glow  = f"box-shadow:0 0 6px {color}60;" if p > 0.57 else ""
    return (
        f"<div style='margin:0.35rem 0;'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.63rem;"
        f"color:#3A5570;margin-bottom:3px;font-family:Chakra Petch,monospace;letter-spacing:0.06em;'>"
        f"<span>{label}</span>"
        f"<span style='color:{color};font-weight:700;font-family:Fira Code,monospace;"
        f"font-size:0.70rem;{glow}'>{pct}%</span></div>"
        f"<div style='background:#06101E;border-radius:2px;height:5px;overflow:hidden;"
        f"border:1px solid #0A1828;'>"
        f"<div style='width:{pct}%;height:100%;background:linear-gradient(90deg,{color}AA,{color});"
        f"border-radius:2px;transition:width 0.5s cubic-bezier(0.4,0,0.2,1);'></div>"
        f"</div></div>"
    )
def regime_badge(label):
    cfg = {
        "Stable":  ("#00FFB2", "#00FFB215"),
        "Mixed":   ("#FFB800", "#FFB80015"),
        "Chaotic": ("#FF3358", "#FF335815"),
    }
    c, bg = cfg.get(label, ("#3A5570", "#3A557010"))
    return (
        f"<span style='background:{bg};border:1px solid {c}50;color:{c};"
        f"padding:2px 8px;border-radius:2px;font-size:0.58rem;letter-spacing:0.10em;"
        f"font-family:Chakra Petch,monospace;font-weight:600;'>{label.upper()}</span>"
    )
def hot_cold_badge(label):
    cfg = {
        "Hot":     ("#FF6B35", "#FF6B3515"),
        "Cold":    ("#00AAFF", "#00AAFF15"),
        "Average": ("#3A5570", "#3A557010"),
    }
    c, bg = cfg.get(label, ("#3A5570", "#3A557010"))
    return (
        f"<span style='background:{bg};border:1px solid {c}50;color:{c};"
        f"padding:2px 8px;border-radius:2px;font-size:0.58rem;letter-spacing:0.10em;"
        f"font-family:Chakra Petch,monospace;font-weight:600;'>{label.upper()}</span>"
    )
def sharpness_badge(score, tier, color):
    """Render composite sharpness score badge."""
    if score is None or tier is None: return ""
    return (
        f"<span style='background:{color}18;border:1px solid {color}55;color:{color};"
        f"padding:2px 8px;border-radius:2px;font-size:0.58rem;letter-spacing:0.08em;"
        f"font-family:Chakra Petch,monospace;font-weight:700;'>⚡ {tier} {score:.0f}</span>"
    )
def trend_badge(label):
    """Render trend convergence badge (Strong Bull / Bull / Neutral / Bear / Strong Bear)."""
    cfg = {
        "Strong Bull": ("#00FFB2", "#00FFB215"),
        "Bull":        ("#00AAFF", "#00AAFF15"),
        "Neutral":     ("#3A5570", "#3A557010"),
        "Bear":        ("#FF9500", "#FF950015"),
        "Strong Bear": ("#FF3358", "#FF335815"),
        "Insufficient":("#3A5570", "#3A557010"),
    }
    c, bg = cfg.get(label, ("#3A5570", "#3A557010"))
    return (
        f"<span style='background:{bg};border:1px solid {c}50;color:{c};"
        f"padding:2px 8px;border-radius:2px;font-size:0.58rem;letter-spacing:0.08em;"
        f"font-family:Chakra Petch,monospace;font-weight:600;'>TREND:{label.upper()}</span>"
    )
def confidence_tier_color(p_cal):
    """Return border color based on calibrated probability confidence tier."""
    if p_cal is None: return "#1E2D3D"
    p = float(p_cal)
    if p >= 0.65: return "#00FFB2"   # Green — high confidence
    if p >= 0.58: return "#00AAFF"   # Blue — solid
    if p >= 0.52: return "#FFB800"   # Amber — moderate
    return "#FF3358"                  # Red — marginal
def mv_badge(mv):
    if not mv or abs(mv.get("pips",0)) < 0.25: return ""
    pips = mv.get("pips",0)
    if mv.get("steam"): col,icon = "#00FFB2","STEAM"
    elif mv.get("fade"): col,icon = "#FF3358","FADE"
    else: col,icon = "#FFB800","MOVE"
    arrow = "UP" if pips > 0 else "DN"
    return f"<span style='color:{col};font-size:0.65rem;'>{icon} {arrow} {abs(pips):.1f}</span>"
# ─── SIDEBAR CSS INJECTION ─────────────────────────────────────
st.html("""
<div style="display:none"><style>
/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #04080F 0%, #070F1C 60%, #060C18 100%) !important;
    border-right: 1px solid #0E1E30 !important;
    min-width: 230px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 0.9rem 1rem 0.9rem !important;
}
/* ── Slider track & thumb ── */
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background: #0E1E30 !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderThumb"] {
    background: #00FFB2 !important;
    border: 2px solid #00FFB2 !important;
    box-shadow: 0 0 8px #00FFB255 !important;
    width: 14px !important;
    height: 14px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(3) {
    background: #00FFB2 !important;
}
/* ── Number input ── */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: #080F1A !important;
    border: 1px solid #0E1E30 !important;
    color: #00FFB2 !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 3px !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: #0E1E30 !important;
    border: 1px solid #1A2F45 !important;
    color: #4A9EBF !important;
}
/* ── Text input ── */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: #080F1A !important;
    border: 1px solid #0E1E30 !important;
    color: #EEF4FF !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.75rem !important;
    border-radius: 3px !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: #00FFB255 !important;
    box-shadow: 0 0 0 1px #00FFB222 !important;
}
/* ── Checkbox ── */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.70rem !important;
    color: #8BA8BF !important;
    letter-spacing: 0.03em !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-testid="stCheckboxWidget"] {
    accent-color: #00FFB2 !important;
}
/* ── Slider label text ── */
[data-testid="stSidebar"] [data-testid="stSlider"] label p,
[data-testid="stSidebar"] [data-testid="stNumberInput"] label p,
[data-testid="stSidebar"] [data-testid="stTextInput"] label p {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.63rem !important;
    color: #4A607A !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-bottom: 2px !important;
}
/* ── Slider value readout ── */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderTickBarMax"] {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.60rem !important;
    color: #2A4060 !important;
}
/* ── Expander header ── */
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-family: 'Chakra Petch', monospace !important;
    font-size: 0.62rem !important;
    color: #4A607A !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    background: #080F1A !important;
    border: 1px solid #0E1E30 !important;
    border-radius: 3px !important;
    padding: 0.35rem 0.6rem !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: #8BA8BF !important;
    border-color: #1A2F45 !important;
}
/* ── Scrollbar ── */
[data-testid="stSidebar"]::-webkit-scrollbar { width: 3px; }
[data-testid="stSidebar"]::-webkit-scrollbar-track { background: #04080F; }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background: #1A2F45; border-radius: 2px; }
/* ── Select box ── */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #080F1A !important;
    border: 1px solid #0E1E30 !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.72rem !important;
    color: #8BA8BF !important;
}
</style></div>
""")
# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    # ── BRAND HEADER ──────────────────────────────────────────
    st.markdown("""
<div style='
    padding: 1rem 0.2rem 0.8rem 0.2rem;
    border-bottom: 1px solid #0E1E30;
    margin-bottom: 0.8rem;
'>
    <div style='font-family:Chakra Petch,monospace;font-size:0.58rem;color:#1E4060;
                letter-spacing:0.25em;text-transform:uppercase;margin-bottom:0.2rem;'>
        QUANTITATIVE ENGINE
    </div>
    <div style='font-family:Chakra Petch,monospace;font-size:1.05rem;font-weight:700;
                color:#EEF4FF;letter-spacing:0.08em;line-height:1.1;'>
        NBA PROP<br>
        <span style='color:#00FFB2;'>ALPHA</span> ENGINE
    </div>
    <div style='display:flex;align-items:center;gap:0.5rem;margin-top:0.45rem;'>
        <div style='width:6px;height:6px;border-radius:50%;background:#00FFB2;
                    box-shadow:0 0 6px #00FFB2;'></div>
        <div style='font-family:Fira Code,monospace;font-size:0.58rem;color:#00FFB2;
                    letter-spacing:0.1em;'>LIVE  ·  v2.2</div>
        <div style='margin-left:auto;font-family:Fira Code,monospace;font-size:0.55rem;
                    color:#2A4060;'>NBA 2024-25</div>
    </div>
</div>
""", unsafe_allow_html=True)
    # ── ACCOUNT ───────────────────────────────────────────────
    _sid_user = st.session_state.get("_auth_user", "")
    _sid_email = _get_user_email(_sid_user)
    st.markdown(f"""
<div style='background:linear-gradient(135deg,#00FFB208,#00AAFF06);
            border:1px solid #0E2840;border-radius:4px;
            padding:0.5rem 0.7rem;margin-bottom:0.6rem;'>
    <div style='font-family:Fira Code,monospace;font-size:0.55rem;
                color:#2A6080;letter-spacing:0.10em;margin-bottom:3px;'>SIGNED IN AS</div>
    <div style='font-family:Chakra Petch,monospace;font-size:0.85rem;
                font-weight:700;color:#00FFB2;letter-spacing:0.06em;'>{_sid_user}</div>
    {f"<div style='font-family:Fira Code,monospace;font-size:0.58rem;color:#3A6080;margin-top:2px;'>{_sid_email}</div>" if _sid_email else ""}
</div>
""", unsafe_allow_html=True)
    # Use authenticated username as user_id
    user_id = _sid_user
    # ── BANKROLL ──────────────────────────────────────────────
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.55rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.4rem;'>
▸ BANKROLL</div>""", unsafe_allow_html=True)
    bankroll = st.number_input("Bankroll ($)", min_value=0.0,
        value=float(st.session_state.get("bankroll", 1000.0)), step=50.0)
    st.session_state["bankroll"] = float(bankroll)
    _lb = st.session_state.get("_last_saved_bankroll")
    if _lb is None or float(_lb) != float(bankroll):
        _bs = load_user_state(user_id)
        _bs["bankroll"] = float(bankroll)
        save_user_state(user_id, _bs)
        st.session_state["_last_saved_bankroll"] = float(bankroll)
    _br_val = float(st.session_state.get("bankroll", 1000.0))
    st.markdown(f"""
<div style='background:#04080F;border:1px solid #0E2840;border-radius:4px;
            padding:0.35rem 0.7rem;margin:0.2rem 0 0.8rem 0;
            display:flex;justify-content:space-between;align-items:center;'>
    <div style='font-family:Fira Code,monospace;font-size:0.55rem;color:#2A6080;'>CAPITAL</div>
    <div style='font-family:Fira Code,monospace;font-size:0.95rem;
                font-weight:700;color:#00FFB2;'>${_br_val:,.2f}</div>
</div>
""", unsafe_allow_html=True)
    # ── AI STATUS ─────────────────────────────────────────────
    _ai_active = _get_anthropic_key()
    if _ai_active:
        st.markdown("""
<div style='background:linear-gradient(90deg,#00FFB208,transparent);
            border:1px solid #00FFB230;border-radius:3px;
            padding:0.3rem 0.6rem;display:flex;align-items:center;gap:0.5rem;
            margin-bottom:0.4rem;'>
    <div style='width:5px;height:5px;border-radius:50%;background:#00FFB2;
                box-shadow:0 0 5px #00FFB2;'></div>
    <div style='font-family:Fira Code,monospace;font-size:0.60rem;color:#00FFB2;
                letter-spacing:0.06em;'>CLAUDE AI  ·  ACTIVE</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style='background:#04080F;border:1px solid #1A2030;border-radius:3px;
            padding:0.3rem 0.6rem;display:flex;align-items:center;gap:0.5rem;
            margin-bottom:0.4rem;'>
    <div style='width:5px;height:5px;border-radius:50%;background:#2A4060;'></div>
    <div style='font-family:Fira Code,monospace;font-size:0.60rem;color:#2A4060;
                letter-spacing:0.06em;'>CLAUDE AI  ·  OFFLINE</div>
</div>""", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:Fira Code,monospace;font-size:0.58rem;
color:#2A5070;margin-bottom:0.6rem;'>→ Configure in <b style='color:#8BA8BF;'>SETTINGS</b> tab</div>""",
        unsafe_allow_html=True)
    # ── SIGN OUT ──────────────────────────────────────────────
    st.markdown("""<div style='margin-top:0.4rem;padding-top:0.6rem;
border-top:1px solid #0E1E30;'></div>""", unsafe_allow_html=True)
    if st.button("SIGN OUT", use_container_width=True, key="sidebar_signout_btn"):
        _cookie_ctrl.remove("auth_user")
        for _k in list(st.session_state.keys()):
            del st.session_state[_k]
        st.rerun()
    # ── FOOTER ────────────────────────────────────────────────
    st.markdown(f"""
<div style='margin-top:1.5rem;padding-top:0.6rem;border-top:1px solid #080F1A;
            text-align:center;'>
    <div style='font-family:Fira Code,monospace;font-size:0.52rem;color:#1A2F45;
                letter-spacing:0.08em;line-height:1.8;'>
        SPORTSBOOK MODEL · -110 VIG REMOVED<br>
        KELLY STAKING · MC CORRELATED<br>
        <span style='color:#0E2840;'>━━━━━━━━━━━━━━━━</span><br>
        <span style='color:#1A3050;'>NBA QUANT ENGINE v2.2</span>
    </div>
</div>
""", unsafe_allow_html=True)
# ─── SESSION STATE INIT ────────────────────────────────────────
# Initialize settings defaults so Settings tab doesn't need to be visited first
_settings_defaults = {
    "market_prior_weight": 0.65,
    "n_games": 10,
    "frac_kelly": 0.25,
    "payout_multi": 3.0,
    "max_risk_per_bet": 3.0,
    "max_daily_loss": 15,
    "max_weekly_loss": 25,
    "exclude_chaotic": True,
    "show_unders": False,
    "max_req_day": 100,
}
for _sk, _sv in _settings_defaults.items():
    if _sk not in st.session_state:
        st.session_state[_sk] = _sv
# Load persisted PrizePicks cookies/JSON from disk so user never has to re-paste
_pp_disk = load_pp_settings()
if "pp_cookies" not in st.session_state:
    st.session_state["pp_cookies"]       = _pp_disk.get("pp_cookies", "")
    st.session_state["pp_relay_url"]     = _pp_disk.get("pp_relay_url", "")
    st.session_state["pp_auto_enabled"]  = _pp_disk.get("pp_auto_enabled", False)
    st.session_state["pp_auto_interval"] = _pp_disk.get("pp_auto_interval", 30)
if "pp_proxy_service" not in st.session_state:
    st.session_state["pp_proxy_service"] = _pp_disk.get("pp_proxy_service", "scrapingbee")
if "pp_proxy_key" not in st.session_state:
    st.session_state["pp_proxy_key"] = _pp_disk.get("pp_proxy_key", "")
if "_odds_api_key_override" not in st.session_state:
    _saved_odds_key = _pp_disk.get("odds_api_key", "")
    st.session_state["_odds_api_key_override"] = _saved_odds_key
    if _saved_odds_key:
        os.environ["ODDS_API_KEY"] = _saved_odds_key
# Restore background auto-fetcher state if it was enabled last session
if st.session_state.get("pp_auto_enabled"):
    set_pp_auto_fetch(
        enabled=True,
        interval_sec=int(st.session_state.get("pp_auto_interval", 30)) * 60,
        cookies=st.session_state.get("pp_cookies", ""),
        relay_url=st.session_state.get("pp_relay_url", ""),
    )
elif _pp_disk.get("pp_auto_enabled"):
    # Disk had it enabled — also restore
    set_pp_auto_fetch(
        enabled=True,
        interval_sec=int(_pp_disk.get("pp_auto_interval", 30)) * 60,
        cookies=_pp_disk.get("pp_cookies", ""),
        relay_url=_pp_disk.get("pp_relay_url", ""),
    )
for k in ["last_results","calibrator_map","scanner_offers","scanner_results"]:
    if k not in st.session_state: st.session_state[k] = None if k != "last_results" else []
# Restore scanner results from disk if session state was just reset (server restart / WS drop)
if st.session_state.get("scanner_results") is None:
    _disk_cache = _load_scanner_cache()
    if _disk_cache.get("scanner_results") is not None:
        for _ck, _cv in _disk_cache.items():
            st.session_state[_ck] = _cv
# Restore alert dedup hashes from disk so re-alerts don't fire after session reset
if "_scanner_alert_hashes" not in st.session_state:
    st.session_state["_scanner_alert_hashes"] = _load_alert_hashes()
# First Basket is a binary market with no numeric line — exclude from MODEL tab selector.
# All shooting volume markets (FGM/FGA/FTM/FTA/3PA) and specialty combos are now re-enabled;
# they're supported via PP/UD direct lines even if Odds API has no matching market key.
_MARKET_EXCLUDE_FROM_UI = {"First Basket", "Alt Points", "Alt Rebounds", "Alt Assists", "Alt 3PM"}
MARKET_OPTIONS = [k for k in ODDS_MARKETS.keys() if k not in _MARKET_EXCLUDE_FROM_UI]
def _daily_pnl(uid):
    h = load_history(uid)
    if h.empty: return 0.0
    try:
        h["ts_d"] = pd.to_datetime(h["ts"],errors="coerce").dt.date
        today_rows = h[h["ts_d"]==date.today()].copy()
        if today_rows.empty: return 0.0
        hits = (today_rows["result"]=="HIT").sum(); miss = (today_rows["result"]=="MISS").sum()
        return float(hits - miss)
    except: return 0.0
def _check_loss_stops(uid, bankroll):
    pnl = _daily_pnl(uid)
    if bankroll > 0 and pnl < 0 and abs(pnl)/bankroll*100 > float(st.session_state.get("max_daily_loss",15)):
        st.error(f"DAILY LOSS STOP HIT ({abs(pnl)/bankroll*100:.1f}%). No new bets recommended today.")
        return True
    return False
# ── Pull settings from session state for use in tab code ──────
bankroll           = float(st.session_state.get("bankroll", 1000.0))
n_games            = int(st.session_state.get("n_games", 10))
frac_kelly         = float(st.session_state.get("frac_kelly", 0.25))
payout_multi       = float(st.session_state.get("payout_multi", 3.0))
market_prior_weight= float(st.session_state.get("market_prior_weight", 0.65))
max_risk_per_bet   = float(st.session_state.get("max_risk_per_bet", 3.0))
max_daily_loss     = int(st.session_state.get("max_daily_loss", 15))
max_weekly_loss    = int(st.session_state.get("max_weekly_loss", 25))
exclude_chaotic    = bool(st.session_state.get("exclude_chaotic", True))
show_unders        = bool(st.session_state.get("show_unders", False))
# ─── TABS ─────────────────────────────────────────────────────
tabs = st.tabs(["MODEL", "RESULTS", "LIVE SCANNER", "PLATFORMS", "HISTORY", "CALIBRATION", "INSIGHTS", "ALERTS", "SETTINGS"])
# Consume staged scanner→model inputs before any widgets render (prevents StreamlitAPIException)
for _si in range(1, 5):
    if f"_staged_pname_{_si}" in st.session_state:
        st.session_state[f"pname_{_si}"]  = st.session_state.pop(f"_staged_pname_{_si}")
    if f"_staged_mkt_{_si}" in st.session_state:
        st.session_state[f"mkt_{_si}"]    = st.session_state.pop(f"_staged_mkt_{_si}")
    if f"_staged_mline_{_si}" in st.session_state:
        st.session_state[f"mline_{_si}"]  = st.session_state.pop(f"_staged_mline_{_si}")
    if f"_staged_manual_{_si}" in st.session_state:
        st.session_state[f"manual_{_si}"] = st.session_state.pop(f"_staged_manual_{_si}")
    if f"_staged_out_{_si}" in st.session_state:
        st.session_state[f"out_{_si}"]    = st.session_state.pop(f"_staged_out_{_si}")
if "_staged_model_date" in st.session_state:
    st.session_state["model_date"] = st.session_state.pop("_staged_model_date")
with tabs[0]:
    _loss_stop_hit = _check_loss_stops(user_id, bankroll)
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>CONFIGURE UP TO 4 LEGS</div>""", unsafe_allow_html=True)
    date_col, book_col = st.columns([2,2])
    with date_col:
        scan_date = st.date_input("Lines Date", value=date.today(), key="model_date")
    with book_col:
        book_choices, book_err = get_sportsbook_choices(scan_date.isoformat())
        if book_err:
            st.caption(f"⚠ Odds API: {book_err} — showing fallback books")
        elif set(book_choices) <= ({"consensus"} | set(_FALLBACK_BOOKS)):
            st.caption("ℹ No live odds for this date — showing default books")
        if "_staged_sportsbook" in st.session_state:
            _sb = st.session_state.pop("_staged_sportsbook")
            if _sb in book_choices:
                st.session_state["model_sportsbook"] = _sb
        sportsbook = st.selectbox("Sportsbook", options=book_choices, index=0, key="model_sportsbook")
    leg_configs = []
    for row_idx in range(2):
        cols = st.columns(2)
        for col_idx in range(2):
            leg_n = row_idx * 2 + col_idx + 1
            tag = f"P{leg_n}"
            with cols[col_idx]:
                st.markdown(f"<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem;'>LEG {leg_n}</div>", unsafe_allow_html=True)
                pname = st.text_input(f"Player", key=f"pname_{leg_n}", placeholder="e.g. LeBron James")
                mkt = st.selectbox(f"Market", options=MARKET_OPTIONS, key=f"mkt_{leg_n}")
                manual = st.checkbox(f"Manual line", key=f"manual_{leg_n}")
                if manual:
                    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;color:#FFB800;letter-spacing:0.10em;margin:-4px 0 2px 0;'>⚠ MANUAL — not from Odds API</div>", unsafe_allow_html=True)
                mline = st.number_input(f"Line", min_value=0.0, value=float(st.session_state.get(f"line_{leg_n}",22.5)), step=0.5, key=f"mline_{leg_n}")
                out_cb = st.checkbox(f"Key teammate OUT?", key=f"out_{leg_n}")
                leg_configs.append((tag, pname, mkt, manual, mline, out_cb))
    if st.session_state.get("_auto_run_model"):
        st.info("⚡ Legs loaded from Live Scanner — running model automatically...")
    run_btn = (st.button("RUN MODEL", use_container_width=True, disabled=_loss_stop_hit)
               or bool(st.session_state.pop("_auto_run_model", False)))
    if run_btn and not _loss_stop_hit:
        results = []; warnings = []; tasks = []
        for (tag, pname, mkt, manual, mline, teammate_out) in leg_configs:
            pname = (pname or "").strip()
            if not pname: continue
            market_key = ODDS_MARKETS.get(mkt)
            if not market_key: warnings.append(f"{tag}: unsupported market {mkt}"); continue
            line = mline; meta = None
            if not manual:
                val, m_meta, ferr = find_player_line_from_events(pname, market_key, scan_date.isoformat(), sportsbook)
                if val is not None:
                    line = float(val); meta = m_meta
                    st.success(f"{tag} - {pname} {mkt}: line {line:.1f} ({sportsbook})")
                else:
                    # Fallback: check stored PrizePicks lines before using manual
                    _pp_df = st.session_state.get("pp_lines")
                    _pp_line = None
                    if _pp_df is not None and not _pp_df.empty:
                        _norm_pname = normalize_name(pname)
                        for _, _r in _pp_df.iterrows():
                            if normalize_name(str(_r.get("player", ""))) == _norm_pname:
                                _r_mkt = map_platform_stat_to_market(_r.get("stat_type", ""))
                                _r_line = _r.get("line")
                                if _r_mkt == mkt and _r_line is not None and not pd.isna(_r_line):
                                    _pp_line = float(_r_line)
                                    break
                    if _pp_line is not None:
                        line = _pp_line
                        meta = {"event_id": None, "home_team": "", "away_team": "",
                                "commence_time": "", "price": 1.909,
                                "book": "prizepicks", "market_key": market_key, "side": "Over"}
                        st.success(f"{tag} - {pname} {mkt}: line {line:.1f} (PrizePicks)")
                    else:
                        st.warning(f"{tag} auto-line failed ({ferr}). Using manual {line:.1f}.")
            if not line or float(line) <= 0:
                warnings.append(f"{tag}: invalid line"); continue
            tasks.append((tag, pname, mkt, float(line), meta, bool(teammate_out)))
        if tasks:
            _inj_map = st.session_state.get("injury_team_map", {})
            with st.spinner("Computing projections..."):
                with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as ex:
                    futs = [ex.submit(compute_leg_projection, pname, mkt, line, meta,
                                      n_games=n_games, key_teammate_out=to,
                                      bankroll=bankroll, frac_kelly=frac_kelly,
                                      max_risk_frac=float(st.session_state.get("max_risk_per_bet",3.0))/100.0,
                                      market_prior_weight=market_prior_weight,
                                      exclude_chaotic=bool(exclude_chaotic),
                                      game_date=scan_date,
                                      injury_team_map=_inj_map)
                            for (tag, pname, mkt, line, meta, to) in tasks]
                    results = []
                    for f in futs:
                        try:
                            results.append(f.result(timeout=60))
                        except TimeoutError:
                            results.append({"player": "Error", "market": "?", "line": 0.0,
                                            "errors": ["thread timeout (≥60s)"],
                                            "gate_ok": False, "gate_reason": "thread timeout"})
                        except Exception as _te:
                            results.append({"player": "Error", "market": "?", "line": 0.0,
                                            "errors": [f"thread error: {type(_te).__name__}: {_te}"],
                                            "gate_ok": False, "gate_reason": "thread error"})
            calib = st.session_state.get("calibrator_map")
            results = [recompute_pricing_fields(dict(leg), calib) for leg in results]
            st.session_state["last_results"] = results
            if warnings:
                for w in warnings: st.warning(w)
    # [FIX 11] Log ALL evaluations (BET and PASS) for calibration
    st.markdown("<hr style='border-color:#1E2D3D;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<span style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#4A607A;letter-spacing:0.12em;'>LOG THIS SLATE</span>", unsafe_allow_html=True)
    c1, c2 = st.columns([2,1])
    with c1: placed = st.radio("Did you place?", ["No","Yes"], horizontal=True, index=0)
    with c2:
        if st.button("Confirm Log"):
            res = st.session_state.get("last_results") or []
            if not res:
                st.warning("Run model first.")
            else:
                decision = "BET" if placed == "Yes" else "PASS"
                result_val = "Pending" if placed == "Yes" else "SKIP"
                append_history(user_id, {
                    "ts":_now_iso(),"user_id":user_id,
                    "legs":json.dumps(res),"n_legs":len(res),
                    "leg_results":json.dumps(["Pending"]*len(res)),
                    "result":result_val,"decision":decision,"notes":""
                })
                st.success(f"Logged ({decision})")
with tabs[1]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>PROJECTION RESULTS & EDGE ANALYSIS</div>""", unsafe_allow_html=True)
    res = st.session_state.get("last_results") or []
    if not res:
        st.markdown(make_card("<span style='color:#4A607A;font-size:0.8rem;'>Run the model to see projections.</span>"), unsafe_allow_html=True)
    else:
        n_gated = sum(1 for l in res if l.get("gate_ok"))
        n_edge  = sum(1 for l in res if (l.get("ev_adj") or 0) > 0.01)
        total_stake = sum(l.get("stake",0) for l in res)
        m1c, m2c, m3c, m4c = st.columns(4)
        m1c.metric("Legs Analyzed", len(res))
        m2c.metric("Passed Gate", n_gated)
        m3c.metric("Positive EV", n_edge)
        m4c.metric("Total Rec. Stake", f"${total_stake:.2f}")
        st.markdown("")
        cols = st.columns(min(4, len(res)))
        for i, leg in enumerate(res):
            c = cols[i % len(cols)]
            with c:
                ec = color_for_edge(leg.get("edge_cat"))
                ev_pct = leg.get("ev_pct")
                ev_str = f"{ev_pct:+.1f}%" if ev_pct is not None else "--"
                proj_disp = f"{leg['proj']:.1f}" if leg.get("proj") is not None else "--"
                p_cal_v = leg.get("p_cal") or leg.get("p_over")
                n_used = leg.get("n_games_used",0)
                rest_d = leg.get("rest_days",2)
                rest_tag = "B2B" if rest_d==0 else f"{rest_d}d rest"
                mv = leg.get("line_movement") or {}
                mv_html = mv_badge(mv)
                sharp = leg.get("sharp_div") or {}
                sharp_html = ""
                if sharp.get("fade_model"):
                    sharp_html = "<div style='color:#FF3358;font-size:0.62rem;'>SHARP FADE</div>"
                elif sharp.get("confirm") == True:
                    sharp_html = "<div style='color:#00FFB2;font-size:0.62rem;'>SHARP CONFIRM</div>"
                hs = leg.get("headshot","")
                hs_html = f"<img src='{hs}' style='width:52px;height:38px;object-fit:cover;border-radius:2px;border:1px solid #1E2D3D;float:right;'>" if hs else ""
                card_html = f"""
<div style='margin-bottom:0.5rem;'>
  {hs_html}
  <div style='font-family:Chakra Petch,monospace;font-size:0.82rem;font-weight:700;color:#EEF4FF;'>{leg["player"]}</div>
  <div style='font-size:0.65rem;color:#4A607A;letter-spacing:0.08em;'>{leg.get("team","??")} vs {leg.get("opp","??")}</div>
  <div style='clear:both;'></div>
</div>
<div style='font-size:0.70rem;color:#4A607A;margin:0.15rem 0;text-transform:uppercase;letter-spacing:0.06em;'>{leg["market"]} | {rest_tag} | {leg.get("position_bucket","?")}</div>
<div style='display:flex;justify-content:space-between;margin:0.6rem 0;'>
  <div style='text-align:center;'><div style='font-size:0.60rem;color:#4A607A;'>LINE</div><div style='font-family:Fira Code,monospace;font-size:1.1rem;color:#EEF4FF;font-weight:500;'>{leg["line"]:.1f}</div></div>
  <div style='text-align:center;'><div style='font-size:0.60rem;color:#4A607A;'>PROJ</div><div style='font-family:Fira Code,monospace;font-size:1.1rem;color:#00AAFF;font-weight:500;'>{proj_disp}</div></div>
  <div style='text-align:center;'><div style='font-size:0.60rem;color:#4A607A;'>EV</div><div style='font-family:Fira Code,monospace;font-size:1.1rem;color:{ec};font-weight:600;'>{ev_str}</div></div>
</div>
{prob_bar_html(p_cal_v, label="P(OVER)")}
{prob_bar_html(leg.get("p_implied"), label="IMPLIED")}
<div style='margin-top:0.6rem;display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center;'>
  {regime_badge(leg.get("regime","?"))}
  {hot_cold_badge(leg.get("hot_cold","Average"))}
  {trend_badge(leg.get("trend_label","Neutral"))}
  {mv_html}
</div>
<div style='margin-top:0.35rem;display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center;'>
  {sharpness_badge(leg.get("sharpness_score"), leg.get("sharpness_tier"), leg.get("sharpness_color","#3A5570"))}
  {"<span style='font-size:0.58rem;color:#FF6B35;font-family:Chakra Petch,monospace;'>🔥 FATIGUE:{fat_lbl}</span>".format(fat_lbl=leg.get("fatigue_label","Normal")) if leg.get("fatigue_label","Normal") != "Normal" else ""}
  {"<span style='font-size:0.58rem;color:#00FFB2;font-family:Chakra Petch,monospace;'>OPP-TIRED:{ofl}</span>".format(ofl=leg.get("opp_fatigue_label","Normal")) if leg.get("opp_fatigue_label","Normal") != "Normal" else ""}
  {"<span style='font-size:0.58rem;color:#B8D0EC;font-family:Chakra Petch,monospace;'>TOT:{gt:.0f}</span>".format(gt=leg["game_total"]) if leg.get("game_total") else ""}
  {"<span style='font-size:0.58rem;color:#FF3358;font-family:Chakra Petch,monospace;'>⚠ BOTH-B2B</span>" if leg.get("both_b2b") else ""}
  {"<span style='font-size:0.58rem;color:#FF9500;font-family:Chakra Petch,monospace;'>✈ {tl}</span>".format(tl=leg.get("travel_label","")) if leg.get("travel_label","Normal") != "Normal" else ""}
  {"<span style='font-size:0.58rem;color:#FFB800;font-family:Chakra Petch,monospace;'>REVERT:{rl}</span>".format(rl=leg.get("reversion_label","")) if leg.get("reversion_label","Normal") not in ("Normal","Insufficient","") else ""}
  {"<span style='font-size:0.58rem;color:#00AAFF;font-family:Chakra Petch,monospace;'>DvP-L10:{dl}</span>".format(dl=leg.get("dvp_l10_label","Avg")) if leg.get("dvp_l10_label","Avg") != "Avg" else ""}
</div>
{sharp_html}
{"<div style='color:#FFA500;font-size:0.62rem;'>🏥 AUTO TEAMMATE OUT: " + (leg.get("auto_inj_player") or "").title() + "</div>" if leg.get("auto_inj") else ""}
<div style='margin-top:0.7rem;font-size:0.64rem;color:#4A607A;'>
  ctx x{leg.get("context_mult",1):.3f} | rest x{leg.get("rest_mult",1):.2f} | ha x{leg.get("ha_mult",1):.2f}<br>
  CV={f"{leg['volatility_cv']:.2f}" if leg.get("volatility_cv") else "--"} | N={n_used} games<br>
  Shrunk mu: {f"{leg['mu_shrunk']:.1f}" if leg.get("mu_shrunk") else "--"} | Trend: L3={f"{leg['l3_avg']:.1f}" if leg.get("l3_avg") else "--"}/L5={f"{leg['l5_avg']:.1f}" if leg.get("l5_avg") else "--"}/L10={f"{leg['l10_avg']:.1f}" if leg.get("l10_avg") else "--"}
</div>"""
                stake = safe_float(leg.get("stake"))
                if stake > 0:
                    card_html += f"<div style='margin-top:0.6rem;background:#00FFB218;border:1px solid #00FFB230;border-radius:2px;padding:0.4rem 0.6rem;font-size:0.72rem;color:#00FFB2;font-family:Fira Code,monospace;'>REC STAKE: ${stake:.2f} ({leg.get('stake_frac',0)*100:.1f}% BR)</div>"
                elif not leg.get("gate_ok"):
                    card_html += f"<div style='margin-top:0.6rem;background:#FF335818;border:1px solid #FF335830;border-radius:2px;padding:0.4rem 0.6rem;font-size:0.65rem;color:#FF3358;'>GATED: {leg.get('gate_reason','')}</div>"
                if leg.get("errors"):
                    card_html += "<div style='margin-top:0.4rem;font-size:0.60rem;color:#FFB800;'>" + "<br>".join(leg["errors"][:2]) + "</div>"
                # [UPGRADE 20] Confidence-tier border overrides edge color
                tier_border = confidence_tier_color(leg.get("p_cal"))
                st.markdown(make_card(card_html, border_color=tier_border, glow=(leg.get("edge_cat") in ["Strong Edge","Solid Edge"])), unsafe_allow_html=True)
                # [UPGRADE 19] Player card expander with per-game bar chart
                with st.expander(f"Drill-down: {leg['player']}", expanded=False):
                    pid = leg.get("player_id")
                    _mkt_exp = leg.get("market", "Points")
                    if pid:
                        gl_exp, _ = fetch_player_gamelog(player_id=pid, max_games=10)
                        if not gl_exp.empty:
                            s_exp = compute_stat_from_gamelog(gl_exp, _mkt_exp.replace("H1 ","").replace("H2 ",""))
                            s_exp = pd.to_numeric(s_exp, errors="coerce").dropna().reset_index(drop=True)
                            if not s_exp.empty:
                                chart_df = pd.DataFrame({
                                    "Game": [f"G-{i+1}" for i in range(len(s_exp))],
                                    "Actual": s_exp.values,
                                    "Line":   [float(leg.get("line", 0))] * len(s_exp),
                                })
                                chart_df = chart_df.set_index("Game")
                                st.caption(f"Last {len(s_exp)} games — {_mkt_exp}")
                                st.bar_chart(chart_df, use_container_width=True, height=180)
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Proj Minutes",
                              f"{leg.get('proj_minutes'):.0f}" if leg.get("proj_minutes") else "--",
                              delta="DNP risk" if leg.get("dnp_risk") else None,
                              delta_color="inverse")
                    d2.metric("vs Opponent", f"{leg.get('opp_specific_factor',1.0):.3f}x",
                              help=f"Based on {leg.get('n_vs_opp',0)} prior meetings")
                    d3.metric("Regime", leg.get("hot_cold","Average"),
                              delta=f"z={leg.get('hot_cold_z',0.0):+.2f}")
                    sigma = leg.get("sigma")
                    proj  = leg.get("proj")
                    if sigma and proj:
                        st.caption(f"Confidence band: {proj:.1f} ± {sigma:.1f} (μ±σ) — "
                                   f"{max(0,proj-sigma):.1f} to {proj+sigma:.1f}")
                    # [UPGRADE NEW] Sharpness components breakdown
                    _sc = leg.get("sharpness_components") or {}
                    if _sc:
                        st.markdown("**Sharpness Breakdown**")
                        _sc_cols = st.columns(4)
                        _sc_items = list(_sc.items())
                        for _ci, (_ck, _cv) in enumerate(_sc_items):
                            _col = _sc_cols[_ci % 4]
                            _clr = "#00FFB2" if _cv > 0 else ("#FF3358" if _cv < 0 else "#4A607A")
                            _col.markdown(
                                f"<div style='font-size:0.60rem;color:#4A607A;'>{_ck}</div>"
                                f"<div style='font-family:Fira Code,monospace;font-size:0.75rem;"
                                f"color:{_clr};font-weight:600;'>{_cv:+.1f}</div>",
                                unsafe_allow_html=True
                            )
                    # [UPGRADE NEW] Trend + game context detail
                    _gt = leg.get("game_total")
                    _gs = leg.get("game_spread")
                    _tl = leg.get("trend_label","Neutral")
                    _ts = leg.get("trend_slope",0.0)
                    _fl = leg.get("fatigue_label","Normal")
                    _ofl = leg.get("opp_fatigue_label","Normal")
                    _rl = leg.get("rolling_min_label", "Normal")
                    _rol = leg.get("reversion_label", "Normal")
                    _or = leg.get("over_rate_l10")
                    _trvl = leg.get("travel_label", "Normal")
                    _dvpl = leg.get("dvp_l10_label", "Avg")
                    _bbf = leg.get("both_b2b", False)
                    st.markdown(
                        f"<div style='margin-top:0.5rem;font-size:0.63rem;color:#4A607A;'>"
                        f"Trend: <span style='color:#EEF4FF;'>{_tl}</span> (slope {_ts:+.2f}/g) | "
                        f"Game Total: <span style='color:#EEF4FF;'>{f'{_gt:.0f}' if _gt else '--'}</span> | "
                        f"Spread: <span style='color:#EEF4FF;'>{f'{_gs:+.1f}' if _gs else '--'}</span><br>"
                        f"Player fatigue: <span style='color:#FF6B35;'>{_fl}</span> | "
                        f"Opp fatigue: <span style='color:#00FFB2;'>{_ofl}</span> | "
                        f"Travel: <span style='color:#FF9500;'>{_trvl}</span><br>"
                        f"Roll-min: <span style='color:#EEF4FF;'>{_rl}</span> | "
                        f"Both-B2B: <span style='color:{'#FF3358' if _bbf else '#4A607A'};'>{'YES' if _bbf else 'No'}</span> | "
                        f"DvP-L10: <span style='color:#00AAFF;'>{_dvpl}</span><br>"
                        f"Over-rate L10: <span style='color:#EEF4FF;'>{f'{_or*100:.0f}%' if _or is not None else '--'}</span> | "
                        f"Reversion signal: <span style='color:#FFB800;'>{_rol}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    # [v3.0] NEW SIGNALS SECTION
                    st.markdown("---")
                    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;color:#00FFB2;letter-spacing:0.18em;text-transform:uppercase;'>v3.0 Advanced Signals</div>", unsafe_allow_html=True)
                    _v3_cols = st.columns(3)
                    # W/L Split
                    _wl_lbl = leg.get("wl_label", "N/A")
                    _w_a = leg.get("w_avg")
                    _l_a = leg.get("l_avg")
                    _wl_col_str = "#00FFB2" if "W-Heavy" in str(_wl_lbl) else ("#FF3358" if "L-Heavy" in str(_wl_lbl) else "#4A607A")
                    _v3_cols[0].markdown(
                        f"<div style='font-size:0.58rem;color:#4A607A;'>WIN/LOSS SPLIT</div>"
                        f"<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_wl_col_str};'>{_wl_lbl}</div>"
                        f"<div style='font-size:0.55rem;color:#3A5570;'>W-prob:{leg.get('expected_win_prob',0.5)*100:.0f}%</div>",
                        unsafe_allow_html=True
                    )
                    # Clutch factor
                    _clutch_lbl = leg.get("clutch_label", "N/A")
                    _clutch_col = "#00FFB2" if "Elite" in str(_clutch_lbl) or "Clutch+" in str(_clutch_lbl) else ("#FF3358" if "Clutch-" in str(_clutch_lbl) else "#4A607A")
                    _v3_cols[1].markdown(
                        f"<div style='font-size:0.58rem;color:#4A607A;'>CLUTCH FACTOR</div>"
                        f"<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_clutch_col};'>{_clutch_lbl}</div>"
                        f"<div style='font-size:0.55rem;color:#3A5570;'>mult:{leg.get('clutch_factor',1.0):.3f}x</div>",
                        unsafe_allow_html=True
                    )
                    # Playoff context
                    _po_lbl = leg.get("playoff_label", "Regular")
                    _po_col = "#FF3358" if "Tanking" in str(_po_lbl) else ("#00FFB2" if "Bubble" in str(_po_lbl) else "#4A607A")
                    _v3_cols[2].markdown(
                        f"<div style='font-size:0.58rem;color:#4A607A;'>PLAYOFF CONTEXT</div>"
                        f"<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_po_col};'>{_po_lbl}</div>"
                        f"<div style='font-size:0.55rem;color:#3A5570;'>mult:{leg.get('playoff_factor',1.0):.3f}x</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
                    _v3_cols2 = st.columns(3)
                    # DNP Probability
                    _dnp_p = leg.get("dnp_prob_score", 0.05)
                    _dnp_lbl = leg.get("dnp_prob_label", "Minimal Risk")
                    _dnp_col = "#FF3358" if _dnp_p >= 0.40 else ("#FFB800" if _dnp_p >= 0.15 else "#00FFB2")
                    _v3_cols2[0].markdown(
                        f"<div style='font-size:0.58rem;color:#4A607A;'>DNP PROBABILITY</div>"
                        f"<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_dnp_col};'>{_dnp_p*100:.0f}%</div>"
                        f"<div style='font-size:0.55rem;color:#3A5570;'>{_dnp_lbl}</div>",
                        unsafe_allow_html=True
                    )
                    # 80% Confidence Interval
                    _ci_lo = leg.get("ci_lower_80")
                    _ci_hi = leg.get("ci_upper_80")
                    _ci_str = f"{_ci_lo:.1f}–{_ci_hi:.1f}" if (_ci_lo is not None and _ci_hi is not None) else "--"
                    _ci_width = (_ci_hi - _ci_lo) if (_ci_lo is not None and _ci_hi is not None) else 0
                    _ci_col = "#00FFB2" if _ci_width < (float(leg.get("line",20))*0.4) else ("#FFB800" if _ci_width < (float(leg.get("line",20))*0.7) else "#FF3358")
                    _v3_cols2[1].markdown(
                        f"<div style='font-size:0.58rem;color:#4A607A;'>80% CONF. INTERVAL</div>"
                        f"<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_ci_col};'>{_ci_str}</div>"
                        f"<div style='font-size:0.55rem;color:#3A5570;'>width:{_ci_width:.1f}</div>",
                        unsafe_allow_html=True
                    )
                    # FTA/Foul Rate
                    _fta_lbl = leg.get("fta_label", "N/A")
                    _fta_col = "#00FFB2" if "Foul-Heavy" in str(_fta_lbl) else ("#FF3358" if "Foul-Light" in str(_fta_lbl) else "#4A607A")
                    _v3_cols2[2].markdown(
                        f"<div style='font-size:0.58rem;color:#4A607A;'>OPP FOUL RATE</div>"
                        f"<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_fta_col};'>{_fta_lbl}</div>"
                        f"<div style='font-size:0.55rem;color:#3A5570;'>mult:{leg.get('fta_factor',1.0):.3f}x</div>",
                        unsafe_allow_html=True
                    )
                    # [v3.0] Referee / HCA / Pos B2B row
                    _v3_cols3 = st.columns(3)
                    _ref_lbl = leg.get("ref_label", "N/A")
                    _ref_tier = leg.get("ref_tier", "N/A")
                    _ref_col = "#00FFB2" if "Foul-Heavy" in str(_ref_tier) else ("#FF3358" if "Foul-Light" in str(_ref_tier) else "#4A607A")
                    _v3_cols3[0].markdown(
                        f"<div style='font-size:0.58rem;color:#4A607A;'>REFEREE CREW</div>"
                        f"<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_ref_col};'>{_ref_lbl}</div>"
                        f"<div style='font-size:0.55rem;color:#3A5570;'>{_ref_tier} | mult:{leg.get('ref_factor',1.0):.3f}x</div>",
                        unsafe_allow_html=True
                    )
                    _hca_lbl = leg.get("hca_label", "N/A")
                    _hca_f = leg.get("hca_factor", 1.0)
                    _hca_col = "#00FFB2" if _hca_f > 1.01 else ("#FF3358" if _hca_f < 0.99 else "#4A607A")
                    _v3_cols3[1].markdown(
                        f"<div style='font-size:0.58rem;color:#4A607A;'>TEAM HCA</div>"
                        f"<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_hca_col};'>{_hca_lbl}</div>"
                        f"<div style='font-size:0.55rem;color:#3A5570;'>mult:{_hca_f:.3f}x</div>",
                        unsafe_allow_html=True
                    )
                    _pb2b = leg.get("pos_b2b_mult", 1.0)
                    _pb2b_col = "#FF3358" if _pb2b < 0.996 else "#4A607A"
                    _pos_bkt = leg.get("pos_bucket", leg.get("position", "?"))
                    _v3_cols3[2].markdown(
                        f"<div style='font-size:0.58rem;color:#4A607A;'>POS B2B PENALTY</div>"
                        f"<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_pb2b_col};'>{_pos_bkt}</div>"
                        f"<div style='font-size:0.55rem;color:#3A5570;'>mult:{_pb2b:.3f}x</div>",
                        unsafe_allow_html=True
                    )
                    # [AUDIT v5.1] Altitude + Shooting Luck row
                    _alt_m = float(leg.get("altitude_mult", 1.0) or 1.0)
                    _sl_m  = float(leg.get("shoot_luck_mult", 1.0) or 1.0)
                    _sl_lbl = str(leg.get("shoot_luck_label", "N/A") or "N/A")
                    if abs(_alt_m - 1.0) > 0.002 or abs(_sl_m - 1.0) > 0.005:
                        _v5_cols = st.columns(2)
                        _alt_col_ui = "#FF3358" if _alt_m < 0.99 else ("#00FFB2" if _alt_m > 1.005 else "#4A607A")
                        _alt_lbl_ui = "At Altitude (DEN)" if _alt_m < 0.99 else ("DEN Away ⬆" if _alt_m > 1.005 else "Sea Level")
                        _v5_cols[0].markdown(
                            f"<div style='font-size:0.58rem;color:#4A607A;'>ALTITUDE</div>"
                            f"<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_alt_col_ui};'>{_alt_lbl_ui}</div>"
                            f"<div style='font-size:0.55rem;color:#3A5570;'>mult:{_alt_m:.3f}x</div>",
                            unsafe_allow_html=True
                        )
                        _sl_col_ui = "#FF3358" if _sl_m < 0.99 else ("#00FFB2" if _sl_m > 1.005 else "#4A607A")
                        _v5_cols[1].markdown(
                            f"<div style='font-size:0.58rem;color:#4A607A;'>SHOOTING LUCK</div>"
                            f"<div style='font-family:Fira Code,monospace;font-size:0.70rem;color:{_sl_col_ui};'>{_sl_lbl[:40]}</div>"
                            f"<div style='font-size:0.55rem;color:#3A5570;'>mult:{_sl_m:.3f}x</div>",
                            unsafe_allow_html=True
                        )
                    # Middle Opportunity Alert
                    if leg.get("middle_exists"):
                        _mid_lo = leg.get("middle_low")
                        _mid_hi = leg.get("middle_high")
                        _mid_p = leg.get("middle_prob", 0)
                        st.markdown(
                            f"<div style='margin-top:0.6rem;background:#FFB80018;border:1px solid #FFB80060;"
                            f"border-radius:4px;padding:0.5rem 0.8rem;'>"
                            f"<span style='font-family:Chakra Petch,monospace;font-size:0.60rem;color:#FFB800;"
                            f"letter-spacing:0.14em;'>⚡ MIDDLE OPPORTUNITY DETECTED</span><br>"
                            f"<span style='font-family:Fira Code,monospace;font-size:0.68rem;color:#EEF4FF;'>"
                            f"Bet OVER {_mid_lo:.1f} AND UNDER {_mid_hi:.1f} across books | "
                            f"Middle prob: ~{_mid_p*100:.1f}%</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    # Alt Line EV Table
                    _alt_evs = leg.get("alt_line_evs", [])
                    if _alt_evs:
                        st.markdown("<div style='margin-top:0.6rem;font-size:0.60rem;color:#4A607A;letter-spacing:0.12em;'>ALT LINE EV COMPARISON</div>", unsafe_allow_html=True)
                        _alt_html = "<div style='display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:0.2rem;'>"
                        for _alt in sorted(_alt_evs, key=lambda x: x["delta"]):
                            _ev_pct = round(_alt["ev"] * 100, 1) if _alt["ev"] is not None else None
                            _ev_col = "#00FFB2" if (_ev_pct and _ev_pct > 4) else ("#FFB800" if (_ev_pct and _ev_pct > 0) else "#FF3358")
                            _alt_html += (
                                f"<div style='background:#06101E;border:1px solid #0A1828;border-radius:3px;"
                                f"padding:0.3rem 0.5rem;text-align:center;min-width:55px;'>"
                                f"<div style='font-family:Fira Code,monospace;font-size:0.58rem;color:#EEF4FF;'>{_alt['line']}</div>"
                                f"<div style='font-family:Fira Code,monospace;font-size:0.62rem;color:{_ev_col};font-weight:700;'>"
                                f"{f'+{_ev_pct:.1f}%' if _ev_pct and _ev_pct > 0 else (f'{_ev_pct:.1f}%' if _ev_pct else '--')}</div>"
                                f"<div style='font-size:0.50rem;color:#2A4060;'>{_alt['edge_cat'][:4]}</div>"
                                f"</div>"
                            )
                        _alt_html += "</div>"
                        st.markdown(_alt_html, unsafe_allow_html=True)
                # 🤖 AI per-leg edge explanation
                if _get_anthropic_key():
                    _ai_leg_key = f"_ai_edge_{i}_{leg.get('player','x').replace(' ','_')[:20]}"
                    if st.button("🤖 AI Edge Analysis", key=f"ai_edge_btn_{i}", use_container_width=True):
                        with st.spinner("Claude analyzing edge…"):
                            _ai_txt = ai_explain_edge(
                                player=str(leg.get("player","?")),
                                market=str(leg.get("market","?")),
                                line=float(leg.get("line") or 0),
                                side=str(leg.get("side","Over")),
                                proj=float(leg.get("proj") or 0),
                                p_cal=float(leg.get("p_cal") or 0),
                                ev_pct=float(leg.get("ev_pct") or 0),
                                edge_cat=str(leg.get("edge_cat","?")),
                                hot_cold=str(leg.get("hot_cold","Average")),
                                rest_days=int(leg.get("rest_days") or 2),
                                dnp_risk=bool(leg.get("dnp_risk",False)),
                                b2b=bool((leg.get("rest_days") or 2) == 0),
                                opp=str(leg.get("opp","?")),
                                vol_cv=float(leg.get("volatility_cv") or 0),
                                n_games=int(leg.get("n_games_used") or 0),
                                errors_str=", ".join((leg.get("errors") or [])[:3]),
                                api_key=_get_anthropic_key(),
                                trend_label=str(leg.get("trend_label","Neutral")),
                                sharpness_score=leg.get("sharpness_score"),
                                sharpness_tier=str(leg.get("sharpness_tier","?")),
                                game_total=leg.get("game_total"),
                                fatigue_label=str(leg.get("fatigue_label","Normal")),
                                opp_fatigue_label=str(leg.get("opp_fatigue_label","Normal")),
                                l3_avg=leg.get("l3_avg"),
                                l5_avg=leg.get("l5_avg"),
                                l10_avg=leg.get("l10_avg"),
                            )
                        st.session_state[_ai_leg_key] = _ai_txt
                    _ai_leg_txt = st.session_state.get(_ai_leg_key)
                    if _ai_leg_txt:
                        st.markdown(
                            f"<div style='background:#00AAFF0D;border:1px solid #00AAFF2A;"
                            f"border-radius:3px;padding:0.55rem 0.75rem;margin-top:0.3rem;"
                            f"font-size:0.65rem;color:#B8D0EC;line-height:1.55;'>"
                            f"<span style='font-family:Chakra Petch,monospace;font-size:0.52rem;"
                            f"color:#00AAFF;letter-spacing:0.10em;display:block;margin-bottom:0.3rem;'>"
                            f"🤖 CLAUDE AI · EDGE ANALYSIS</span>{_ai_leg_txt}</div>",
                            unsafe_allow_html=True,
                        )
                # [FEATURE] Under flip card — zero extra API calls, reuses p_cal from above
                if bool(st.session_state.get("show_unders", False)):
                    _p_cal_u = leg.get("p_cal")
                    _p_imp_u_base = leg.get("p_implied")
                    if _p_cal_u is not None and _p_imp_u_base is not None:
                        _p_under = 1.0 - float(_p_cal_u)
                        _p_imp_u = 1.0 - float(_p_imp_u_base)
                        _vol_cv_u = leg.get("volatility_cv")
                        _skew_u = leg.get("stat_skewness")
                        _ev_u = (_p_under / _p_imp_u - 1.0) if _p_imp_u > 0 else None
                        _gate_u, _reason_u = passes_volatility_gate(_vol_cv_u, _ev_u, skew=_skew_u, bet_type="Under")
                        if _gate_u and not leg.get("dnp_risk") and _p_under >= 0.52:
                            _ev_u_str = f"{_ev_u*100:+.1f}%" if _ev_u is not None else "--"
                            _ec_u = color_for_edge(classify_edge(_ev_u))
                            _adv_u = round(_p_under - _p_imp_u, 3)
                            # Fractional Kelly stake for Under
                            _pd_u = leg.get("price_decimal")
                            _stake_u = 0.0
                            if _ev_u and _ev_u > 0 and _pd_u and float(_pd_u) > 1:
                                _kf_u = _ev_u / (float(_pd_u) - 1.0)
                                _stake_u = min(bankroll * frac_kelly * _kf_u, bankroll * 0.05)
                            _under_html = f"""<div style='background:#FF335808;border:1px solid #FF335830;border-radius:3px;padding:0.5rem 0.6rem;margin-top:0.3rem;'>
<div style='font-family:Chakra Petch,monospace;font-size:0.58rem;color:#FF8080;letter-spacing:0.12em;margin-bottom:0.3rem;'>↓ UNDER FLIP</div>
<div style='font-size:0.68rem;color:#EEF4FF;font-weight:600;'>{leg["player"]} U{leg["line"]:.1f}</div>
{prob_bar_html(_p_under, label="P(UNDER)")}
{prob_bar_html(_p_imp_u, label="IMPLIED")}
<div style='display:flex;justify-content:space-between;margin-top:0.35rem;font-size:0.64rem;'>
  <span style='color:#4A607A;'>EDGE</span><span style='color:{_ec_u};font-family:Fira Code,monospace;font-weight:600;'>{_ev_u_str}</span>
  <span style='color:#4A607A;'>ADV</span><span style='color:#EEF4FF;font-family:Fira Code,monospace;'>{_adv_u:+.3f}</span>
</div>
{f"<div style='margin-top:0.35rem;background:#00FFB218;border:1px solid #00FFB230;border-radius:2px;padding:0.25rem 0.5rem;font-size:0.63rem;color:#00FFB2;font-family:Fira Code,monospace;'>REC STAKE: ${_stake_u:.2f}</div>" if _stake_u > 0 else ""}
</div>"""
                            st.markdown(_under_html, unsafe_allow_html=True)
        # Multi-leg combo
        if len(res) >= 2:
            st.markdown("<hr style='border-color:#1E2D3D;margin:1rem 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.72rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.8rem;'>MULTI-LEG JOINT MONTE CARLO (CORRELATED)</div>", unsafe_allow_html=True)
            try:
                from scipy.stats import norm as _norm_mc
                valid_legs = [l for l in res if l.get("gate_ok") and l.get("p_cal") is not None]
                if len(valid_legs) < 2:
                    st.caption("Need >=2 gated legs for combo.")
                else:
                    n = len(valid_legs)
                    probs = np.array([float(l["p_cal"]) for l in valid_legs])
                    corr_mat = np.eye(n)
                    for i in range(n):
                        for j in range(i+1, n):
                            c = estimate_player_correlation(valid_legs[i], valid_legs[j])
                            corr_mat[i,j] = corr_mat[j,i] = c
                    evals, evecs = np.linalg.eigh(corr_mat)
                    evals = np.clip(evals, 1e-6, None)
                    corr_psd = evecs @ np.diag(evals) @ evecs.T
                    rng2 = np.random.default_rng(99)
                    z = rng2.multivariate_normal(np.zeros(n), corr_psd, 10000)
                    u = _norm_mc.cdf(z)
                    joint = float((u < probs).all(axis=1).mean())
                    payout_mult = float(st.session_state.get("payout_multi",3.0))
                    ev_combo = payout_mult * joint - 1.0
                    naive = float(np.prod(probs))
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Joint Hit Prob (MC)", f"{joint*100:.1f}%")
                    mc2.metric("Naive (uncorr)", f"{naive*100:.1f}%")
                    mc3.metric(f"Combo EV (x{payout_mult})", f"{ev_combo*100:+.1f}%")
            except ImportError:
                st.caption("scipy not available - joint MC skipped.")
            except Exception as e:
                st.caption(f"Joint MC error: {type(e).__name__}: {e}")
        with st.expander("Raw Data Table", expanded=False):
            display_cols = ["player","market","line","proj","p_cal","p_implied","advantage",
                            "ev_pct","edge_cat","gate_ok","stake","volatility_label","volatility_cv",
                            "regime","rest_days","position_bucket","context_mult","n_games_used",
                            "usage_rate","pos_def_mult","half_factor","pace_adj"]
            disp_df = pd.DataFrame([{k:l.get(k) for k in display_cols} for l in res])
            st.dataframe(disp_df, use_container_width=True)
        # ── Parlay Optimizer ──────────────────────────────────
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>PARLAY OPTIMIZER</div>", unsafe_allow_html=True)
        po_col1, po_col2 = st.columns(2)
        with po_col1:
            po_max_legs = st.slider("Max combo legs", 2, 4, 3, key="po_max_legs")
        with po_col2:
            po_payout = st.number_input("Payout multiplier (x)", 1.5, 20.0, float(st.session_state.get("payout_multi",3.0)), 0.5, key="po_payout")
        if st.button("Optimize Parlay Combos", use_container_width=True):
            combos = kelly_parlay_optimizer(res, po_payout, max_legs=po_max_legs, bankroll=bankroll, frac_kelly=frac_kelly)
            if combos:
                st.dataframe(pd.DataFrame(combos), use_container_width=True)
            else:
                st.warning("Need 2+ gated legs with P > 50% to generate combos.")
        # ── 🎰 DFS Entry Builder ─────────────────────────────────
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.72rem;color:#FFB800;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>🎰 DFS ENTRY BUILDER — PRIZEPICKS / UNDERDOG / SLEEPER</div>", unsafe_allow_html=True)
        dfs_c1, dfs_c2, dfs_c3 = st.columns(3)
        with dfs_c1:
            dfs_platform = st.selectbox("Platform", ["prizepicks","underdog","sleeper"], key="dfs_platform_sel")
        with dfs_c2:
            dfs_max_legs = st.slider("Max legs per entry", 2, 6,
                                     min(4, max(2, len([l for l in res if float(l.get("p_cal") or 0) > 0.50]))),
                                     key="dfs_max_legs")
        with dfs_c3:
            dfs_entry_type = st.radio("Entry type", ["Power Play", "Flex", "Best of both"],
                                      key="dfs_entry_type_radio", horizontal=True)
        # Show the platform's payout table inline
        _tbl = {"prizepicks": DFS_PP_PAYOUTS, "underdog": DFS_UD_PAYOUTS, "sleeper": DFS_SLEEPER_PAYOUTS}
        _pt = _tbl.get(dfs_platform, DFS_PP_PAYOUTS)
        _payout_cols = st.columns(len(_pt))
        for _pi, (_n, _x) in enumerate(_pt.items()):
            _payout_cols[_pi].metric(f"{_n}-leg PP", f"{_x}x",
                                     help=f"Must hit all {_n} legs to win {_x}x your entry")
        if st.button("🎰 Find Best DFS Entries", use_container_width=True, key="dfs_optimizer_btn"):
            _gated = [l for l in res if float(l.get("p_cal") or 0) > 0.50]
            if len(_gated) < 2:
                st.warning("Need 2+ legs with P(Over) > 50% to build entries.")
            else:
                with st.spinner(f"Optimizing {dfs_platform.title()} entries with MC simulation…"):
                    dfs_results = dfs_entry_optimizer(
                        _gated, platform=dfs_platform, max_legs=dfs_max_legs,
                        n_sims=5000, bankroll=bankroll, frac_kelly=frac_kelly,
                    )
                st.session_state["_dfs_entry_results"] = dfs_results
        _dfs_res = st.session_state.get("_dfs_entry_results")
        if _dfs_res:
            # Filter by entry type selection
            if dfs_entry_type == "Power Play":
                _dfs_show = [r for r in _dfs_res if r.get("rec_mode") == "PowerPlay"]
            elif dfs_entry_type == "Flex":
                _dfs_show = [r for r in _dfs_res if r.get("rec_mode") == "Flex"]
            else:
                _dfs_show = _dfs_res
            if not _dfs_show:
                st.caption("No entries matched that filter — try 'Best of both'.")
            else:
                dfs_df = pd.DataFrame(_dfs_show)[
                    ["n_legs","combo","rec_mode","joint_prob_%","pp_payout_x",
                     "pp_ev_%","flex_ev_%","best_ev_%","avg_leg_edge_%","rec_stake_$"]
                ].rename(columns={
                    "n_legs": "Legs", "combo": "Entry", "rec_mode": "Mode",
                    "joint_prob_%": "Joint%", "pp_payout_x": "PP Payout",
                    "pp_ev_%": "PP EV%", "flex_ev_%": "Flex EV%",
                    "best_ev_%": "Best EV%", "avg_leg_edge_%": "Avg Edge%",
                    "rec_stake_$": "Rec Stake $"
                })
                st.dataframe(dfs_df, use_container_width=True, hide_index=True)
                # Visual breakdown of top entry
                _best = _dfs_show[0]
                _pp_ev_val = _best.get("pp_ev_%")
                _fl_ev_val = _best.get("flex_ev_%")
                _pp_ev_num = _pp_ev_val if isinstance(_pp_ev_val, (int, float)) else 0
                _fl_ev_num = _fl_ev_val if isinstance(_fl_ev_val, (int, float)) else 0
                st.markdown(f"""
<div style='background:linear-gradient(135deg,#FFB80018,#0D1B2A);border:1px solid #FFB80050;
border-radius:6px;padding:0.9rem 1.1rem;margin-top:0.5rem;'>
<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;color:#FFB800;
letter-spacing:0.12em;margin-bottom:0.5rem;'>🎰 TOP {dfs_platform.upper()} ENTRY</div>
<div style='font-size:0.78rem;color:#EEF4FF;font-weight:600;margin-bottom:0.4rem;'>
{_best.get("combo","")}</div>
<div style='display:flex;gap:1.2rem;font-size:0.65rem;flex-wrap:wrap;'>
  <span style='color:#4A607A;'>Legs: <span style='color:#EEF4FF;'>{_best.get("n_legs")}</span></span>
  <span style='color:#4A607A;'>Mode: <span style='color:#FFB800;'>{_best.get("rec_mode")}</span></span>
  <span style='color:#4A607A;'>Joint Prob: <span style='color:#00AAFF;font-family:Fira Code,monospace;'>{_best.get("joint_prob_%")}%</span></span>
  <span style='color:#4A607A;'>PP Payout: <span style='color:#EEF4FF;font-family:Fira Code,monospace;'>{_best.get("pp_payout_x")}x</span></span>
  <span style='color:#4A607A;'>PP EV: <span style='color:{"#00FFB2" if _pp_ev_num > 0 else "#FF3358"};font-family:Fira Code,monospace;'>{_pp_ev_val}%</span></span>
  <span style='color:#4A607A;'>Flex EV: <span style='color:{"#00FFB2" if _fl_ev_num > 0 else "#FF3358"};font-family:Fira Code,monospace;'>{_fl_ev_val}%</span></span>
  <span style='color:#4A607A;'>Rec Stake: <span style='color:#00FFB2;font-family:Fira Code,monospace;'>${_best.get("rec_stake_$")}</span></span>
</div>
<div style='margin-top:0.5rem;font-size:0.60rem;color:#4A607A;'>
Individual legs 50% breakeven on {dfs_platform.title()} — edge is purely model probability above the 50% floor.
</div>
</div>""", unsafe_allow_html=True)
        # 🎯 AI Parlay Optimizer
        # [v3.0] AI DEEP DIVE — multi-signal comprehensive analysis
        if _get_anthropic_key() and len(res) >= 1:
            st.markdown("<hr style='border-color:#1E2D3D;margin:0.5rem 0;'>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;"
                "letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;'>"
                "🔬 AI DEEP DIVE — v3.0 FULL SIGNAL ANALYSIS</div>",
                unsafe_allow_html=True
            )
            if st.button("🔬 Run AI Deep Dive (All Signals)", use_container_width=True, key="ai_deepdive_btn"):
                with st.spinner("Claude running deep signal analysis…"):
                    _dd_data = [
                        {
                            "player": l.get("player"), "market": l.get("market"),
                            "line": l.get("line"), "side": l.get("side", "Over"),
                            "proj": round(float(l.get("proj") or 0), 2),
                            "p_cal_%": round(float(l.get("p_cal") or 0) * 100, 1),
                            "ev_%": round(float(l.get("ev_pct") or 0), 1),
                            "edge_cat": l.get("edge_cat"),
                            "sharpness": l.get("sharpness_score"),
                            "sharpness_tier": l.get("sharpness_tier"),
                            "hot_cold": l.get("hot_cold"),
                            "trend": l.get("trend_label"),
                            "wl_label": l.get("wl_label"),
                            "expected_win_prob_%": round(float(l.get("expected_win_prob", 0.5)) * 100, 1),
                            "clutch_label": l.get("clutch_label"),
                            "playoff_label": l.get("playoff_label"),
                            "dnp_prob_%": round(float(l.get("dnp_prob_score", 0)) * 100, 1),
                            "dnp_risk_label": l.get("dnp_prob_label"),
                            "ci_80": f"{l.get('ci_lower_80','?'):.1f}–{l.get('ci_upper_80','?'):.1f}" if l.get("ci_lower_80") else "--",
                            "middle_alert": f"YES: {l.get('middle_low')}–{l.get('middle_high')}" if l.get("middle_exists") else "No",
                            "fatigue": l.get("fatigue_label"),
                            "opp_fatigue": l.get("opp_fatigue_label"),
                            "both_b2b": l.get("both_b2b"),
                            "travel": l.get("travel_label"),
                            "fta_label": l.get("fta_label"),
                            "reversion": l.get("reversion_label"),
                            "over_rate_l10_%": round(float(l.get("over_rate_l10") or 0) * 100, 0),
                            "gate_ok": l.get("gate_ok"),
                        }
                        for l in res
                    ]
                    _deepdive_txt = ai_edge_deepdive(
                        json.dumps(_dd_data, indent=2), api_key=_get_anthropic_key()
                    )
                st.session_state["_ai_deepdive_result"] = _deepdive_txt
            _deepdive_txt = st.session_state.get("_ai_deepdive_result")
            if _deepdive_txt:
                _dd_html = _html.escape(_deepdive_txt).replace("\n", "<br>")
                st.markdown(
                    f"<div style='background:#00FFB20A;border:1px solid #00FFB228;"
                    f"border-radius:4px;padding:0.9rem 1.1rem;margin-top:0.5rem;"
                    f"font-size:0.68rem;color:#B0E8D0;line-height:1.65;'>"
                    f"<span style='font-family:Chakra Petch,monospace;font-size:0.55rem;"
                    f"color:#00FFB2;letter-spacing:0.1em;display:block;margin-bottom:0.6rem;'>"
                    f"🔬 CLAUDE AI · v3.0 DEEP DIVE</span>{_dd_html}</div>",
                    unsafe_allow_html=True,
                )
        if _get_anthropic_key() and len(res) >= 2:
            st.markdown("<hr style='border-color:#1E2D3D;margin:0.5rem 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;'>🎯 AI PARLAY OPTIMIZER</div>", unsafe_allow_html=True)
            if st.button("Generate AI Parlay Recommendations", use_container_width=True, key="ai_parlay_btn"):
                with st.spinner("Claude optimizing parlay combinations…"):
                    _legs_data = [
                        {
                            "player": l.get("player"), "market": l.get("market"),
                            "line": l.get("line"), "side": "Over",
                            "p_cal_%": round(float(l.get("p_cal") or 0) * 100, 1),
                            "ev_%": round(float(l.get("ev_pct") or 0), 1),
                            "edge_cat": l.get("edge_cat"), "team": l.get("team"),
                            "opp": l.get("opp"), "hot_cold": l.get("hot_cold"),
                            "gate_ok": l.get("gate_ok"),
                        }
                        for l in res
                    ]
                    _parlay_ai = ai_parlay_optimizer(json.dumps(_legs_data, indent=2), api_key=_get_anthropic_key())
                st.session_state["_ai_parlay_result"] = _parlay_ai
            _parlay_ai_txt = st.session_state.get("_ai_parlay_result")
            if _parlay_ai_txt:
                _parlay_html = _html.escape(_parlay_ai_txt).replace("\n", "<br>")
                st.markdown(
                    f"<div style='background:#00FFB20A;border:1px solid #00FFB228;"
                    f"border-radius:4px;padding:0.8rem 1rem;margin-top:0.5rem;"
                    f"font-size:0.68rem;color:#B0E8D0;line-height:1.6;'>"
                    f"<span style='font-family:Chakra Petch,monospace;font-size:0.55rem;"
                    f"color:#00FFB2;letter-spacing:0.1em;display:block;margin-bottom:0.5rem;'>"
                    f"🎯 CLAUDE AI · PARLAY OPTIMIZER</span>{_parlay_html}</div>",
                    unsafe_allow_html=True,
                )
        # ── PrizePicks AI Helper ──────────────────────────────
        if _get_anthropic_key():
            st.markdown("<hr style='border-color:#1E2D3D;margin:0.5rem 0;'>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#FFB800;"
                "letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;'>"
                "🏆 PRIZEPICKS AI HELPER — POWER PLAY vs FLEX</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Uses your sportsbook-calibrated model results (52.4% breakeven) as the selection "
                "filter, then determines the optimal PP entry mode. Core model math is unchanged."
            )
            _pp_dfs_res = st.session_state.get("_dfs_entry_results")
            if not _pp_dfs_res:
                st.info("Run 🎰 Find Best DFS Entries above first to generate PP entry combinations.")
            else:
                if st.button("🏆 Get PP Power Play / Flex Recommendation", use_container_width=True, key="ai_pp_helper_btn"):
                    with st.spinner("Claude analyzing Power Play vs Flex…"):
                        _pp_top = _pp_dfs_res[:6]  # Top 6 combos for context
                        _pp_legs_data = [
                            {
                                "player": l.get("player"),
                                "market": l.get("market"),
                                "line": l.get("line"),
                                "p_cal_%": round(float(l.get("p_cal") or 0) * 100, 1),
                                "sb_ev_%": round(float(l.get("ev_pct") or 0), 1),
                                "edge_cat": l.get("edge_cat"),
                                "team": l.get("team"),
                                "opp": l.get("opp"),
                                "hot_cold": l.get("hot_cold"),
                                "vol_cv": l.get("vol_cv"),
                                "dnp_risk": l.get("dnp_risk"),
                            }
                            for l in res if float(l.get("p_cal") or 0) > 0.50
                        ]
                        _pp_ai = ai_prizepicks_helper(
                            json.dumps(_pp_top, indent=2),
                            json.dumps(_pp_legs_data, indent=2),
                            api_key=_get_anthropic_key(),
                        )
                    st.session_state["_ai_pp_helper_result"] = _pp_ai
                _pp_ai_txt = st.session_state.get("_ai_pp_helper_result")
                if _pp_ai_txt:
                    _pp_html = _html.escape(_pp_ai_txt).replace("\n", "<br>")
                    st.markdown(
                        f"<div style='background:#FFB80009;border:1px solid #FFB80040;"
                        f"border-radius:4px;padding:0.8rem 1rem;margin-top:0.5rem;"
                        f"font-size:0.68rem;color:#F0D8A0;line-height:1.6;'>"
                        f"<span style='font-family:Chakra Petch,monospace;font-size:0.55rem;"
                        f"color:#FFB800;letter-spacing:0.1em;display:block;margin-bottom:0.5rem;'>"
                        f"🏆 CLAUDE AI · PRIZEPICKS HELPER</span>{_pp_html}</div>",
                        unsafe_allow_html=True,
                    )
        # ── Monte Carlo Simulation ────────────────────────────
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>MONTE CARLO GAME SIMULATION</div>", unsafe_allow_html=True)
        if st.button("Run MC Simulation (all legs)", use_container_width=True):
            mc = monte_carlo_game_sim(res, n_sims=20000, payout_mult=float(po_payout))
            if mc and "error" not in mc:
                mc_c1, mc_c2, mc_c3 = st.columns(3)
                mc_c1.metric("Joint Hit Prob (MC)", f"{mc['joint_prob_%']:.2f}%")
                mc_c2.metric("Naive (uncorr)", f"{mc['naive_joint_%']:.2f}%")
                mc_c3.metric(f"Combo EV (x{po_payout:.1f})", f"{mc['ev_%']:+.2f}%")
                if mc.get("per_leg_sim_%"):
                    st.caption("Per-leg simulated hit rates: " + " | ".join(
                        f"{res[i].get('player','?')} {p}%" for i, p in enumerate(mc["per_leg_sim_%"]) if i < len(res)
                    ))
            elif mc and "error" in mc:
                st.warning(f"MC error: {mc['error']}")
# ─── LIVE SCANNER TAB [FIX 13: persistent] [FIX 14: week-ahead] ───
with tabs[2]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>LIVE SCANNER - SWEEP ALL PLAYER PROPS FOR EDGES</div>""", unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns([2,2,2])
    with sc1:
        # [FIX 14] Week-ahead: date range selection
        scan_start = st.date_input("Start Date", value=date.today(), key="scan_start")
        scan_days = st.slider("Days ahead", 0, 7, 0, key="scan_days_ahead")
        scan_end = scan_start + timedelta(days=scan_days)
        if scan_days > 0:
            st.caption(f"Scanning {scan_start.isoformat()} to {scan_end.isoformat()}")
    with sc2:
        _qb1, _qb2, _qb3, _qb4 = st.columns(4)
        if _qb1.button("Core", key="qs_core", use_container_width=True):
            st.session_state["scanner_markets_sel"] = ["Points","Rebounds","Assists","3PM","PRA","PR","PA","RA"]
            st.rerun()
        if _qb2.button("+Halves", key="qs_half", use_container_width=True):
            st.session_state["scanner_markets_sel"] = [
                "Points","Rebounds","Assists","3PM","PRA",
                "H1 Points","H1 Rebounds","H1 Assists","H1 3PM","H1 PRA",
                "H2 Points","H2 Rebounds","H2 Assists","H2 PRA",
                "Q1 Points",
            ]
            st.rerun()
        if _qb3.button("+Shoot", key="qs_shoot", use_container_width=True):
            st.session_state["scanner_markets_sel"] = [
                "Points","Rebounds","Assists","3PM","PRA",
                "FGM","FGA","FTM","FTA","3PA","Blocks","Steals",
            ]
            st.rerun()
        if _qb4.button("ALL", key="qs_all", use_container_width=True):
            st.session_state["scanner_markets_sel"] = list(MARKET_OPTIONS)
            st.rerun()
        # Default: use whatever is in session state (buttons write directly to widget key)
        _qs_default = st.session_state.get("scanner_markets_sel", ["Points","Rebounds","Assists","3PM","PRA"])
        markets_sel = st.multiselect("Markets", options=MARKET_OPTIONS, default=_qs_default, key="scanner_markets_sel")
    with sc3:
        book_choices2, _book_err2 = get_sportsbook_choices(scan_start.isoformat())
        if "prizepicks" not in book_choices2:
            book_choices2 = ["prizepicks"] + book_choices2
        if _book_err2:
            st.caption(f"⚠ Odds API: {_book_err2} — showing fallback books")
        elif set(book_choices2) - {"prizepicks"} <= ({"consensus"} | set(_FALLBACK_BOOKS)):
            st.caption("ℹ No live odds for this date — showing default books")
        sportsbook2 = st.selectbox("Book", options=["all"]+book_choices2, index=0)
    sf1, sf2, sf3 = st.columns(3)
    with sf1:
        min_prob = st.slider("Min P(Over)", 0.50, 0.80, 0.53, 0.01)
        st.caption("PP/UD breakeven = 0.52 · Sportsbook (-110) breakeven = 0.524")
    with sf2:
        min_adv  = st.slider("Min Advantage vs Implied", 0.00, 0.12, 0.01, 0.005)
        st.caption("PP/UD p_implied = 0.50 · vs book = 0.524")
    with sf3:
        min_ev   = st.slider("Min EV (adj)", -0.05, 0.25, 0.01, 0.005)
    max_rows = st.slider("Max Results", 10, 200, 60, 10)
    # ── Platform Lines (PP / UD) inline in scanner ──────────────────────
    with st.expander("📱 Platform Lines — PrizePicks / Underdog / Sleeper (scan against platform lines)", expanded=False):
        _plat_c1, _plat_c2, _plat_c3, _plat_c4 = st.columns(4)
        with _plat_c1:
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;'>PRIZEPICKS</div>", unsafe_allow_html=True)
            _pp_auto_data, _pp_age, _ = get_pp_auto_lines()
            _pp_pkey = st.session_state.get("pp_proxy_key", "")
            if _pp_auto_data and _pp_age and _pp_age < 1800:
                st.markdown(f"<div style='font-size:0.58rem;color:#00FFB2;'>✓ {len(_pp_auto_data)} lines · {_pp_age//60}m ago</div>", unsafe_allow_html=True)
            elif _pp_pkey:
                st.caption("Proxy ready — click Fetch")
            else:
                st.caption("Set up in Settings → PP Connection")
            if st.button("Fetch PP Lines", key="scanner_fetch_pp_btn", use_container_width=True):
                with st.spinner("Fetching PrizePicks…"):
                    _pp_rows, _pp_err = fetch_prizepicks_lines()
                if _pp_err:
                    st.error(f"PP: {_pp_err}")
                elif _pp_rows:
                    st.session_state["pp_lines"] = pd.DataFrame(_pp_rows)
                    st.success(f"✓ {len(_pp_rows)} PP props")
                else:
                    st.warning("PP: 0 props — slate may not be posted yet")
            _pp_loaded = st.session_state.get("pp_lines")
            if _pp_loaded is not None and not _pp_loaded.empty:
                st.markdown(f"<div style='font-size:0.58rem;color:#00FFB2;'>✓ {len(_pp_loaded)} ready</div>", unsafe_allow_html=True)
            _scanner_auto = st.toggle(
                "Auto-refresh PP",
                value=st.session_state.get("pp_auto_enabled", False),
                key="scanner_pp_auto_toggle",
                help=f"Fetches every {st.session_state.get('pp_auto_interval', 30)} min. Configure interval in Platforms tab.",
            )
            if _scanner_auto != st.session_state.get("pp_auto_enabled", False):
                st.session_state["pp_auto_enabled"] = _scanner_auto
                save_pp_settings(pp_auto_enabled=_scanner_auto)
                set_pp_auto_fetch(
                    enabled=_scanner_auto,
                    interval_sec=int(st.session_state.get("pp_auto_interval", 30)) * 60,
                    cookies=st.session_state.get("pp_cookies", ""),
                    relay_url=st.session_state.get("pp_relay_url", ""),
                    proxy_service=st.session_state.get("pp_proxy_service", ""),
                    proxy_key=st.session_state.get("pp_proxy_key", ""),
                )
                st.rerun()
        with _plat_c2:
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#FFB800;'>UNDERDOG</div>", unsafe_allow_html=True)
            if st.button("Fetch UD Lines", key="scanner_fetch_ud_btn", use_container_width=True):
                with st.spinner("Fetching Underdog…"):
                    _ud_rows, _ud_err = fetch_underdog_lines()
                if _ud_err:
                    st.error(f"UD: {_ud_err}")
                elif _ud_rows:
                    st.session_state["ud_lines"] = pd.DataFrame(_ud_rows)
                    st.success(f"✓ {len(_ud_rows)} UD props")
                else:
                    st.warning("UD: 0 props found")
            _ud_loaded = st.session_state.get("ud_lines")
            if _ud_loaded is not None and not _ud_loaded.empty:
                st.markdown(f"<div style='font-size:0.58rem;color:#FFB800;'>✓ {len(_ud_loaded)} ready</div>", unsafe_allow_html=True)
        with _plat_c3:
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#9B59F5;'>SLEEPER</div>", unsafe_allow_html=True)
            _sl_loaded_sc = st.session_state.get("sl_lines")
            if _sl_loaded_sc is not None and not _sl_loaded_sc.empty:
                st.markdown(f"<div style='font-size:0.58rem;color:#9B59F5;'>✓ {len(_sl_loaded_sc)} ready</div>", unsafe_allow_html=True)
            else:
                st.caption("Import via Platforms → Sleeper tab")
        with _plat_c4:
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#4A607A;'>SCAN SOURCE</div>", unsafe_allow_html=True)
            _scan_source = st.radio(
                "Lines to scan",
                ["Odds API only", "PP + UD only", "All sources"],
                index=int(st.session_state.get("scan_source_idx", 0)),
                key="scan_source_radio",
                label_visibility="collapsed",
            )
            st.session_state["scan_source_idx"] = ["Odds API only","PP + UD only","All sources"].index(_scan_source)
            st.caption("PP/UD/Sleeper: 50% implied (no vig)  Odds API: from market price")
    # ── Last-fetch indicator + Force Refresh ──────────────────────
    _last_fetch_ts  = st.session_state.get("_scanner_lines_fetch_ts")
    _last_fetch_lbl = ""
    _lines_stale    = True
    if _last_fetch_ts:
        _age_sec = time.time() - _last_fetch_ts
        _age_min = int(_age_sec / 60)
        if _age_min < 60:
            _last_fetch_lbl = f"Lines fetched {_age_min}m ago"
        else:
            _last_fetch_lbl = f"Lines fetched {_age_min//60}h {_age_min%60}m ago"
        _lines_stale = _age_sec > _LINES_CACHE_TTL   # stale after 2 hours
    fetch_col, refresh_col, scan_col = st.columns([3, 2, 3])
    if _last_fetch_ts and not _lines_stale:
        fetch_col.info(f"✓ {_last_fetch_lbl} (cached 2h)")
    _do_fetch = fetch_col.button(
        "Fetch Live Lines (Odds API)" if not _last_fetch_ts else "Re-fetch Lines",
        use_container_width=True,
        help="Lines are cached 2 hours to preserve API quota. Use Force Refresh to bypass.",
    )
    if refresh_col.button("Force Refresh", use_container_width=True,
                          help="Clears 2-hour cache and re-fetches from Odds API. Uses API credits."):
        odds_get_events.clear()
        odds_get_events_range.clear()
        odds_get_event_odds.clear()
        st.session_state.pop("_scanner_lines_fetch_ts", None)
        st.session_state.pop("scanner_offers", None)
        st.toast("Cache cleared — click Fetch Live Lines to refresh.")
    if _do_fetch:
        selected_keys = list(dict.fromkeys(ODDS_MARKETS.get(m) for m in markets_sel if ODDS_MARKETS.get(m)))
        if not selected_keys:
            st.warning("Select at least one market.")
        else:
            # [FIX 14] Fetch across date range
            if scan_days > 0:
                evs, err = odds_get_events_range(scan_start.isoformat(), scan_end.isoformat())
            else:
                evs, err = odds_get_events(scan_start.isoformat())
            if err: st.error(err)
            elif not evs: st.warning("No events found for that date. Note: late games (7–10 PM ET) are now mapped to Eastern date correctly.")
            else:
                offers = []
                _fetch_errors = []
                # Warn about markets with no confirmed Odds API key (PP/UD/Sleeper only)
                _unsupported_sel = [m for m in markets_sel if m in ODDS_API_UNSUPPORTED_MARKETS]
                _supported_sel   = [m for m in markets_sel if m not in ODDS_API_UNSUPPORTED_MARKETS]
                if _unsupported_sel:
                    st.warning(
                        f"**{', '.join(_unsupported_sel)}** have no confirmed Odds API key and will be skipped. "
                        "Use PP / UD / Sleeper source for these markets."
                    )
                if not _supported_sel:
                    st.warning("No Odds-API-supported markets selected. Switch to PP + UD source or add standard markets.")
                else:
                    # Rebuild selected_keys from supported markets only
                    selected_keys = list(dict.fromkeys(
                        ODDS_MARKETS.get(m) for m in _supported_sel if ODDS_MARKETS.get(m)
                    ))
                    # [FIX H1/H2/ALT] Split market keys into standard and specialty batches
                    # Odds API supports max ~6 markets per call reliably; specialty markets
                    # (H1/H2/Q1/Alt/Fantasy) need separate calls as not all books offer them
                    std_keys  = [k for k in selected_keys if k not in SPECIALTY_MARKET_KEYS]
                    spec_keys = [k for k in selected_keys if k in SPECIALTY_MARKET_KEYS]
                    market_batches = []
                    BATCH_SIZE = 6
                    if std_keys:
                        for i in range(0, len(std_keys), BATCH_SIZE):
                            market_batches.append(std_keys[i:i+BATCH_SIZE])
                    # Specialty markets individually (each book may only carry a subset)
                    for sk in spec_keys:
                        market_batches.append([sk])
                    # Inform user about specialty market timing
                    _SPEC_TIMING_MARKETS = {
                        "H1 Points","H1 Rebounds","H1 Assists","H1 3PM","H1 PRA",
                        "H2 Points","H2 Rebounds","H2 Assists","H2 PRA",
                        "Q1 Points","Q1 Rebounds","Q1 Assists",
                        "Double Double","Triple Double","Fantasy Score",
                    }
                    _sel_spec_timing = [m for m in markets_sel if m in _SPEC_TIMING_MARKETS]
                    if _sel_spec_timing and _scan_source not in ("PP + UD only",) and sportsbook2 not in ("prizepicks", "underdog", "sleeper"):
                        st.info(
                            f"**{', '.join(_sel_spec_timing)}** — these specialty markets typically open on "
                            "DraftKings/FanDuel **1–2 hours before tip-off**. If you see 0 lines, try "
                            "Force Refresh closer to game time. PP/UD may have them all day."
                        )
                    # Track per-specialty-market counts for debug output
                    _spec_market_counts: dict[str, int] = {m: 0 for m in _supported_sel if ODDS_MARKETS.get(m) in SPECIALTY_MARKET_KEYS}
                    for ev in evs:
                        eid = ev.get("id")
                        if not eid: continue
                        for batch_keys in market_batches:
                            # For specialty markets try all regions for better coverage
                            regions = "us,us2,eu,uk" if any(k in SPECIALTY_MARKET_KEYS for k in batch_keys) else REGION_US
                            odds, oerr = odds_get_event_odds(eid, tuple(batch_keys), regions=regions)
                            if oerr:
                                # H1/H2/Q1 markets commonly return errors pre-tip — expected, not alarming
                                _is_half_batch = any(k in batch_keys for k in [
                                    "player_points_q1q2", "player_rebounds_q1q2", "player_assists_q1q2",
                                    "player_threes_q1q2", "player_points_rebounds_assists_q1q2",
                                    "player_points_q3q4", "player_rebounds_q3q4", "player_assists_q3q4",
                                    "player_points_rebounds_assists_q3q4",
                                    "player_points_q1", "player_rebounds_q1", "player_assists_q1",
                                ])
                                if _is_half_batch:
                                    _fetch_errors.append(f"H1/H2/Q1 ({','.join(batch_keys)}): {oerr} [EXPECTED — opens 1-2h pre-tip; use PP/UD for all-day H1/H2 lines]")
                                else:
                                    _fetch_errors.append(f"{','.join(batch_keys)}: {oerr}")
                                continue
                            if not odds: continue
                            _is_spec_batch = any(k in SPECIALTY_MARKET_KEYS for k in batch_keys)
                            for m in _supported_sel:
                                mk = ODDS_MARKETS.get(m)
                                if not mk or mk not in batch_keys: continue
                                # For specialty markets fetch all books — many books (DK, FD, etc.)
                                # carry different subsets of H1/H2/FGM/DD/TD etc.
                                # Applying the book filter here would silently drop all lines when
                                # the selected book doesn't carry that particular specialty market.
                                bf = None if _is_spec_batch else (sportsbook2 if sportsbook2 not in ("all", "prizepicks") else None)
                                parsed, _ = _parse_player_prop_outcomes(odds, mk, book_filter=bf)
                                if parsed and m in _spec_market_counts:
                                    _spec_market_counts[m] += len(parsed)
                                offers.extend([{**r,"market":m} for r in parsed])
                    if offers:
                        # [FIX 13] Store in session state - persists across tab switches
                        st.session_state["scanner_offers"] = pd.DataFrame(offers)
                        st.session_state["_scanner_lines_fetch_ts"] = time.time()
                        # Auto-save to prop line history DB + [UPGRADE 10] opening line capture
                        for r2 in offers:
                            save_prop_line(r2.get("player",""), r2.get("market",""),
                                           r2.get("line"), r2.get("price"), r2.get("book"),
                                           event_id=r2.get("event_id"))
                            pn2 = normalize_name(r2.get("player",""))
                            mk2 = r2.get("market_key", ODDS_MARKETS.get(r2.get("market",""), ""))
                            side2 = r2.get("side", "Over")
                            save_opening_line(pn2, mk2, side2, r2.get("line", 0), r2.get("price"))
                        st.success(f"Fetched {len(offers)} raw prop outcomes — opening lines captured. Cached for 2 hours.")
                    else:
                        st.warning("No offers returned.")
                    # Show per-specialty-market debug info for any that returned 0 lines
                    _empty_spec = [m for m, cnt in _spec_market_counts.items() if cnt == 0]
                    if _empty_spec or _fetch_errors:
                        with st.expander(f"Debug ({len(_fetch_errors)} API errors, {len(_empty_spec)} empty specialty markets)"):
                            if _empty_spec:
                                st.caption(
                                    "**Specialty markets with 0 lines** — H1/H2/DD/TD/Fantasy Score open "
                                    "1–2h before tip-off on most books. Use Force Refresh near game time: "
                                    + ", ".join(_empty_spec)
                                )
                            for _fe in _fetch_errors:
                                st.caption(_fe)
    # ── Bulk game log loader (recommended before large scans) ──────
    bulk_loaded = _fetch_bulk_gamelogs() is not None
    _bulk_label = "✓ All Game Logs Loaded" if bulk_loaded else "Load All Game Logs (Recommended)"
    if scan_col.button(_bulk_label, use_container_width=True, disabled=bulk_loaded):
        with st.spinner("Loading all NBA player game logs (one-time, cached 6h)..."):
            _fetch_bulk_gamelogs.clear()
            result = _fetch_bulk_gamelogs()
        if result is not None:
            st.success(f"Loaded {len(result):,} game log rows — scans are now near-instant.")
        else:
            st.warning("Bulk load failed — scanner will fall back to per-player fetches.")
    if scan_col.button("Run Scan", use_container_width=True):
        _scan_source = st.session_state.get("scan_source_radio", "Odds API only")
        df = st.session_state.get("scanner_offers")
        _use_odds_api = _scan_source in ("Odds API only", "All sources") and sportsbook2 != "prizepicks"
        _use_platforms = _scan_source in ("PP + UD only", "All sources") or sportsbook2 == "prizepicks"
        # Validate we have something to scan
        _odds_has_data = df is not None and not (hasattr(df, 'empty') and df.empty)
        _pp_has_data   = bool(st.session_state.get("pp_lines") is not None and not st.session_state["pp_lines"].empty)
        _ud_has_data   = bool(st.session_state.get("ud_lines") is not None and not st.session_state["ud_lines"].empty)
        if _use_odds_api and not _odds_has_data and not _use_platforms:
            st.warning("Fetch Odds API lines first, or switch to PP + UD source.")
        elif _use_platforms and not _pp_has_data and not _ud_has_data and not _use_odds_api:
            st.warning("Load PrizePicks or Underdog lines first (expand 'Platform Lines' above).")
        else:
            candidates = []
            # ── Odds API candidates ────────────────────────────────────────
            if _use_odds_api and _odds_has_data:
                df2 = df[df["side"].str.lower().isin(["over","o"])].copy()
                if df2.empty:
                    st.warning("No OVER props found in Odds API offers. Fetch lines again.")
                for _, r in df2.iterrows():
                    pname = (r.get("player") or "").strip()
                    mkt   = (r.get("market") or "").strip()
                    line  = r.get("line")
                    if not pname or pd.isna(line) or not mkt: continue
                    if mkt not in markets_sel: continue
                    meta = {"event_id":r.get("event_id"),"home_team":r.get("home_team"),
                            "away_team":r.get("away_team"),"commence_time":r.get("commence_time"),
                            "price":r.get("price"),"book":r.get("book"),
                            "market_key":ODDS_MARKETS.get(mkt),"side":r.get("side","Over")}
                    candidates.append((pname, mkt, float(line), meta))
            # ── Platform candidates (PP + UD + Sleeper) ────────────────────
            _unmapped_stats = set()   # Track stat_types with no mapping
            _half_candidate_count = 0  # Count H1/H2/Q1 candidates built
            if _use_platforms:
                for _plat_store, _plat_label in [("pp_lines","prizepicks"), ("ud_lines","underdog"), ("sl_lines","sleeper")]:
                    if sportsbook2 == "prizepicks" and _plat_label != "prizepicks":
                        continue
                    _plat_df = st.session_state.get(_plat_store)
                    if _plat_df is None or _plat_df.empty:
                        continue
                    for _, r in _plat_df.iterrows():
                        pname = (r.get("player") or "").strip()
                        stat_t = r.get("stat_type","")
                        mkt = map_platform_stat_to_market(stat_t)
                        if not mkt:
                            _unmapped_stats.add(stat_t)
                            continue
                        if mkt not in MARKET_OPTIONS:
                            continue
                        if mkt not in markets_sel:
                            continue
                        line = r.get("line")
                        if not pname or line is None or pd.isna(line):
                            continue
                        # Skip PrizePicks goblin/demon alternate lines — adjusted multipliers
                        # make EV% misleading (goblins pay ~0.85x, not 1x; demons pay ~1.25x).
                        if _plat_label == "prizepicks":
                            _ot = str(r.get("odds_type", "standard") or "standard").lower()
                            if _ot not in ("standard", ""):
                                continue
                        # Platforms use true 50/50 (no vig): decimal 2.0 = even money.
                        # IMPORTANT: using 1.909 (-110) was WRONG — it set p_implied=0.524
                        # instead of 0.50, making every PP leg appear ~2.4% less sharp
                        # and causing high-variance markets (3PM, Blocks, Steals) to be
                        # dropped by the p_cal < min_prob gate even when they had real edges.
                        _plat_price = 2.0 if _plat_label in ("prizepicks", "underdog", "sleeper") else 1.909
                        meta = {
                            "event_id": None, "home_team": "", "away_team": "",
                            "commence_time": "", "price": _plat_price,
                            "book": _plat_label,
                            "market_key": ODDS_MARKETS.get(mkt), "side": "Over",
                        }
                        candidates.append((pname, mkt, float(line), meta))
                        if any(h in mkt for h in ("H1", "H2", "Q1")):
                            _half_candidate_count += 1
            # ── Diagnostics: unmapped stats + half-game candidate count ─────
            if _unmapped_stats:
                _unmapped_display = sorted(_unmapped_stats)[:12]
                st.caption(
                    f"⚠ {len(_unmapped_stats)} unmapped PP/UD stat type(s): "
                    f"{', '.join(_unmapped_display)}"
                    f"{'...' if len(_unmapped_stats) > 12 else ''}"
                )
            _selected_half = [m for m in markets_sel if any(h in m for h in ("H1", "H2", "Q1"))]
            if _selected_half:
                if _half_candidate_count > 0:
                    st.caption(f"📊 {_half_candidate_count} half/quarter candidates from platform lines")
                else:
                    st.warning(
                        f"0 half/quarter candidates found for {', '.join(_selected_half)}. "
                        "Check: (1) PP/UD lines are loaded, (2) today's slate includes H1/H2 props, "
                        "(3) look at the unmapped stats above for PP's exact stat_type strings."
                    )
            out_rows, dropped = [], []
            all_computed_legs = []  # [FEATURE] Stores all computed legs for Under scan
            if candidates:
                _inj_map = st.session_state.get("injury_team_map", {})
                # Auto-load bulk game logs if not already cached (one-time, ~15-30s)
                bulk_ready = _fetch_bulk_gamelogs() is not None
                if not bulk_ready:
                    with st.spinner("Loading all NBA game logs (one-time ~20s)..."):
                        _fetch_bulk_gamelogs.clear()
                        bulk_ready = _fetch_bulk_gamelogs() is not None
                # Pre-warm player caches before launching threads
                with st.spinner(f"Pre-warming caches for {len(candidates)} candidates..."):
                    pre_warm_scanner_caches(candidates, n_games)
                _scan_workers = min(32, len(candidates)) if bulk_ready else min(10, len(candidates))
                if not bulk_ready:
                    st.warning(
                        f"Bulk game log load failed — scanning with {_scan_workers} workers "
                        f"(individual NBA API calls). Click **Load All Game Logs** above for faster scans."
                    )
                from concurrent.futures import as_completed as _as_completed
                with st.spinner(f"Scanning {len(candidates)} candidates ({_scan_workers} workers)..."):
                    with ThreadPoolExecutor(max_workers=_scan_workers) as ex:
                        _fut_map = {
                            ex.submit(compute_leg_projection, pname, mkt, line, meta,
                                      n_games=n_games, key_teammate_out=False,
                                      bankroll=bankroll, frac_kelly=frac_kelly,
                                      max_risk_frac=float(st.session_state.get("max_risk_per_bet",3.0))/100.0,
                                      market_prior_weight=market_prior_weight,
                                      exclude_chaotic=bool(exclude_chaotic),
                                      game_date=scan_start,
                                      injury_team_map=_inj_map,
                                      skip_expensive_signals=True,
                                      skip_halfgame_boxscores=True): (pname, mkt, line, meta)
                            for pname, mkt, line, meta in candidates
                        }
                        _completed_legs = []
                        for _fut in _as_completed(_fut_map):
                            _pname, _mkt, _line, _meta = _fut_map[_fut]
                            try:
                                _leg = _fut.result(timeout=60)
                                _completed_legs.append((_pname, _mkt, _line, _meta, _leg))
                            except TimeoutError:
                                dropped.append({"player": _pname, "market": _mkt, "reason": "thread timeout (NBA API ≥60s)"})
                            except Exception as _te:
                                dropped.append({"player": _pname, "market": _mkt, "reason": f"thread error: {type(_te).__name__}: {_te}"})
                    # Process all results in one batch (prevents flooding mobile WebSocket)
                    for pname, mkt, line, meta, leg in _completed_legs:
                        leg = recompute_pricing_fields(leg, st.session_state.get("calibrator_map"))
                        # [FEATURE] Capture every computed leg for Under scan (before gate filter)
                        all_computed_legs.append((pname, mkt, float(line), meta, leg))
                        if not leg.get("gate_ok"):
                            dropped.append({"player":pname,"market":mkt,"reason":leg.get("gate_reason","gated")}); continue
                        # Use p_cal exclusively after recompute_pricing_fields;
                        # p_over fallback could use uncalibrated bootstrap prob which bypasses isotonic correction
                        pc = leg.get("p_cal")
                        if pc is None:
                            dropped.append({"player":pname,"market":mkt,"reason":"p_cal None (calibration failed)"}); continue
                        pc = float(pc)
                        pi = leg.get("p_implied")
                        ev = leg.get("ev_adj")
                        if pi is None or ev is None:
                            dropped.append({"player":pname,"market":mkt,"reason":"no price/EV"}); continue
                        adv = pc - float(pi)
                        if pc < min_prob: dropped.append({"player":pname,"market":mkt,"reason":f"p_cal<{min_prob:.2f}"}); continue
                        if adv < min_adv: dropped.append({"player":pname,"market":mkt,"reason":f"adv<{min_adv:.3f}"}); continue
                        if float(ev) < min_ev: dropped.append({"player":pname,"market":mkt,"reason":f"ev<{min_ev:.3f}"}); continue
                        mv = leg.get("line_movement") or {}
                        inj_flag = ("🏥 " + (leg.get("auto_inj_player") or "").title()
                                    if leg.get("auto_inj") else "")
                        _src_badge = {"prizepicks": "PP", "underdog": "UD"}.get(
                            str(meta.get("book","")).lower(), meta.get("book","") or "odds"
                        )
                        out_rows.append({
                            "side": "Over",
                            "src": _src_badge,
                            "player":pname,"market":mkt,"line":line,
                            "p_cal":round(pc,3),"p_implied":round(float(pi),3),
                            "advantage":round(adv,3),"ev_adj_pct":round(float(ev)*100,2),
                            "proj":safe_round(leg.get("proj")),
                            "edge_cat":leg.get("edge_cat",""),"regime":leg.get("regime",""),
                            "hot_cold":leg.get("hot_cold","Average"),
                            "team":leg.get("team",""),"opp":leg.get("opp",""),
                            "b2b": "B2B" if leg.get("b2b") else "",
                            "dnp_risk": "DNP?" if leg.get("dnp_risk") else "",
                            "vol_cv":safe_round(leg.get("volatility_cv")),
                            "rest_d":int(leg.get("rest_days",2)),
                            "line_mv":mv.get("direction","--"),
                            "mv_pips":float(mv.get("pips",0.0)),
                            "steam": "STEAM" if mv.get("steam") else ("FADE" if mv.get("fade") else ""),
                            "stake_$":round(leg.get("stake",0),2),
                            "n_games":int(leg.get("n_games_used",0)),
                            "inj_boost": inj_flag,
                            "min_proj": safe_round(leg.get("proj_minutes"),0),
                            "pp_edge_%": round((pc - 0.50) * 100, 1),
                            "pp_2leg_ev_%": round((DFS_PP_PAYOUTS[2] * pc**2 - 1.0) * 100, 1),
                            "sharp": safe_round(leg.get("sharpness_score"), 0),
                            "sharp_tier": leg.get("sharpness_tier", ""),
                            "trend": leg.get("trend_label", ""),
                            "fatigue": leg.get("fatigue_label", "Normal"),
                            "game_tot": safe_round(leg.get("game_total"), 0),
                            "l3": safe_round(leg.get("l3_avg"), 1),
                            "l5": safe_round(leg.get("l5_avg"), 1),
                        })
            out_df = pd.DataFrame(out_rows)
            if not out_df.empty:
                # [UPGRADE NEW] Sort by composite sharpness score (when available), then by EV as tiebreaker
                if "sharp" in out_df.columns and out_df["sharp"].notna().any():
                    out_df["_sort_key"] = (
                        out_df["sharp"].fillna(0).astype(float) * 0.6 +
                        out_df["ev_adj_pct"].fillna(0).astype(float) * 0.4
                    )
                    out_df = out_df.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])
                else:
                    out_df = out_df.sort_values("ev_adj_pct", ascending=False)
                out_df = out_df.head(max_rows)
                # [FIX 13] Persist scanner results in session state
                # [AUDIT FIX] scan_id ties results to their dropped list — consecutive scans can't corrupt display
                _scan_id = _now_iso()
                st.session_state["scanner_results"] = out_df
                st.session_state["scanner_dropped"] = dropped
                st.session_state["scanner_scan_id"] = _scan_id
                _save_scanner_cache()  # persist to disk — survives server restarts / WS drops
                # Auto-send Discord/Telegram alerts for strong edges
                _dw = st.session_state.get("discord_webhook","")
                _tt = st.session_state.get("tg_token","")
                _tc = st.session_state.get("tg_chat","")
                _et = float(st.session_state.get("discord_ev_thresh", 5.0))
                if (_dw or (_tt and _tc)):
                    # [AUDIT FIX #16] Hash-based dedup: consecutive scans won't re-send same alert
                    _sent_hashes = st.session_state.get("_scanner_alert_hashes", set())
                    _sent_count = 0
                    strong = [r for _, r in out_df.iterrows() if float(r.get("ev_adj_pct") or 0) >= _et]
                    for r in strong:
                        _msg = format_edge_alert(dict(r))
                        _mhash = hashlib.md5(_msg.encode()).hexdigest()
                        if _mhash not in _sent_hashes:
                            if _dw: send_discord_alert(_dw, _msg)
                            if _tt and _tc: send_telegram_alert(_tt, _tc, _msg)
                            _sent_hashes.add(_mhash)
                            _sent_count += 1
                    # [AUDIT FIX #13] Also alert Under edges when show_unders is on
                    if bool(st.session_state.get("show_unders", False)):
                        _under_res = st.session_state.get("scanner_under_results", pd.DataFrame())
                        if not _under_res.empty:
                            for _, _ur in _under_res.iterrows():
                                if float(_ur.get("ev_adj_pct") or 0) >= _et:
                                    _umsg = format_edge_alert(dict(_ur)) + " ↓ UNDER"
                                    _uhash = hashlib.md5(_umsg.encode()).hexdigest()
                                    if _uhash not in _sent_hashes:
                                        if _dw: send_discord_alert(_dw, _umsg)
                                        if _tt and _tc: send_telegram_alert(_tt, _tc, _umsg)
                                        _sent_hashes.add(_uhash)
                                        _sent_count += 1
                    st.session_state["_scanner_alert_hashes"] = _sent_hashes
                    _save_alert_hashes(_sent_hashes)  # persist to disk — survives session resets
                    if _sent_count > 0:
                        st.success(f"Auto-sent {_sent_count} new alerts.")
            else:
                st.session_state["scanner_results"] = pd.DataFrame()
                st.session_state["scanner_dropped"] = dropped
                st.warning("No legs met threshold criteria.")
            # [FEATURE] Under scan: derive Under edges from cached computed legs (zero extra API calls)
            if bool(st.session_state.get("show_unders", False)) and all_computed_legs:
                under_rows = []
                for _pn, _mk, _ln, _mt, _leg in all_computed_legs:
                    _pc = float(_leg.get("p_cal") or _leg.get("p_over") or 0)
                    _pi = _leg.get("p_implied")
                    if _pi is None:
                        continue
                    _p_under = 1.0 - _pc
                    _p_imp_u = 1.0 - float(_pi)
                    if _p_imp_u <= 0 or _p_imp_u >= 1:
                        continue
                    _ev_u = (_p_under / _p_imp_u - 1.0)
                    _gate_u, _ = passes_volatility_gate(
                        _leg.get("volatility_cv"), _ev_u,
                        skew=_leg.get("stat_skewness"), bet_type="Under"
                    )
                    if not _gate_u or _leg.get("dnp_risk"):
                        continue
                    if _p_under < min_prob:
                        continue
                    _adv_u = _p_under - _p_imp_u
                    if _adv_u < min_adv or _ev_u < min_ev:
                        continue
                    _umv = _leg.get("line_movement") or {}
                    under_rows.append({
                        "side": "Under",
                        "player": _pn, "market": _mk, "line": _ln,
                        "p_cal": round(_p_under, 3),
                        "p_implied": round(_p_imp_u, 3),
                        "advantage": round(_adv_u, 3),
                        "ev_adj_pct": round(_ev_u * 100, 2),
                        "proj": safe_round(_leg.get("proj")),
                        "edge_cat": classify_edge(_ev_u),
                        "regime": _leg.get("regime", ""),
                        "hot_cold": _leg.get("hot_cold", "Average"),
                        "team": _leg.get("team", ""),
                        "opp": _leg.get("opp", ""),
                        "b2b": "B2B" if _leg.get("b2b") else "",
                        # [AUDIT FIX] Add columns OVER rows have to maintain schema parity
                        "dnp_risk": "DNP?" if _leg.get("dnp_risk") else "",
                        "vol_cv": safe_round(_leg.get("volatility_cv")),
                        "rest_d": int(_leg.get("rest_days", 2)),
                        "line_mv": _umv.get("direction", "--"),
                        "mv_pips": float(_umv.get("pips", 0.0)),
                        "steam": "STEAM" if _umv.get("steam") else ("FADE" if _umv.get("fade") else ""),
                        "stake_$": 0.0,
                        "n_games": int(_leg.get("n_games_used", 0)),
                        "inj_boost": "🏥 " + (_leg.get("auto_inj_player") or "").title() if _leg.get("auto_inj") else "",
                        "min_proj": safe_round(_leg.get("proj_minutes"), 0),
                        "src": {"prizepicks": "PP", "underdog": "UD"}.get(
                            str(_mt.get("book", "")).lower(), _mt.get("book", "") or "odds"
                        ),
                        "pp_edge_%": round((_p_under - 0.50) * 100, 1),
                        "pp_2leg_ev_%": round((DFS_PP_PAYOUTS[2] * _p_under**2 - 1.0) * 100, 1),
                        "sharp": safe_round(_leg.get("sharpness_score"), 0),
                        "sharp_tier": _leg.get("sharpness_tier", ""),
                        "trend": _leg.get("trend_label", ""),
                        "fatigue": _leg.get("fatigue_label", "Normal"),
                        "game_tot": safe_round(_leg.get("game_total"), 0),
                        "l3": safe_round(_leg.get("l3_avg"), 1),
                        "l5": safe_round(_leg.get("l5_avg"), 1),
                    })
                if under_rows:
                    under_df = pd.DataFrame(under_rows).sort_values("ev_adj_pct", ascending=False)
                    st.session_state["scanner_under_results"] = under_df
                else:
                    st.session_state["scanner_under_results"] = pd.DataFrame()
                _save_scanner_cache()  # re-save with under results included
    # [FIX 13] Always show last scanner results (persists across tab switches + server restarts)
    scanner_out = st.session_state.get("scanner_results")
    if scanner_out is not None and not scanner_out.empty:
        _scan_ts = st.session_state.get("scanner_scan_id", "")
        _ts_label = f" · scanned {_scan_ts}" if _scan_ts else ""
        st.markdown(f"<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.10em;margin-bottom:0.6rem;'>{len(scanner_out)} EDGES FOUND{_ts_label}</div>", unsafe_allow_html=True)
        # [UPGRADE 20] Color-code rows by confidence tier
        def _style_scanner_row(row):
            p = float(row.get("p_cal") or 0)
            if p >= 0.65:   bg = "background-color:#00FFB215;"
            elif p >= 0.58: bg = "background-color:#00AAFF12;"
            elif p >= 0.52: bg = "background-color:#FFB80010;"
            else:           bg = "background-color:#FF335810;"
            return [bg] * len(row)
        try:
            styled = scanner_out.style.apply(_style_scanner_row, axis=1)
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(scanner_out, use_container_width=True)
        # 📊 AI Slate Briefing
        if _get_anthropic_key():
            _sl_btn_col, _sl_clear_col = st.columns([3, 1])
            with _sl_btn_col:
                if st.button("📊 AI Slate Briefing", use_container_width=True, key="ai_slate_btn",
                             help="Claude Sonnet analyzes today's top edges and writes an institutional-grade slate brief"):
                    with st.spinner("Claude analyzing slate…"):
                        _top_edges = scanner_out.head(15).to_dict(orient="records")
                        _slate_payload = json.dumps([
                            {k: v for k, v in r.items()
                             if k in ["player","market","line","side","p_cal","ev_adj_pct",
                                      "edge_cat","hot_cold","dnp_risk","opp","src","b2b",
                                      "sharp","sharp_tier","trend","fatigue","game_tot",
                                      "l3","l5","steam","min_proj","inj_boost"]}
                            for r in _top_edges
                        ], indent=2)
                        _slate_ai = ai_slate_briefing(_slate_payload, api_key=_get_anthropic_key())
                    st.session_state["_ai_slate_result"] = _slate_ai
            with _sl_clear_col:
                if st.button("Clear", key="ai_slate_clear_btn"):
                    st.session_state.pop("_ai_slate_result", None)
            _slate_ai_txt = st.session_state.get("_ai_slate_result")
            if _slate_ai_txt:
                _slate_html = _html.escape(_slate_ai_txt).replace("\n", "<br>")
                st.markdown(
                    f"<div style='background:#00AAFF0A;border:1px solid #00AAFF25;"
                    f"border-radius:4px;padding:0.85rem 1.1rem;margin:0.5rem 0;"
                    f"font-size:0.68rem;color:#B0C8E8;line-height:1.65;'>"
                    f"<span style='font-family:Chakra Petch,monospace;font-size:0.58rem;"
                    f"color:#00AAFF;letter-spacing:0.1em;display:block;margin-bottom:0.5rem;'>"
                    f"📊 CLAUDE AI · SLATE BRIEFING</span>{_slate_html}</div>",
                    unsafe_allow_html=True,
                )
        # [FEATURE] Under opportunities panel (shown when toggle is on)
        if bool(st.session_state.get("show_unders", False)):
            under_out = st.session_state.get("scanner_under_results")
            if under_out is not None and not under_out.empty:
                st.markdown(
                    f"<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;"
                    f"color:#FF8080;letter-spacing:0.10em;margin:0.8rem 0 0.4rem;'>"
                    f"↓ {len(under_out)} UNDER OPPORTUNITIES (from cached projections — 0 extra API calls)</div>",
                    unsafe_allow_html=True
                )
                try:
                    def _style_under_row(row):
                        p = float(row.get("p_cal") or 0)
                        if p >= 0.65:   return ["background-color:#FF335815;"] * len(row)
                        elif p >= 0.58: return ["background-color:#FF335810;"] * len(row)
                        else:           return ["background-color:#FF33580A;"] * len(row)
                    st.dataframe(under_out.style.apply(_style_under_row, axis=1), use_container_width=True)
                except Exception:
                    st.dataframe(under_out, use_container_width=True)
            elif under_out is not None:
                st.caption("↓ No Under edges found meeting threshold criteria.")
        # ── SEND TO MODEL ────────────────────────────────────────────────
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00AAFF;"
            "letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;'>"
            "▶ SEND TO MODEL TAB</div>",
            unsafe_allow_html=True
        )
        # Build labels for both OVER and Under results combined
        _model_labels = [
            f"{r['player']} — {r['market']} {'O' if str(r.get('side','')).lower() != 'under' else 'U'}{r['line']} ({r.get('ev_adj_pct',0):+.1f}%)"
            for _, r in scanner_out.iterrows()
        ]
        _under_res_for_model = st.session_state.get("scanner_under_results")
        if bool(st.session_state.get("show_unders", False)) and _under_res_for_model is not None and not _under_res_for_model.empty:
            _model_labels += [
                f"{r['player']} — {r['market']} U{r['line']} ({r.get('ev_adj_pct',0):+.1f}%) [UNDER]"
                for _, r in _under_res_for_model.iterrows()
            ]
        _legs_for_model = st.multiselect(
            "Select up to 4 legs to run through full model",
            options=_model_labels,
            max_selections=4,
            key="scanner_model_pick",
            help="Selected legs will populate MODEL tab inputs and auto-run projections"
        )
        if _legs_for_model and st.button("▶ Send to MODEL + Run", key="send_to_model_btn", use_container_width=True):
            # Build combined row lookup (Over rows first, then Under)
            _all_model_rows = list(scanner_out.iterrows())
            if bool(st.session_state.get("show_unders", False)) and _under_res_for_model is not None and not _under_res_for_model.empty:
                _all_model_rows += list(_under_res_for_model.iterrows())
            _row_by_label = {lbl: row for lbl, (_, row) in zip(_model_labels, _all_model_rows)}
            for i, lbl in enumerate(_legs_for_model[:4], 1):
                r = _row_by_label.get(lbl)
                if r is None:
                    continue
                # Use staging keys — widget-owned keys (pname_{i} etc.) cannot be set
                # after the Model tab widgets have already rendered this cycle.
                # Staged values are consumed before widgets render on the next rerun.
                st.session_state[f"_staged_pname_{i}"]  = str(r.get("player", ""))
                _mkt_v = str(r.get("market", "Points"))
                if _mkt_v in MARKET_OPTIONS:
                    st.session_state[f"_staged_mkt_{i}"] = _mkt_v
                st.session_state[f"_staged_mline_{i}"]  = float(r.get("line", 22.5))
                # If source is PrizePicks, use the scanned line directly — PP lines are not on Odds API
                _is_pp_source = (sportsbook2 == "prizepicks") or (str(r.get("source","")).lower() == "prizepicks")
                st.session_state[f"_staged_manual_{i}"] = _is_pp_source
                st.session_state[f"_staged_out_{i}"]    = False
            # Clear unused legs beyond selection count
            for i in range(len(_legs_for_model) + 1, 5):
                st.session_state[f"_staged_pname_{i}"] = ""
            # Sync scanner date and book to MODEL tab
            st.session_state["_staged_model_date"] = scan_start
            if sportsbook2 and sportsbook2 != "all":
                st.session_state["_staged_sportsbook"] = sportsbook2
            st.session_state["_auto_run_model"] = True   # MODEL tab will detect and auto-run
            st.rerun()
        # [UPGRADE 21] One-click parlay builder from scanner results
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;'>ONE-CLICK PARLAY BUILDER</div>", unsafe_allow_html=True)
        player_labels = [f"{r['player']} — {r['market']} O{r['line']} ({r.get('ev_adj_pct',0):+.1f}%)"
                         for _, r in scanner_out.iterrows()]
        selected_legs = st.multiselect("Select legs to parlay", options=player_labels, key="parlay_picker")
        if selected_legs:
            sel_indices = [player_labels.index(s) for s in selected_legs if s in player_labels]
            sel_rows = [scanner_out.iloc[i] for i in sel_indices]
            probs = [float(r.get("p_cal") or 0) for r in sel_rows]
            naive_joint = float(np.prod(probs)) if probs else 0.0
            # [AUDIT FIX] Correlation-adjusted joint prob: naive assumes independence;
            # teammates and game-script-linked players inflate it. Estimate adjustment.
            _corrs = []
            for _ci in range(len(sel_rows)):
                for _cj in range(_ci + 1, len(sel_rows)):
                    _c = estimate_player_correlation(dict(sel_rows[_ci]), dict(sel_rows[_cj]))
                    _corrs.append(float(_c or 0.0))
            avg_corr = float(np.mean(_corrs)) if _corrs else 0.0
            corr_adj_factor = max(0.60, 1.0 - avg_corr * 0.5)
            corr_joint = naive_joint * corr_adj_factor
            # [AUDIT FIX] Add payout input directly in scanner so user can adjust
            pm = st.number_input("Payout multiplier (x)", 1.5, 20.0,
                                 value=float(st.session_state.get("payout_multi", 3.0)),
                                 step=0.5, key="scanner_pm_input")
            st.session_state["payout_multi"] = pm
            ev_parlay = pm * corr_joint - 1.0
            pb1, pb2, pb3 = st.columns(3)
            pb1.metric("Joint Prob (corr-adj)", f"{corr_joint*100:.1f}%",
                       delta=f"naive {naive_joint*100:.1f}%", delta_color="off")
            pb2.metric(f"EV @ {pm:.1f}x payout", f"{ev_parlay*100:+.1f}%")
            rec_stake_parlay = float(st.session_state.get("bankroll", 1000)) * float(st.session_state.get("frac_kelly", 0.25)) * max(0, ev_parlay / (pm - 1)) if pm > 1 else 0
            pb3.metric("Rec Stake", f"${min(rec_stake_parlay, float(st.session_state.get('bankroll',1000))*0.05):.2f}")
            if st.button("Log This Parlay to History", use_container_width=True, key="log_parlay_btn"):
                parlay_legs = []
                for r in sel_rows:
                    parlay_legs.append({k: v for k, v in r.items()})
                append_history(st.session_state.get("user_id","trader"), {
                    "ts": _now_iso(), "user_id": st.session_state.get("user_id","trader"),
                    "legs": json.dumps(parlay_legs), "n_legs": len(parlay_legs),
                    "result": "Pending", "decision": "BET", "notes": "parlay-builder",
                })
                st.success(f"Logged {len(parlay_legs)}-leg parlay to history.")
        # [UPGRADE 35] Live Steam Check
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#FFB800;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;'>STEAM DETECTOR — CHECK LINE MOVES VS OPENING</div>", unsafe_allow_html=True)
        sc_steam1, sc_steam2 = st.columns([3,1])
        with sc_steam2:
            steam_thresh = st.number_input("Move threshold", 0.1, 2.0, 0.5, 0.1, key="steam_thresh")
        if sc_steam1.button("Run Steam Check (vs Opening Lines)", use_container_width=True, key="steam_check_btn"):
            steam_alerts = []
            for _, row in scanner_out.iterrows():
                pn = normalize_name(str(row.get("player","")))
                mk = ODDS_MARKETS.get(str(row.get("market","")), "")
                cur_line = float(row.get("line", 0) or 0)
                open_line, _ = get_opening_line(pn, mk, "Over")
                if open_line is not None:
                    delta = cur_line - float(open_line)
                    if abs(delta) >= steam_thresh:
                        direction = "UP" if delta > 0 else "DOWN"
                        steam_type = "STEAM" if (delta > 0) else "FADE"
                        steam_alerts.append({
                            "player": row.get("player"), "market": row.get("market"),
                            "open": open_line, "current": cur_line,
                            "move": f"{direction} {abs(delta):.1f}",
                            "type": steam_type,
                            "ev_%": row.get("ev_adj_pct"),
                        })
            if steam_alerts:
                st.warning(f"**{len(steam_alerts)} line move(s) detected vs opening:**")
                st.dataframe(pd.DataFrame(steam_alerts), use_container_width=True)
                # Auto-alert significant steam
                _dw2 = st.session_state.get("discord_webhook","")
                _tt2 = st.session_state.get("tg_token","")
                _tc2 = st.session_state.get("tg_chat","")
                for sa in steam_alerts:
                    msg = (f"**STEAM ALERT** — {sa['player']} {sa['market']}\n"
                           f"Line moved {sa['move']} ({sa['open']} → {sa['current']}) [{sa['type']}]\n"
                           f"Current EV: {sa.get('ev_%',0):+.1f}%")
                    if _dw2: send_discord_alert(_dw2, msg)
                    if _tt2 and _tc2: send_telegram_alert(_tt2, _tc2, msg)
                if _dw2 or (_tt2 and _tc2):
                    st.success(f"Steam alerts fired to Discord/Telegram.")
            else:
                st.success("No significant line moves vs opening. Lines are stable.")
    scanner_dropped = st.session_state.get("scanner_dropped", [])
    if scanner_dropped:
        with st.expander(f"Excluded ({len(scanner_dropped)})", expanded=False):
            st.dataframe(pd.DataFrame(scanner_dropped).head(200), use_container_width=True)
# ─── PLATFORMS TAB ─────────────────────────────────────────────
with tabs[3]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>PLATFORMS — PRIZEPICKS / UNDERDOG / LINE SHOPPING</div>""", unsafe_allow_html=True)
    plat_tabs = st.tabs(["PrizePicks", "Underdog", "Sleeper", "Line History", "Best Available"])
    with plat_tabs[0]:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>PRIZEPICKS NBA LINES</div>", unsafe_allow_html=True)
        # Show cookie status inline — saves user a trip to Alerts tab
        _pp_ck = st.session_state.get("pp_cookies", "")
        if _pp_ck:
            st.markdown(f"<div style='font-size:0.60rem;color:#00FFB2;margin-bottom:4px;'>🔑 Cookie auth active ({len(_pp_ck)} chars) — auto-fetch will use your cookies.</div>", unsafe_allow_html=True)
        else:
            with st.expander("🔑 Set PrizePicks Cookies (needed for auto-fetch on cloud)", expanded=False):
                st.caption("PrizePicks blocks Streamlit Cloud IPs. Paste your browser cookies here to bypass — or use Manual Import below.")
                _new_ck = st.text_area("Cookie string", value="", height=60, placeholder="_pxmvid=...; __cf_bm=...", key="pp_ck_platforms")
                if st.button("Save Cookies", key="pp_ck_save_plat"):
                    _ck_val2 = _new_ck.strip()
                    st.session_state["pp_cookies"] = _ck_val2
                    save_pp_settings(pp_cookies=_ck_val2)
                    _fetch_prizepicks_lines_cached.clear()
                    st.session_state["_pp_last_cookies_used"] = ""
                    st.success("Saved to disk — persists across restarts.")
                    st.rerun()
        pp_load_tab, pp_manual_tab = st.tabs(["Auto Fetch", "Manual Import"])
        with pp_load_tab:
            # ── Auto-refresh component (triggers rerun every N seconds) ──
            try:
                from streamlit_autorefresh import st_autorefresh
                _ar_interval_ms = int(st.session_state.get("pp_auto_interval", 30)) * 60 * 1000
                if st.session_state.get("pp_auto_enabled"):
                    st_autorefresh(interval=_ar_interval_ms, key="pp_auto_rerun")
            except ImportError:
                pass
            # ── Auto-refresh toggle with credit-aware defaults ──
            _af_col1, _af_col2, _af_col3 = st.columns([2, 2, 2])

            _auto_on = _af_col1.toggle(
                "Auto-Refresh Lines",
                value=st.session_state.get("pp_auto_enabled", False),
                key="pp_auto_toggle",
                help="Background thread fetches fresh PP lines on a timer. Uses proxy credits each time.",
            )

            # Smart default: 30 min for ScraperAPI premium (conserve credits), 10 min for others
            _cur_proxy_svc = st.session_state.get("pp_proxy_service", "scraperapi")
            _default_interval = 30 if _cur_proxy_svc == "scraperapi" else 10
            _saved_interval = int(st.session_state.get("pp_auto_interval", _default_interval))

            _auto_min = _af_col2.number_input(
                "Interval (minutes)",
                min_value=5,
                max_value=120,
                value=max(5, _saved_interval),
                step=5,
                key="pp_auto_interval_input",
                help="How often to fetch fresh lines. ScraperAPI premium: 30+ min recommended.",
            )

            # Credit usage estimate
            _svc_info = _PROXY_SERVICES.get(_cur_proxy_svc, {})
            _creds_per = _svc_info.get("credits_per_req", 1)
            _creds_per_fetch = _creds_per * 2  # 2 API calls per fetch (single_stat true + false)
            _fetches_per_day = (24 * 60) / max(_auto_min, 1)
            _daily_credits = _fetches_per_day * _creds_per_fetch
            _monthly_credits = _daily_credits * 30

            if _cur_proxy_svc == "scraperapi":
                _budget = 5000
            elif _cur_proxy_svc == "scrapingbee":
                _budget = 1000
            else:
                _budget = 1000

            _usage_pct = (_monthly_credits / _budget * 100) if _budget > 0 else 0
            _usage_color = "#FF3358" if _usage_pct > 100 else ("#FFB800" if _usage_pct > 70 else "#00FFB2")
            _status_label = "OVER BUDGET" if _usage_pct > 100 else ("TIGHT" if _usage_pct > 70 else "OK")

            _af_col3.markdown(f"""
<div style='background:#04080F;border:1px solid #0E1E30;border-radius:4px;padding:0.4rem 0.6rem;margin-top:0.3rem;'>
<div style='font-family:Fira Code,monospace;font-size:0.52rem;color:#2A6080;'>CREDIT USAGE</div>
<div style='font-family:Fira Code,monospace;font-size:0.72rem;color:{_usage_color};font-weight:600;'>{_daily_credits:.0f}/day · {_monthly_credits:.0f}/mo</div>
<div style='font-family:Fira Code,monospace;font-size:0.50rem;color:{_usage_color};'>{_status_label} ({_usage_pct:.0f}% of {_budget:,} free)</div>
</div>""", unsafe_allow_html=True)

            if _auto_on and _usage_pct > 100:
                st.warning(
                    f"⚠ At {_auto_min}-min intervals, auto-refresh will use ~{_monthly_credits:.0f} "
                    f"credits/month — exceeds your {_budget:,} free tier. "
                    f"Increase interval to {int(max(30, _auto_min))}+ min or switch to ScrapingBee."
                )

            if (_auto_on != st.session_state.get("pp_auto_enabled") or
               int(_auto_min) != int(st.session_state.get("pp_auto_interval", _default_interval))):
                st.session_state["pp_auto_enabled"]  = _auto_on
                st.session_state["pp_auto_interval"] = int(_auto_min)
                save_pp_settings(pp_auto_enabled=_auto_on, pp_auto_interval=int(_auto_min))
                set_pp_auto_fetch(
                    enabled=_auto_on,
                    interval_sec=int(_auto_min) * 60,
                    cookies=st.session_state.get("pp_cookies", ""),
                    relay_url=st.session_state.get("pp_relay_url", ""),
                    proxy_service=st.session_state.get("pp_proxy_service", ""),
                    proxy_key=st.session_state.get("pp_proxy_key", ""),
                )
            # ── Relay URL (optional — for cloud deployments) ──
            with st.expander("Local Relay URL (optional, for Streamlit Cloud)", expanded=False):
                st.caption(
                    "Run `pp_relay.py` on your local machine (residential IP bypasses PerimeterX). "
                    "Expose it via [ngrok](https://ngrok.com): `ngrok http 8765` then paste the URL below."
                )
                _relay_new = st.text_input(
                    "Relay URL", value=st.session_state.get("pp_relay_url", ""),
                    placeholder="http://localhost:8765/lines  or  https://xxxx.ngrok.io/lines",
                    key="pp_relay_url_input",
                )
                if st.button("Save Relay URL", key="pp_relay_save"):
                    st.session_state["pp_relay_url"] = _relay_new.strip()
                    save_pp_settings(pp_relay_url=_relay_new.strip())
                    set_pp_auto_fetch(
                        enabled=st.session_state.get("pp_auto_enabled", False),
                        interval_sec=int(st.session_state.get("pp_auto_interval", 30)) * 60,
                        cookies=st.session_state.get("pp_cookies", ""),
                        relay_url=_relay_new.strip(),
                    )
                    st.success("Relay URL saved.")
            st.markdown("---")
            # ── Status + manual fetch ──
            _auto_rows, _auto_age, _auto_err = get_pp_auto_lines()
            if _auto_rows:
                if _auto_age is not None and _auto_age < 60:
                    st.success(f"Lines loaded — {len(_auto_rows)} props (updated {_auto_age}s ago)")
                elif _auto_age is not None:
                    _m, _s = divmod(_auto_age, 60)
                    st.info(f"Lines loaded — {len(_auto_rows)} props (last updated {_m}m {_s}s ago)")
                    if _auto_age > 900:
                        st.warning("Lines are >15 min old. Click Fetch Now to refresh.")
            elif _auto_err:
                st.error(f"Auto-fetch error: {_auto_err}")
            if st.button("Fetch Now", use_container_width=True, key="pp_fetch_now_btn"):
                with st.spinner("Fetching PrizePicks..."):
                    pp_lines, pp_err = fetch_prizepicks_lines()
                if pp_err:
                    st.error(f"PrizePicks: {pp_err}")
                    st.info(
                        "**If running on Streamlit Cloud:** PerimeterX blocks cloud IPs.\n\n"
                        "**Fix options:**\n"
                        "1. Run `pp_relay.py` locally + expose via ngrok, paste URL above\n"
                        "2. Run the app locally (`streamlit run app.py`) — auto-fetch works from home IP\n"
                        "3. Paste browser cookies in Settings → PrizePicks Cookies\n"
                        "4. Use **Manual Import** tab"
                    )
                elif not pp_lines:
                    st.warning("No lines returned.")
                else:
                    pp_df = pd.DataFrame(pp_lines)
                    st.session_state["pp_lines"] = pp_df
                    st.success(f"Fetched {len(pp_df)} PrizePicks props.")
                    if _auto_on:
                        # Also push to background state so auto-fetcher has fresh baseline
                        _s = _pp_auto_state()
                        with _s["lock"]:
                            _s["rows"] = pp_lines
                            _s["ts"]   = time.time()
                            _s["err"]  = None
                        _save_pp_disk_cache(pp_lines)
        with pp_manual_tab:
            st.markdown("""<div style='font-size:0.68rem;color:#4A607A;margin-bottom:0.5rem;'>
            <b>How to get the JSON:</b> Open prizepicks.com → DevTools (F12) → Network tab → filter
            <code>projections</code> → click the largest request (~200-300 kB) →
            <b>Response</b> tab → right-click body → Copy &rarr; Copy response. Paste below.<br>
            The JSON starts with <code>{"data":[</code>
            </div>""", unsafe_allow_html=True)
            pp_upload = st.file_uploader("Upload CSV", type=["csv"], key="pp_csv_upload")
            pp_paste = st.text_area("Or paste PrizePicks API JSON response", height=120, key="pp_json_paste",
                                    placeholder='{"data":[{"id":"...","type":"Projection",...}],"included":[...]}')
            if st.button("Load Data", use_container_width=True, key="pp_manual_load"):
                rows = []
                err_msg = None
                if pp_upload is not None:
                    try:
                        df_up = pd.read_csv(pp_upload)
                        df_up.columns = [c.strip().lower().replace(" ","_") for c in df_up.columns]
                        col_map = {}
                        for need, alts in [("player",["player","name","player_name","athlete"]),
                                           ("stat_type",["stat_type","stat","market","type","display_stat","category"]),
                                           ("line",["line","line_score","value","projection","o_u","ou","over_under"])]:
                            # Exact match first, then fuzzy match via difflib
                            for a in alts:
                                if a in df_up.columns:
                                    col_map[need] = a; break
                            if need not in col_map:
                                best = difflib.get_close_matches(alts[0], df_up.columns, n=1, cutoff=0.75)
                                if best: col_map[need] = best[0]
                        if all(k in col_map for k in ("player","stat_type","line")):
                            for _, r in df_up.iterrows():
                                rows.append({"player": str(r[col_map["player"]]),
                                             "stat_type": str(r[col_map["stat_type"]]),
                                             "line": float(r[col_map["line"]]),
                                             "source": "prizepicks"})
                        else:
                            err_msg = f"CSV must have player, stat_type, line columns. Found: {list(df_up.columns)}"
                    except Exception as e:
                        err_msg = f"CSV parse error: {e}"
                elif pp_paste.strip():
                    try:
                        raw = pp_paste.strip()
                        data = json.loads(raw)
                        if not isinstance(data, dict):
                            raise ValueError("Expected a JSON object starting with {. Make sure you copied the full Response body, not just part of it.")
                        if "data" not in data:
                            raise ValueError('JSON is missing a "data" key. Copy the Response tab (not Preview or Headers).')
                        # Use _parse_pp_response_all to accept all leagues + all specialty markets
                        rows = _parse_pp_response_all(data)
                        if not rows:
                            _types_seen = list({str(p.get("type","")) for p in data.get("data",[]) if isinstance(p,dict)})[:8]
                            err_msg = (
                                f"No projections found in the JSON. Types seen: {_types_seen}. "
                                "Copy the full Response body from the Network tab (not just headers or preview)."
                            )
                    except json.JSONDecodeError as e:
                        err_msg = f'Invalid JSON: {e}. Make sure you copied the entire response (it should start with {{"data":[).'
                    except ValueError as e:
                        err_msg = str(e)
                    except Exception as e:
                        err_msg = f"Parse error: {type(e).__name__}: {e}"
                else:
                    err_msg = "Upload a CSV or paste JSON."
                if err_msg:
                    st.error(err_msg)
                elif rows:
                    pp_df_manual = pd.DataFrame(rows)
                    st.session_state["pp_lines"] = pp_df_manual
                    st.success(f"Loaded {len(pp_df_manual)} PrizePicks props.")
                    # Offer to save JSON so future fetches auto-use it without re-pasting
                    if pp_paste.strip() and pp_paste.strip().startswith("{"):
                        if st.button("💾 Save this JSON (auto-load on next visit)", key="pp_save_json_btn"):
                            st.session_state["pp_cookies"] = pp_paste.strip()
                            save_pp_settings(pp_cookies=pp_paste.strip())
                            st.success("JSON saved — 'Fetch PP Lines' will use it automatically from now on.")
        pp_df = st.session_state.get("pp_lines")
        if pp_df is not None and not pp_df.empty:
            # Filter out alternate (goblin/demon) lines — their adjusted multipliers make
            # EV% misleading. Goblins pay ~0.85x, demons ~1.25x vs standard 1x.
            if "odds_type" in pp_df.columns:
                pp_df_std = pp_df[pp_df["odds_type"].str.lower().isin(["standard", ""])]
                _n_filtered = len(pp_df) - len(pp_df_std)
                if _n_filtered > 0:
                    st.caption(f"⚡ {_n_filtered} alternate (goblin/demon) lines hidden — unplayable multipliers.")
            else:
                pp_df_std = pp_df
            # Run model on PrizePicks lines
            pp_col1, pp_col2 = st.columns([3,1])
            with pp_col1:
                pp_filter = st.text_input("Filter player", key="pp_filter")
            with pp_col2:
                pp_min_ev = st.number_input("Min EV%", -5.0, 30.0, 2.0, 0.5, key="pp_min_ev")
            display_df = pp_df_std
            if pp_filter:
                display_df = pp_df_std[pp_df_std["player"].str.strip().str.contains(pp_filter.strip(), case=False, na=False)]
            st.dataframe(display_df, use_container_width=True)
            # ── Stat-type mapping diagnostic ──────────────────────────────
            if not display_df.empty and "stat_type" in display_df.columns:
                _all_types = sorted(display_df["stat_type"].dropna().unique().tolist())
                _half_types = [t for t in _all_types if any(h in t.lower() for h in
                    ["half", "1h", "2h", "h1", "h2", "1st", "2nd", "quarter", "q1", "1q"])]
                if _half_types:
                    st.caption(f"📊 PP half/quarter stat_types found: {', '.join(_half_types)}")
                else:
                    st.caption("ℹ No half/quarter stat_types in current PP data — slate may not include them yet.")
                _mapped_count = sum(1 for _st in _all_types if map_platform_stat_to_market(_st))
                _unmapped_types = [_st for _st in _all_types if not map_platform_stat_to_market(_st)]
                if _unmapped_types:
                    with st.expander(f"⚠ {len(_unmapped_types)} unmapped stat types (click to see)", expanded=False):
                        st.write(_unmapped_types)
                st.caption(f"Mapped: {_mapped_count}/{len(_all_types)} stat types")
            if st.button("Scan PrizePicks vs Model", use_container_width=True):
                pp_candidates = []
                _pp_meta_base = {"event_id": None, "home_team": "", "away_team": "",
                                 "commence_time": "", "price": 2.0,
                                 "book": "prizepicks", "side": "Over"}
                for _, r in display_df.iterrows():
                    mkt = map_platform_stat_to_market(r.get("stat_type",""))
                    if mkt and r.get("line"):
                        _m = dict(_pp_meta_base, market_key=ODDS_MARKETS.get(mkt))
                        pp_candidates.append((r["player"], mkt, float(r["line"]), _m))
                if pp_candidates:
                    _inj_map = st.session_state.get("injury_team_map", {})
                    with st.spinner(f"Scanning {len(pp_candidates)} PrizePicks props..."):
                        with ThreadPoolExecutor(max_workers=min(16, len(pp_candidates))) as ex:
                            futs_pp = [ex.submit(compute_leg_projection, pn, mk, ln, mt,
                                                 n_games=n_games, key_teammate_out=False,
                                                 bankroll=bankroll, frac_kelly=frac_kelly,
                                                 market_prior_weight=market_prior_weight,
                                                 exclude_chaotic=bool(exclude_chaotic),
                                                 game_date=date.today(),
                                                 injury_team_map=_inj_map)
                                       for pn, mk, ln, mt in pp_candidates]
                            pp_results = []
                            for fut in futs_pp:
                                try:
                                    pp_results.append(fut.result(timeout=60))
                                except TimeoutError:
                                    pass
                                except Exception as _e:
                                    pass
                    calib = st.session_state.get("calibrator_map")
                    pp_results = [recompute_pricing_fields(dict(l), calib) for l in pp_results]
                    pp_edges = [l for l in pp_results if l.get("gate_ok") and float(l.get("ev_adj",0) or 0)*100 >= pp_min_ev]
                    if pp_edges:
                        st.success(f"{len(pp_edges)} edges vs PrizePicks lines")
                        pp_out = pd.DataFrame([{
                            "player": l["player"], "market": l["market"], "line": l["line"],
                            "proj": safe_round(l.get("proj")), "p_cal": safe_round(l.get("p_cal"),3),
                            "ev_%": safe_round(l.get("ev_adj",0)*100,1), "edge_cat": l.get("edge_cat",""),
                            "hot_cold": l.get("hot_cold",""), "dnp?": "⚠" if l.get("dnp_risk") else "",
                            "source": "prizepicks",
                        } for l in pp_edges])
                        st.dataframe(pp_out.sort_values("ev_%", ascending=False), use_container_width=True)
                    else:
                        st.warning("No edges found vs PrizePicks lines.")
    with plat_tabs[1]:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>UNDERDOG FANTASY NBA LINES</div>", unsafe_allow_html=True)
        _ud_ck = st.session_state.get("ud_cookies","")
        if _ud_ck:
            st.caption(f"🔑 UD Cookie active ({len(_ud_ck)} chars) — auto-fetch will use your cookies.")
        else:
            with st.expander("🔑 Set Underdog Cookies (needed if auto-fetch is blocked)", expanded=False):
                st.caption("If Underdog returns 403, paste your browser cookies here to bypass the block.")
                _new_ud_ck = st.text_area("Cookie string", value="", height=60,
                                          placeholder="_ud_session=...; __cf_bm=...", key="ud_ck_platforms")
                if st.button("Save UD Cookies", key="ud_ck_save_plat"):
                    st.session_state["ud_cookies"] = _new_ud_ck.strip()
                    _fetch_underdog_lines_cached.clear()
                    st.session_state["_ud_last_cookies_used"] = ""
                    st.success("Underdog cookies saved.")
                    st.rerun()
        ud_auto_tab, ud_manual_tab = st.tabs(["Auto Fetch", "Manual Import"])
        with ud_auto_tab:
            if st.button("Fetch Underdog Lines", use_container_width=True):
                with st.spinner("Fetching Underdog..."):
                    ud_lines, ud_err = fetch_underdog_lines()
                if ud_err:
                    st.error(f"Underdog: {ud_err}")
                    st.info(
                        "If blocked:\n\n"
                        "1. Paste your Underdog browser cookies in the field above and retry, **OR**\n"
                        "2. Use **Manual Import**:\n"
                        "   - Open [underdogfantasy.com](https://underdogfantasy.com) → NBA board\n"
                        "   - DevTools (F12) → Network → filter `over_under_lines`\n"
                        "   - Click the request → Response tab → Copy → paste in Manual Import"
                    )
                elif not ud_lines:
                    st.warning("No lines returned. The NBA slate may not be posted yet.")
                else:
                    ud_df = pd.DataFrame(ud_lines)
                    st.session_state["ud_lines"] = ud_df
                    st.success(f"Fetched {len(ud_df)} Underdog props.")
        with ud_manual_tab:
            st.caption("Paste the raw JSON from Underdog's over_under_lines API response, or upload a CSV with columns: player, stat_type, line")
            ud_upload = st.file_uploader("Upload CSV", type=["csv"], key="ud_csv_upload")
            ud_paste = st.text_area("Or paste Underdog API JSON", height=100, key="ud_json_paste",
                                    placeholder='{"over_under_lines":[...],"appearances":[...],"players":[...]}')
            if st.button("Load Underdog Data", use_container_width=True, key="ud_manual_load"):
                ud_rows = []; ud_err_msg = None
                if ud_upload is not None:
                    try:
                        df_ud = pd.read_csv(ud_upload)
                        df_ud.columns = [c.strip().lower() for c in df_ud.columns]
                        col_map = {}
                        for need, alts in [("player",["player","name","player_name"]),
                                           ("stat_type",["stat_type","stat","market","type","display_stat"]),
                                           ("line",["line","stat_value","value","projection"])]:
                            for a in alts:
                                if a in df_ud.columns: col_map[need] = a; break
                        if all(k in col_map for k in ("player","stat_type","line")):
                            for _, r in df_ud.iterrows():
                                ud_rows.append({"player": str(r[col_map["player"]]),
                                                "stat_type": str(r[col_map["stat_type"]]),
                                                "line": float(r[col_map["line"]]), "source": "underdog"})
                        else:
                            ud_err_msg = f"CSV missing required columns. Found: {list(df_ud.columns)}"
                    except Exception as e:
                        ud_err_msg = f"CSV error: {e}"
                elif ud_paste.strip():
                    try:
                        data = json.loads(ud_paste.strip())
                        appearances = {str(a.get("id","")): a for a in data.get("appearances",[])}
                        players_map = {str(p.get("id","")): p for p in data.get("players",[])}
                        for line in data.get("over_under_lines", data.get("lines", [])):
                            try:
                                ou = line.get("over_under", line)
                                app_stat = ou.get("appearance_stat", {})
                                app = appearances.get(str(app_stat.get("appearance_id","")), {})
                                # Sport filter: only NBA/basketball
                                sport = str(app.get("sport_id", app.get("sport",""))).lower().strip()
                                if sport and sport not in _UD_BASKETBALL_SPORT_IDS:
                                    continue
                                player = players_map.get(str(app.get("player_id","")), {})
                                pname = f"{player.get('first_name','')} {player.get('last_name','')}".strip()
                                stat = app_stat.get("display_stat", app_stat.get("stat",""))
                                val = line.get("stat_value", ou.get("stat_value"))
                                if pname and stat and val is not None:
                                    ud_rows.append({"player": pname, "stat_type": stat,
                                                    "line": float(val), "source": "underdog"})
                            except Exception: continue
                        if not ud_rows:
                            ud_err_msg = "No lines found in JSON. Check structure — needs over_under_lines, appearances, players."
                    except Exception as e:
                        ud_err_msg = f"JSON parse error: {e}"
                else:
                    ud_err_msg = "Upload a CSV or paste JSON."
                if ud_err_msg:
                    st.error(ud_err_msg)
                elif ud_rows:
                    st.session_state["ud_lines"] = pd.DataFrame(ud_rows)
                    st.success(f"Loaded {len(ud_rows)} Underdog props.")
                    st.rerun()
        ud_df = st.session_state.get("ud_lines")
        if ud_df is not None and not ud_df.empty:
            ud_filter = st.text_input("Filter player", key="ud_filter")
            display_ud = ud_df
            if ud_filter:
                display_ud = ud_df[ud_df["player"].str.strip().str.contains(ud_filter.strip(), case=False, na=False)]
            st.dataframe(display_ud, use_container_width=True)
            if st.button("Scan Underdog vs Model", use_container_width=True):
                ud_candidates = []
                _ud_meta_base = {"event_id": None, "home_team": "", "away_team": "",
                                 "commence_time": "", "price": 2.0,
                                 "book": "underdog", "side": "Over"}
                for _, r in display_ud.iterrows():
                    mkt = map_platform_stat_to_market(r.get("stat_type",""))
                    if mkt and r.get("line"):
                        _m = dict(_ud_meta_base, market_key=ODDS_MARKETS.get(mkt))
                        ud_candidates.append((r["player"], mkt, float(r["line"]), _m))
                if ud_candidates:
                    _inj_map = st.session_state.get("injury_team_map", {})
                    with st.spinner(f"Scanning {len(ud_candidates)} Underdog props..."):
                        with ThreadPoolExecutor(max_workers=min(16, len(ud_candidates))) as ex:
                            futs_ud = [ex.submit(compute_leg_projection, pn, mk, ln, mt,
                                                 n_games=n_games, key_teammate_out=False,
                                                 bankroll=bankroll, frac_kelly=frac_kelly,
                                                 market_prior_weight=market_prior_weight,
                                                 exclude_chaotic=bool(exclude_chaotic),
                                                 game_date=date.today(),
                                                 injury_team_map=_inj_map)
                                       for pn, mk, ln, mt in ud_candidates]
                            ud_results = []
                            for fut in futs_ud:
                                try:
                                    ud_results.append(fut.result(timeout=60))
                                except TimeoutError:
                                    pass
                                except Exception:
                                    pass
                    calib = st.session_state.get("calibrator_map")
                    ud_results = [recompute_pricing_fields(dict(l), calib) for l in ud_results]
                    ud_edges = [l for l in ud_results if l.get("gate_ok") and float(l.get("ev_adj",0) or 0) > 0]
                    if ud_edges:
                        st.success(f"{len(ud_edges)} edges vs Underdog lines")
                        ud_out = pd.DataFrame([{
                            "player": l["player"], "market": l["market"], "line": l["line"],
                            "proj": safe_round(l.get("proj")), "p_cal": safe_round(l.get("p_cal"),3),
                            "ev_%": safe_round(l.get("ev_adj",0)*100,1), "edge_cat": l.get("edge_cat",""),
                            "hot_cold": l.get("hot_cold",""), "dnp?": "⚠" if l.get("dnp_risk") else "",
                            "source": "underdog",
                        } for l in ud_edges])
                        st.dataframe(ud_out.sort_values("ev_%", ascending=False), use_container_width=True)
                    else:
                        st.warning("No edges found vs Underdog lines.")
    with plat_tabs[2]:
        # ── SLEEPER FANTASY ─────────────────────────────────────
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#9B59F5;letter-spacing:0.12em;'>SLEEPER FANTASY NBA LINES</div>", unsafe_allow_html=True)
        st.caption("Sleeper Picks lines — paste your JSON export or upload CSV to sync with the scanner.")
        sl_auto_tab, sl_manual_tab = st.tabs(["Auto Fetch", "Manual Import"])
        with sl_auto_tab:
            st.info("Sleeper's picks API is not publicly documented. Use **Manual Import** to paste your lines, or export from the Sleeper app.")
            if st.button("Attempt Sleeper Fetch", key="sl_auto_fetch_btn"):
                # Try known Sleeper endpoints (unofficial)
                _sl_urls = [
                    "https://api.sleeper.app/v1/stats/nba/projections/regular/2025/1",
                    "https://api.sleeper.app/projections/nba",
                    "https://api.sleeper.com/picks/nba",
                ]
                _sl_found = False
                for _sl_url in _sl_urls:
                    try:
                        from curl_cffi import requests as cffi_requests
                        _sl_r = cffi_requests.get(_sl_url, impersonate="chrome120",
                                                   headers={"Accept": "application/json"}, timeout=10)
                    except ImportError:
                        _sl_r = requests.get(_sl_url, timeout=10)
                    except Exception:
                        continue
                    if _sl_r.ok:
                        try:
                            _sl_data = _sl_r.json()
                            st.json(_sl_data if isinstance(_sl_data, dict) else {"items": len(_sl_data)})
                            st.success(f"Got response from {_sl_url} — paste in Manual Import to process.")
                            _sl_found = True
                            break
                        except Exception:
                            pass
                if not _sl_found:
                    st.error("Could not reach Sleeper API. Use Manual Import below.")
        with sl_manual_tab:
            sl_import_mode = st.radio("Import format", ["JSON paste", "CSV upload"], horizontal=True, key="sl_import_mode")
            if sl_import_mode == "JSON paste":
                sl_json_raw = st.text_area("Paste Sleeper JSON", height=200, key="sl_json_raw",
                                           placeholder='[{"player_name":"LeBron James","stat_type":"Points","line":24.5}, ...]')
                if st.button("Parse Sleeper JSON", key="sl_parse_json_btn"):
                    try:
                        _sl_parsed = json.loads(sl_json_raw.strip())
                        if isinstance(_sl_parsed, dict):
                            _sl_parsed = _sl_parsed.get("picks", _sl_parsed.get("lines",
                                         _sl_parsed.get("projections", [_sl_parsed])))
                        sl_rows = []
                        for item in _sl_parsed:
                            _pn = (item.get("player_name") or item.get("name") or item.get("player","")).strip()
                            _st = (item.get("stat_type") or item.get("stat") or item.get("category","")).strip()
                            _ln = item.get("line") or item.get("line_score") or item.get("value")
                            if _pn and _st and _ln is not None:
                                sl_rows.append({"player": _pn, "stat_type": _st,
                                                "line": float(_ln), "source": "sleeper"})
                        if sl_rows:
                            _sl_df = pd.DataFrame(sl_rows)
                            st.session_state["sl_lines"] = _sl_df
                            st.success(f"✓ {len(sl_rows)} Sleeper lines imported")
                            st.dataframe(_sl_df.head(20), use_container_width=True)
                        else:
                            st.warning("No valid props found — check JSON structure.")
                    except Exception as e:
                        st.error(f"JSON parse error: {e}")
            else:
                sl_file = st.file_uploader("Upload Sleeper CSV", type="csv", key="sl_csv_upload")
                if sl_file:
                    try:
                        _sl_csv = pd.read_csv(sl_file)
                        _sl_csv.columns = [c.strip().lower().replace(" ","_") for c in _sl_csv.columns]
                        _sl_col_map = {}
                        for _need, _alts in [
                            ("player",    ["player","player_name","name","athlete"]),
                            ("stat_type", ["stat_type","stat","category","market","prop"]),
                            ("line",      ["line","line_score","value","over_under"]),
                        ]:
                            for _a in _alts:
                                if _a in _sl_csv.columns:
                                    _sl_col_map[_need] = _a; break
                            if _need not in _sl_col_map:
                                _best = difflib.get_close_matches(_alts[0], _sl_csv.columns, n=1, cutoff=0.7)
                                if _best: _sl_col_map[_need] = _best[0]
                        if all(k in _sl_col_map for k in ("player","stat_type","line")):
                            _sl_rows2 = []
                            for _, row in _sl_csv.iterrows():
                                _pn2 = str(row.get(_sl_col_map["player"],"")).strip()
                                _st2 = str(row.get(_sl_col_map["stat_type"],"")).strip()
                                _ln2 = row.get(_sl_col_map["line"])
                                if _pn2 and _st2 and _ln2 is not None:
                                    try: _sl_rows2.append({"player":_pn2,"stat_type":_st2,"line":float(_ln2),"source":"sleeper"})
                                    except: pass
                            if _sl_rows2:
                                _sl_df2 = pd.DataFrame(_sl_rows2)
                                st.session_state["sl_lines"] = _sl_df2
                                st.success(f"✓ {len(_sl_rows2)} Sleeper lines from CSV")
                                st.dataframe(_sl_df2.head(20), use_container_width=True)
                            else:
                                st.warning("No valid rows parsed.")
                        else:
                            st.warning(f"Could not map columns. Found: {list(_sl_csv.columns)}. Need: player, stat_type, line")
                    except Exception as e:
                        st.error(f"CSV error: {e}")
        _sl_loaded = st.session_state.get("sl_lines")
        if _sl_loaded is not None and not _sl_loaded.empty:
            st.markdown(f"<div style='font-size:0.62rem;color:#9B59F5;margin-top:4px;'>✓ {len(_sl_loaded)} Sleeper lines loaded — available in Scanner (select 'PP + UD only' or 'All sources')</div>", unsafe_allow_html=True)
    with plat_tabs[3]:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>PROP LINE HISTORY</div>", unsafe_allow_html=True)
        ph_col1, ph_col2 = st.columns(2)
        with ph_col1:
            ph_player = st.text_input("Player name", key="ph_player")
        with ph_col2:
            ph_market = st.selectbox("Market", [""] + list(ODDS_MARKETS.keys()), key="ph_market")
        ph_df = load_prop_line_history(
            player=ph_player if ph_player else None,
            market=ph_market if ph_market else None,
        )
        if ph_df.empty:
            st.info("No prop line history yet. Lines are auto-saved when you run the Live Scanner.")
        else:
            st.dataframe(ph_df, use_container_width=True)
            ph_csv = ph_df.to_csv(index=False)
            st.download_button("Export Line History CSV", ph_csv, "prop_line_history.csv", "text/csv")
    with plat_tabs[4]:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>BEST AVAILABLE LINE SHOPPING</div>", unsafe_allow_html=True)
        st.caption("Checks all available books for the highest price on any scanner result. Requires a valid Odds API key.")
        scanner_out_shop = st.session_state.get("scanner_results")
        if scanner_out_shop is None or scanner_out_shop.empty:
            st.info("Run the Live Scanner first to populate results.")
        else:
            has_odds_key = bool(odds_api_key())
            if not has_odds_key:
                st.warning("No Odds API key configured — best_price column will be empty. Add ODDS_API_KEY in Settings to enable price lookups.")
            if st.button("Find Best Lines for Scanner Results", use_container_width=True):
                shop_rows = []
                for _, r in scanner_out_shop.iterrows():
                    eid = r.get("event_id") if hasattr(r,"get") else None
                    mk = ODDS_MARKETS.get(r.get("market","") if hasattr(r,"get") else "")
                    pn = normalize_name(str(r.get("player","") if hasattr(r,"get") else ""))
                    best_p, best_b = None, None
                    if eid and mk and pn and has_odds_key:
                        best_p, best_b = get_best_available_price(eid, mk, pn, "Over")
                    shop_rows.append({
                        "player": r.get("player",""), "market": r.get("market",""),
                        "line": r.get("line"), "book": r.get("book",""),
                        "ev_%": r.get("ev_adj_pct"),
                        "best_price": safe_round(best_p) if best_p else "—",
                        "best_book": best_b or ("no key" if not has_odds_key else "not found"),
                    })
                if shop_rows:
                    st.dataframe(pd.DataFrame(shop_rows), use_container_width=True)
                else:
                    st.info("No scanner results to display.")
# ─── HISTORY TAB [FIX 12: export button] ──────────────────────
with tabs[4]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>BET HISTORY & CLV TRACKER</div>""", unsafe_allow_html=True)
    h = load_history(user_id)
    if h.empty:
        st.markdown(make_card("<span style='color:#4A607A;'>No bets logged yet. Log from the Model tab.</span>"), unsafe_allow_html=True)
    else:
        settled = h[h["result"] != "Pending"].copy() if not h.empty else pd.DataFrame()
        n_hit = (settled["result"]=="HIT").sum() if not settled.empty else 0
        n_miss = (settled["result"]=="MISS").sum() if not settled.empty else 0
        n_pend = (h["result"]=="Pending").sum() if not h.empty else 0
        hit_rate = n_hit/(n_hit+n_miss) if (n_hit+n_miss)>0 else None
        hc1,hc2,hc3,hc4 = st.columns(4)
        hc1.metric("Parlay Hit Rate", f"{hit_rate*100:.1f}%" if hit_rate else "--")
        hc2.metric("Parlay Wins", n_hit)
        hc3.metric("Parlay Losses", n_miss)
        hc4.metric("Pending", n_pend)
        # Per-leg accuracy (only legs with individual results logged)
        h_legs_df = _expand_history_legs(h)
        settled_legs_df = h_legs_df[h_legs_df["y"].notna()].copy() if not h_legs_df.empty else pd.DataFrame()
        if not settled_legs_df.empty:
            n_leg_hit = int((settled_legs_df["y"]==1).sum())
            n_leg_miss = int((settled_legs_df["y"]==0).sum())
            leg_hr = n_leg_hit/(n_leg_hit+n_leg_miss) if (n_leg_hit+n_leg_miss)>0 else None
            lc1,lc2,lc3 = st.columns(3)
            lc1.metric("Per-Leg Hit Rate", f"{leg_hr*100:.1f}%" if leg_hr else "--",
                       help="Individual leg accuracy — requires per-leg results to be marked below")
            lc2.metric("Leg Hits", n_leg_hit)
            lc3.metric("Leg Misses", n_leg_miss)
        else:
            st.caption("Per-leg accuracy will appear once you mark individual leg results below.")
        st.dataframe(h, use_container_width=True)
        # [FIX 12] Export button
        csv_data = h.to_csv(index=False)
        _exp_col, _del_col = st.columns([2, 1])
        _exp_col.download_button("Export History CSV", data=csv_data,
                           file_name=f"history_{user_id}.csv", mime="text/csv",
                           use_container_width=True)
        # Delete individual bets or clear all
        with st.expander("🗑 Delete / Remove Entries", expanded=False):
            st.caption("Row numbers shown in the table above (0-indexed). Deletion is permanent.")
            _d_col1, _d_col2 = st.columns([2, 1])
            with _d_col1:
                _del_idx = st.number_input("Row to delete", 0, max(0, len(h)-1), 0, 1, key="del_row_idx")
            with _d_col2:
                st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)
                if st.button("Delete Row", key="del_row_btn", type="primary"):
                    try:
                        h2 = h.drop(index=int(_del_idx)).reset_index(drop=True)
                        h2.to_csv(history_path(user_id), index=False)
                        st.success(f"Row {int(_del_idx)} deleted.")
                        st.rerun()
                    except Exception as _de:
                        st.error(f"Delete failed: {_de}")
            st.markdown("<hr style='border-color:#1E2D3D;margin:0.5rem 0;'>", unsafe_allow_html=True)
            if "confirm_clear_all" not in st.session_state:
                st.session_state["confirm_clear_all"] = False
            if not st.session_state["confirm_clear_all"]:
                if st.button("Clear ALL History", key="clear_all_btn"):
                    st.session_state["confirm_clear_all"] = True
                    st.rerun()
            else:
                st.warning("Are you sure? This will permanently delete all bet history.")
                _ca1, _ca2 = st.columns(2)
                if _ca1.button("Yes — delete everything", key="confirm_clear_yes", type="primary"):
                    try:
                        pd.DataFrame().to_csv(history_path(user_id), index=False)
                        st.session_state["confirm_clear_all"] = False
                        st.success("All history cleared.")
                        st.rerun()
                    except Exception as _ce:
                        st.error(f"Clear failed: {_ce}")
                if _ca2.button("Cancel", key="confirm_clear_no"):
                    st.session_state["confirm_clear_all"] = False
                    st.rerun()
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        # ── Per-leg result update (fixes calibration skew for multi-leg bets) ──
        st.markdown("<span style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.10em;'>UPDATE PER-LEG RESULTS</span>", unsafe_allow_html=True)
        st.caption("For multi-leg bets, mark each leg individually so calibration uses accurate per-leg outcomes.")
        leg_row_idx = st.number_input("Bet row to update legs", 0, max(0, len(h)-1), 0, 1, key="leg_row_idx")
        try:
            _legs_to_show = json.loads(h.loc[int(leg_row_idx), "legs"]) if isinstance(h.loc[int(leg_row_idx), "legs"], str) else []
            _leg_res_stored = json.loads(h.loc[int(leg_row_idx), "leg_results"]) if "leg_results" in h.columns and isinstance(h.loc[int(leg_row_idx), "leg_results"], str) else ["Pending"]*len(_legs_to_show)
            if len(_leg_res_stored) < len(_legs_to_show):
                _leg_res_stored = _leg_res_stored + ["Pending"]*(len(_legs_to_show)-len(_leg_res_stored))
        except Exception:
            _legs_to_show = []
            _leg_res_stored = []
        if _legs_to_show:
            new_leg_results = []
            for _li, _lleg in enumerate(_legs_to_show):
                _default = _leg_res_stored[_li] if _li < len(_leg_res_stored) else "Pending"
                _opts = ["Pending","HIT","MISS","PUSH"]
                _di = _opts.index(_default) if _default in _opts else 0
                _lcols = st.columns([3,2])
                _lcols[0].markdown(f"<span style='font-size:0.75rem;color:#A0B8C8;'>{_lleg.get('player','?')} — {_lleg.get('market','?')} {_lleg.get('line','')}</span>", unsafe_allow_html=True)
                _sel = _lcols[1].selectbox("", _opts, index=_di, key=f"legres_{leg_row_idx}_{_li}", label_visibility="collapsed")
                new_leg_results.append(_sel)
            if st.button("Save Leg Results", key="save_leg_results"):
                h2 = h.copy()
                h2.loc[int(leg_row_idx), "leg_results"] = json.dumps(new_leg_results)
                # Auto-derive parlay result: HIT only if all legs HIT, MISS if any MISS, else Pending
                if all(r=="HIT" for r in new_leg_results):
                    h2.loc[int(leg_row_idx), "result"] = "HIT"
                elif any(r=="MISS" for r in new_leg_results):
                    h2.loc[int(leg_row_idx), "result"] = "MISS"
                elif all(r in ("HIT","PUSH") for r in new_leg_results):
                    h2.loc[int(leg_row_idx), "result"] = "PUSH"
                h2.to_csv(history_path(user_id), index=False)
                st.success("Leg results saved — calibration will now use individual leg outcomes.")
        else:
            st.caption("No legs found for this row.")
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        uc1, uc2 = st.columns(2)
        with uc1:
            st.markdown("<span style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#4A607A;letter-spacing:0.10em;'>UPDATE PARLAY RESULT</span>", unsafe_allow_html=True)
            idx = st.number_input("Row index", 0, max(0,len(h)-1), 0, 1)
            new_res = st.selectbox("Result", ["Pending","HIT","MISS","PUSH"])
            if st.button("Update Result"):
                h2 = h.copy(); h2.loc[int(idx),"result"] = new_res
                h2.to_csv(history_path(user_id), index=False)
                st.success("Updated")
        with uc2:
            st.markdown("<span style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#4A607A;letter-spacing:0.10em;'>CLV UPDATE</span>", unsafe_allow_html=True)
            idx2 = st.number_input("CLV Row index", 0, max(0,len(h)-1), 0, 1, key="clv_idx")
            if st.button("Fetch & Update CLV"):
                try:
                    h2 = h.copy()
                    legs = json.loads(h2.loc[int(idx2),"legs"])
                    if not isinstance(legs,list) or not legs:
                        st.warning("No legs on that row.")
                    else:
                        legs2, errs = apply_clv_update_to_legs(legs)
                        h2.loc[int(idx2),"legs"] = json.dumps(legs2)
                        h2.to_csv(history_path(user_id), index=False)
                        for e in errs[:5]: st.warning(e)
                        st.success("CLV updated")
                except Exception as e:
                    st.error(f"CLV update failed: {e}")
# ─── CALIBRATION TAB ─────────────────────────────────────────
with tabs[5]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>CALIBRATION ENGINE</div>""", unsafe_allow_html=True)
    h = load_history(user_id)
    legs_df = _expand_history_legs(h)
    # Only use settled legs for calibration metrics
    settled_df = legs_df[legs_df["y"].notna()].copy() if not legs_df.empty else pd.DataFrame()
    if settled_df.empty:
        st.markdown(make_card("<span style='color:#4A607A;font-size:0.78rem;'>No settled bets yet. Log bets and mark results to enable calibration.<br><span style='font-size:0.65rem;'>Minimum ~80 settled legs needed.</span></span>"), unsafe_allow_html=True)
    else:
        y = settled_df["y"].values.astype(float)
        p_raw = settled_df["p_raw"].values.astype(float)
        brier = float(np.mean((p_raw - y)**2))
        hit_rate_cal = float(y.mean())
        n_settled = len(settled_df)
        n_pass_logged = len(legs_df[legs_df.get("decision","")=="PASS"]) if "decision" in legs_df.columns else 0
        cc1,cc2,cc3,cc4 = st.columns(4)
        cc1.metric("Settled Legs", n_settled)
        cc2.metric("Actual Hit Rate", f"{hit_rate_cal*100:.1f}%")
        cc3.metric("Brier Score (raw)", f"{brier:.4f}")
        cc4.metric("Calibrator Fitted", "Yes" if st.session_state.get("calibrator_map") else "No")
        if "clv_line_fav" in settled_df.columns and settled_df["clv_line_fav"].notna().any():
            clv_line_rate = float(settled_df["clv_line_fav"].dropna().astype(int).mean())
            st.metric("CLV (line) favorable %", f"{clv_line_rate*100:.1f}%",
                      delta="Edge exists" if clv_line_rate > 0.52 else "No edge vs closing line",
                      delta_color="normal" if clv_line_rate > 0.52 else "inverse")
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>ROI BY MARKET</div>", unsafe_allow_html=True)
        if "market" in settled_df.columns:
            mkt_grp = settled_df.groupby("market").agg(
                bets=("y","size"), hit_rate=("y","mean"),
                avg_ev=("ev_adj","mean")
            ).reset_index()
            mkt_grp["hit_rate_pct"] = (mkt_grp["hit_rate"]*100).round(1)
            mkt_grp["avg_ev_pct"] = (mkt_grp["avg_ev"]*100).round(2)
            mkt_grp = mkt_grp.sort_values("hit_rate_pct", ascending=False)
            st.dataframe(mkt_grp, use_container_width=True)
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>RELIABILITY TABLE (p_raw vs actual)</div>", unsafe_allow_html=True)
        n_bins = st.slider("Bins", 6, 20, 10)
        settled_df["bin"] = pd.cut(settled_df["p_raw"], bins=n_bins, labels=False, include_lowest=True)
        rel = settled_df.groupby("bin",dropna=True).agg(
            p_mean=("p_raw","mean"), win_rate=("y","mean"), n=("y","size")).reset_index()
        rel["calibration_error"] = (rel["p_mean"] - rel["win_rate"]).abs().round(3)
        st.dataframe(rel, use_container_width=True)
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>FIT CALIBRATOR</div>", unsafe_allow_html=True)
        st.caption("Monotone isotonic calibration maps p_raw -> p_cal using your settled history.")
        if st.button("Fit Calibrator from History", use_container_width=True):
            calib = fit_monotone_calibrator(settled_df, n_bins=int(n_bins))
            if calib is None:
                st.warning(f"Need ~80+ quality legs (currently {n_settled}). Identity calibration used.")
                st.session_state["calibrator_map"] = None
            else:
                st.session_state["calibrator_map"] = calib
                st.success(f"Calibrator fitted on {calib.get('n','?')} legs (range: {calib.get('training_min',0):.2f}-{calib.get('training_max',1):.2f})")
        calib = st.session_state.get("calibrator_map")
        if calib:
            settled_df["p_cal_fit"] = settled_df["p_raw"].apply(lambda p: apply_calibrator(p, calib))
            brier_cal = float(np.mean((settled_df["p_cal_fit"].values.astype(float)-y)**2))
            st.metric("Brier Score (calibrated)", f"{brier_cal:.4f}",
                      delta=f"{(brier_cal-brier)*100:.2f}% vs raw",
                      delta_color="inverse")
            # [FIX 9] Show training range
            st.caption(f"Training range: [{calib.get('training_min',0):.3f}, {calib.get('training_max',1):.3f}]")
            settled_df["bin2"] = pd.cut(settled_df["p_cal_fit"], bins=n_bins, labels=False, include_lowest=True)
            rel2 = settled_df.groupby("bin2",dropna=True).agg(
                p_mean=("p_cal_fit","mean"),win_rate=("y","mean"),n=("y","size")).reset_index()
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00AAFF;letter-spacing:0.12em;text-transform:uppercase;margin:0.6rem 0;'>POST-CALIBRATION RELIABILITY</div>", unsafe_allow_html=True)
            st.dataframe(rel2, use_container_width=True)
            st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#4A607A;'>POLICY AUDIT</div>", unsafe_allow_html=True)
            if hit_rate_cal < 0.48:
                st.error("Hit rate below 48% - cut volume, tighten EV threshold, review market selection.")
            elif hit_rate_cal > 0.58:
                st.success("Strong hit rate - consider increasing Kelly fraction gradually.")
            else:
                st.info("Moderate hit rate. Continue collecting data, focus on CLV tracking.")
            if brier_cal > brier:
                st.warning("Calibrator is WORSENING Brier score - needs more data. Reset to identity.")
    # ── Rolling Brier Score ────────────────────────────────────
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>ROLLING BRIER SCORE (TRAILING WINDOWS)</div>", unsafe_allow_html=True)
    rb = compute_rolling_brier(settled_df if not settled_df.empty else pd.DataFrame())
    if rb:
        rb_cols = st.columns(3)
        for i, w in enumerate([25, 50, 100]):
            key = f"last_{w}"
            if key in rb:
                rb_cols[i].metric(f"Last {w} Legs", f"{rb[key]:.4f}", help="Brier score: lower = better calibrated")
        if "rolling_series" in rb and len(rb["rolling_series"]) > 5:
            st.caption("Trailing 25-leg rolling Brier (lower = better):")
            import pandas as _pd_rb
            series_df = _pd_rb.DataFrame({"Brier": rb["rolling_series"]})
            st.line_chart(series_df, use_container_width=True, height=150)
    else:
        st.caption("Need 10+ settled legs for rolling Brier.")
# ─── INSIGHTS TAB (CLV leaderboard, book efficiency, prop breakdown, Bayesian priors) ───
with tabs[6]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>INSIGHTS — EDGE ANALYTICS & INTELLIGENCE</div>""", unsafe_allow_html=True)
    h_ins = load_history(st.session_state.get("user_id","trader"))
    legs_ins = _expand_history_legs(h_ins)
    ins_tabs = st.tabs(["CLV Leaderboard", "Book Efficiency", "Prop Breakdown", "Bayesian Priors"])
    with ins_tabs[0]:
        # [UPGRADE 31] CLV Leaderboard
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;'>TOP CLOSING LINE VALUE PLAYS</div>", unsafe_allow_html=True)
        st.caption("Ranks your best bets by no-vig price CLV. Positive = you beat the closing line. This is the gold-standard long-term edge indicator.")
        clv_lb = compute_clv_leaderboard(h_ins, top_n=25)
        if clv_lb.empty:
            st.info("No CLV data yet. Fetch CLV updates from the History tab after games close.")
        else:
            clv_pos = clv_lb[clv_lb["clv_price"].notna() & (clv_lb["clv_price"] > 0)]
            clv_neg = clv_lb[clv_lb["clv_price"].notna() & (clv_lb["clv_price"] <= 0)]
            n_clv = len(clv_lb)
            rate = len(clv_pos) / max(n_clv, 1)
            c1, c2, c3 = st.columns(3)
            c1.metric("Beats Closing Line", f"{rate*100:.0f}%", help="CLV price > 0 = bought above close")
            c2.metric("Avg CLV (price)", f"{clv_lb['clv_price'].mean():.4f}" if not clv_lb.empty else "--")
            c3.metric("Avg Line CLV", f"{clv_lb['clv_line'].mean():.2f}" if not clv_lb.empty else "--")
            if n_clv < 10:
                st.warning(f"Only {n_clv} CLV data point{'s' if n_clv != 1 else ''} — rate is not yet statistically meaningful. Keep logging bets and fetching CLV updates.")
            st.dataframe(clv_lb, use_container_width=True)
            if not clv_pos.empty:
                st.markdown("<div style='font-size:0.65rem;color:#00FFB2;margin-top:0.4rem;'>Positive CLV = model found real edge vs closing price. Keep targeting these players/markets.</div>", unsafe_allow_html=True)
    with ins_tabs[1]:
        # [UPGRADE 32] Book Efficiency Score
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;'>PER-BOOK MARKET EFFICIENCY</div>", unsafe_allow_html=True)
        st.caption("Books with higher hit rate = your model has edge there. Books that never move = likely pricing your bets correctly. Focus on soft books with high hit rate.")
        book_eff = compute_book_efficiency(h_ins)
        if book_eff.empty:
            st.info("Need settled bets with CLV data to compute book efficiency.")
        else:
            st.dataframe(book_eff, use_container_width=True)
            best_book = book_eff.iloc[0]["book"] if not book_eff.empty else None
            if best_book:
                st.markdown(f"<div style='color:#00FFB2;font-size:0.68rem;margin-top:0.4rem;'>Best book by hit rate: <b>{best_book}</b> — prioritize this book when line shopping.</div>", unsafe_allow_html=True)
    with ins_tabs[2]:
        # [UPGRADE 33] Prop-type edge breakdown
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;'>EDGE BREAKDOWN BY MARKET TYPE</div>", unsafe_allow_html=True)
        st.caption("Hit rate and EV by stat type. If rebounds hit at 58% and points at 48%, focus on rebounds.")
        if legs_ins.empty:
            st.info("No settled legs yet.")
        else:
            settled_ins = legs_ins[legs_ins["y"].notna()].copy()
            if settled_ins.empty:
                st.info("No settled legs yet.")
            else:
                mkt_breakdown = settled_ins.groupby("market").agg(
                    bets=("y","size"),
                    hit_rate=("y","mean"),
                    avg_ev_adj=("ev_adj","mean"),
                    avg_cv=("cv","mean"),
                ).reset_index()
                mkt_breakdown["hit_%"]  = (mkt_breakdown["hit_rate"] * 100).round(1)
                mkt_breakdown["ev_%"]   = (mkt_breakdown["avg_ev_adj"] * 100).round(2)
                mkt_breakdown["avg_cv"] = mkt_breakdown["avg_cv"].round(3)
                mkt_breakdown = mkt_breakdown.sort_values("hit_%", ascending=False)
                st.dataframe(mkt_breakdown[["market","bets","hit_%","ev_%","avg_cv"]], use_container_width=True)
                # Highlight best market
                best_mkt = mkt_breakdown.iloc[0]["market"] if not mkt_breakdown.empty else None
                if best_mkt:
                    st.markdown(f"<div style='color:#00FFB2;font-size:0.68rem;'>Best market: <b>{best_mkt}</b> ({mkt_breakdown.iloc[0]['hit_%']:.1f}% hit rate). Increase allocation here.</div>", unsafe_allow_html=True)
                # Worst market
                worst_mkt = mkt_breakdown.iloc[-1]["market"] if len(mkt_breakdown) > 1 else None
                if worst_mkt and mkt_breakdown.iloc[-1]["hit_%"] < 48:
                    st.markdown(f"<div style='color:#FF3358;font-size:0.68rem;'>Weakest market: <b>{worst_mkt}</b> ({mkt_breakdown.iloc[-1]['hit_%']:.1f}% hit rate). Consider avoiding.</div>", unsafe_allow_html=True)
    with ins_tabs[3]:
        # [UPGRADE 34] Bayesian Prior Update from history
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;'>BAYESIAN PRIOR UPDATE FROM YOUR HISTORY</div>", unsafe_allow_html=True)
        st.caption("Updates positional priors (the baseline expected stat means) using your personal hit/miss record. Markets you consistently beat get slightly higher priors.")
        pos_for_prior = st.selectbox("Position bucket", ["Guard","Wing","Big","Unknown"], key="prior_pos_sel")
        if not legs_ins.empty and len(legs_ins[legs_ins["y"].notna()]) >= 20:
            updated_priors = compute_history_based_priors(legs_ins, pos_for_prior)
            base_priors    = POSITIONAL_PRIORS.get(pos_for_prior, POSITIONAL_PRIORS["Unknown"])
            prior_rows = []
            for mkt in base_priors:
                orig = base_priors[mkt]
                upd  = updated_priors.get(mkt, orig)
                prior_rows.append({
                    "market": mkt,
                    "original_prior": round(orig, 2),
                    "updated_prior":  round(upd, 2),
                    "delta_%": round((upd/orig - 1)*100, 1) if orig else 0,
                })
            prior_df = pd.DataFrame(prior_rows).sort_values("delta_%", ascending=False)
            st.dataframe(prior_df, use_container_width=True)
            if st.button("Apply Updated Priors to Session", use_container_width=True, key="apply_priors_btn"):
                st.session_state["_custom_priors"] = {pos_for_prior: updated_priors}
                st.success(f"Updated {pos_for_prior} priors applied to this session. Model will use these for the next run.")
        else:
            st.info("Need 20+ settled legs to compute personalised prior updates.")
# ─── ALERTS TAB ───────────────────────────────────────────────
with tabs[7]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>ALERTS — DISCORD / TELEGRAM</div>""", unsafe_allow_html=True)
    # ── PrizePicks cookie auth ─────────────────────────────────
    with st.expander("PrizePicks Cookie Auth (bypasses 403 block)", expanded=False):
        st.markdown("""<div style='font-size:0.68rem;color:#4A607A;line-height:1.6;'>
<b>How to get your cookies (one-time setup, lasts days):</b><br>
1. Open <a href='https://app.prizepicks.com' target='_blank' style='color:#00FFB2;'>app.prizepicks.com</a> in Chrome and log in<br>
2. Press <b>F12</b> → Network tab → refresh the page<br>
3. Click any <code>projections</code> request → Headers tab<br>
4. Find the <b>Cookie:</b> request header → copy the entire value<br>
5. Paste below and click Save
</div>""", unsafe_allow_html=True)
        pp_cookies_val = st.text_area(
            "PrizePicks Cookie String",
            value=st.session_state.get("pp_cookies", ""),
            height=80,
            placeholder="_pxmvid=...; _px3=...; __cf_bm=...",
            key="pp_cookies_input",
        )
        if st.button("Save PrizePicks Cookies", use_container_width=True):
            _ck_val = pp_cookies_val.strip()
            st.session_state["pp_cookies"] = _ck_val
            save_pp_settings(pp_cookies=_ck_val)
            _fetch_prizepicks_lines_cached.clear()
            st.session_state["_pp_last_cookies_used"] = ""
            _is_json = _ck_val.startswith("{") and '"data"' in _ck_val
            st.success(f"{'JSON response' if _is_json else 'Cookies'} saved to disk — will persist across restarts.")
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
    al_col1, al_col2 = st.columns(2)
    with al_col1:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>DISCORD WEBHOOK</div>", unsafe_allow_html=True)
        discord_webhook = st.text_input("Webhook URL", value=st.session_state.get("discord_webhook",""), type="password", key="discord_wh_input")
        st.session_state["discord_webhook"] = discord_webhook
        if st.button("Test Discord", use_container_width=True):
            ok, err = send_discord_alert(discord_webhook, "NBA Quant Engine — Discord alert test ✅")
            st.success("Discord OK") if ok else st.error(f"Discord failed: {err}")
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
        st.caption("Auto-alert on strong edges (EV > threshold):")
        discord_ev_thresh = st.slider("Min EV% for auto-alert", 0.0, 25.0, float(st.session_state.get("discord_ev_thresh",5.0)), 0.5, key="d_ev_thresh")
        st.session_state["discord_ev_thresh"] = float(discord_ev_thresh)
    with al_col2:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>TELEGRAM BOT</div>", unsafe_allow_html=True)
        tg_token = st.text_input("Bot Token", value=st.session_state.get("tg_token",""), type="password", key="tg_token_input")
        st.session_state["tg_token"] = tg_token
        tg_chat = st.text_input("Chat ID", value=st.session_state.get("tg_chat",""), key="tg_chat_input")
        st.session_state["tg_chat"] = tg_chat
        if st.button("Test Telegram", use_container_width=True):
            ok, err = send_telegram_alert(tg_token, tg_chat, "NBA Quant Engine — Telegram alert test ✅")
            st.success("Telegram OK") if ok else st.error(f"Telegram failed: {err}")
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>SEND SCANNER EDGES AS ALERTS</div>", unsafe_allow_html=True)
    scanner_for_alerts = st.session_state.get("scanner_results")
    if scanner_for_alerts is None or scanner_for_alerts.empty:
        st.info("Run Live Scanner first.")
    else:
        alert_thresh = st.slider("Min EV% to include", 0.0, 20.0, 3.0, 0.5, key="alert_thresh_send")
        alerted = [r for _, r in scanner_for_alerts.iterrows() if float(r.get("ev_adj_pct") or 0) >= alert_thresh]
        st.write(f"{len(alerted)} edges above {alert_thresh:.1f}% EV threshold")
        if st.button(f"Send {len(alerted)} alerts to Discord + Telegram", use_container_width=True):
            sent_d, sent_t, errs_d, errs_t = 0, 0, [], []
            for r in alerted:
                msg = format_edge_alert(dict(r))
                if discord_webhook:
                    ok, e = send_discord_alert(discord_webhook, msg)
                    if ok: sent_d += 1
                    elif e: errs_d.append(e)
                if tg_token and tg_chat:
                    ok, e = send_telegram_alert(tg_token, tg_chat, msg)
                    if ok: sent_t += 1
                    elif e: errs_t.append(e)
            st.success(f"Sent — Discord: {sent_d} | Telegram: {sent_t}")
            for e in (errs_d + errs_t)[:5]:
                st.warning(e)
    # [UPGRADE 23] Alert Digest Mode
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>DAILY DIGEST MODE</div>", unsafe_allow_html=True)
    scanner_for_digest = st.session_state.get("scanner_results")
    if scanner_for_digest is None or scanner_for_digest.empty:
        st.info("Run Live Scanner first to generate digest content.")
    else:
        digest_ev_thresh = st.slider("Min EV% for digest", 0.0, 20.0, 3.0, 0.5, key="digest_thresh")
        digest_legs = [dict(r) for _, r in scanner_for_digest.iterrows()
                       if float(r.get("ev_adj_pct") or 0) >= digest_ev_thresh]
        digest_msg  = format_digest_message(digest_legs)
        st.text_area("Digest Preview", value=digest_msg, height=200, key="digest_preview")
        dig_c1, dig_c2 = st.columns(2)
        if dig_c1.button("Send Digest to Discord", use_container_width=True, key="send_digest_discord"):
            _dw3 = st.session_state.get("discord_webhook","")
            if _dw3:
                ok, err = send_discord_alert(_dw3, digest_msg)
                st.success("Digest sent to Discord.") if ok else st.error(f"Discord error: {err}")
            else:
                st.warning("No Discord webhook configured.")
        if dig_c2.button("Send Digest to Telegram", use_container_width=True, key="send_digest_tg"):
            _tt3 = st.session_state.get("tg_token",""); _tc3 = st.session_state.get("tg_chat","")
            if _tt3 and _tc3:
                ok, err = send_telegram_alert(_tt3, _tc3, digest_msg)
                st.success("Digest sent to Telegram.") if ok else st.error(f"Telegram error: {err}")
            else:
                st.warning("No Telegram bot configured.")
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#4A607A;letter-spacing:0.10em;'>INJURY REPORT</div>", unsafe_allow_html=True)
    if st.button("Fetch Today's Injury Report", use_container_width=True):
        fetch_injury_report.clear()   # force fresh data
        with st.spinner("Fetching from ESPN..."):
            injuries = fetch_injury_report()
        if not injuries:
            st.info("No injury data found — ESPN may not have posted today's report yet.")
        else:
            # Build and store the map for auto key_teammate_out
            inj_map = build_injury_team_map(injuries)
            st.session_state["injury_team_map"] = inj_map
            out_count = sum(len(v) for v in inj_map.values())
            st.success(f"Loaded {out_count} OUT/DOUBTFUL player(s) — auto teammate-out active in scanner & model.")
            inj_rows = []
            for team, players in injuries.items():
                for p in players:
                    inj_rows.append({"team": team, **p})
            st.dataframe(
                pd.DataFrame(inj_rows).sort_values(["team","status"]),
                use_container_width=True,
            )
    # Show current injury map status
    cur_inj = st.session_state.get("injury_team_map", {})
    if cur_inj:
        n_out = sum(len(v) for v in cur_inj.values())
        st.caption(f"Auto-injury active: {n_out} OUT/DOUBTFUL players across {len(cur_inj)} teams")
    # [UPGRADE 9] Rotowire News Feed
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#FFB800;letter-spacing:0.10em;text-transform:uppercase;'>ROTOWIRE NBA NEWS — FAST INJURY INTEL</div>", unsafe_allow_html=True)
    st.caption("Scrapes Rotowire's live NBA injury page for faster intel than ESPN's 15-min cache.")
    rw_col1, rw_col2 = st.columns([3,1])
    with rw_col2:
        rw_filter = st.text_input("Filter player", key="rw_filter_input")
    if rw_col1.button("Fetch Rotowire News", use_container_width=True, key="rw_fetch_btn"):
        fetch_rotowire_news.clear()
        with st.spinner("Scraping Rotowire..."):
            rw_rows, rw_err = fetch_rotowire_news()
        if rw_err:
            st.error(f"Rotowire: {rw_err}")
        elif not rw_rows:
            st.warning("No news found — Rotowire page structure may have changed.")
        else:
            st.session_state["rw_news"] = rw_rows
            st.success(f"Fetched {len(rw_rows)} Rotowire reports.")
    rw_news = st.session_state.get("rw_news", [])
    if rw_news:
        rw_df = pd.DataFrame(rw_news)
        if rw_filter:
            rw_df = rw_df[rw_df["player"].str.strip().str.contains(rw_filter.strip(), case=False, na=False)]
        st.dataframe(rw_df, use_container_width=True)
        # Cross-reference with watchlist
        wl_cur = load_watchlist(st.session_state.get("user_id","trader"))
        if wl_cur:
            wl_norms = [normalize_name(p) for p in wl_cur]
            rw_wl = rw_df[rw_df["player"].apply(lambda p: normalize_name(p) in wl_norms)]
            if not rw_wl.empty:
                st.markdown("<div style='color:#FF3358;font-size:0.68rem;font-weight:600;margin-top:0.4rem;'>WATCHLIST ALERT — these players have Rotowire news:</div>", unsafe_allow_html=True)
                st.dataframe(rw_wl, use_container_width=True)
# ── SETTINGS TAB (tabs[8]) ──────────────────────────────────
with tabs[8]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;
color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;
margin-bottom:1.2rem;'>ACCOUNT & ENGINE SETTINGS</div>""", unsafe_allow_html=True)
    _set_col1, _set_col2 = st.columns(2)
    with _set_col1:
        # ── ACCOUNT INFO ────────────────────────────────────
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;
margin-bottom:0.6rem;padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>
▸ ACCOUNT</div>""", unsafe_allow_html=True)
        _acct_user = st.session_state.get("_auth_user", "")
        _acct_email = _get_user_email(_acct_user)
        st.markdown(f"""
<div style='background:#060D18;border:1px solid #0E1E30;border-radius:4px;
            padding:0.7rem 0.9rem;margin-bottom:0.8rem;'>
    <div style='font-family:Fira Code,monospace;font-size:0.58rem;color:#2A6080;
                margin-bottom:4px;'>USERNAME</div>
    <div style='font-family:Chakra Petch,monospace;font-size:1rem;font-weight:700;
                color:#00FFB2;'>{_acct_user}</div>
    {f"<div style='font-family:Fira Code,monospace;font-size:0.62rem;color:#4A7090;margin-top:4px;'>{_acct_email}</div>" if _acct_email else ""}
</div>
""", unsafe_allow_html=True)
        # ── BANKROLL ─────────────────────────────────────────
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;
margin-bottom:0.6rem;margin-top:0.8rem;padding-bottom:0.4rem;
border-bottom:1px solid #0E1E30;'>▸ BANKROLL</div>""", unsafe_allow_html=True)
        _set_br = st.number_input("Bankroll ($)", min_value=0.0,
            value=float(st.session_state.get("bankroll", 1000.0)), step=50.0,
            key="settings_bankroll")
        st.session_state["bankroll"] = float(_set_br)
        _bs2 = load_user_state(user_id)
        _bs2["bankroll"] = float(_set_br)
        save_user_state(user_id, _bs2)
        # ── MODEL PARAMETERS ────────────────────────────────
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;
margin-bottom:0.6rem;margin-top:0.8rem;padding-bottom:0.4rem;
border-bottom:1px solid #0E1E30;'>▸ MODEL PARAMETERS</div>""", unsafe_allow_html=True)
        _mpw = st.slider("Model Weight vs Market", 0.0, 1.0,
            float(st.session_state.get("market_prior_weight", 0.65)), 0.05,
            help="1.0 = pure statistical model · 0.0 = pure market implied prob",
            key="settings_mpw")
        st.session_state["market_prior_weight"] = float(_mpw)
        _ng = st.slider("Sample Window (games)", 5, 30,
            int(st.session_state.get("n_games", 10)), key="settings_ng")
        st.session_state["n_games"] = _ng
        _fk = st.slider("Fractional Kelly", 0.0, 1.0,
            float(st.session_state.get("frac_kelly", 0.25)), 0.05, key="settings_fk")
        st.session_state["frac_kelly"] = _fk
        _pm = st.number_input("Parlay Payout (x)", min_value=1.0,
            value=float(st.session_state.get("payout_multi", 3.0)), step=0.1, key="settings_pm")
        st.session_state["payout_multi"] = _pm
        # ── FILTERS ──────────────────────────────────────────
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;
margin-bottom:0.6rem;margin-top:0.8rem;padding-bottom:0.4rem;
border-bottom:1px solid #0E1E30;'>▸ FILTERS</div>""", unsafe_allow_html=True)
        _exc = st.checkbox("Block Chaotic Regime",
            value=bool(st.session_state.get("exclude_chaotic", True)),
            help="Filters high-CV / blowout-risk environments",
            key="settings_exc")
        st.session_state["exclude_chaotic"] = bool(_exc)
        _su = st.checkbox("Show Under Opportunities",
            value=bool(st.session_state.get("show_unders", False)),
            help="Surface high-probability Unders alongside each leg",
            key="settings_su")
        st.session_state["show_unders"] = bool(_su)
    with _set_col2:
        # ── RISK CONTROLS ────────────────────────────────────
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;
margin-bottom:0.6rem;padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>
▸ RISK CONTROLS</div>""", unsafe_allow_html=True)
        _mrpb = st.slider("Max Bet Size (% Bankroll)", 0.0, 10.0,
            float(st.session_state.get("max_risk_per_bet", 3.0)), 0.5, key="settings_mrpb")
        st.session_state["max_risk_per_bet"] = float(_mrpb)
        _mdl = st.slider("Daily Loss Stop (%)", 0, 50,
            int(st.session_state.get("max_daily_loss", 15)), key="settings_mdl")
        st.session_state["max_daily_loss"] = _mdl
        _mwl = st.slider("Weekly Loss Stop (%)", 0, 50,
            int(st.session_state.get("max_weekly_loss", 25)), key="settings_mwl")
        st.session_state["max_weekly_loss"] = _mwl
        _rclr = "#FF3358" if _mrpb >= 7 else ("#FFB800" if _mrpb >= 4 else "#00FFB2")
        st.markdown(f"""
<div style='background:#04080F;border:1px solid #0E1E30;border-radius:4px;
            padding:0.5rem 0.7rem;margin:0.4rem 0 0.8rem 0;
            display:flex;gap:1rem;align-items:center;'>
    <div style='flex:1;text-align:center;'>
        <div style='font-family:Fira Code,monospace;font-size:0.55rem;color:#2A6080;'>DAILY STOP</div>
        <div style='font-family:Fira Code,monospace;font-size:0.85rem;color:#FFB800;font-weight:600;'>{_mdl}%</div>
    </div>
    <div style='width:1px;height:28px;background:#0E1E30;'></div>
    <div style='flex:1;text-align:center;'>
        <div style='font-family:Fira Code,monospace;font-size:0.55rem;color:#2A6080;'>WEEKLY STOP</div>
        <div style='font-family:Fira Code,monospace;font-size:0.85rem;color:#FFB800;font-weight:600;'>{_mwl}%</div>
    </div>
    <div style='width:1px;height:28px;background:#0E1E30;'></div>
    <div style='flex:1;text-align:center;'>
        <div style='font-family:Fira Code,monospace;font-size:0.55rem;color:#2A6080;'>MAX BET</div>
        <div style='font-family:Fira Code,monospace;font-size:0.85rem;color:{_rclr};font-weight:600;'>{_mrpb:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)
        # ── ODDS API ─────────────────────────────────────────
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;
margin-bottom:0.6rem;padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>
▸ ODDS API</div>""", unsafe_allow_html=True)
        _hdr_s = st.session_state.get("_odds_headers_last", {})
        _rem_s = _hdr_s.get("remaining", "?"); _used_s = _hdr_s.get("used", "?")
        try:
            _ri = int(_rem_s); _ui = int(_used_s); _tot = _ri + _ui
            _pct = (_ui / _tot * 100) if _tot > 0 else 0
            _rc = "#FF3358" if _ri < 10 else ("#FFB800" if _ri < 30 else "#00FFB2")
            _bc = "#FF3358" if _pct > 85 else ("#FFB800" if _pct > 60 else "#00FFB2")
            _bw = f"{min(_pct, 100):.0f}%"
        except Exception:
            _rc = "#4A607A"; _bc = "#1E2D3D"; _bw = "0%"
        st.markdown(f"""
<div style='background:#04080F;border:1px solid #0E1E30;border-radius:4px;
            padding:0.5rem 0.7rem;margin-bottom:0.5rem;'>
    <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem;'>
        <div style='font-family:Fira Code,monospace;font-size:0.58rem;color:#2A6080;'>QUOTA USAGE</div>
        <div style='font-family:Fira Code,monospace;font-size:0.60rem;'>
            <span style='color:#4A607A;'>used </span><span style='color:#8BA8BF;'>{_used_s}</span>
            <span style='color:#2A4060;'> / rem </span><span style='color:{_rc};font-weight:600;'>{_rem_s}</span>
        </div>
    </div>
    <div style='background:#0A1628;border-radius:2px;height:3px;'>
        <div style='background:{_bc};width:{_bw};height:3px;border-radius:2px;'></div>
    </div>
</div>
""", unsafe_allow_html=True)
        _scan_book_override_s = st.text_input("Book override (blank=auto)", value="", key="settings_book_override")
        _mrd = st.number_input("Max API requests/day", 1, 500,
            int(st.session_state.get("max_req_day", 100)), 10, key="settings_mrd")
        st.session_state["max_req_day"] = int(_mrd)
        # ── PRIZEPICKS CONNECTION ─────────────────────────────
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;
margin-top:0.8rem;margin-bottom:0.6rem;padding-bottom:0.4rem;
border-bottom:1px solid #0E1E30;'>▸ PRIZEPICKS CONNECTION</div>""", unsafe_allow_html=True)
        # Status indicator
        _pp_conn_rows, _pp_conn_age, _ = get_pp_auto_lines()
        _pp_conn_ok = bool(_pp_conn_rows and _pp_conn_age is not None and _pp_conn_age < 1800)
        _pp_dot_col = "#00FFB2" if _pp_conn_ok else "#FF3358"
        _pp_conn_label = f"CONNECTED · {len(_pp_conn_rows)} lines · {_pp_conn_age//60}m ago" if _pp_conn_ok else "DISCONNECTED"
        st.markdown(f"""<div style='background:#04080F;border:1px solid #0E1E30;border-radius:4px;
padding:0.4rem 0.7rem;display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;'>
<div style='width:7px;height:7px;border-radius:50%;background:{_pp_dot_col};box-shadow:0 0 6px {_pp_dot_col};'></div>
<div style='font-family:Fira Code,monospace;font-size:0.60rem;color:{_pp_dot_col};'>{_pp_conn_label}</div>
</div>""", unsafe_allow_html=True)
        # Method 1: Scraping Proxy
        with st.expander("Method ① Scraping Proxy (Recommended — free, works from any server)", expanded=not _pp_conn_ok):
            st.markdown("""<div style='font-family:Fira Code,monospace;font-size:0.60rem;color:#4A7090;'>
Routes PP requests through residential IPs — bypasses the datacenter IP block permanently.
<br><b>ScraperAPI free tier:</b> 5,000 credits/month. PrizePicks requires <b>premium mode</b> (10 credits/req) → ~250 fetches/month (~8/day).
<br><b>Recommended:</b> Auto-refresh every 30 min or fetch manually before each scan.
<br><b>ScrapingBee</b> is a good alternative if ScraperAPI quota runs out.</div>""", unsafe_allow_html=True)
            _proxy_svc_opts = list(_PROXY_SERVICES.keys())
            _cur_psvc = st.session_state.get("pp_proxy_service", "scraperapi")
            _psvc_idx = _proxy_svc_opts.index(_cur_psvc) if _cur_psvc in _proxy_svc_opts else 0
            _new_psvc = st.selectbox("Service", options=_proxy_svc_opts, index=_psvc_idx, key="settings_pp_proxy_svc")
            _svc_info = _PROXY_SERVICES.get(_new_psvc, {})
            st.caption(f"Free tier: {_svc_info.get('free_tier','?')} — [Sign up free]({_svc_info.get('signup','')})")
            _new_pkey = st.text_input("API Key", value=st.session_state.get("pp_proxy_key",""), type="password", key="settings_pp_proxy_key")
            _pc1, _pc2 = st.columns(2)
            if _pc1.button("Save Proxy Config", key="save_proxy_cfg", use_container_width=True):
                st.session_state["pp_proxy_service"] = _new_psvc
                st.session_state["pp_proxy_key"] = _new_pkey
                save_pp_settings(pp_proxy_service=_new_psvc, pp_proxy_key=_new_pkey)
                st.success("Saved!")
            if _pc2.button("Test Proxy", key="test_proxy_btn", use_container_width=True):
                if _new_pkey:
                    with st.spinner("Testing..."):
                        _fetch_pp_via_proxy.clear()
                        _tr, _te = _fetch_pp_via_proxy(_new_psvc, _new_pkey)
                    if _tr:
                        st.success(f"✓ Proxy works! {len(_tr)} PP props fetched.")
                    else:
                        st.error(f"Proxy failed: {_te}")
                else:
                    st.warning("Enter API key first.")
        # Method 2: Local Relay
        with st.expander("Method ② Local Relay (pp_relay.py + ngrok)", expanded=False):
            _relay_val = st.text_input("Relay URL", value=st.session_state.get("pp_relay_url",""), key="settings_relay_url", placeholder="https://xxxx.ngrok.io/lines")
            if st.button("Save Relay URL", key="save_relay_url"):
                st.session_state["pp_relay_url"] = _relay_val
                save_pp_settings(pp_relay_url=_relay_val)
                st.success("Saved!")
            st.code("python pp_relay.py", language="bash")
            st.caption("Run on your local machine, expose via ngrok, paste the URL above.")
        # Method 3: Manual JSON
        with st.expander("Method ③ Manual JSON Import (last resort)", expanded=False):
            st.markdown("""<div style='font-size:0.62rem;color:#4A607A;'>Paste the raw JSON from https://api.prizepicks.com/projections?per_page=500 (open in your browser).</div>""", unsafe_allow_html=True)
            _manual_json = st.text_area("Paste PP JSON", value="", height=100, key="settings_manual_pp_json")
            if st.button("Import JSON", key="import_pp_json"):
                try:
                    _mj_data = json.loads(_manual_json)
                    _mj_rows = _parse_pp_response(_mj_data)
                    if _mj_rows:
                        st.session_state["pp_lines"] = pd.DataFrame(_mj_rows)
                        _save_pp_disk_cache(_mj_rows)
                        st.success(f"✓ Imported {len(_mj_rows)} PP props")
                    else:
                        st.warning("0 props found in JSON — check format")
                except Exception as _mje:
                    st.error(f"JSON parse error: {_mje}")
        with st.expander("Background scraper service (always-fresh lines)", expanded=False):
            st.code("python pp_scraper.py --interval 600", language="bash")
            st.caption("Fetches every 10 min, writes to shared DB. Fastest option when running locally.")
        # ── ODDS API ─────────────────────────────────────────
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;
margin-top:0.8rem;margin-bottom:0.6rem;padding-bottom:0.4rem;
border-bottom:1px solid #0E1E30;'>▸ ODDS API</div>""", unsafe_allow_html=True)
        _odds_stored = st.session_state.get("_odds_api_key_override", "") or load_pp_settings().get("odds_api_key","")
        if _odds_stored:
            st.markdown("""
<div style='background:linear-gradient(90deg,#00FFB208,transparent);
            border:1px solid #00FFB230;border-radius:3px;
            padding:0.4rem 0.7rem;display:flex;align-items:center;gap:0.5rem;
            margin-bottom:0.5rem;'>
    <div style='width:6px;height:6px;border-radius:50%;background:#00FFB2;
                box-shadow:0 0 6px #00FFB2;'></div>
    <div style='font-family:Fira Code,monospace;font-size:0.62rem;color:#00FFB2;'>
        ODDS API KEY ACTIVE</div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("No Odds API key configured — live line shopping and book price comparisons will be unavailable.")
        with st.expander("Set Odds API Key", expanded=not bool(_odds_stored)):
            _odds_input = st.text_input(
                "The Odds API Key",
                value=_odds_stored,
                type="password",
                help="Get your key at the-odds-api.com",
                key="settings_odds_key_input"
            )
            if st.button("Save Odds API Key", key="save_odds_api_key_btn"):
                _odds_val = _odds_input.strip()
                st.session_state["_odds_api_key_override"] = _odds_val
                os.environ["ODDS_API_KEY"] = _odds_val
                save_pp_settings(odds_api_key=_odds_val)
                st.success("✓ Odds API key saved" if _odds_val else "✓ Odds API key cleared")
                st.rerun()
        # ── CLAUDE AI ────────────────────────────────────────
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;
margin-top:0.8rem;margin-bottom:0.6rem;padding-bottom:0.4rem;
border-bottom:1px solid #0E1E30;'>▸ CLAUDE AI ENGINE</div>""", unsafe_allow_html=True)
        _ai_key_s = _get_anthropic_key()
        if _ai_key_s:
            st.markdown("""
<div style='background:linear-gradient(90deg,#00FFB208,transparent);
            border:1px solid #00FFB230;border-radius:3px;
            padding:0.4rem 0.7rem;display:flex;align-items:center;gap:0.5rem;
            margin-bottom:0.5rem;'>
    <div style='width:6px;height:6px;border-radius:50%;background:#00FFB2;
                box-shadow:0 0 6px #00FFB2;'></div>
    <div style='font-family:Fira Code,monospace;font-size:0.62rem;color:#00FFB2;'>
        CLAUDE AI ACTIVE — all users have access</div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("Claude AI key not configured. Add ANTHROPIC_API_KEY to Streamlit secrets to enable AI for all users.")
        with st.expander("Override API Key (optional)", expanded=False):
            _ai_ov_stored = st.session_state.get("_anthropic_key_override", "")
            _ai_ov_input = st.text_input(
                "Personal Anthropic API Key",
                value=_ai_ov_stored,
                type="password",
                help="Leave blank to use the app's shared key.",
                key="settings_ai_key_input"
            )
            if _ai_ov_input != _ai_ov_stored:
                st.session_state["_anthropic_key_override"] = _ai_ov_input
                if _ai_ov_input:
                    os.environ["ANTHROPIC_API_KEY"] = _ai_ov_input
                for _k in list(st.session_state.keys()):
                    if _k.startswith("_ai_"):
                        st.session_state.pop(_k, None)
            elif _ai_ov_input and not os.environ.get("ANTHROPIC_API_KEY"):
                os.environ["ANTHROPIC_API_KEY"] = _ai_ov_input
        # ── WATCHLIST ────────────────────────────────────────
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.60rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;
margin-top:0.8rem;margin-bottom:0.6rem;padding-bottom:0.4rem;
border-bottom:1px solid #0E1E30;'>▸ WATCHLIST</div>""", unsafe_allow_html=True)
        _wl_s = load_watchlist(st.session_state.get("user_id", "trader"))
        _wl_add_s = st.text_input("Add player to watchlist", placeholder="LeBron James",
                                   key="settings_wl_add")
        _wl_c1, _wl_c2 = st.columns(2)
        if _wl_c1.button("Add", key="settings_wl_add_btn"):
            if _wl_add_s.strip() and _wl_add_s.strip() not in _wl_s:
                _wl_s.append(_wl_add_s.strip())
                save_watchlist(st.session_state.get("user_id", "trader"), _wl_s)
                st.rerun()
        if _wl_s:
            _wl_rm_s = st.selectbox("Remove player", ["--"] + _wl_s, key="settings_wl_rm")
            if _wl_c2.button("Remove", key="settings_wl_rm_btn") and _wl_rm_s != "--":
                _wl_s = [p for p in _wl_s if p != _wl_rm_s]
                save_watchlist(st.session_state.get("user_id", "trader"), _wl_s)
                st.rerun()
        if _wl_s:
            _wl_pills = "".join([
                f"<div style='display:inline-block;background:#0A1828;border:1px solid #1A3050;"
                f"border-radius:3px;padding:0.15rem 0.5rem;margin:0.15rem;font-family:Fira Code,monospace;"
                f"font-size:0.62rem;color:#6AABCF;'>{p}</div>"
                for p in _wl_s
            ])
            st.markdown(f"<div style='margin-top:0.4rem;'>{_wl_pills}</div>",
                        unsafe_allow_html=True)
    # ── SAVE SETTINGS NOTICE ─────────────────────────────────
    st.markdown("""<div style='margin-top:1rem;font-family:Fira Code,monospace;font-size:0.60rem;
color:#2A5070;border-top:1px solid #0E1E30;padding-top:0.6rem;'>
Settings are applied immediately and saved to your account.</div>""",
        unsafe_allow_html=True)
    # Save settings to user state
    _settings_to_save = {
        "market_prior_weight": st.session_state.get("market_prior_weight", 0.65),
        "n_games": st.session_state.get("n_games", 10),
        "frac_kelly": st.session_state.get("frac_kelly", 0.25),
        "payout_multi": st.session_state.get("payout_multi", 3.0),
        "max_risk_per_bet": st.session_state.get("max_risk_per_bet", 3.0),
        "max_daily_loss": st.session_state.get("max_daily_loss", 15),
        "max_weekly_loss": st.session_state.get("max_weekly_loss", 25),
        "exclude_chaotic": st.session_state.get("exclude_chaotic", True),
        "show_unders": st.session_state.get("show_unders", False),
        "max_req_day": st.session_state.get("max_req_day", 100),
        "bankroll": st.session_state.get("bankroll", 1000.0),
    }
    _cur_saved = load_user_state(st.session_state.get("user_id", "trader"))
    if any(_cur_saved.get(k) != v for k, v in _settings_to_save.items()):
        _cur_saved.update(_settings_to_save)
        save_user_state(st.session_state.get("user_id", "trader"), _cur_saved)
# ── FOOTER ──────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:2rem;padding-top:0.8rem;border-top:1px solid #1E2D3D;
font-family:Fira Code,monospace;font-size:0.60rem;color:#2A3A4A;
display:flex;flex-wrap:wrap;gap:0.3rem 1rem;justify-content:space-between;'>
  <span>NBA QUANT ENGINE v4.0</span>
  <span style='flex:1;min-width:0;word-break:break-word;'>EXP DECAY | OPP SPLITS | HOT/COLD | O/U ASYMMETRY | CLV LEADERBOARD | BOOK EFFICIENCY | STEAM DETECTOR | ROTOWIRE | PARLAY BUILDER | DIGEST ALERTS | Q1/FANTASY MKTS</span>
  <span>Powered by Kamal</span>
</div>
""", unsafe_allow_html=True)

