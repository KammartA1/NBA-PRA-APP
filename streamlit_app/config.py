"""
streamlit_app/config.py
=======================
Application configuration, page setup, theme constants.
"""
import os
import streamlit as st

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_TITLE = "NBA Prop Alpha Engine"
APP_VERSION = "v5.0-refactored"
APP_ICON = "basketball"
LAYOUT = "wide"

# ---------------------------------------------------------------------------
# Tab names (order matters -- matches tab index)
# ---------------------------------------------------------------------------
TAB_NAMES = [
    "MODEL",
    "RESULTS",
    "LIVE SCANNER",
    "PLATFORMS",
    "HISTORY",
    "CALIBRATION",
    "INSIGHTS",
    "ALERTS",
    "QUANT SYSTEM",
    "CLV SYSTEM",
    "DATA QUALITY",
    "EDGE SOURCES",
    "EDGE DECOMP",
    "EDGE ATTRIBUTION",
    "SETTINGS",
]

# ---------------------------------------------------------------------------
# Market definitions (mirrors app.py ODDS_MARKETS)
# ---------------------------------------------------------------------------
ODDS_MARKETS = {
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
    "H1 Points":       "player_points_q1q2",
    "H1 Rebounds":     "player_rebounds_q1q2",
    "H1 Assists":      "player_assists_q1q2",
    "H1 3PM":          "player_threes_q1q2",
    "H1 PRA":          "player_points_rebounds_assists_q1q2",
    "H2 Points":       "player_points_q3q4",
    "H2 Rebounds":     "player_rebounds_q3q4",
    "H2 Assists":      "player_assists_q3q4",
    "H2 PRA":          "player_points_rebounds_assists_q3q4",
    "Q1 Points":       "player_points_q1",
    "Q1 Rebounds":     "player_rebounds_q1",
    "Q1 Assists":      "player_assists_q1",
    "FGM":             "player_field_goals_made",
    "FGA":             "player_field_goals_attempted",
    "FTM":             "player_free_throws_made",
    "FTA":             "player_free_throws_attempted",
    "3PA":             "player_threes_attempted",
    "Double Double":   "player_double_double",
    "Triple Double":   "player_triple_double",
    "Fantasy Score":   "player_fantasy_score",
    "First Basket":    "player_first_basket",
    "Alt Points":      "player_alternate_points",
    "Alt Rebounds":    "player_alternate_rebounds",
    "Alt Assists":     "player_alternate_assists",
    "Alt 3PM":         "player_alternate_threes",
}

# Markets excluded from the MODEL tab UI selector
_MARKET_EXCLUDE_FROM_UI = {
    "First Basket", "Alt Points", "Alt Rebounds", "Alt Assists", "Alt 3PM",
}
MARKET_OPTIONS = [k for k in ODDS_MARKETS if k not in _MARKET_EXCLUDE_FROM_UI]

# ---------------------------------------------------------------------------
# Default settings (shared reference for services and UI)
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "n_games": 10,
    "frac_kelly": 0.25,
    "payout_multi": 3.0,
    "market_prior_weight": 0.65,
    "max_risk_per_bet": 3.0,
    "max_daily_loss": 15,
    "max_weekly_loss": 25,
    "exclude_chaotic": True,
    "show_unders": False,
    "max_req_day": 100,
    "bankroll": 1000.0,
}

# ---------------------------------------------------------------------------
# Theme / style constants
# ---------------------------------------------------------------------------
COLOR_PRIMARY = "#00FFB2"
COLOR_WARNING = "#FFB800"
COLOR_DANGER = "#FF3358"
COLOR_MUTED = "#4A607A"
COLOR_BG_DARK = "#04080F"
COLOR_BORDER = "#0E1E30"
FONT_MONO = "Fira Code, monospace"
FONT_DISPLAY = "Chakra Petch, monospace"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_page():
    """Configure Streamlit page settings.  Must be the first Streamlit call."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":basketball:",
        layout=LAYOUT,
        initial_sidebar_state="expanded",
    )
    # Inject shared CSS
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def get_anthropic_key() -> str:
    """Resolve Claude API key from session state, secrets, or env."""
    override = st.session_state.get("_anthropic_key_override", "")
    if override:
        return override
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


def get_user_id() -> str:
    """Return the current user ID.  Falls back to 'default'."""
    return st.session_state.get("_auth_user", "") or st.session_state.get("user_id", "default")


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """
<style>
/* Chakra Petch + Fira Code from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=Fira+Code:wght@400;600&display=swap');

/* Subtle dark theme tweaks */
.stApp { font-family: 'Fira Code', monospace; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Chakra Petch', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="stMetric"] label {
    font-family: 'Fira Code', monospace;
    font-size: 0.6rem;
    color: #4A607A;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
</style>
"""
