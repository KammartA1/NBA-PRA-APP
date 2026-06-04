"""
core/sports.py — Multi-sport registry for the Sports Quant Engine.

Single source of truth for which sports are live, their display metadata,
the markets offered within each sport, and the market display->code mapping.
The Streamlit app reads this to drive the sport selector and the per-sport
market dropdowns; the projection layer reads it to route to the right engine.
"""
from __future__ import annotations

# MLB market dropdown (display -> internal code consumed by the MLB engine).
MLB_MARKETS = {
    # Batter
    "Total Bases": "TB",
    "Hits": "H",
    "Runs": "R",
    "RBIs": "RBI",
    "Hits+Runs+RBIs": "HRR",
    "Home Runs": "HR",
    "Stolen Bases": "SB",
    "Walks": "BB",
    "Singles": "1B",
    "Doubles": "2B",
    "Triples": "3B",
    # Pitcher
    "Pitcher Strikeouts": "K",
    "Pitching Outs": "OUTS",
    "Earned Runs": "ER",
    "Hits Allowed": "HA",
    "Walks Allowed": "BB_A",
    # Both
    "Fantasy Score": "MLB_FS",
}

SPORTS = {
    "NBA": {
        "display_name": "NBA",
        "icon": "🏀",
        "subtitle": "NBA 2024-25",
        "engine": "native",          # projected by app.py's built-in NBA pipeline
        "markets": None,             # uses app.py ODDS_MARKETS
        "season_label": "NBA 2024-25",
    },
    "MLB": {
        "display_name": "MLB",
        "icon": "⚾",
        "subtitle": "MLB 2026",
        "engine": "mlb_sim",         # simulation/mlb projection pipeline
        "markets": MLB_MARKETS,
        "season_label": "MLB 2026",
    },
}

DEFAULT_SPORT = "NBA"
ENABLED_SPORTS = ["NBA", "MLB"]


def sport_options() -> list[str]:
    """Labels for the sport selector, e.g. ['🏀 NBA', '⚾ MLB']."""
    return [f"{SPORTS[s]['icon']} {SPORTS[s]['display_name']}" for s in ENABLED_SPORTS]


def sport_from_label(label: str) -> str:
    """Map a selector label back to the canonical sport key."""
    for s in ENABLED_SPORTS:
        if SPORTS[s]["display_name"] in label:
            return s
    return DEFAULT_SPORT


def get_markets(sport: str) -> dict | None:
    return SPORTS.get(sport, {}).get("markets")


def is_native(sport: str) -> bool:
    return SPORTS.get(sport, {}).get("engine") == "native"
