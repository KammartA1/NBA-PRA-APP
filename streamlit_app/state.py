"""
streamlit_app/state.py
======================
Thin session-state manager.

Rules
-----
* ``st.session_state`` holds ONLY transient UI values:
  - Form inputs being typed (before the user clicks "Save")
  - UI toggle / expander states
  - Temporary display data (e.g. last model run results for the current page view)
* All *persistent* data is read from / written to the database via services.
* On crash / restart, the DB is the single source of truth -- session state
  is rebuilt from it.
"""
from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from services import settings_service

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transient UI state helpers
# ---------------------------------------------------------------------------

def get_ui_state(key: str, default: Any = None) -> Any:
    """Read a transient UI state value.  Returns *default* if missing."""
    return st.session_state.get(key, default)


def set_ui_state(key: str, value: Any) -> None:
    """Write a transient UI state value."""
    st.session_state[key] = value


def toggle_ui_state(key: str) -> bool:
    """Toggle a boolean UI state key and return the new value."""
    current = bool(st.session_state.get(key, False))
    st.session_state[key] = not current
    return not current


# ---------------------------------------------------------------------------
# User settings  (DB-backed via settings_service)
# ---------------------------------------------------------------------------

def load_user_settings(user_id: str) -> dict:
    """Load all persistent settings from settings_service into session_state
    so that Streamlit widgets can display them.

    Only populates keys that are NOT already in session_state, so user edits
    in the current form are not overwritten.
    """
    all_settings = settings_service.get_all_settings(user_id)
    for key, value in all_settings.items():
        if key.startswith("_"):
            continue  # skip internal keys
        if key not in st.session_state:
            st.session_state[key] = value
    return all_settings


def save_user_settings(user_id: str) -> None:
    """Persist the current session_state settings to the database.

    Only saves keys that correspond to known model / risk / bankroll params.
    """
    known_keys = {
        "bankroll", "n_games", "frac_kelly", "payout_multi",
        "market_prior_weight", "max_risk_per_bet", "max_daily_loss",
        "max_weekly_loss", "exclude_chaotic", "show_unders", "max_req_day",
    }
    params = {}
    for key in known_keys:
        if key in st.session_state:
            params[key] = st.session_state[key]

    # Bankroll gets its own setter (has history tracking)
    bankroll = params.pop("bankroll", None)
    if bankroll is not None:
        settings_service.set_bankroll(user_id, float(bankroll))

    # Remaining model params
    if params:
        settings_service.set_model_params(user_id, params)


def get_model_settings(user_id: str) -> dict:
    """Return the current model settings dict, merging DB defaults with any
    session-state overrides.  Suitable for passing to projection_service."""
    db_params = settings_service.get_model_params(user_id)
    # Session state overrides (user may have changed a slider but not saved yet)
    override_keys = [
        "n_games", "frac_kelly", "payout_multi", "market_prior_weight",
        "max_risk_per_bet", "exclude_chaotic", "bankroll",
    ]
    for k in override_keys:
        if k in st.session_state:
            db_params[k] = st.session_state[k]
    return db_params


# ---------------------------------------------------------------------------
# Convenience: ensure session defaults on first run
# ---------------------------------------------------------------------------

def init_session_defaults(user_id: str) -> None:
    """Populate session_state with DB-persisted settings on first page load.

    Safe to call on every rerun -- only sets keys that are missing.
    """
    if get_ui_state("_session_initialized"):
        return

    load_user_settings(user_id)
    set_ui_state("_session_initialized", True)
    log.debug("Session defaults loaded for user %s", user_id)
