"""
Settings Service — persistent user settings (replaces session_state for durable data).

Stores settings in JSON files on disk (one per user). Thread-safe via file locking.
Returns plain Python types.
"""
import json
import logging
import os
import re
import threading
from datetime import datetime

log = logging.getLogger(__name__)

_SETTINGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "user_settings",
)
_lock = threading.Lock()

# Default model parameters
_MODEL_DEFAULTS = {
    "n_games": 10,
    "frac_kelly": 0.25,
    "payout_multi": 3.0,
    "market_prior_weight": 0.65,
    "max_risk_per_bet": 3.0,
    "exclude_chaotic": True,
    "show_unders": False,
}

# Default risk limits
_RISK_DEFAULTS = {
    "max_daily_loss": 15,
    "max_weekly_loss": 25,
    "max_risk_per_bet": 3.0,
    "max_concurrent_exposure": 10.0,
    "min_ev_threshold": 0.05,
    "min_sharpness": 50,
}


def _sanitize_uid(uid: str) -> str:
    """Sanitize user ID for use as filename."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", uid or "default")


def _settings_path(uid: str) -> str:
    """Return the settings file path for a user."""
    os.makedirs(_SETTINGS_DIR, exist_ok=True)
    return os.path.join(_SETTINGS_DIR, f"settings_{_sanitize_uid(uid)}.json")


def _load_settings_file(uid: str) -> dict:
    """Load settings from disk."""
    fp = _settings_path(uid)
    try:
        if os.path.exists(fp):
            with open(fp) as f:
                return json.load(f) or {}
    except Exception as exc:
        log.warning("Failed to load settings for %s: %s", uid, exc)
    return {}


def _save_settings_file(uid: str, data: dict):
    """Save settings to disk atomically."""
    fp = _settings_path(uid)
    tmp = fp + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, fp)
    except Exception as exc:
        log.error("Failed to save settings for %s: %s", uid, exc)
        # Clean up temp file
        try:
            os.unlink(tmp)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_setting(user_id: str, key: str, default=None):
    """
    Get a single setting value for a user.
    Returns the stored value or default if not found.
    """
    with _lock:
        data = _load_settings_file(user_id)
    return data.get(key, default)


def set_setting(user_id: str, key: str, value):
    """
    Set a single setting value for a user. Persists to disk immediately.
    """
    with _lock:
        data = _load_settings_file(user_id)
        data[key] = value
        data["_updated_at"] = datetime.utcnow().isoformat()
        _save_settings_file(user_id, data)


def get_all_settings(user_id: str) -> dict:
    """
    Get all settings for a user as a dict.
    Returns merged defaults + user overrides.
    """
    with _lock:
        data = _load_settings_file(user_id)

    # Merge with defaults
    result = {}
    result.update(_MODEL_DEFAULTS)
    result.update(_RISK_DEFAULTS)
    result.update(data)
    return result


def get_bankroll(user_id: str) -> float:
    """Get the user's current bankroll."""
    return float(get_setting(user_id, "bankroll", 1000.0))


def set_bankroll(user_id: str, amount: float):
    """Set the user's bankroll amount."""
    if amount < 0:
        log.warning("Attempted to set negative bankroll for %s: %.2f", user_id, amount)
        amount = 0.0
    set_setting(user_id, "bankroll", float(amount))
    # Also log bankroll history for tracking
    with _lock:
        data = _load_settings_file(user_id)
        history = data.get("_bankroll_history", [])
        history.append({
            "amount": float(amount),
            "ts": datetime.utcnow().isoformat(),
        })
        # Keep last 365 entries
        data["_bankroll_history"] = history[-365:]
        data["bankroll"] = float(amount)
        data["_updated_at"] = datetime.utcnow().isoformat()
        _save_settings_file(user_id, data)


def get_model_params(user_id: str) -> dict:
    """
    Return all model parameters for the user.
    Falls back to defaults for any missing keys.
    """
    all_settings = get_all_settings(user_id)
    params = {}
    for key in _MODEL_DEFAULTS:
        params[key] = all_settings.get(key, _MODEL_DEFAULTS[key])
    # Also include bankroll since projection needs it
    params["bankroll"] = float(all_settings.get("bankroll", 1000.0))
    return params


def set_model_params(user_id: str, params: dict):
    """
    Set multiple model parameters at once.
    Only keys that match known model params are stored.
    """
    known_keys = set(_MODEL_DEFAULTS.keys()) | {"bankroll"}
    with _lock:
        data = _load_settings_file(user_id)
        for key, value in params.items():
            if key in known_keys:
                data[key] = value
        data["_updated_at"] = datetime.utcnow().isoformat()
        _save_settings_file(user_id, data)


def get_risk_limits(user_id: str) -> dict:
    """
    Return risk management limits for the user.
    Falls back to defaults for any missing keys.
    """
    all_settings = get_all_settings(user_id)
    limits = {}
    for key in _RISK_DEFAULTS:
        limits[key] = all_settings.get(key, _RISK_DEFAULTS[key])

    # Add computed fields
    bankroll = float(all_settings.get("bankroll", 1000.0))
    limits["bankroll"] = bankroll
    limits["max_daily_loss_amount"] = round(
        bankroll * float(limits.get("max_daily_loss", 15)) / 100.0, 2
    )
    limits["max_weekly_loss_amount"] = round(
        bankroll * float(limits.get("max_weekly_loss", 25)) / 100.0, 2
    )
    limits["max_single_bet_amount"] = round(
        bankroll * float(limits.get("max_risk_per_bet", 3.0)) / 100.0, 2
    )
    return limits
