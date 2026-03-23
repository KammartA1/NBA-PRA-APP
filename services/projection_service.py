"""
Projection Service — wraps compute_leg_projection() and related model computation.

All computation reads/writes database, not session state.
Returns plain dicts so they're Streamlit-compatible.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, date, timedelta

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports from the monolith app.py — avoids circular imports and heavy
# module-level NBA API loads.  Every public function in this module calls
# _ensure_imports() first.
# ---------------------------------------------------------------------------
_app = None


def _ensure_imports():
    """Import computation functions from app.py on first use."""
    global _app
    if _app is not None:
        return
    import app as _app_module
    _app = _app_module


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def _get_quant_session():
    """Return a quant_system DB session."""
    try:
        from quant_system.db.schema import get_session
        return get_session()
    except Exception:
        return None


def _store_signal(result: dict):
    """Persist a projection result into the quant_system signals/bet_log table."""
    session = _get_quant_session()
    if session is None:
        return
    try:
        from quant_system.db.schema import BetLog
        from datetime import datetime
        import json, uuid
        # Only store if we have meaningful data
        if result.get("proj") is None or result.get("p_cal") is None:
            return
        row = BetLog(
            bet_id=f"signal_{uuid.uuid4().hex[:12]}",
            sport="nba",
            timestamp=datetime.utcnow(),
            player=result.get("player", ""),
            bet_type=str(result.get("side", "over")).lower(),
            stat_type=str(result.get("market", "")),
            line=float(result.get("line", 0)),
            direction=str(result.get("side", "over")).lower(),
            model_prob=float(result.get("p_cal", 0)),
            market_prob=float(result.get("p_implied", 0) or 0),
            edge=float(result.get("ev_adj", 0) or 0),
            stake=float(result.get("stake", 0) or 0),
            kelly_fraction=float(result.get("stake_frac", 0) or 0),
            odds_american=-110,
            odds_decimal=float(result.get("price_decimal", 1.909) or 1.909),
            model_projection=float(result.get("proj", 0) or 0),
            model_std=float(result.get("sigma", 0) or 0),
            confidence_score=float(result.get("sharpness_score", 0) or 0) / 100.0,
            engine_agreement=0.0,
            status="signal",
            features_snapshot=json.dumps({
                k: result.get(k) for k in (
                    "regime", "hot_cold", "trend_label", "fatigue_label",
                    "volatility_cv", "gate_ok", "gate_reason", "edge_cat",
                )
            }),
            notes="auto-signal from projection_service",
        )
        session.add(row)
        session.commit()
    except Exception as exc:
        log.warning("Failed to store signal: %s", exc)
        try:
            session.rollback()
        except Exception:
            pass
    finally:
        try:
            session.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_projection(
    player_name: str,
    market: str,
    line: float,
    book: str | None = None,
    game_date=None,
    settings_dict: dict | None = None,
) -> dict:
    """
    Run the full model projection for a single leg.

    Parameters
    ----------
    player_name : str
    market : str  – e.g. "Points", "Rebounds", "PRA"
    line : float
    book : str | None – sportsbook key, or None for manual
    game_date : date | None – defaults to today
    settings_dict : dict | None – model params overriding defaults:
        n_games, bankroll, frac_kelly, max_risk_per_bet,
        market_prior_weight, exclude_chaotic, injury_team_map

    Returns
    -------
    dict – full projection result (same schema as compute_leg_projection)
    """
    _ensure_imports()

    settings = settings_dict or {}
    game_date = game_date or date.today()
    n_games = int(settings.get("n_games", 10))
    bankroll = float(settings.get("bankroll", 1000.0))
    frac_kelly = float(settings.get("frac_kelly", 0.25))
    max_risk_frac = float(settings.get("max_risk_per_bet", 3.0)) / 100.0
    market_prior_weight = float(settings.get("market_prior_weight", 0.65))
    exclude_chaotic = bool(settings.get("exclude_chaotic", True))
    injury_team_map = settings.get("injury_team_map", {})

    # Build meta dict if book is provided
    meta = None
    if book and book not in ("manual", ""):
        market_key = _app.ODDS_MARKETS.get(market)
        if market_key:
            try:
                val, m_meta, _err = _app.find_player_line_from_events(
                    player_name, market_key, game_date.isoformat(), book
                )
                if val is not None:
                    line = float(val)
                    meta = m_meta
            except Exception as exc:
                log.warning("Line lookup failed for %s/%s: %s", player_name, market, exc)

    if meta is None:
        # Manual line — assume standard -110
        meta = {
            "event_id": None, "home_team": "", "away_team": "",
            "commence_time": "", "price": 1.909,
            "book": book or "manual",
            "market_key": _app.ODDS_MARKETS.get(market, ""),
            "side": "Over",
        }

    try:
        result = _app.compute_leg_projection(
            player_name, market, line, meta,
            n_games=n_games,
            key_teammate_out=bool(settings.get("key_teammate_out", False)),
            bankroll=bankroll,
            frac_kelly=frac_kelly,
            max_risk_frac=max_risk_frac,
            market_prior_weight=market_prior_weight,
            exclude_chaotic=exclude_chaotic,
            game_date=game_date,
            injury_team_map=injury_team_map,
        )
    except Exception as exc:
        log.error("compute_leg_projection failed: %s", exc, exc_info=True)
        return {
            "player": player_name, "market": market, "line": float(line),
            "errors": [f"projection error: {type(exc).__name__}: {exc}"],
            "gate_ok": False, "gate_reason": "computation error",
        }

    # Apply calibration if available
    calib = settings.get("calibrator_map")
    if calib is not None:
        try:
            result = _app.recompute_pricing_fields(dict(result), calib)
        except Exception as exc:
            log.warning("recompute_pricing_fields failed: %s", exc)

    # Store to DB
    try:
        _store_signal(result)
    except Exception as exc:
        log.debug("Signal storage skipped: %s", exc)

    return result


def run_multi_leg_projection(
    legs: list[dict],
    settings: dict | None = None,
) -> list[dict]:
    """
    Parallel computation of multiple legs.

    Parameters
    ----------
    legs : list[dict]
        Each dict must have: player_name, market, line.
        Optional: book, game_date, key_teammate_out
    settings : dict | None – shared model settings

    Returns
    -------
    list[dict] – projection results (one per leg)
    """
    settings = settings or {}
    max_workers = min(8, len(legs)) if legs else 1

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for leg in legs:
            fut = executor.submit(
                run_projection,
                player_name=leg["player_name"],
                market=leg["market"],
                line=float(leg["line"]),
                book=leg.get("book"),
                game_date=leg.get("game_date"),
                settings_dict={
                    **settings,
                    "key_teammate_out": leg.get("key_teammate_out", False),
                },
            )
            futures.append(fut)

        for fut in futures:
            try:
                results.append(fut.result(timeout=90))
            except FuturesTimeout:
                results.append({
                    "player": "Error", "market": "?", "line": 0.0,
                    "errors": ["thread timeout (>=90s)"],
                    "gate_ok": False, "gate_reason": "thread timeout",
                })
            except Exception as exc:
                results.append({
                    "player": "Error", "market": "?", "line": 0.0,
                    "errors": [f"thread error: {type(exc).__name__}: {exc}"],
                    "gate_ok": False, "gate_reason": "thread error",
                })

    return results


def get_cached_projection(
    player: str,
    market: str,
    max_age_minutes: int = 10,
) -> dict | None:
    """
    Check the signals table for a recent projection (< max_age_minutes).

    Returns the most recent matching signal as a dict, or None.
    """
    session = _get_quant_session()
    if session is None:
        return None
    try:
        from quant_system.db.schema import BetLog
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        row = (
            session.query(BetLog)
            .filter(
                BetLog.sport == "nba",
                BetLog.player == player,
                BetLog.stat_type == market,
                BetLog.timestamp >= cutoff,
                BetLog.status == "signal",
            )
            .order_by(BetLog.timestamp.desc())
            .first()
        )
        if row is None:
            return None
        return {
            "player": row.player,
            "market": row.stat_type,
            "line": float(row.line),
            "proj": float(row.model_projection),
            "p_cal": float(row.model_prob),
            "p_implied": float(row.market_prob),
            "ev_adj": float(row.edge),
            "stake": float(row.stake),
            "stake_frac": float(row.kelly_fraction),
            "sigma": float(row.model_std),
            "sharpness_score": float(row.confidence_score) * 100.0,
            "price_decimal": float(row.odds_decimal),
            "side": row.direction,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "cached": True,
        }
    except Exception as exc:
        log.warning("get_cached_projection failed: %s", exc)
        return None
    finally:
        try:
            session.close()
        except Exception:
            pass
