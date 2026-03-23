"""
workers/signal_worker.py
========================
Generates trading signals whenever new line data arrives.

Checks for new ``LineMovement`` rows since the last run, computes model
projections, calculates edge vs market line, and writes ``Signal`` records.

Only generates signals for events starting within the next 24 hours.

Run standalone:
    python -m workers.signal_worker          # one-shot
    python -m workers.signal_worker --loop   # 10-min loop
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is on sys.path so app-level imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from database.connection import session_scope, init_db
from database.models import LineMovement, Signal, Event, ModelVersion, WorkerStatus
from workers.base import BaseWorker, standalone_main

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Lazy-loaded projection engine
# ---------------------------------------------------------------------------
# Importing app.py directly would pull in Streamlit; instead we import only
# the pure-computation helpers we need.  If that fails (e.g. nba_api not
# installed) we fall back to a simplified heuristic model.

_COMPUTE_AVAILABLE = False
_compute_leg_projection = None


def _ensure_compute_engine():
    """Try to load the heavy projection engine from app.py once."""
    global _COMPUTE_AVAILABLE, _compute_leg_projection

    if _compute_leg_projection is not None:
        return

    try:
        # Monkey-patch Streamlit dependencies that app.py expects at import
        # time so we can import compute_leg_projection without a running
        # Streamlit server.
        import types
        _fake_st = types.ModuleType("streamlit")
        _fake_st.cache_data = lambda *a, **kw: (lambda f: f)
        _fake_st.secrets = {}
        _fake_st.session_state = {}

        if "streamlit" not in sys.modules:
            sys.modules["streamlit"] = _fake_st
        if "streamlit_cookies_controller" not in sys.modules:
            _fake_cc = types.ModuleType("streamlit_cookies_controller")
            _fake_cc.CookieController = type("CookieController", (), {"__init__": lambda s: None})
            sys.modules["streamlit_cookies_controller"] = _fake_cc

        # Dynamic import of just the function we need
        import importlib
        app_mod = importlib.import_module("app")
        _compute_leg_projection = getattr(app_mod, "compute_leg_projection", None)
        if _compute_leg_projection is not None:
            _COMPUTE_AVAILABLE = True
            log.info("Loaded compute_leg_projection from app.py")
        else:
            log.warning("compute_leg_projection not found in app module")
    except Exception as exc:
        log.warning("Could not load app.py compute engine: %s", exc)
        _COMPUTE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback lightweight projection
# ---------------------------------------------------------------------------

def _simple_projection(
    player: str,
    market: str,
    current_line: float,
) -> Dict[str, Any]:
    """Minimal heuristic when the full projection engine is unavailable.

    Uses the market line as the baseline (efficient market assumption) and
    adds a small noise term so the system can still generate signals when
    the line deviates from its recent average.
    """
    # Without historical data, assume the line IS the fair value.
    # This means zero edge -- signals will only fire if we later overlay
    # additional information (CLV, sharp movements, etc.).
    proj = current_line
    p_over = 0.50
    sigma = current_line * 0.15  # rough vol assumption
    return {
        "proj": proj,
        "p_cal": p_over,
        "edge_pct": 0.0,
        "sigma": sigma,
        "confidence": 0.0,
        "kelly_stake": 0.0,
        "direction": "over",
        "model_version": "fallback_v0",
        "gate_ok": False,
        "errors": ["Using fallback simple projection -- full engine unavailable"],
    }


def _run_projection(
    player: str,
    market: str,
    line: float,
    event_name: str = "",
) -> Dict[str, Any]:
    """Run the full projection if available, else fall back to simple model."""
    _ensure_compute_engine()

    if _COMPUTE_AVAILABLE and _compute_leg_projection is not None:
        try:
            result = _compute_leg_projection(
                player_name=player,
                market_name=market,
                line=line,
                meta=None,
                n_games=12,
                key_teammate_out=False,
                bankroll=10000.0,
                frac_kelly=0.25,
                max_risk_frac=0.05,
            )
            if result and result.get("proj") is not None:
                proj = result["proj"]
                p_cal = result.get("p_cal") or 0.50
                ev_adj = result.get("ev_adj") or 0.0
                direction = result.get("side", "over")
                sigma = result.get("sigma")
                confidence = result.get("confidence_score") or result.get("sharpness_score", 0)
                if confidence and confidence > 1:
                    confidence = confidence / 100.0

                # Kelly fraction
                kelly = result.get("stake_frac", 0.0)

                # Edge percentage
                edge_pct = ev_adj * 100.0 if ev_adj else 0.0

                # Get model version
                mv = result.get("model_version", "app_v4.0")

                return {
                    "proj": proj,
                    "p_cal": p_cal,
                    "edge_pct": edge_pct,
                    "sigma": sigma,
                    "confidence": confidence,
                    "kelly_stake": kelly,
                    "direction": direction,
                    "model_version": mv,
                    "gate_ok": result.get("gate_ok", False),
                    "errors": result.get("errors", []),
                }
        except Exception as exc:
            log.warning("Full projection failed for %s %s: %s", player, market, exc)

    return _simple_projection(player, market, line)


# ---------------------------------------------------------------------------
# Active model version lookup
# ---------------------------------------------------------------------------

def _get_active_model_version(session) -> str:
    """Return the version string of the currently active model."""
    mv = (
        session.query(ModelVersion)
        .filter(ModelVersion.sport == "NBA", ModelVersion.is_active == True)
        .order_by(ModelVersion.created_at.desc())
        .first()
    )
    if mv:
        return mv.version
    return "app_v4.0"


# ===================================================================
# Worker class
# ===================================================================

class SignalWorker(BaseWorker):
    """Checks for new line_movements and generates signals every 10 minutes."""

    def __init__(self, **kwargs):
        super().__init__(
            name="signal_worker",
            interval_seconds=int(os.environ.get("SIGNAL_INTERVAL", "600")),
            max_retries=2,
            retry_delay=15.0,
            **kwargs,
        )

    def execute(self) -> Dict[str, Any]:
        now = _utcnow()
        signals_created = 0
        signals_with_edge = 0
        errors: List[str] = []

        with session_scope() as session:
            # Determine cutoff: lines since last successful run
            ws = (
                session.query(WorkerStatus)
                .filter(WorkerStatus.worker_name == self.name)
                .first()
            )
            if ws and ws.last_success:
                cutoff = ws.last_success
                # Ensure timezone aware
                if cutoff.tzinfo is None:
                    cutoff = cutoff.replace(tzinfo=timezone.utc)
            else:
                cutoff = now - timedelta(minutes=self.interval_seconds // 60 + 5)

            self.logger.info(
                "Checking for new line_movements since %s",
                cutoff.isoformat(),
            )

            # Get new line movements
            new_movements = (
                session.query(LineMovement)
                .filter(LineMovement.timestamp >= cutoff)
                .order_by(LineMovement.timestamp.desc())
                .all()
            )

            if not new_movements:
                self.logger.info("No new line movements found")
                return {"ok": True, "signals_created": 0, "reason": "no_new_data"}

            self.logger.info("Found %d new line movements", len(new_movements))

            # Only process events starting within next 24 hours
            event_cutoff = now + timedelta(hours=24)

            # Deduplicate: latest line per (player, market, book)
            latest_lines: Dict[tuple, LineMovement] = {}
            for lm in new_movements:
                key = (lm.player, lm.market, lm.book)
                if key not in latest_lines:
                    latest_lines[key] = lm

            # Further deduplicate to unique (player, market) -- pick best book
            player_markets: Dict[tuple, LineMovement] = {}
            for key, lm in latest_lines.items():
                pm_key = (lm.player, lm.market)
                if pm_key not in player_markets:
                    player_markets[pm_key] = lm

            active_version = _get_active_model_version(session)

            for (player, market), lm in player_markets.items():
                # Check if event is within 24h window
                if lm.event:
                    ev = (
                        session.query(Event)
                        .filter(Event.event_name == lm.event, Event.sport == "NBA")
                        .first()
                    )
                    if ev and ev.start_time:
                        st = ev.start_time
                        if st.tzinfo is None:
                            st = st.replace(tzinfo=timezone.utc)
                        if st > event_cutoff:
                            continue  # Event too far in the future

                # Run projection
                try:
                    result = _run_projection(
                        player=player,
                        market=market,
                        line=lm.line,
                        event_name=lm.event or "",
                    )
                except Exception as exc:
                    errors.append(f"{player}/{market}: {exc}")
                    continue

                proj = result.get("proj")
                p_cal = result.get("p_cal", 0.50)
                edge_pct = result.get("edge_pct", 0.0)
                confidence = result.get("confidence", 0.0)
                kelly = result.get("kelly_stake", 0.0)
                direction = result.get("direction", "over")
                mv = result.get("model_version", active_version)

                # Signal value = model projection
                signal_val = proj if proj is not None else lm.line

                sig = Signal(
                    sport="NBA",
                    event=lm.event or "",
                    market=market,
                    player=player,
                    signal_value=signal_val,
                    confidence=confidence,
                    generated_at=now,
                    model_version=mv,
                    direction=direction,
                    edge_pct=edge_pct,
                    kelly_stake=kelly,
                )
                session.add(sig)
                signals_created += 1

                if abs(edge_pct) >= 3.0 and result.get("gate_ok"):
                    signals_with_edge += 1

            self.logger.info(
                "Generated %d signals (%d with edge >= 3%%)",
                signals_created,
                signals_with_edge,
            )

        return {
            "ok": True,
            "signals_created": signals_created,
            "signals_with_edge": signals_with_edge,
            "line_movements_processed": len(new_movements),
            "errors": errors if errors else None,
        }


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    standalone_main(SignalWorker)
