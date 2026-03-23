"""
UI Bridge — Single interface between backend systems and Streamlit UI
======================================================================
Every public method wraps backend calls in try/except and returns sensible
defaults on failure.  The UI should never see an unhandled exception from
this layer.

Usage::

    bridge = UIBridge()
    data = bridge.get_command_center_data()
"""

import copy
import logging
import statistics
from datetime import datetime, date, timedelta
from typing import Any, Optional

from sqlalchemy import func, desc

from database.models import (
    Bet,
    CalibrationSnapshot,
    EdgeReport,
    ModelVersion,
    Signal,
    SystemState,
    WorkerStatus,
)
from database.connection import session_scope

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sport constant
# ---------------------------------------------------------------------------

_SPORT = "NBA"

# ---------------------------------------------------------------------------
# Lazy imports — keep module-level fast; import backend services only when
# needed so a broken import in one service doesn't block the whole bridge.
# ---------------------------------------------------------------------------


def _import_bet_service():
    from services import bet_service
    return bet_service


def _import_event_service():
    from services import event_service
    return event_service


def _import_odds_service():
    from services import odds_service
    return odds_service


def _import_projection_service():
    from services import projection_service
    return projection_service


def _import_report_service():
    from services import report_service
    return report_service


def _import_scanner_service():
    from services import scanner_service
    return scanner_service


def _import_settings_service():
    from services import settings_service
    return settings_service


def _import_player_service():
    from services import player_service
    return player_service


def _import_connection():
    from database import connection
    return connection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_PNL = {
    "period": "daily",
    "total_pnl": 0,
    "total_staked": 0,
    "roi": 0,
    "win_rate": 0,
    "bets_settled": 0,
    "bets_won": 0,
    "bets_lost": 0,
    "bets_pushed": 0,
    "avg_stake": 0,
    "best_bet": 0,
    "worst_bet": 0,
}

_EMPTY_CLV = {
    "avg_clv": 0,
    "median_clv": 0,
    "pct_positive": 0,
    "n_bets": 0,
    "clv_by_market": {},
}


def _safe(fallback):
    """Decorator: catch all exceptions, log them, and return *fallback*."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                log.exception("UIBridge.%s failed — returning default", fn.__name__)
                # Return a deep copy if mutable to prevent shared-state bugs
                if isinstance(fallback, (dict, list)):
                    return copy.deepcopy(fallback)
                return fallback
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


def _today_start() -> datetime:
    """Return midnight today (UTC)."""
    t = date.today()
    return datetime(t.year, t.month, t.day)


def _compute_clv(bet_line, closing_line, direction: str) -> float:
    """Compute CLV for a single bet given line values and direction."""
    if bet_line is None or closing_line is None:
        return 0.0
    d = direction.upper() if direction else ""
    if d == "OVER":
        return bet_line - closing_line
    elif d == "UNDER":
        return closing_line - bet_line
    else:
        return closing_line - bet_line


# =========================================================================
# UIBridge
# =========================================================================

class UIBridge:
    """Single entry point for all UI data needs."""

    # -- System state helpers ----------------------------------------------

    def _get_system_state_row(self) -> Optional[dict]:
        """Fetch the latest SystemState row, or None."""
        try:
            with session_scope() as session:
                row = (
                    session.query(SystemState)
                    .filter(SystemState.sport == _SPORT)
                    .order_by(desc(SystemState.changed_at))
                    .first()
                )
                return row.to_dict() if row else None
        except Exception:
            log.exception("Failed to read SystemState")
            return None

    def _get_active_model_version(self) -> str:
        """Return the version string of the currently active model."""
        try:
            with session_scope() as session:
                row = (
                    session.query(ModelVersion)
                    .filter(
                        ModelVersion.sport == _SPORT,
                        ModelVersion.is_active == True,  # noqa: E712
                    )
                    .order_by(desc(ModelVersion.created_at))
                    .first()
                )
                return row.version if row else "unknown"
        except Exception:
            log.exception("Failed to read ModelVersion")
            return "unknown"

    def _get_worker_statuses(self) -> list[dict]:
        """Return all worker status rows normalised to the UI contract."""
        try:
            with session_scope() as session:
                rows = session.query(WorkerStatus).all()
                return [
                    {
                        "name": r.worker_name or "",
                        "status": r.status or "unknown",
                        "last_run": r.last_run.isoformat() if r.last_run else None,
                        "last_success": r.last_success.isoformat() if r.last_success else None,
                        "last_error": r.last_error or "",
                    }
                    for r in rows
                ]
        except Exception:
            log.exception("Failed to read WorkerStatus")
            return []

    def _get_kill_switches(self) -> list[dict]:
        """
        Kill switches are derived from SystemState rows whose state != ACTIVE.
        Returns list of {name, status, reason}.
        """
        try:
            with session_scope() as session:
                rows = (
                    session.query(SystemState)
                    .filter(SystemState.sport == _SPORT)
                    .order_by(desc(SystemState.changed_at))
                    .all()
                )
                if not rows:
                    return [{"name": "system", "status": "UNKNOWN", "reason": "No state record found"}]

                switches = []
                for r in rows:
                    if r.state != "ACTIVE":
                        switches.append({
                            "name": f"kill_switch_{r.id}",
                            "status": r.state,
                            "reason": r.reason or "No reason recorded",
                        })
                return switches
        except Exception:
            log.exception("Failed to read kill switches")
            return [{"name": "system", "status": "UNKNOWN", "reason": "Failed to query state"}]

    def _get_todays_signals(self) -> list[dict]:
        """Query Signal table for signals generated today."""
        try:
            today = _today_start()
            with session_scope() as session:
                rows = (
                    session.query(Signal)
                    .filter(
                        Signal.sport == _SPORT,
                        Signal.generated_at >= today,
                    )
                    .order_by(desc(Signal.edge_pct))
                    .all()
                )
                return [
                    {
                        "player": r.player or "",
                        "event": r.event or "",
                        "market": r.market or "",
                        "edge_pct": r.edge_pct or 0.0,
                        "rec_stake": r.kelly_stake or 0.0,
                        "action": r.direction or "",
                    }
                    for r in rows
                ]
        except Exception:
            log.exception("Failed to read today's signals")
            return []

    def _get_exposure(self) -> dict:
        """Sum stakes from today's pending + settled bets."""
        try:
            bet_svc = _import_bet_service()
            pending = bet_svc.get_pending_bets()
            settled = bet_svc.get_settled_bets(days=1)
            all_bets = (pending or []) + (settled or [])

            total_wagered = sum(b.get("stake", 0) or 0 for b in all_bets)

            by_market: dict[str, float] = {}
            by_confidence: dict[str, float] = {}
            for b in all_bets:
                # Group by market
                mkt = b.get("market", "unknown") or "unknown"
                by_market[mkt] = by_market.get(mkt, 0) + (b.get("stake", 0) or 0)

                # Group by confidence tier
                conf = b.get("confidence_score")
                if conf is not None:
                    if conf >= 0.7:
                        bucket = "high"
                    elif conf >= 0.4:
                        bucket = "medium"
                    else:
                        bucket = "low"
                else:
                    bucket = "unknown"
                by_confidence[bucket] = by_confidence.get(bucket, 0) + (b.get("stake", 0) or 0)

            return {
                "total_wagered": round(total_wagered, 2),
                "by_market": by_market,
                "by_confidence": by_confidence,
            }
        except Exception:
            log.exception("Failed to compute exposure")
            return {"total_wagered": 0, "by_market": {}, "by_confidence": {}}

    def _get_edge_status(self) -> bool:
        """True if latest edge report shows positive edge or system is ACTIVE."""
        try:
            report_svc = _import_report_service()
            report = report_svc.get_latest_report("daily")
            if report:
                data = report.get("data", {})
                if isinstance(data, dict):
                    edge = data.get("edge", data.get("avg_edge", data.get("total_edge")))
                    if edge is not None:
                        return float(edge) > 0
            # Fallback: system state is ACTIVE
            state_row = self._get_system_state_row()
            if state_row:
                return state_row.get("state") == "ACTIVE"
            return False
        except Exception:
            log.exception("Failed to determine edge status")
            return False

    def _get_data_quality_score(self) -> float:
        """
        Heuristic data quality score (0-100).
        Checks worker health, recent signal freshness, and DB connectivity.
        """
        score = 0.0
        try:
            # DB connectivity: 30 points
            conn = _import_connection()
            hc = conn.health_check()
            if hc.get("ok"):
                score += 30.0

            # Workers healthy: 40 points
            workers = self._get_worker_statuses()
            if workers:
                healthy = sum(
                    1 for w in workers
                    if w.get("status") in ("idle", "running", "success")
                )
                score += 40.0 * (healthy / len(workers))

            # Fresh signals today: 30 points
            signals = self._get_todays_signals()
            if signals:
                score += 30.0

        except Exception:
            log.exception("Failed to compute data quality score")
        return round(min(score, 100.0), 1)

    def _get_simulation_status(self) -> str:
        """Check latest report for simulation pass/fail."""
        try:
            report_svc = _import_report_service()
            report = report_svc.get_latest_report("simulation")
            if report:
                data = report.get("data", {})
                if isinstance(data, dict):
                    status = data.get("status", data.get("result", "UNKNOWN"))
                    return str(status).upper()
            return "UNKNOWN"
        except Exception:
            log.exception("Failed to read simulation status")
            return "UNKNOWN"

    def _get_calibration_data(self) -> list[dict]:
        """Query CalibrationSnapshot for the most recent snapshot date."""
        try:
            with session_scope() as session:
                latest_date = (
                    session.query(func.max(CalibrationSnapshot.snapshot_date))
                    .filter(CalibrationSnapshot.sport == _SPORT)
                    .scalar()
                )
                if not latest_date:
                    return []

                rows = (
                    session.query(CalibrationSnapshot)
                    .filter(
                        CalibrationSnapshot.sport == _SPORT,
                        CalibrationSnapshot.snapshot_date == latest_date,
                    )
                    .order_by(CalibrationSnapshot.prob_lower)
                    .all()
                )
                return [
                    {
                        "bucket": r.bucket_label,
                        "predicted": round(r.predicted_avg, 4),
                        "actual": round(r.actual_rate, 4),
                        "count": r.n_bets,
                    }
                    for r in rows
                ]
        except Exception:
            log.exception("Failed to read calibration data")
            return []

    # ==================================================================
    # Public API
    # ==================================================================

    @_safe({
        "bankroll": 0,
        "today_pnl": 0,
        "today_pnl_delta": 0,
        "edge_status": False,
        "clv_50bet": 0.0,
        "kill_switches": [],
        "worker_statuses": [],
        "todays_signals": [],
        "exposure": {"total_wagered": 0, "by_market": {}, "by_confidence": {}},
        "system_state": "UNKNOWN",
        "model_version": "unknown",
        "data_quality_score": 0.0,
        "simulation_status": "UNKNOWN",
    })
    def get_command_center_data(self) -> dict:
        """
        Returns everything the command-center dashboard needs in one call.

        Keys:
            bankroll, today_pnl, today_pnl_delta, edge_status (bool),
            clv_50bet (float), kill_switches (list of {name, status, reason}),
            worker_statuses (list of {name, status, last_run, last_success, last_error}),
            todays_signals (list of dicts with player, event, market, edge_pct,
                            rec_stake, action),
            exposure (dict with total_wagered, by_market, by_confidence),
            system_state (str: ACTIVE/REDUCED/SUSPENDED/KILLED),
            model_version (str), data_quality_score (float 0-100),
            simulation_status (str: PASS/FAIL/UNKNOWN)
        """
        settings_svc = _import_settings_service()
        bet_svc = _import_bet_service()

        # Bankroll
        try:
            bankroll = float(settings_svc.get_bankroll("default"))
        except Exception:
            try:
                bankroll = float(settings_svc.get_setting("default", "bankroll", 1000.0))
            except Exception:
                bankroll = 1000.0

        # Today's P&L
        try:
            pnl = bet_svc.get_pnl_summary("daily")
            today_pnl = pnl.get("total_pnl", 0) if pnl else 0
        except Exception:
            pnl = dict(_EMPTY_PNL)
            today_pnl = 0

        # P&L delta (today vs yesterday): approximate with roi
        today_pnl_delta = pnl.get("roi", 0) if pnl else 0

        # CLV 50-bet window
        try:
            clv = bet_svc.get_clv_summary(window=50)
            clv_50bet = clv.get("avg_clv", 0.0) if clv else 0.0
        except Exception:
            clv_50bet = 0.0

        # System state
        state_row = self._get_system_state_row()
        system_state = state_row.get("state", "UNKNOWN") if state_row else "UNKNOWN"

        return {
            "bankroll": bankroll,
            "today_pnl": today_pnl,
            "today_pnl_delta": today_pnl_delta,
            "edge_status": self._get_edge_status(),
            "clv_50bet": clv_50bet,
            "kill_switches": self._get_kill_switches(),
            "worker_statuses": self._get_worker_statuses(),
            "todays_signals": self._get_todays_signals(),
            "exposure": self._get_exposure(),
            "system_state": system_state,
            "model_version": self._get_active_model_version(),
            "data_quality_score": self._get_data_quality_score(),
            "simulation_status": self._get_simulation_status(),
        }

    @_safe([])
    def get_signals_data(
        self,
        sport_filter: str = "all",
        confidence_filter: str = "all",
        sort_by: str = "edge_pct",
        search: str = "",
    ) -> list[dict]:
        """
        Returns list of signal dicts with: timestamp, player, event, market,
        line, model_prob, market_prob, edge_pct, clv_expected, confidence,
        rec_stake, direction.

        Supports filtering by sport, confidence tier, sorting, and text search.
        """
        with session_scope() as session:
            q = session.query(Signal).filter(Signal.sport == _SPORT)

            if sport_filter and sport_filter != "all":
                q = q.filter(Signal.sport == sport_filter.upper())

            if confidence_filter and confidence_filter != "all":
                thresholds = {
                    "high": (0.7, 1.0),
                    "medium": (0.4, 0.7),
                    "low": (0.0, 0.4),
                }
                lo, hi = thresholds.get(confidence_filter.lower(), (0.0, 1.0))
                q = q.filter(Signal.confidence >= lo, Signal.confidence < hi)

            # Sorting
            sort_map = {
                "edge_pct": Signal.edge_pct.desc(),
                "confidence": Signal.confidence.desc(),
                "timestamp": Signal.generated_at.desc(),
                "player": Signal.player.asc(),
            }
            order = sort_map.get(sort_by, Signal.edge_pct.desc())
            q = q.order_by(order)

            rows = q.limit(500).all()

        results = []
        for r in rows:
            player_name = r.player or ""
            if search and search.lower() not in player_name.lower():
                continue
            results.append({
                "timestamp": r.generated_at.isoformat() if r.generated_at else "",
                "player": player_name,
                "event": r.event or "",
                "market": r.market or "",
                "line": r.signal_value or 0.0,
                "model_prob": 0.0,
                "market_prob": 0.0,
                "edge_pct": r.edge_pct or 0.0,
                "clv_expected": 0.0,
                "confidence": r.confidence or 0.0,
                "rec_stake": r.kelly_stake or 0.0,
                "direction": r.direction or "",
            })

        return results

    @_safe({
        "clv_history": [],
        "roi_history": [],
        "drawdown_history": [],
        "calibration_data": [],
        "summary": {
            "total_bets": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "avg_clv": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        },
    })
    def get_performance_data(self, days: int = 90) -> dict:
        """
        Returns cumulative performance metrics computed from bet history.

        Keys:
            clv_history: list of {bet_num, cumulative_clv}
            roi_history: list of {bet_num, cumulative_roi}
            drawdown_history: list of {bet_num, drawdown_pct}
            calibration_data: list of {bucket, predicted, actual, count}
            summary: dict with total_bets, win_rate, roi, avg_clv, sharpe,
                     max_drawdown
        """
        bet_svc = _import_bet_service()
        bets = bet_svc.get_bet_history({"days": days, "limit": 5000})

        if not bets:
            return {
                "clv_history": [],
                "roi_history": [],
                "drawdown_history": [],
                "calibration_data": self._get_calibration_data(),
                "summary": {
                    "total_bets": 0,
                    "win_rate": 0.0,
                    "roi": 0.0,
                    "avg_clv": 0.0,
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                },
            }

        # Sort by timestamp ascending for cumulative computation
        bets.sort(key=lambda b: b.get("timestamp") or "")

        # Filter to settled bets for cumulative stats
        settled = [b for b in bets if b.get("status") in ("won", "lost", "push")]

        # --- Cumulative CLV ---
        clv_history = []
        cum_clv = 0.0
        for i, b in enumerate(settled, 1):
            cl = b.get("closing_line")
            bl = b.get("bet_line")
            if cl is not None and bl is not None:
                cum_clv += _compute_clv(bl, cl, b.get("direction", ""))
            clv_history.append({"bet_num": i, "cumulative_clv": round(cum_clv, 4)})

        # --- Cumulative ROI ---
        roi_history = []
        cum_pnl = 0.0
        cum_staked = 0.0
        for i, b in enumerate(settled, 1):
            pnl = b.get("pnl") or b.get("profit") or 0
            stake = b.get("stake") or 0
            cum_pnl += pnl
            cum_staked += stake
            roi = (cum_pnl / cum_staked) if cum_staked > 0 else 0.0
            roi_history.append({"bet_num": i, "cumulative_roi": round(roi, 4)})

        # --- Drawdown ---
        drawdown_history = []
        peak = 0.0
        running = 0.0
        max_dd = 0.0
        for i, b in enumerate(settled, 1):
            pnl = b.get("pnl") or b.get("profit") or 0
            running += pnl
            if running > peak:
                peak = running
            dd = ((peak - running) / peak) if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            drawdown_history.append({"bet_num": i, "drawdown_pct": round(dd, 4)})

        # --- Summary ---
        total_bets = len(settled)
        wins = sum(1 for b in settled if b.get("status") == "won")
        win_rate = (wins / total_bets) if total_bets > 0 else 0.0

        total_staked = sum((b.get("stake") or 0) for b in settled)
        total_pnl = sum((b.get("pnl") or b.get("profit") or 0) for b in settled)
        roi = (total_pnl / total_staked) if total_staked > 0 else 0.0

        # Average CLV
        clv_vals = []
        for b in settled:
            cl = b.get("closing_line")
            bl = b.get("bet_line")
            if cl is not None and bl is not None:
                clv_vals.append(_compute_clv(bl, cl, b.get("direction", "")))
        avg_clv = (sum(clv_vals) / len(clv_vals)) if clv_vals else 0.0

        # Sharpe ratio (per-bet returns approximation)
        pnl_series = [(b.get("pnl") or b.get("profit") or 0) for b in settled]
        if len(pnl_series) >= 2:
            mean_ret = statistics.mean(pnl_series)
            std_ret = statistics.stdev(pnl_series)
            sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "clv_history": clv_history,
            "roi_history": roi_history,
            "drawdown_history": drawdown_history,
            "calibration_data": self._get_calibration_data(),
            "summary": {
                "total_bets": total_bets,
                "win_rate": round(win_rate, 4),
                "roi": round(roi, 4),
                "avg_clv": round(avg_clv, 4),
                "sharpe": round(sharpe, 4),
                "max_drawdown": round(max_dd, 4),
            },
        }

    @_safe([])
    def get_history_data(self, filters: Optional[dict] = None) -> list[dict]:
        """
        Returns list of bet dicts with: date, player, event, market, bet_line,
        close_line, clv, model_prob, result (W/L/P), pnl, stake.
        """
        bet_svc = _import_bet_service()
        raw = bet_svc.get_bet_history(filters or {})

        results = []
        for b in (raw or []):
            # Compute CLV
            cl = b.get("closing_line")
            bl = b.get("bet_line")
            if cl is not None and bl is not None:
                clv = _compute_clv(bl, cl, b.get("direction", ""))
            else:
                clv = None

            # Map status to W/L/P
            status = b.get("status", "")
            result_map = {"won": "W", "lost": "L", "push": "P"}
            result = result_map.get(status, status.upper()[:1] if status else "")

            ts = b.get("timestamp") or b.get("settled_at") or ""
            if isinstance(ts, str) and "T" in ts:
                date_str = ts.split("T")[0]
            elif isinstance(ts, datetime):
                date_str = ts.strftime("%Y-%m-%d")
            else:
                date_str = str(ts)[:10] if ts else ""

            results.append({
                "date": date_str,
                "player": b.get("player", ""),
                "event": b.get("event", ""),
                "market": b.get("market", ""),
                "bet_line": b.get("bet_line"),
                "close_line": cl,
                "clv": round(clv, 4) if clv is not None else None,
                "model_prob": b.get("predicted_prob"),
                "result": result,
                "pnl": b.get("pnl") or b.get("profit") or 0,
                "stake": b.get("stake") or 0,
            })

        return results

    @_safe({
        "worker_statuses": [],
        "kill_switches": [],
        "model_version": "unknown",
        "data_quality_score": 0.0,
        "db_health": {"ok": False, "error": "unknown"},
    })
    def get_system_health(self) -> dict:
        """
        Returns system health overview.

        Keys:
            worker_statuses, kill_switches, model_version,
            data_quality_score, db_health
        """
        # DB health
        try:
            conn = _import_connection()
            db_health = conn.health_check()
        except Exception:
            db_health = {"ok": False, "error": "Could not reach database"}

        return {
            "worker_statuses": self._get_worker_statuses(),
            "kill_switches": self._get_kill_switches(),
            "model_version": self._get_active_model_version(),
            "data_quality_score": self._get_data_quality_score(),
            "db_health": db_health,
        }
