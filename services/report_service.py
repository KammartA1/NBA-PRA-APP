"""
Report Service — read calibration data, system state, edge summaries, daily reports.

Reads from the quant_system database tables.
Returns plain dicts.
"""
import logging
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


def _get_session():
    try:
        from quant_system.db.schema import get_session
        return get_session()
    except Exception:
        return None


def _safe_close(session):
    if session:
        try:
            session.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_latest_report(report_type: str = "daily") -> dict:
    """
    Get the latest report of the given type.

    report_type: 'daily', 'weekly', 'calibration', 'system_state'

    Returns a dict with report data, or empty dict if none found.
    """
    if report_type == "calibration":
        return _get_latest_calibration_report()
    elif report_type == "system_state":
        return get_system_state()
    elif report_type == "daily":
        return _get_daily_report()
    elif report_type == "weekly":
        return _get_weekly_report()
    return {}


def get_report_history(report_type: str, days: int = 30) -> list[dict]:
    """
    Get historical reports of the given type over the past N days.
    """
    if report_type == "calibration":
        return get_calibration_data()
    elif report_type == "system_state":
        return _get_system_state_history(days)
    elif report_type in ("daily", "weekly"):
        return _get_pnl_history(days)
    return []


def get_calibration_data(sport: str = "NBA") -> list[dict]:
    """
    Get calibration bucket data from the calibration_log table.
    Returns latest calibration snapshot as list of bucket dicts.
    """
    session = _get_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import CalibrationLog

        # Get the most recent report_date
        latest = (
            session.query(CalibrationLog.report_date)
            .filter(CalibrationLog.sport == sport.lower())
            .order_by(CalibrationLog.report_date.desc())
            .first()
        )
        if latest is None:
            return []

        latest_date = latest[0]
        rows = (
            session.query(CalibrationLog)
            .filter(
                CalibrationLog.sport == sport.lower(),
                CalibrationLog.report_date == latest_date,
            )
            .order_by(CalibrationLog.prob_lower)
            .all()
        )

        return [
            {
                "bucket_label": r.bucket_label,
                "prob_lower": float(r.prob_lower),
                "prob_upper": float(r.prob_upper),
                "predicted_avg": float(r.predicted_avg),
                "actual_rate": float(r.actual_rate),
                "n_bets": int(r.n_bets),
                "calibration_error": float(r.calibration_error),
                "is_overconfident": bool(r.is_overconfident),
                "report_date": r.report_date.isoformat() if r.report_date else None,
            }
            for r in rows
        ]
    except Exception as exc:
        log.error("get_calibration_data failed: %s", exc)
        return []
    finally:
        _safe_close(session)


def get_system_state(sport: str = "NBA") -> dict:
    """
    Get the current system state (active, reduced, suspended, killed).

    Returns dict with: state, last_change, reason, clv, bankroll, drawdown
    """
    session = _get_session()
    if session is None:
        return {"state": "active", "reason": "no DB connection"}
    try:
        from quant_system.db.schema import SystemStateLog

        latest = (
            session.query(SystemStateLog)
            .filter(SystemStateLog.sport == sport.lower())
            .order_by(SystemStateLog.timestamp.desc())
            .first()
        )
        if latest is None:
            return {
                "state": "active",
                "last_change": None,
                "reason": "no state changes recorded",
                "clv": None,
                "bankroll": None,
                "drawdown": None,
            }

        return {
            "state": latest.new_state,
            "previous_state": latest.previous_state,
            "last_change": latest.timestamp.isoformat() if latest.timestamp else None,
            "reason": latest.reason or "",
            "clv": float(latest.clv_at_change) if latest.clv_at_change is not None else None,
            "bankroll": float(latest.bankroll_at_change) if latest.bankroll_at_change is not None else None,
            "drawdown": float(latest.drawdown_at_change) if latest.drawdown_at_change is not None else None,
        }
    except Exception as exc:
        log.error("get_system_state failed: %s", exc)
        return {"state": "unknown", "reason": str(exc)}
    finally:
        _safe_close(session)


def get_edge_summary() -> dict:
    """
    Compute an edge performance summary from recent bet signals.

    Returns dict with: avg_edge, total_signals, gated_pct, avg_sharpness,
    top_markets, edge_distribution
    """
    session = _get_session()
    if session is None:
        return {}
    try:
        from quant_system.db.schema import BetLog
        cutoff = datetime.utcnow() - timedelta(days=7)

        rows = (
            session.query(BetLog)
            .filter(
                BetLog.sport == "nba",
                BetLog.timestamp >= cutoff,
            )
            .all()
        )
        if not rows:
            return {
                "avg_edge": 0.0, "total_signals": 0,
                "gated_pct": 0.0, "avg_sharpness": 0.0,
                "top_markets": [], "edge_distribution": {},
            }

        # Compute stats
        edges = [float(r.edge or 0) for r in rows]
        sharpness_scores = [float(r.confidence_score or 0) * 100 for r in rows]
        positive_edge = [e for e in edges if e > 0]

        # Market breakdown
        from collections import Counter
        market_counts = Counter()
        market_edges = {}
        for r in rows:
            mkt = r.stat_type or "unknown"
            market_counts[mkt] += 1
            market_edges.setdefault(mkt, []).append(float(r.edge or 0))

        top_markets = []
        for mkt, count in market_counts.most_common(10):
            mkt_edges = market_edges[mkt]
            top_markets.append({
                "market": mkt,
                "count": count,
                "avg_edge": round(sum(mkt_edges) / len(mkt_edges) * 100, 2),
            })

        # Edge distribution buckets
        edge_dist = {"no_edge": 0, "lean": 0, "solid": 0, "strong": 0}
        for e in edges:
            if e <= 0:
                edge_dist["no_edge"] += 1
            elif e < 0.05:
                edge_dist["lean"] += 1
            elif e < 0.10:
                edge_dist["solid"] += 1
            else:
                edge_dist["strong"] += 1

        # Settled bet performance (if any)
        settled = [r for r in rows if r.status in ("won", "lost", "push")]
        settled_pnl = sum(float(r.pnl or 0) for r in settled)

        return {
            "avg_edge": round(sum(edges) / max(len(edges), 1) * 100, 2),
            "total_signals": len(rows),
            "positive_edge_count": len(positive_edge),
            "gated_pct": round(len(positive_edge) / max(len(rows), 1) * 100, 1),
            "avg_sharpness": round(sum(sharpness_scores) / max(len(sharpness_scores), 1), 1),
            "top_markets": top_markets,
            "edge_distribution": edge_dist,
            "settled_count": len(settled),
            "settled_pnl": round(settled_pnl, 2),
            "period_days": 7,
        }
    except Exception as exc:
        log.error("get_edge_summary failed: %s", exc)
        return {}
    finally:
        _safe_close(session)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_latest_calibration_report() -> dict:
    """Get the most recent calibration report as a summary dict."""
    buckets = get_calibration_data()
    if not buckets:
        return {"status": "no calibration data"}

    total_bets = sum(b["n_bets"] for b in buckets)
    avg_error = sum(abs(b["calibration_error"]) * b["n_bets"] for b in buckets) / max(total_bets, 1)
    overconfident_pct = sum(
        b["n_bets"] for b in buckets if b["is_overconfident"]
    ) / max(total_bets, 1) * 100

    return {
        "report_date": buckets[0].get("report_date") if buckets else None,
        "n_buckets": len(buckets),
        "total_bets": total_bets,
        "avg_calibration_error": round(avg_error, 4),
        "overconfident_pct": round(overconfident_pct, 1),
        "buckets": buckets,
    }


def _get_daily_report() -> dict:
    """Generate today's daily report from bet data."""
    session = _get_session()
    if session is None:
        return {}
    try:
        from quant_system.db.schema import BetLog
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        rows = (
            session.query(BetLog)
            .filter(
                BetLog.sport == "nba",
                BetLog.timestamp >= today_start,
            )
            .all()
        )

        signals = len(rows)
        bets_placed = sum(1 for r in rows if r.status != "signal")
        settled = [r for r in rows if r.status in ("won", "lost", "push")]
        pnl = sum(float(r.pnl or 0) for r in settled)
        wins = sum(1 for r in settled if r.status == "won")
        losses = sum(1 for r in settled if r.status == "lost")

        return {
            "report_type": "daily",
            "date": today_start.date().isoformat(),
            "total_signals": signals,
            "bets_placed": bets_placed,
            "settled": len(settled),
            "wins": wins,
            "losses": losses,
            "pnl": round(pnl, 2),
            "win_rate": round(wins / max(wins + losses, 1) * 100, 1),
        }
    except Exception as exc:
        log.error("_get_daily_report failed: %s", exc)
        return {}
    finally:
        _safe_close(session)


def _get_weekly_report() -> dict:
    """Generate this week's report."""
    session = _get_session()
    if session is None:
        return {}
    try:
        from quant_system.db.schema import BetLog
        week_start = datetime.utcnow() - timedelta(days=7)

        rows = (
            session.query(BetLog)
            .filter(
                BetLog.sport == "nba",
                BetLog.timestamp >= week_start,
            )
            .all()
        )

        settled = [r for r in rows if r.status in ("won", "lost", "push")]
        pnl = sum(float(r.pnl or 0) for r in settled)
        stakes = sum(float(r.stake or 0) for r in settled)
        wins = sum(1 for r in settled if r.status == "won")
        losses = sum(1 for r in settled if r.status == "lost")

        return {
            "report_type": "weekly",
            "start_date": week_start.date().isoformat(),
            "end_date": datetime.utcnow().date().isoformat(),
            "total_signals": len(rows),
            "bets_placed": sum(1 for r in rows if r.status != "signal"),
            "settled": len(settled),
            "wins": wins,
            "losses": losses,
            "pnl": round(pnl, 2),
            "roi_pct": round(pnl / max(stakes, 1) * 100, 2),
            "win_rate": round(wins / max(wins + losses, 1) * 100, 1),
        }
    except Exception as exc:
        log.error("_get_weekly_report failed: %s", exc)
        return {}
    finally:
        _safe_close(session)


def _get_system_state_history(days: int = 30) -> list[dict]:
    """Get system state change history."""
    session = _get_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import SystemStateLog
        cutoff = datetime.utcnow() - timedelta(days=days)

        rows = (
            session.query(SystemStateLog)
            .filter(
                SystemStateLog.sport == "nba",
                SystemStateLog.timestamp >= cutoff,
            )
            .order_by(SystemStateLog.timestamp.desc())
            .all()
        )
        return [
            {
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "previous_state": r.previous_state,
                "new_state": r.new_state,
                "reason": r.reason or "",
                "clv": float(r.clv_at_change) if r.clv_at_change is not None else None,
                "bankroll": float(r.bankroll_at_change) if r.bankroll_at_change is not None else None,
                "drawdown": float(r.drawdown_at_change) if r.drawdown_at_change is not None else None,
            }
            for r in rows
        ]
    except Exception as exc:
        log.error("_get_system_state_history failed: %s", exc)
        return []
    finally:
        _safe_close(session)


def _get_pnl_history(days: int = 30) -> list[dict]:
    """Get daily PnL breakdown for the last N days."""
    session = _get_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import BetLog
        from collections import defaultdict

        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = (
            session.query(BetLog)
            .filter(
                BetLog.sport == "nba",
                BetLog.status.in_(["won", "lost", "push"]),
                BetLog.timestamp >= cutoff,
            )
            .order_by(BetLog.timestamp.asc())
            .all()
        )

        daily = defaultdict(lambda: {"pnl": 0.0, "bets": 0, "wins": 0, "losses": 0})
        for r in rows:
            d = r.timestamp.date().isoformat() if r.timestamp else "unknown"
            daily[d]["pnl"] += float(r.pnl or 0)
            daily[d]["bets"] += 1
            if r.status == "won":
                daily[d]["wins"] += 1
            elif r.status == "lost":
                daily[d]["losses"] += 1

        return [
            {"date": d, **vals}
            for d, vals in sorted(daily.items())
        ]
    except Exception as exc:
        log.error("_get_pnl_history failed: %s", exc)
        return []
    finally:
        _safe_close(session)
