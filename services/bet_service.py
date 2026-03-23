"""
Bet Service — bet lifecycle management: place, settle, history, PnL, CLV.

All reads/writes go to the quant_system bets tables.
Returns plain dicts (not SQLAlchemy objects).
"""
import json
import logging
import uuid
from datetime import datetime, date, timedelta

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

def place_bet(bet_data: dict) -> int:
    """
    Create a new bet record from a projection result or manual entry.

    bet_data keys:
        player, market (stat_type), line, direction (over/under),
        stake, price_decimal, model_prob, market_prob, edge,
        kelly_fraction, model_projection, model_std,
        sharpness_score, book, notes

    Returns the row ID, or -1 on failure.
    """
    session = _get_session()
    if session is None:
        log.error("place_bet: no DB session")
        return -1
    try:
        from quant_system.db.schema import BetLog

        bet_id = f"bet_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        price_dec = float(bet_data.get("price_decimal", 1.909) or 1.909)
        # Convert decimal to American
        if price_dec >= 2.0:
            odds_american = int(round((price_dec - 1.0) * 100))
        else:
            odds_american = int(round(-100.0 / (price_dec - 1.0))) if price_dec > 1.0 else -110

        row = BetLog(
            bet_id=bet_id,
            sport="nba",
            timestamp=datetime.utcnow(),
            player=bet_data.get("player", ""),
            bet_type=str(bet_data.get("direction", "over")).lower(),
            stat_type=str(bet_data.get("market", bet_data.get("stat_type", ""))),
            line=float(bet_data.get("line", 0)),
            direction=str(bet_data.get("direction", "over")).lower(),
            model_prob=float(bet_data.get("model_prob", bet_data.get("p_cal", 0)) or 0),
            market_prob=float(bet_data.get("market_prob", bet_data.get("p_implied", 0)) or 0),
            edge=float(bet_data.get("edge", bet_data.get("ev_adj", 0)) or 0),
            stake=float(bet_data.get("stake", 0) or 0),
            kelly_fraction=float(bet_data.get("kelly_fraction", bet_data.get("stake_frac", 0)) or 0),
            odds_american=odds_american,
            odds_decimal=price_dec,
            model_projection=float(bet_data.get("model_projection", bet_data.get("proj", 0)) or 0),
            model_std=float(bet_data.get("model_std", bet_data.get("sigma", 0)) or 0),
            confidence_score=float(bet_data.get("sharpness_score", 0) or 0) / 100.0,
            engine_agreement=float(bet_data.get("engine_agreement", 0) or 0),
            status="pending",
            notes=bet_data.get("notes", ""),
        )
        session.add(row)
        session.commit()
        row_id = row.id
        log.info("Placed bet %s for %s %s %.1f (%s)",
                 bet_id, row.player, row.stat_type, row.line, row.direction)
        return row_id
    except Exception as exc:
        log.error("place_bet failed: %s", exc)
        try:
            session.rollback()
        except Exception:
            pass
        return -1
    finally:
        _safe_close(session)


def settle_bet(bet_id: int, actual_result: float) -> dict:
    """
    Settle a bet with the actual stat result.
    Computes CLV if closing line data is available.

    Returns dict with settlement details: status, pnl, clv_raw, etc.
    """
    session = _get_session()
    if session is None:
        return {"error": "no DB session"}
    try:
        from quant_system.db.schema import BetLog, CLVLog

        row = session.query(BetLog).filter(BetLog.id == bet_id).first()
        if row is None:
            return {"error": f"bet id {bet_id} not found"}

        actual = float(actual_result)
        line = float(row.line)
        direction = str(row.direction).lower()

        # Determine outcome
        if abs(actual - line) < 0.001:
            status = "push"
            pnl = 0.0
        elif (direction == "over" and actual > line) or (direction == "under" and actual < line):
            status = "won"
            pnl = float(row.stake) * (float(row.odds_decimal) - 1.0)
        else:
            status = "lost"
            pnl = -float(row.stake)

        row.status = status
        row.actual_result = actual
        row.settled_at = datetime.utcnow()
        row.pnl = pnl

        # CLV calculation if we have closing line info
        clv_data = {}
        closing_line = row.closing_line
        if closing_line is not None:
            try:
                cl = float(closing_line)
                bet_line = float(row.line)
                # CLV = how much the line moved in our favor after we bet
                if direction == "over":
                    clv_raw = cl - bet_line  # positive = line went up = good for Over
                else:
                    clv_raw = bet_line - cl  # positive = line went down = good for Under

                clv_entry = CLVLog(
                    bet_id=row.bet_id,
                    sport="nba",
                    opening_line=bet_line,
                    bet_line=bet_line,
                    closing_line=cl,
                    line_movement=cl - bet_line,
                    clv_raw=clv_raw,
                    clv_cents=round(clv_raw / max(bet_line, 0.5) * 100, 2),
                    beat_close=clv_raw > 0,
                    calculated_at=datetime.utcnow(),
                )
                session.add(clv_entry)
                clv_data = {
                    "clv_raw": clv_raw,
                    "clv_cents": round(clv_raw / max(bet_line, 0.5) * 100, 2),
                    "beat_close": clv_raw > 0,
                }
            except Exception as clv_exc:
                log.warning("CLV calc failed: %s", clv_exc)

        session.commit()

        return {
            "bet_id": bet_id,
            "status": status,
            "pnl": round(pnl, 2),
            "actual_result": actual,
            "line": line,
            "direction": direction,
            **clv_data,
        }
    except Exception as exc:
        log.error("settle_bet failed: %s", exc)
        try:
            session.rollback()
        except Exception:
            pass
        return {"error": str(exc)}
    finally:
        _safe_close(session)


def get_pending_bets() -> list[dict]:
    """Get all unsettled bets."""
    session = _get_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import BetLog
        rows = (
            session.query(BetLog)
            .filter(BetLog.status == "pending", BetLog.sport == "nba")
            .order_by(BetLog.timestamp.desc())
            .all()
        )
        return [_bet_to_dict(r) for r in rows]
    except Exception as exc:
        log.error("get_pending_bets failed: %s", exc)
        return []
    finally:
        _safe_close(session)


def get_settled_bets(days: int = 30) -> list[dict]:
    """Get settled bets within the last N days."""
    session = _get_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import BetLog
        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = (
            session.query(BetLog)
            .filter(
                BetLog.sport == "nba",
                BetLog.status.in_(["won", "lost", "push"]),
                BetLog.timestamp >= cutoff,
            )
            .order_by(BetLog.timestamp.desc())
            .all()
        )
        return [_bet_to_dict(r) for r in rows]
    except Exception as exc:
        log.error("get_settled_bets failed: %s", exc)
        return []
    finally:
        _safe_close(session)


def get_bet_history(filters: dict | None = None) -> list[dict]:
    """
    Get bet history with optional filters.

    Supported filter keys: player, market, status, min_date, max_date,
                           min_edge, direction, book
    """
    session = _get_session()
    if session is None:
        return []
    filters = filters or {}
    try:
        from quant_system.db.schema import BetLog
        query = session.query(BetLog).filter(BetLog.sport == "nba")

        if filters.get("player"):
            query = query.filter(BetLog.player.ilike(f"%{filters['player']}%"))
        if filters.get("market"):
            query = query.filter(BetLog.stat_type == filters["market"])
        if filters.get("status"):
            query = query.filter(BetLog.status == filters["status"])
        if filters.get("direction"):
            query = query.filter(BetLog.direction == filters["direction"])
        if filters.get("min_date"):
            query = query.filter(BetLog.timestamp >= filters["min_date"])
        if filters.get("max_date"):
            query = query.filter(BetLog.timestamp <= filters["max_date"])
        if filters.get("min_edge"):
            query = query.filter(BetLog.edge >= float(filters["min_edge"]))

        rows = query.order_by(BetLog.timestamp.desc()).limit(500).all()
        return [_bet_to_dict(r) for r in rows]
    except Exception as exc:
        log.error("get_bet_history failed: %s", exc)
        return []
    finally:
        _safe_close(session)


def get_pnl_summary(period: str = "daily") -> dict:
    """
    Compute PnL summary for the given period.
    period: 'daily', 'weekly', 'monthly', 'all'

    Returns: {total_pnl, n_bets, win_rate, avg_stake, roi_pct,
              wins, losses, pushes, best_day, worst_day}
    """
    session = _get_session()
    if session is None:
        return {}
    try:
        from quant_system.db.schema import BetLog
        import numpy as np

        if period == "daily":
            cutoff = datetime.utcnow() - timedelta(days=1)
        elif period == "weekly":
            cutoff = datetime.utcnow() - timedelta(days=7)
        elif period == "monthly":
            cutoff = datetime.utcnow() - timedelta(days=30)
        else:
            cutoff = datetime(2020, 1, 1)

        rows = (
            session.query(BetLog)
            .filter(
                BetLog.sport == "nba",
                BetLog.status.in_(["won", "lost", "push"]),
                BetLog.timestamp >= cutoff,
            )
            .all()
        )
        if not rows:
            return {
                "total_pnl": 0.0, "n_bets": 0, "win_rate": 0.0,
                "avg_stake": 0.0, "roi_pct": 0.0,
                "wins": 0, "losses": 0, "pushes": 0,
            }

        pnls = [float(r.pnl or 0) for r in rows]
        stakes = [float(r.stake or 0) for r in rows]
        wins = sum(1 for r in rows if r.status == "won")
        losses = sum(1 for r in rows if r.status == "lost")
        pushes = sum(1 for r in rows if r.status == "push")
        total_staked = sum(stakes)

        # Daily PnL breakdown
        daily_pnl = {}
        for r in rows:
            d = r.timestamp.date().isoformat() if r.timestamp else "unknown"
            daily_pnl[d] = daily_pnl.get(d, 0.0) + float(r.pnl or 0)

        return {
            "total_pnl": round(sum(pnls), 2),
            "n_bets": len(rows),
            "win_rate": round(wins / max(wins + losses, 1) * 100, 1),
            "avg_stake": round(total_staked / len(rows), 2) if rows else 0.0,
            "roi_pct": round(sum(pnls) / max(total_staked, 1) * 100, 2),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "best_day": max(daily_pnl.values()) if daily_pnl else 0.0,
            "worst_day": min(daily_pnl.values()) if daily_pnl else 0.0,
            "period": period,
        }
    except Exception as exc:
        log.error("get_pnl_summary failed: %s", exc)
        return {}
    finally:
        _safe_close(session)


def get_clv_summary(window: int = 100) -> dict:
    """
    CLV performance over the last N settled bets.

    Returns: {avg_clv_cents, beat_close_pct, n_bets, total_clv_raw}
    """
    session = _get_session()
    if session is None:
        return {}
    try:
        from quant_system.db.schema import CLVLog
        rows = (
            session.query(CLVLog)
            .filter(CLVLog.sport == "nba")
            .order_by(CLVLog.calculated_at.desc())
            .limit(window)
            .all()
        )
        if not rows:
            return {
                "avg_clv_cents": 0.0, "beat_close_pct": 0.0,
                "n_bets": 0, "total_clv_raw": 0.0,
            }

        clv_cents = [float(r.clv_cents or 0) for r in rows]
        beat_count = sum(1 for r in rows if r.beat_close)

        return {
            "avg_clv_cents": round(sum(clv_cents) / len(clv_cents), 2),
            "beat_close_pct": round(beat_count / len(rows) * 100, 1),
            "n_bets": len(rows),
            "total_clv_raw": round(sum(float(r.clv_raw or 0) for r in rows), 2),
        }
    except Exception as exc:
        log.error("get_clv_summary failed: %s", exc)
        return {}
    finally:
        _safe_close(session)


def log_bet_from_quant_engine(decision: dict) -> int:
    """
    Log a bet that was auto-generated by the quant engine.
    Maps quant engine decision dict fields to place_bet format.

    Returns row ID or -1 on failure.
    """
    bet_data = {
        "player": decision.get("player", ""),
        "market": decision.get("stat_type", decision.get("market", "")),
        "line": decision.get("line", 0),
        "direction": decision.get("direction", "over"),
        "stake": decision.get("stake", 0),
        "price_decimal": decision.get("odds_decimal", 1.909),
        "model_prob": decision.get("model_prob", 0),
        "market_prob": decision.get("market_prob", 0),
        "edge": decision.get("edge", 0),
        "kelly_fraction": decision.get("kelly_fraction", 0),
        "model_projection": decision.get("model_projection", decision.get("projection", 0)),
        "model_std": decision.get("model_std", 0),
        "sharpness_score": decision.get("confidence_score", 0) * 100,
        "engine_agreement": decision.get("engine_agreement", 0),
        "notes": f"quant_engine_auto: {decision.get('notes', '')}",
    }
    return place_bet(bet_data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bet_to_dict(row) -> dict:
    """Convert a BetLog SQLAlchemy row to a plain dict."""
    return {
        "id": row.id,
        "bet_id": row.bet_id,
        "sport": row.sport,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "player": row.player,
        "bet_type": row.bet_type,
        "stat_type": row.stat_type,
        "line": float(row.line),
        "direction": row.direction,
        "model_prob": float(row.model_prob),
        "market_prob": float(row.market_prob),
        "edge": float(row.edge),
        "stake": float(row.stake),
        "kelly_fraction": float(row.kelly_fraction),
        "odds_american": row.odds_american,
        "odds_decimal": float(row.odds_decimal),
        "model_projection": float(row.model_projection),
        "model_std": float(row.model_std),
        "confidence_score": float(row.confidence_score or 0),
        "engine_agreement": float(row.engine_agreement or 0),
        "status": row.status,
        "actual_result": float(row.actual_result) if row.actual_result is not None else None,
        "closing_line": float(row.closing_line) if row.closing_line is not None else None,
        "closing_odds": row.closing_odds,
        "settled_at": row.settled_at.isoformat() if row.settled_at else None,
        "pnl": float(row.pnl or 0),
        "model_version": row.model_version,
        "notes": row.notes,
    }
