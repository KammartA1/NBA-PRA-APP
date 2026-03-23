"""
Odds Service — line data access, line movement detection, platform-specific lines.

All reads/writes go through the database layer. Returns plain dicts.
"""
import logging
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


def _get_quant_session():
    try:
        from quant_system.db.schema import get_session
        return get_session()
    except Exception:
        return None


def _get_pp_session():
    try:
        from services.database import get_session
        return get_session()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_current_lines(sport: str = "NBA") -> list[dict]:
    """
    Read latest lines from the quant_system line_snapshots table.
    Returns lines captured within the last 6 hours.
    """
    session = _get_quant_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import LineSnapshot
        cutoff = datetime.utcnow() - timedelta(hours=6)
        rows = (
            session.query(LineSnapshot)
            .filter(
                LineSnapshot.sport == sport.lower(),
                LineSnapshot.captured_at >= cutoff,
            )
            .order_by(LineSnapshot.captured_at.desc())
            .all()
        )
        results = []
        seen = set()
        for r in rows:
            key = (r.player, r.stat_type, r.source)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "id": r.id,
                "player": r.player,
                "stat_type": r.stat_type,
                "source": r.source,
                "line": float(r.line),
                "odds_american": r.odds_american,
                "over_prob_implied": float(r.over_prob_implied) if r.over_prob_implied else None,
                "under_prob_implied": float(r.under_prob_implied) if r.under_prob_implied else None,
                "captured_at": r.captured_at.isoformat() if r.captured_at else None,
                "is_opening": bool(r.is_opening),
                "is_closing": bool(r.is_closing),
            })
        return results
    except Exception as exc:
        log.error("get_current_lines failed: %s", exc)
        return []
    finally:
        try:
            session.close()
        except Exception:
            pass


def get_lines_for_player(player_name: str, market: str | None = None) -> list[dict]:
    """
    Get all current lines for a specific player across all books.
    Optionally filter by market (stat_type).
    """
    session = _get_quant_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import LineSnapshot
        cutoff = datetime.utcnow() - timedelta(hours=12)
        query = (
            session.query(LineSnapshot)
            .filter(
                LineSnapshot.sport == "nba",
                LineSnapshot.player.ilike(f"%{player_name}%"),
                LineSnapshot.captured_at >= cutoff,
            )
        )
        if market:
            query = query.filter(LineSnapshot.stat_type == market.lower())
        rows = query.order_by(LineSnapshot.captured_at.desc()).all()
        results = []
        seen = set()
        for r in rows:
            key = (r.player, r.stat_type, r.source)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "player": r.player,
                "stat_type": r.stat_type,
                "source": r.source,
                "line": float(r.line),
                "odds_american": r.odds_american,
                "over_prob_implied": float(r.over_prob_implied) if r.over_prob_implied else None,
                "captured_at": r.captured_at.isoformat() if r.captured_at else None,
                "is_opening": bool(r.is_opening),
            })
        return results
    except Exception as exc:
        log.error("get_lines_for_player failed: %s", exc)
        return []
    finally:
        try:
            session.close()
        except Exception:
            pass


def get_line_history(
    player: str,
    market: str,
    book: str | None = None,
    hours: int = 24,
) -> list[dict]:
    """
    Historical line movements for a player/market within the time window.
    """
    session = _get_quant_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import LineSnapshot
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        query = (
            session.query(LineSnapshot)
            .filter(
                LineSnapshot.sport == "nba",
                LineSnapshot.player.ilike(f"%{player}%"),
                LineSnapshot.stat_type == market.lower(),
                LineSnapshot.captured_at >= cutoff,
            )
        )
        if book:
            query = query.filter(LineSnapshot.source == book.lower())
        rows = query.order_by(LineSnapshot.captured_at.asc()).all()
        return [
            {
                "player": r.player,
                "stat_type": r.stat_type,
                "source": r.source,
                "line": float(r.line),
                "odds_american": r.odds_american,
                "captured_at": r.captured_at.isoformat() if r.captured_at else None,
            }
            for r in rows
        ]
    except Exception as exc:
        log.error("get_line_history failed: %s", exc)
        return []
    finally:
        try:
            session.close()
        except Exception:
            pass


def get_best_available(player: str, market: str) -> dict:
    """
    Best price across all books for a player/market.
    Returns the line with the highest over implied probability (best for Over bets).
    """
    lines = get_lines_for_player(player, market)
    if not lines:
        return {}
    # Best for bettor = highest implied probability of the Over hitting
    best = max(
        lines,
        key=lambda x: float(x.get("over_prob_implied") or 0),
    )
    return best


def detect_sharp_movements(minutes: int = 30, threshold: float = 1.0) -> list[dict]:
    """
    Detect sharp line movements in the last `minutes` window.
    A sharp movement = line change >= threshold points within the window.
    """
    session = _get_quant_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import LineSnapshot
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        rows = (
            session.query(LineSnapshot)
            .filter(
                LineSnapshot.sport == "nba",
                LineSnapshot.captured_at >= cutoff,
            )
            .order_by(LineSnapshot.captured_at.asc())
            .all()
        )
        # Group by (player, stat_type, source) and find movements
        from collections import defaultdict
        groups = defaultdict(list)
        for r in rows:
            key = (r.player, r.stat_type, r.source)
            groups[key].append(r)

        movements = []
        for (player, stat, source), snapshots in groups.items():
            if len(snapshots) < 2:
                continue
            first_line = float(snapshots[0].line)
            last_line = float(snapshots[-1].line)
            delta = last_line - first_line
            if abs(delta) >= threshold:
                movements.append({
                    "player": player,
                    "stat_type": stat,
                    "source": source,
                    "opening_line": first_line,
                    "current_line": last_line,
                    "delta": round(delta, 2),
                    "direction": "UP" if delta > 0 else "DOWN",
                    "first_seen": snapshots[0].captured_at.isoformat(),
                    "last_seen": snapshots[-1].captured_at.isoformat(),
                    "n_snapshots": len(snapshots),
                })
        # Sort by magnitude of movement
        movements.sort(key=lambda x: abs(x["delta"]), reverse=True)
        return movements
    except Exception as exc:
        log.error("detect_sharp_movements failed: %s", exc)
        return []
    finally:
        try:
            session.close()
        except Exception:
            pass


def store_lines(lines: list[dict]):
    """
    Bulk insert line snapshots into quant_system.

    Each dict should have: player, stat_type, source, line.
    Optional: odds_american, over_prob_implied, under_prob_implied, is_opening
    """
    session = _get_quant_session()
    if session is None:
        log.warning("store_lines: no DB session available")
        return
    try:
        from quant_system.db.schema import LineSnapshot
        now = datetime.utcnow()
        objects = []
        for ln in lines:
            obj = LineSnapshot(
                sport="nba",
                player=ln.get("player", ""),
                stat_type=str(ln.get("stat_type", ln.get("market", ""))).lower(),
                source=str(ln.get("source", ln.get("book", "unknown"))).lower(),
                line=float(ln.get("line", 0)),
                odds_american=ln.get("odds_american"),
                over_prob_implied=ln.get("over_prob_implied"),
                under_prob_implied=ln.get("under_prob_implied"),
                captured_at=now,
                is_opening=bool(ln.get("is_opening", False)),
                is_closing=bool(ln.get("is_closing", False)),
            )
            objects.append(obj)
        session.bulk_save_objects(objects)
        session.commit()
        log.info("Stored %d line snapshots", len(objects))
    except Exception as exc:
        log.error("store_lines failed: %s", exc)
        try:
            session.rollback()
        except Exception:
            pass
    finally:
        try:
            session.close()
        except Exception:
            pass


def get_prizepicks_lines() -> list[dict]:
    """
    Get PrizePicks-specific lines from the nba_prizepicks_lines DB table.
    Returns only the latest batch (is_latest=True).
    """
    session = _get_pp_session()
    if session is None:
        return []
    try:
        from services.database import NbaPrizePicksLine
        rows = (
            session.query(NbaPrizePicksLine)
            .filter(NbaPrizePicksLine.is_latest == True)
            .order_by(NbaPrizePicksLine.fetched_at.desc())
            .all()
        )
        return [
            {
                "id": r.id,
                "player": r.player_name,
                "stat_type": r.stat_type,
                "line": float(r.line_score),
                "start_time": r.start_time,
                "odds_type": r.odds_type,
                "league": r.league,
                "fetched_at": r.fetched_at.isoformat() if r.fetched_at else None,
                "source": "prizepicks",
            }
            for r in rows
        ]
    except Exception as exc:
        log.error("get_prizepicks_lines failed: %s", exc)
        return []
    finally:
        try:
            session.close()
        except Exception:
            pass


def get_underdog_lines() -> list[dict]:
    """
    Get Underdog Fantasy lines from the quant_system line_snapshots table.
    """
    session = _get_quant_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import LineSnapshot
        cutoff = datetime.utcnow() - timedelta(hours=12)
        rows = (
            session.query(LineSnapshot)
            .filter(
                LineSnapshot.sport == "nba",
                LineSnapshot.source == "underdog",
                LineSnapshot.captured_at >= cutoff,
            )
            .order_by(LineSnapshot.captured_at.desc())
            .all()
        )
        seen = set()
        results = []
        for r in rows:
            key = (r.player, r.stat_type)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "player": r.player,
                "stat_type": r.stat_type,
                "line": float(r.line),
                "source": "underdog",
                "captured_at": r.captured_at.isoformat() if r.captured_at else None,
            })
        return results
    except Exception as exc:
        log.error("get_underdog_lines failed: %s", exc)
        return []
    finally:
        try:
            session.close()
        except Exception:
            pass


def get_sleeper_lines() -> list[dict]:
    """
    Get Sleeper lines from the quant_system line_snapshots table.
    """
    session = _get_quant_session()
    if session is None:
        return []
    try:
        from quant_system.db.schema import LineSnapshot
        cutoff = datetime.utcnow() - timedelta(hours=12)
        rows = (
            session.query(LineSnapshot)
            .filter(
                LineSnapshot.sport == "nba",
                LineSnapshot.source == "sleeper",
                LineSnapshot.captured_at >= cutoff,
            )
            .order_by(LineSnapshot.captured_at.desc())
            .all()
        )
        seen = set()
        results = []
        for r in rows:
            key = (r.player, r.stat_type)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "player": r.player,
                "stat_type": r.stat_type,
                "line": float(r.line),
                "source": "sleeper",
                "captured_at": r.captured_at.isoformat() if r.captured_at else None,
            })
        return results
    except Exception as exc:
        log.error("get_sleeper_lines failed: %s", exc)
        return []
    finally:
        try:
            session.close()
        except Exception:
            pass
