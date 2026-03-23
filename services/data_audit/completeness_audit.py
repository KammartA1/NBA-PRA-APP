"""
services/data_audit/completeness_audit.py
==========================================
Validates data completeness across the system.

Checks:
  - What % of events have full line movement history?
  - What % of bets have complete data (signal, bet, close)?
  - What data is systematically missing? (certain books, market types)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Set

from sqlalchemy import func as sa_func

from quant_system.db.schema import get_engine, get_session, BetLog, CLVLog, LineSnapshot
from services.clv_system.models import (
    CLVLineMovement,
    CLVBetSnapshot,
    CLVClosingLine,
    Base,
)

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CompletenessAuditor:
    """Validates data completeness across all system dimensions.

    Produces a detailed findings dict with:
      - event_coverage_pct: % of events with full line movement history
      - bet_completeness_pct: % of bets with complete data chain
      - missing_opening_pct: % of bets missing opening lines
      - missing_closing_pct: % of settled bets missing closing lines
      - missing_snapshot_pct: % of bets missing bet-time snapshots
      - systematic_gaps: data that is systematically missing
      - issues: list of human-readable issue descriptions
      - score: 0-100 composite completeness score
    """

    def __init__(self, sport: str = "NBA", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    def audit(self) -> Dict[str, Any]:
        """Run all completeness checks and return consolidated results."""
        bet_complete = self._check_bet_completeness()
        event_coverage = self._check_event_coverage()
        systematic = self._check_systematic_gaps()
        line_history = self._check_line_history_depth()

        all_issues: List[str] = []
        all_issues.extend(bet_complete.get("issues", []))
        all_issues.extend(event_coverage.get("issues", []))
        all_issues.extend(systematic.get("issues", []))
        all_issues.extend(line_history.get("issues", []))

        # Composite score
        bet_pct = bet_complete.get("completeness_pct", 100.0)
        event_pct = event_coverage.get("coverage_pct", 100.0)
        gap_penalty = min(len(systematic.get("gaps", [])) * 5, 25)
        history_pct = line_history.get("adequate_pct", 100.0)

        score = max(0.0, min(100.0,
            bet_pct * 0.35
            + event_pct * 0.25
            + (100 - gap_penalty) * 0.15
            + history_pct * 0.25
        ))

        return {
            "score": round(score, 1),
            "bet_completeness_pct": round(bet_pct, 1),
            "total_bets": bet_complete.get("total_bets", 0),
            "settled_bets": bet_complete.get("settled_bets", 0),
            "complete_bets": bet_complete.get("complete_bets", 0),
            "missing_opening_pct": round(bet_complete.get("missing_opening_pct", 0.0), 1),
            "missing_closing_pct": round(bet_complete.get("missing_closing_pct", 0.0), 1),
            "missing_snapshot_pct": round(bet_complete.get("missing_snapshot_pct", 0.0), 1),
            "missing_clv_pct": round(bet_complete.get("missing_clv_pct", 0.0), 1),
            "event_coverage_pct": round(event_pct, 1),
            "events_with_lines": event_coverage.get("with_lines", 0),
            "events_without_lines": event_coverage.get("without_lines", 0),
            "line_history_adequate_pct": round(history_pct, 1),
            "systematic_gaps": systematic.get("gaps", []),
            "book_coverage": systematic.get("book_coverage", {}),
            "market_coverage": systematic.get("market_coverage", {}),
            "issues": all_issues,
        }

    # ── Bet completeness ──────────────────────────────────────────

    def _check_bet_completeness(self) -> Dict[str, Any]:
        """Check that every bet has the complete data chain.

        A complete bet needs:
          1. Opening line (first observed line for this player/market)
          2. Bet-time snapshot (CLVBetSnapshot)
          3. Closing line (CLVClosingLine or BetLog.closing_line)
          4. CLV calculation (CLVLog entry)
        """
        session = self._session()
        try:
            issues = []

            bets = (
                session.query(BetLog)
                .filter(BetLog.sport == self.sport.lower())
                .all()
            )

            total = len(bets)
            if total == 0:
                return {
                    "completeness_pct": 100.0,
                    "total_bets": 0,
                    "settled_bets": 0,
                    "complete_bets": 0,
                    "missing_opening_pct": 0.0,
                    "missing_closing_pct": 0.0,
                    "missing_snapshot_pct": 0.0,
                    "missing_clv_pct": 0.0,
                    "issues": ["No bets in the system"],
                }

            settled = [b for b in bets if b.status in ("won", "lost", "push")]
            settled_count = len(settled)

            missing_opening = 0
            missing_closing = 0
            missing_snapshot = 0
            missing_clv = 0
            complete_count = 0

            for bet in bets:
                is_complete = True

                # Check opening line
                has_opening = (
                    session.query(sa_func.count(CLVLineMovement.id))
                    .filter(
                        CLVLineMovement.sport == self.sport,
                        CLVLineMovement.player == bet.player,
                        CLVLineMovement.market_type == bet.stat_type,
                        CLVLineMovement.is_opening == True,  # noqa: E712
                    )
                    .scalar()
                ) or 0

                if has_opening == 0:
                    missing_opening += 1
                    is_complete = False

                # Check bet-time snapshot
                has_snapshot = (
                    session.query(sa_func.count(CLVBetSnapshot.id))
                    .filter(CLVBetSnapshot.bet_id == bet.bet_id)
                    .scalar()
                ) or 0

                if has_snapshot == 0:
                    missing_snapshot += 1
                    is_complete = False

                # Check closing line (only for settled bets)
                if bet.status in ("won", "lost", "push"):
                    has_closing = bet.closing_line is not None
                    if not has_closing:
                        # Also check CLVClosingLine table
                        has_closing_record = (
                            session.query(sa_func.count(CLVClosingLine.id))
                            .filter(
                                CLVClosingLine.sport == self.sport,
                                CLVClosingLine.player == bet.player,
                                CLVClosingLine.market_type == bet.stat_type,
                            )
                            .scalar()
                        ) or 0
                        has_closing = has_closing_record > 0

                    if not has_closing:
                        missing_closing += 1
                        is_complete = False

                    # Check CLV calculation
                    has_clv = (
                        session.query(sa_func.count(CLVLog.id))
                        .filter(CLVLog.bet_id == bet.bet_id)
                        .scalar()
                    ) or 0

                    if has_clv == 0:
                        missing_clv += 1
                        is_complete = False

                if is_complete:
                    complete_count += 1

            completeness_pct = (complete_count / total * 100) if total > 0 else 100.0
            missing_opening_pct = (missing_opening / total * 100) if total > 0 else 0.0
            missing_closing_pct = (
                (missing_closing / settled_count * 100) if settled_count > 0 else 0.0
            )
            missing_snapshot_pct = (missing_snapshot / total * 100) if total > 0 else 0.0
            missing_clv_pct = (
                (missing_clv / settled_count * 100) if settled_count > 0 else 0.0
            )

            if missing_opening > 0:
                issues.append(
                    f"MISSING_OPENING_LINES: {missing_opening}/{total} bets "
                    f"({missing_opening_pct:.1f}%) have no opening line recorded"
                )
            if missing_snapshot > 0:
                issues.append(
                    f"MISSING_SNAPSHOTS: {missing_snapshot}/{total} bets "
                    f"({missing_snapshot_pct:.1f}%) have no bet-time market snapshot"
                )
            if missing_closing > 0:
                issues.append(
                    f"MISSING_CLOSING_LINES: {missing_closing}/{settled_count} settled bets "
                    f"({missing_closing_pct:.1f}%) have no closing line"
                )
            if missing_clv > 0:
                issues.append(
                    f"MISSING_CLV: {missing_clv}/{settled_count} settled bets "
                    f"({missing_clv_pct:.1f}%) have no CLV calculation"
                )

            return {
                "completeness_pct": completeness_pct,
                "total_bets": total,
                "settled_bets": settled_count,
                "complete_bets": complete_count,
                "missing_opening_pct": missing_opening_pct,
                "missing_closing_pct": missing_closing_pct,
                "missing_snapshot_pct": missing_snapshot_pct,
                "missing_clv_pct": missing_clv_pct,
                "issues": issues,
            }
        finally:
            session.close()

    # ── Event coverage ────────────────────────────────────────────

    def _check_event_coverage(self) -> Dict[str, Any]:
        """Check what % of events have line movement data.

        Events without line data cannot be used for CLV analysis.
        """
        session = self._session()
        try:
            issues = []

            # Get all unique event_ids from closing lines (events we should have data for)
            events_with_closings = set(
                row[0] for row in
                session.query(CLVClosingLine.event_id)
                .filter(CLVClosingLine.sport == self.sport)
                .distinct()
                .all()
            )

            # Get all unique event_ids that have line movement data
            events_with_movements = set(
                row[0] for row in
                session.query(CLVLineMovement.event_id)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.event_id.isnot(None),
                )
                .distinct()
                .all()
            )

            # Events with closings but no line movements = data gap
            all_events = events_with_closings | events_with_movements
            if not all_events:
                # Fall back: count bets with unique event references
                bet_count = (
                    session.query(sa_func.count(BetLog.id))
                    .filter(BetLog.sport == self.sport.lower())
                    .scalar()
                ) or 0

                movement_count = (
                    session.query(sa_func.count(CLVLineMovement.id))
                    .filter(CLVLineMovement.sport == self.sport)
                    .scalar()
                ) or 0

                if bet_count > 0 and movement_count == 0:
                    issues.append(
                        f"NO_LINE_DATA: {bet_count} bets exist but no line "
                        f"movement data has been captured"
                    )
                    return {
                        "coverage_pct": 0.0,
                        "with_lines": 0,
                        "without_lines": bet_count,
                        "issues": issues,
                    }

                return {
                    "coverage_pct": 100.0,
                    "with_lines": 0,
                    "without_lines": 0,
                    "issues": [],
                }

            with_lines = len(events_with_movements)
            without_lines = len(all_events) - with_lines
            coverage_pct = (with_lines / len(all_events) * 100) if all_events else 100.0

            if without_lines > 0:
                issues.append(
                    f"EVENTS_WITHOUT_LINES: {without_lines}/{len(all_events)} events "
                    f"have no line movement history"
                )

            return {
                "coverage_pct": coverage_pct,
                "with_lines": with_lines,
                "without_lines": without_lines,
                "issues": issues,
            }
        finally:
            session.close()

    # ── Systematic gaps ───────────────────────────────────────────

    def _check_systematic_gaps(self) -> Dict[str, Any]:
        """Identify data that is systematically missing.

        Looks for patterns like:
          - Certain books consistently missing from line movements
          - Certain market types with no data
          - Time periods with complete data blackouts
        """
        session = self._session()
        try:
            issues = []
            gaps: List[Dict[str, str]] = []

            # Book coverage: which books appear in line movements
            book_counts = (
                session.query(
                    CLVLineMovement.book,
                    sa_func.count(CLVLineMovement.id),
                )
                .filter(CLVLineMovement.sport == self.sport)
                .group_by(CLVLineMovement.book)
                .all()
            )

            book_coverage: Dict[str, int] = {
                book: count for book, count in book_counts
            }

            if book_coverage:
                max_count = max(book_coverage.values())
                for book, count in book_coverage.items():
                    if count < max_count * 0.1:  # Less than 10% of the most covered book
                        gaps.append({
                            "type": "book",
                            "name": book,
                            "detail": f"Only {count} observations "
                                      f"({count/max_count*100:.1f}% of best-covered book)",
                        })
                        issues.append(
                            f"SPARSE_BOOK: {book} has only {count} line observations "
                            f"({count/max_count*100:.1f}% of best-covered book)"
                        )

            # Market type coverage
            market_counts = (
                session.query(
                    CLVLineMovement.market_type,
                    sa_func.count(CLVLineMovement.id),
                )
                .filter(CLVLineMovement.sport == self.sport)
                .group_by(CLVLineMovement.market_type)
                .all()
            )

            market_coverage: Dict[str, int] = {
                market: count for market, count in market_counts
            }

            # Check which markets have bets but no line data
            bet_markets = set(
                row[0] for row in
                session.query(BetLog.stat_type)
                .filter(BetLog.sport == self.sport.lower())
                .distinct()
                .all()
            )

            tracked_markets = set(market_coverage.keys())
            untracked = bet_markets - tracked_markets

            for market in untracked:
                gaps.append({
                    "type": "market",
                    "name": market,
                    "detail": "Bets placed but no line movement data captured",
                })
                issues.append(
                    f"UNTRACKED_MARKET: {market} has bets placed but no "
                    f"line movement data is being captured"
                )

            # Time-based gaps: check for days with zero data in recent 30 days
            cutoff = _utcnow() - timedelta(days=30)
            daily_counts_raw = (
                session.query(
                    sa_func.date(CLVLineMovement.timestamp),
                    sa_func.count(CLVLineMovement.id),
                )
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.timestamp >= cutoff,
                )
                .group_by(sa_func.date(CLVLineMovement.timestamp))
                .all()
            )

            if daily_counts_raw:
                days_with_data = {str(row[0]) for row in daily_counts_raw}
                now = _utcnow()
                all_days = set()
                for i in range(30):
                    day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
                    all_days.add(day)

                missing_days = all_days - days_with_data
                if len(missing_days) > 3:
                    gaps.append({
                        "type": "time",
                        "name": "data_blackout",
                        "detail": f"{len(missing_days)} days with zero data in last 30 days",
                    })
                    issues.append(
                        f"DATA_BLACKOUTS: {len(missing_days)} days with zero "
                        f"line movement data in the last 30 days"
                    )

            return {
                "gaps": gaps,
                "book_coverage": book_coverage,
                "market_coverage": market_coverage,
                "issues": issues,
            }
        finally:
            session.close()

    # ── Line history depth ────────────────────────────────────────

    def _check_line_history_depth(self) -> Dict[str, Any]:
        """Check if line history has enough data points for meaningful analysis.

        A bet with only 1-2 line observations for its player/market has
        insufficient history for reliable CLV calculation.
        """
        session = self._session()
        try:
            issues = []

            bets = (
                session.query(BetLog)
                .filter(BetLog.sport == self.sport.lower())
                .all()
            )

            if not bets:
                return {"adequate_pct": 100.0, "issues": []}

            adequate = 0
            inadequate = 0

            for bet in bets:
                obs_count = (
                    session.query(sa_func.count(CLVLineMovement.id))
                    .filter(
                        CLVLineMovement.sport == self.sport,
                        CLVLineMovement.player == bet.player,
                        CLVLineMovement.market_type == bet.stat_type,
                    )
                    .scalar()
                ) or 0

                if obs_count >= 5:  # At least 5 observations = adequate
                    adequate += 1
                else:
                    inadequate += 1

            total = adequate + inadequate
            adequate_pct = (adequate / total * 100) if total > 0 else 100.0

            if inadequate > 0:
                issues.append(
                    f"SHALLOW_HISTORY: {inadequate}/{total} bets have fewer than "
                    f"5 line observations — insufficient for reliable CLV analysis"
                )

            return {
                "adequate_pct": adequate_pct,
                "adequate": adequate,
                "inadequate": inadequate,
                "issues": issues,
            }
        finally:
            session.close()
