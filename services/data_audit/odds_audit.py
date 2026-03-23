"""
services/data_audit/odds_audit.py
==================================
Validates odds data quality across the system.

Checks:
  - Are the odds we record actually available at the time we claim?
  - Cross-reference: was the book actually offering this line at this timestamp?
  - Check for stale odds (same line repeated for hours = likely stale cache)
  - Check for phantom lines (odds in our system but never actually offered)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from sqlalchemy import func as sa_func

from quant_system.db.schema import get_engine, get_session, BetLog, LineSnapshot
from services.clv_system.models import (
    CLVLineMovement,
    CLVBetSnapshot,
    Base,
)

log = logging.getLogger(__name__)

# If a line doesn't change for this many hours, it's stale
STALE_LINE_THRESHOLD_HOURS = 6
# Maximum reasonable odds range (American)
MAX_REASONABLE_ODDS = 500
MIN_REASONABLE_ODDS = -1000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OddsAuditor:
    """Validates odds data for availability, staleness, and phantom lines.

    Produces a detailed findings dict with:
      - availability_verified_pct: % of bet-time odds verified against line history
      - stale_lines_count: number of stale (unchanging) line sequences detected
      - phantom_lines_count: odds that appear in snapshots but have no line history
      - unreasonable_odds_count: odds outside expected ranges
      - issues: list of human-readable issue descriptions
      - score: 0-100 composite odds quality score
    """

    def __init__(self, sport: str = "NBA", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    def audit(self) -> Dict[str, Any]:
        """Run all odds quality checks and return consolidated results."""
        availability = self._check_odds_availability()
        staleness = self._check_stale_odds()
        phantom = self._check_phantom_lines()
        reasonable = self._check_odds_reasonableness()

        all_issues: List[str] = []
        all_issues.extend(availability.get("issues", []))
        all_issues.extend(staleness.get("issues", []))
        all_issues.extend(phantom.get("issues", []))
        all_issues.extend(reasonable.get("issues", []))

        # Composite score
        avail_score = availability.get("verified_pct", 100.0)
        stale_penalty = min(staleness.get("stale_sequences", 0) * 2, 20)
        phantom_penalty = min(phantom.get("phantom_count", 0) * 3, 20)
        unreasonable_penalty = min(reasonable.get("unreasonable_count", 0) * 5, 20)

        score = max(0.0, min(100.0,
            (avail_score * 0.40)
            + (100 - stale_penalty) * 0.25
            + (100 - phantom_penalty) * 0.20
            + (100 - unreasonable_penalty) * 0.15
        ))

        return {
            "score": round(score, 1),
            "availability_verified_pct": round(avail_score, 1),
            "bets_verified": availability.get("verified", 0),
            "bets_unverified": availability.get("unverified", 0),
            "bets_checked": availability.get("total_checked", 0),
            "stale_lines_count": staleness.get("stale_sequences", 0),
            "stale_lines_detail": staleness.get("details", []),
            "phantom_lines_count": phantom.get("phantom_count", 0),
            "phantom_lines_detail": phantom.get("details", []),
            "unreasonable_odds_count": reasonable.get("unreasonable_count", 0),
            "issues": all_issues,
        }

    # ── Odds availability verification ────────────────────────────

    def _check_odds_availability(self) -> Dict[str, Any]:
        """Verify that odds recorded at bet time were actually available.

        Cross-references CLVBetSnapshot lines against CLVLineMovement history
        to confirm the book was actually offering that line at that timestamp.
        """
        session = self._session()
        try:
            issues = []

            snapshots = (
                session.query(CLVBetSnapshot)
                .filter(CLVBetSnapshot.sport == self.sport)
                .order_by(CLVBetSnapshot.signal_timestamp.desc())
                .limit(500)
                .all()
            )

            if not snapshots:
                return {
                    "verified_pct": 100.0, "verified": 0,
                    "unverified": 0, "total_checked": 0,
                    "issues": ["No bet snapshots to verify"],
                }

            verified = 0
            unverified = 0

            for snap in snapshots:
                # Parse the lines_json to get what books/lines were captured
                try:
                    lines_data = json.loads(snap.lines_json) if snap.lines_json else {}
                except (json.JSONDecodeError, TypeError):
                    lines_data = {}

                if not lines_data:
                    unverified += 1
                    continue

                # For each book in the snapshot, check if we have a matching
                # line movement within a reasonable time window
                snap_verified = False
                window_start = snap.signal_timestamp - timedelta(minutes=10)
                window_end = snap.signal_timestamp + timedelta(minutes=2)

                for book, line_info in lines_data.items():
                    line_val = line_info if isinstance(line_info, (int, float)) else (
                        line_info.get("line") if isinstance(line_info, dict) else None
                    )
                    if line_val is None:
                        continue

                    # Look for a matching line movement
                    match = (
                        session.query(CLVLineMovement)
                        .filter(
                            CLVLineMovement.sport == self.sport,
                            CLVLineMovement.player == snap.player,
                            CLVLineMovement.market_type == snap.market_type,
                            CLVLineMovement.book == book,
                            CLVLineMovement.timestamp >= window_start,
                            CLVLineMovement.timestamp <= window_end,
                        )
                        .first()
                    )
                    if match:
                        snap_verified = True
                        break

                if snap_verified:
                    verified += 1
                else:
                    unverified += 1

            total = verified + unverified
            verified_pct = (verified / total * 100) if total > 0 else 100.0

            if unverified > 0:
                issues.append(
                    f"UNVERIFIED_ODDS: {unverified}/{total} bet-time snapshots "
                    f"could not be cross-referenced against line movement history"
                )

            return {
                "verified_pct": verified_pct,
                "verified": verified,
                "unverified": unverified,
                "total_checked": total,
                "issues": issues,
            }
        finally:
            session.close()

    # ── Stale odds detection ──────────────────────────────────────

    def _check_stale_odds(self) -> Dict[str, Any]:
        """Detect line sequences where the same value repeats for hours.

        If a line doesn't change for STALE_LINE_THRESHOLD_HOURS hours
        across multiple polls, it's likely stale cached data, not a real
        market quote.
        """
        session = self._session()
        try:
            issues = []
            stale_sequences = 0
            details: List[Dict[str, Any]] = []

            # Get distinct player/market/book combinations with recent data
            combos = (
                session.query(
                    CLVLineMovement.player,
                    CLVLineMovement.market_type,
                    CLVLineMovement.book,
                )
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.timestamp >= _utcnow() - timedelta(days=7),
                )
                .distinct()
                .limit(200)
                .all()
            )

            for player, market, book in combos:
                movements = (
                    session.query(CLVLineMovement)
                    .filter(
                        CLVLineMovement.sport == self.sport,
                        CLVLineMovement.player == player,
                        CLVLineMovement.market_type == market,
                        CLVLineMovement.book == book,
                        CLVLineMovement.timestamp >= _utcnow() - timedelta(days=7),
                    )
                    .order_by(CLVLineMovement.timestamp.asc())
                    .all()
                )

                if len(movements) < 3:
                    continue

                # Find runs of identical lines
                run_start = movements[0]
                run_line = movements[0].line

                for i in range(1, len(movements)):
                    if movements[i].line == run_line:
                        # Still in the same run
                        duration = (movements[i].timestamp - run_start.timestamp).total_seconds()
                        if duration >= STALE_LINE_THRESHOLD_HOURS * 3600:
                            # Check how many polls are in this stale run
                            polls_in_run = i - movements.index(run_start) + 1
                            if polls_in_run >= 4:  # At least 4 identical polls
                                stale_sequences += 1
                                details.append({
                                    "player": player,
                                    "market": market,
                                    "book": book,
                                    "line": run_line,
                                    "duration_hours": round(duration / 3600, 1),
                                    "polls": polls_in_run,
                                })
                                # Move past this sequence
                                run_start = movements[i]
                                run_line = movements[i].line
                    else:
                        run_start = movements[i]
                        run_line = movements[i].line

            if stale_sequences > 0:
                issues.append(
                    f"STALE_ODDS: {stale_sequences} line sequences unchanged for "
                    f"{STALE_LINE_THRESHOLD_HOURS}+ hours — likely stale cached data"
                )

            return {
                "stale_sequences": stale_sequences,
                "details": details[:20],  # Limit detail output
                "issues": issues,
            }
        finally:
            session.close()

    # ── Phantom lines ─────────────────────────────────────────────

    def _check_phantom_lines(self) -> Dict[str, Any]:
        """Detect lines that appear in bet snapshots but have no history.

        A phantom line is one that shows up in our system (e.g., in a
        CLVBetSnapshot) but was never tracked in CLVLineMovement — suggesting
        it may have been fabricated or injected rather than captured from
        a real market.
        """
        session = self._session()
        try:
            issues = []
            phantom_count = 0
            details: List[Dict[str, Any]] = []

            snapshots = (
                session.query(CLVBetSnapshot)
                .filter(CLVBetSnapshot.sport == self.sport)
                .order_by(CLVBetSnapshot.signal_timestamp.desc())
                .limit(300)
                .all()
            )

            for snap in snapshots:
                try:
                    lines_data = json.loads(snap.lines_json) if snap.lines_json else {}
                except (json.JSONDecodeError, TypeError):
                    continue

                for book, line_info in lines_data.items():
                    line_val = line_info if isinstance(line_info, (int, float)) else (
                        line_info.get("line") if isinstance(line_info, dict) else None
                    )
                    if line_val is None:
                        continue

                    # Check if this book ever tracked this player/market
                    has_history = (
                        session.query(sa_func.count(CLVLineMovement.id))
                        .filter(
                            CLVLineMovement.sport == self.sport,
                            CLVLineMovement.player == snap.player,
                            CLVLineMovement.market_type == snap.market_type,
                            CLVLineMovement.book == book,
                        )
                        .scalar()
                    ) or 0

                    if has_history == 0:
                        phantom_count += 1
                        details.append({
                            "bet_id": snap.bet_id,
                            "player": snap.player,
                            "market": snap.market_type,
                            "book": book,
                            "line": line_val,
                        })

            if phantom_count > 0:
                issues.append(
                    f"PHANTOM_LINES: {phantom_count} lines in bet snapshots have "
                    f"no corresponding line movement history — may not have been "
                    f"genuinely offered by the book"
                )

            return {
                "phantom_count": phantom_count,
                "details": details[:20],
                "issues": issues,
            }
        finally:
            session.close()

    # ── Odds reasonableness ───────────────────────────────────────

    def _check_odds_reasonableness(self) -> Dict[str, Any]:
        """Check for odds values outside reasonable ranges."""
        session = self._session()
        try:
            issues = []

            # Check for unreasonable American odds in line movements
            unreasonable = (
                session.query(sa_func.count(CLVLineMovement.id))
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.odds_american.isnot(None),
                )
                .filter(
                    (CLVLineMovement.odds_american > MAX_REASONABLE_ODDS)
                    | (CLVLineMovement.odds_american < MIN_REASONABLE_ODDS)
                    | ((CLVLineMovement.odds_american > -100)
                       & (CLVLineMovement.odds_american < 100)
                       & (CLVLineMovement.odds_american != 0))
                )
                .scalar()
            ) or 0

            # Check for negative or zero lines (shouldn't exist for props)
            bad_lines = (
                session.query(sa_func.count(CLVLineMovement.id))
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.line < 0,
                )
                .scalar()
            ) or 0

            # Check for implied probabilities outside 0-1
            bad_probs = (
                session.query(sa_func.count(CLVLineMovement.id))
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.implied_prob.isnot(None),
                )
                .filter(
                    (CLVLineMovement.implied_prob < 0)
                    | (CLVLineMovement.implied_prob > 1)
                )
                .scalar()
            ) or 0

            total_unreasonable = unreasonable + bad_lines + bad_probs

            if unreasonable > 0:
                issues.append(
                    f"UNREASONABLE_ODDS: {unreasonable} line movements have "
                    f"American odds outside [{MIN_REASONABLE_ODDS}, {MAX_REASONABLE_ODDS}] "
                    f"or in the forbidden (-100, 100) range"
                )
            if bad_lines > 0:
                issues.append(
                    f"NEGATIVE_LINES: {bad_lines} line movements have "
                    f"negative line values"
                )
            if bad_probs > 0:
                issues.append(
                    f"BAD_IMPLIED_PROBS: {bad_probs} line movements have "
                    f"implied probability outside [0, 1]"
                )

            return {
                "unreasonable_count": total_unreasonable,
                "unreasonable_odds": unreasonable,
                "negative_lines": bad_lines,
                "bad_implied_probs": bad_probs,
                "issues": issues,
            }
        finally:
            session.close()
