"""
services/data_audit/timestamp_audit.py
=======================================
Validates timestamp accuracy across all data in the system.

Checks:
  - Are timestamps accurate to seconds? (not rounded to minutes/hours)
  - Are timezone conversions correct?
  - Are there impossible timestamps? (future dates, before season start)
  - Are data ingestion timestamps consistent with source update frequency?
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from sqlalchemy import func as sa_func, text

from quant_system.db.schema import get_engine, get_session, BetLog, LineSnapshot
from services.clv_system.models import (
    CLVLineMovement,
    CLVBetSnapshot,
    CLVClosingLine,
    Base,
)

log = logging.getLogger(__name__)

# NBA season typically runs October through June
NBA_SEASON_START_MONTH = 10  # October
NBA_SEASON_END_MONTH = 7    # July (including playoffs/finals)

# Expected polling interval for odds ingestion (5 minutes)
EXPECTED_POLL_INTERVAL_SEC = 300
# Maximum acceptable gap before flagging as missed poll
MAX_POLL_GAP_SEC = 900  # 15 minutes (3x expected interval)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampAuditor:
    """Validates all timestamps in the system for accuracy and consistency.

    Produces a detailed findings dict with:
      - second_precision_pct: % of timestamps accurate to the second
      - timezone_issues: count of timezone-related problems
      - impossible_timestamps: count of future/pre-season timestamps
      - ingestion_regularity_pct: % of polls arriving on expected schedule
      - issues: list of human-readable issue descriptions
      - score: 0-100 composite timestamp quality score
    """

    def __init__(self, sport: str = "NBA", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    def audit(self) -> Dict[str, Any]:
        """Run all timestamp checks and return a consolidated result."""
        precision = self._check_second_precision()
        impossible = self._check_impossible_timestamps()
        regularity = self._check_ingestion_regularity()
        timezone_issues = self._check_timezone_consistency()
        closing_timing = self._check_closing_line_timing()

        # Merge all issues
        all_issues: List[str] = []
        all_issues.extend(precision.get("issues", []))
        all_issues.extend(impossible.get("issues", []))
        all_issues.extend(regularity.get("issues", []))
        all_issues.extend(timezone_issues.get("issues", []))
        all_issues.extend(closing_timing.get("issues", []))

        # Composite score
        precision_score = precision.get("precision_pct", 100.0)
        impossible_penalty = min(impossible.get("count", 0) * 5, 30)
        regularity_score = regularity.get("regularity_pct", 100.0)
        tz_penalty = min(timezone_issues.get("count", 0) * 3, 15)
        closing_penalty = min(closing_timing.get("late_count", 0) * 2, 20)

        score = max(0.0, min(100.0,
            (precision_score * 0.25)
            + (regularity_score * 0.25)
            + (100 - impossible_penalty) * 0.20
            + (100 - tz_penalty) * 0.15
            + (100 - closing_penalty) * 0.15
        ))

        return {
            "score": round(score, 1),
            "second_precision_pct": round(precision.get("precision_pct", 100.0), 1),
            "rounded_timestamps": precision.get("rounded_count", 0),
            "total_timestamps_checked": precision.get("total_checked", 0),
            "impossible_timestamps": impossible.get("count", 0),
            "future_timestamps": impossible.get("future_count", 0),
            "pre_season_timestamps": impossible.get("pre_season_count", 0),
            "ingestion_regularity_pct": round(regularity.get("regularity_pct", 100.0), 1),
            "missed_polls": regularity.get("missed_polls", 0),
            "total_intervals": regularity.get("total_intervals", 0),
            "timezone_issues": timezone_issues.get("count", 0),
            "late_closing_captures": closing_timing.get("late_count", 0),
            "avg_closing_delta_sec": round(closing_timing.get("avg_delta_sec", 0.0), 1),
            "issues": all_issues,
        }

    # ── Second precision ──────────────────────────────────────────

    def _check_second_precision(self) -> Dict[str, Any]:
        """Check if timestamps have second-level precision or are rounded."""
        session = self._session()
        try:
            issues = []
            movements = (
                session.query(CLVLineMovement.timestamp)
                .filter(CLVLineMovement.sport == self.sport)
                .order_by(CLVLineMovement.timestamp.desc())
                .limit(5000)
                .all()
            )

            if not movements:
                return {"precision_pct": 100.0, "rounded_count": 0,
                        "total_checked": 0, "issues": ["No line movements to check"]}

            total = len(movements)
            rounded_to_minute = 0
            rounded_to_hour = 0

            for (ts,) in movements:
                if ts is None:
                    continue
                if ts.second == 0 and ts.microsecond == 0:
                    rounded_to_minute += 1
                    if ts.minute == 0:
                        rounded_to_hour += 1

            # If >50% of timestamps have :00 seconds, likely rounded
            # Normal distribution would give ~1.67% chance of :00 seconds
            rounded_pct = (rounded_to_minute / total * 100) if total > 0 else 0
            precision_pct = 100.0 - min(rounded_pct, 100.0)

            if rounded_to_hour > total * 0.05:
                issues.append(
                    f"ROUNDED_HOURS: {rounded_to_hour}/{total} timestamps "
                    f"({rounded_to_hour/total*100:.1f}%) are rounded to the hour"
                )
            elif rounded_to_minute > total * 0.10:
                issues.append(
                    f"ROUNDED_MINUTES: {rounded_to_minute}/{total} timestamps "
                    f"({rounded_pct:.1f}%) have zero seconds — likely rounded to minutes"
                )

            return {
                "precision_pct": precision_pct,
                "rounded_count": rounded_to_minute,
                "total_checked": total,
                "issues": issues,
            }
        finally:
            session.close()

    # ── Impossible timestamps ─────────────────────────────────────

    def _check_impossible_timestamps(self) -> Dict[str, Any]:
        """Check for timestamps that cannot be real."""
        session = self._session()
        try:
            now = _utcnow()
            issues = []

            # Future timestamps (more than 5 minutes ahead)
            future_cutoff = now + timedelta(minutes=5)
            future_count = (
                session.query(sa_func.count(CLVLineMovement.id))
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.timestamp > future_cutoff,
                )
                .scalar()
            ) or 0

            if future_count > 0:
                issues.append(
                    f"FUTURE_TIMESTAMPS: {future_count} line movements "
                    f"have timestamps in the future"
                )

            # Future timestamps in bet logs
            future_bets = (
                session.query(sa_func.count(BetLog.id))
                .filter(
                    BetLog.sport == self.sport.lower(),
                    BetLog.timestamp > future_cutoff,
                )
                .scalar()
            ) or 0

            if future_bets > 0:
                issues.append(
                    f"FUTURE_BET_TIMESTAMPS: {future_bets} bets have "
                    f"timestamps in the future"
                )

            # Pre-season timestamps: for current season, check for data
            # before October of the previous year (rough heuristic)
            current_year = now.year
            season_start = datetime(
                current_year - 1 if now.month < NBA_SEASON_START_MONTH else current_year,
                NBA_SEASON_START_MONTH, 1,
                tzinfo=timezone.utc,
            )

            pre_season_count = (
                session.query(sa_func.count(CLVLineMovement.id))
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.timestamp < season_start,
                )
                .scalar()
            ) or 0

            if pre_season_count > 0:
                issues.append(
                    f"PRE_SEASON: {pre_season_count} line movements have "
                    f"timestamps before season start ({season_start.strftime('%Y-%m-%d')})"
                )

            # Extremely old timestamps (before 2020 = data error)
            ancient_cutoff = datetime(2020, 1, 1, tzinfo=timezone.utc)
            ancient_count = (
                session.query(sa_func.count(CLVLineMovement.id))
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.timestamp < ancient_cutoff,
                )
                .scalar()
            ) or 0

            if ancient_count > 0:
                issues.append(
                    f"ANCIENT_TIMESTAMPS: {ancient_count} line movements "
                    f"have timestamps before 2020 — likely data errors"
                )

            total = future_count + future_bets + pre_season_count + ancient_count

            return {
                "count": total,
                "future_count": future_count + future_bets,
                "pre_season_count": pre_season_count + ancient_count,
                "issues": issues,
            }
        finally:
            session.close()

    # ── Ingestion regularity ──────────────────────────────────────

    def _check_ingestion_regularity(self) -> Dict[str, Any]:
        """Check if data ingestion polls arrive at expected intervals."""
        session = self._session()
        try:
            issues = []

            # Get timestamps of the last 1000 line movements, ordered
            timestamps = (
                session.query(CLVLineMovement.timestamp)
                .filter(CLVLineMovement.sport == self.sport)
                .order_by(CLVLineMovement.timestamp.desc())
                .limit(1000)
                .all()
            )

            if len(timestamps) < 10:
                return {
                    "regularity_pct": 100.0,
                    "missed_polls": 0,
                    "total_intervals": 0,
                    "issues": ["Insufficient data for regularity check"],
                }

            # Deduplicate to unique poll timestamps (group by minute)
            seen_minutes = set()
            unique_timestamps = []
            for (ts,) in timestamps:
                if ts is None:
                    continue
                minute_key = ts.replace(second=0, microsecond=0)
                if minute_key not in seen_minutes:
                    seen_minutes.add(minute_key)
                    unique_timestamps.append(ts)

            unique_timestamps.sort()

            if len(unique_timestamps) < 2:
                return {
                    "regularity_pct": 100.0,
                    "missed_polls": 0,
                    "total_intervals": 0,
                    "issues": [],
                }

            # Count gaps exceeding expected interval
            missed_polls = 0
            total_intervals = len(unique_timestamps) - 1
            large_gaps = []

            for i in range(1, len(unique_timestamps)):
                gap = (unique_timestamps[i] - unique_timestamps[i - 1]).total_seconds()
                if gap > MAX_POLL_GAP_SEC:
                    missed_polls += 1
                    large_gaps.append(gap)

            regularity_pct = (
                (1 - missed_polls / total_intervals) * 100
                if total_intervals > 0 else 100.0
            )

            if missed_polls > 0:
                max_gap_min = max(large_gaps) / 60 if large_gaps else 0
                issues.append(
                    f"MISSED_POLLS: {missed_polls}/{total_intervals} intervals "
                    f"exceeded {MAX_POLL_GAP_SEC/60:.0f}-minute threshold "
                    f"(max gap: {max_gap_min:.1f} min)"
                )

            return {
                "regularity_pct": regularity_pct,
                "missed_polls": missed_polls,
                "total_intervals": total_intervals,
                "issues": issues,
            }
        finally:
            session.close()

    # ── Timezone consistency ──────────────────────────────────────

    def _check_timezone_consistency(self) -> Dict[str, Any]:
        """Check for timezone-related issues in the data."""
        session = self._session()
        try:
            issues = []
            tz_issue_count = 0

            # Check for closing lines where captured_at and event_start_time
            # have suspiciously round differences (e.g., exactly 4, 5, or 8 hours)
            # suggesting a timezone conversion error
            closing_lines = (
                session.query(CLVClosingLine)
                .filter(CLVClosingLine.sport == self.sport)
                .limit(1000)
                .all()
            )

            suspicious_offsets = {4 * 3600, 5 * 3600, 8 * 3600, -4 * 3600,
                                  -5 * 3600, -8 * 3600}

            for cl in closing_lines:
                if cl.event_start_time and cl.captured_at:
                    delta_sec = (cl.captured_at - cl.event_start_time).total_seconds()
                    # Closing should be captured near event start (within ~30 min)
                    # If the delta is exactly a common timezone offset, it's suspicious
                    abs_delta = abs(delta_sec)
                    for offset in suspicious_offsets:
                        if abs(abs_delta - abs(offset)) < 60:
                            tz_issue_count += 1
                            break

            if tz_issue_count > 0:
                issues.append(
                    f"TIMEZONE_OFFSET: {tz_issue_count} closing lines have "
                    f"capture-to-event deltas matching common timezone offsets "
                    f"(4h, 5h, 8h) — possible timezone conversion error"
                )

            # Check bet timestamps vs snapshot timestamps for consistency
            snapshots = (
                session.query(CLVBetSnapshot)
                .filter(CLVBetSnapshot.sport == self.sport)
                .limit(500)
                .all()
            )

            bet_tz_mismatches = 0
            for snap in snapshots:
                bet = (
                    session.query(BetLog)
                    .filter(BetLog.bet_id == snap.bet_id)
                    .first()
                )
                if bet and bet.timestamp and snap.signal_timestamp:
                    delta = abs((bet.timestamp - snap.signal_timestamp).total_seconds())
                    # Bet and snapshot should be within seconds of each other
                    # If off by hours, it's a timezone issue
                    if delta > 3600 and any(abs(delta - o) < 120 for o in
                                             [4*3600, 5*3600, 8*3600]):
                        bet_tz_mismatches += 1

            if bet_tz_mismatches > 0:
                tz_issue_count += bet_tz_mismatches
                issues.append(
                    f"BET_SNAPSHOT_TZ_MISMATCH: {bet_tz_mismatches} bets have "
                    f"timestamps offset from their snapshots by exact timezone amounts"
                )

            return {
                "count": tz_issue_count,
                "issues": issues,
            }
        finally:
            session.close()

    # ── Closing line capture timing ───────────────────────────────

    def _check_closing_line_timing(self) -> Dict[str, Any]:
        """Check if closing lines are captured consistently at event start."""
        session = self._session()
        try:
            issues = []
            closing_lines = (
                session.query(CLVClosingLine)
                .filter(CLVClosingLine.sport == self.sport)
                .all()
            )

            if not closing_lines:
                return {"late_count": 0, "avg_delta_sec": 0.0, "issues": []}

            deltas = []
            late_count = 0
            very_early_count = 0

            for cl in closing_lines:
                if cl.event_start_time and cl.captured_at:
                    delta = abs((cl.captured_at - cl.event_start_time).total_seconds())
                    deltas.append(delta)
                    if delta > 300:  # More than 5 minutes from event start
                        late_count += 1
                    if delta > 1800:  # More than 30 minutes — very early capture
                        very_early_count += 1

            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

            if late_count > 0:
                issues.append(
                    f"LATE_CLOSINGS: {late_count}/{len(closing_lines)} closing lines "
                    f"captured more than 5 minutes from event start"
                )

            if very_early_count > 0:
                issues.append(
                    f"VERY_EARLY_CLOSINGS: {very_early_count} closing lines captured "
                    f"30+ minutes before event start — likely stale pre-close data"
                )

            return {
                "late_count": late_count,
                "very_early_count": very_early_count,
                "avg_delta_sec": avg_delta,
                "total_closing_lines": len(closing_lines),
                "issues": issues,
            }
        finally:
            session.close()
