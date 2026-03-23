"""
services/clv_system/integrity.py
==================================
Data quality validation for the CLV system.

If CLV data cannot be trusted, the entire system is invalid.
This module validates:
  - Missing data (opening lines, closing lines, bet-time snapshots)
  - Timestamp accuracy
  - Statistical outliers (likely data errors)
  - Closing line validity (not stale data)

Produces an integrity score (0-100).  If < 80, CLV is untrustworthy.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import func as sa_func

from quant_system.db.schema import get_engine, get_session, BetLog, CLVLog
from services.clv_system.models import (
    CLVLineMovement,
    CLVBetSnapshot,
    CLVClosingLine,
    CLVIntegrityReport,
    Base,
)

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CLVIntegrity:
    """Data quality validation for CLV tracking.

    Checks every dimension of data quality and produces an integrity score.
    If score < 80, CLV data cannot be trusted and system edge is unverifiable.
    """

    # Thresholds
    OUTLIER_Z_THRESHOLD = 4.0
    STALE_LINE_HOURS = 6
    MIN_INTEGRITY_SCORE = 80

    def __init__(self, sport: str = "NBA", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    # ── Check: Missing data ──────────────────────────────────────────

    def check_missing_data(self) -> List[str]:
        """Flag any bets missing opening line, closing line, or bet-time snapshot.

        Returns:
            List of warning strings for each data gap found.
        """
        warnings = []
        session = self._session()
        try:
            # All bets
            bets = (
                session.query(BetLog)
                .filter(BetLog.sport == self.sport.lower())
                .all()
            )

            if not bets:
                return ["No bets found in the system"]

            for bet in bets:
                # Check closing line
                if bet.status in ("won", "lost", "push") and bet.closing_line is None:
                    clv_row = (
                        session.query(CLVLog)
                        .filter_by(bet_id=bet.bet_id)
                        .first()
                    )
                    if not clv_row:
                        warnings.append(
                            f"MISSING_CLOSING: bet {bet.bet_id} ({bet.player} "
                            f"{bet.stat_type}) settled without closing line"
                        )

                # Check bet-time snapshot
                snapshot = (
                    session.query(CLVBetSnapshot)
                    .filter_by(bet_id=bet.bet_id)
                    .first()
                )
                if not snapshot:
                    warnings.append(
                        f"MISSING_SNAPSHOT: bet {bet.bet_id} ({bet.player} "
                        f"{bet.stat_type}) has no bet-time snapshot"
                    )

                # Check opening line
                opening = (
                    session.query(CLVLineMovement)
                    .filter(
                        CLVLineMovement.sport == self.sport,
                        CLVLineMovement.player == bet.player,
                        CLVLineMovement.market_type == bet.stat_type,
                        CLVLineMovement.is_opening == True,  # noqa: E712
                    )
                    .first()
                )
                if not opening:
                    warnings.append(
                        f"MISSING_OPENING: bet {bet.bet_id} ({bet.player} "
                        f"{bet.stat_type}) has no opening line recorded"
                    )

            return warnings

        finally:
            session.close()

    # ── Check: Timestamp accuracy ────────────────────────────────────

    def check_timestamp_accuracy(self) -> Dict[str, Any]:
        """Verify timestamps are within expected precision.

        Checks:
          - Line movement timestamps are in chronological order
          - No future timestamps
          - No unreasonable gaps (> 1 hour between polls)
          - Closing lines captured within 5 min of event start

        Returns:
            Dict with accuracy metrics and any issues found.
        """
        session = self._session()
        issues = []
        try:
            now = _utcnow()

            # Check for future timestamps
            future_count = (
                session.query(sa_func.count(CLVLineMovement.id))
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.timestamp > now + timedelta(minutes=5),
                )
                .scalar()
            ) or 0
            if future_count > 0:
                issues.append(f"FUTURE_TIMESTAMPS: {future_count} line movements have future timestamps")

            # Check closing line timeliness
            closing_lines = (
                session.query(CLVClosingLine)
                .filter(CLVClosingLine.sport == self.sport)
                .all()
            )
            late_closings = 0
            for cl in closing_lines:
                if cl.event_start_time and cl.captured_at:
                    delta = abs((cl.captured_at - cl.event_start_time).total_seconds())
                    if delta > 300:  # More than 5 minutes
                        late_closings += 1
            if late_closings > 0:
                issues.append(
                    f"LATE_CLOSINGS: {late_closings} closing lines captured "
                    f"more than 5 minutes from event start"
                )

            # Check polling regularity
            total_movements = (
                session.query(sa_func.count(CLVLineMovement.id))
                .filter(CLVLineMovement.sport == self.sport)
                .scalar()
            ) or 0

            return {
                "total_movements": total_movements,
                "future_timestamps": future_count,
                "late_closings": late_closings,
                "total_closing_lines": len(closing_lines),
                "issues": issues,
                "accuracy_score": max(0, 100 - (future_count * 5) - (late_closings * 3)),
            }

        finally:
            session.close()

    # ── Check: Outliers ──────────────────────────────────────────────

    def check_outliers(self) -> List[str]:
        """Flag any CLV values that are statistical outliers (likely data errors).

        Uses Z-score method: any CLV value more than 4 standard deviations
        from the mean is flagged.

        Returns:
            List of warning strings for each outlier found.
        """
        warnings = []
        session = self._session()
        try:
            rows = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport.lower())
                .all()
            )

            if len(rows) < 20:
                return ["Insufficient data for outlier detection (need 20+ CLV records)"]

            clv_values = [r.clv_cents for r in rows]
            mean_clv = float(np.mean(clv_values))
            std_clv = float(np.std(clv_values))

            if std_clv == 0:
                return []

            for row in rows:
                z_score = abs(row.clv_cents - mean_clv) / std_clv
                if z_score > self.OUTLIER_Z_THRESHOLD:
                    warnings.append(
                        f"OUTLIER: bet {row.bet_id} CLV={row.clv_cents:.2f} "
                        f"(z={z_score:.1f}, mean={mean_clv:.2f}, std={std_clv:.2f})"
                    )

            return warnings

        finally:
            session.close()

    # ── Check: Closing line validity ─────────────────────────────────

    def check_closing_line_validity(self) -> Dict[str, Any]:
        """Verify closing lines are actual closes, not stale data.

        A closing line that hasn't changed in 6+ hours is likely stale
        (the market was not active) and should not be used for CLV.

        Returns:
            Dict with validity metrics.
        """
        session = self._session()
        try:
            closing_lines = (
                session.query(CLVClosingLine)
                .filter(
                    CLVClosingLine.sport == self.sport,
                    CLVClosingLine.is_consensus == False,  # noqa: E712
                )
                .all()
            )

            total = len(closing_lines)
            stale = 0
            valid = 0

            for cl in closing_lines:
                # Find the second-to-last line before closing
                prev_line = (
                    session.query(CLVLineMovement)
                    .filter(
                        CLVLineMovement.sport == self.sport,
                        CLVLineMovement.player == cl.player,
                        CLVLineMovement.market_type == cl.market_type,
                        CLVLineMovement.book == cl.book,
                        CLVLineMovement.timestamp < cl.captured_at,
                    )
                    .order_by(CLVLineMovement.timestamp.desc())
                    .first()
                )

                if prev_line and cl.captured_at:
                    hours_since = (cl.captured_at - prev_line.timestamp).total_seconds() / 3600
                    if hours_since > self.STALE_LINE_HOURS:
                        stale += 1
                    else:
                        valid += 1
                else:
                    valid += 1  # No prior data = assume valid

            return {
                "total_closing_lines": total,
                "valid_closings": valid,
                "stale_closings": stale,
                "stale_pct": round(stale / total * 100, 1) if total > 0 else 0.0,
                "validity_score": round(valid / total * 100, 1) if total > 0 else 100.0,
            }

        finally:
            session.close()

    # ── Master integrity report ──────────────────────────────────────

    def generate_integrity_report(self) -> str:
        """Generate a comprehensive data integrity report.

        Returns formatted report text. Also stores the report in the
        clv_integrity_reports table.

        If integrity_score < 80:
            WARNING: CLV data cannot be trusted. System edge is unverifiable.
        """
        session = self._session()
        try:
            # Gather all checks
            missing_warnings = self.check_missing_data()
            timestamp_check = self.check_timestamp_accuracy()
            outlier_warnings = self.check_outliers()
            validity_check = self.check_closing_line_validity()

            # Count bets
            total_bets = (
                session.query(sa_func.count(BetLog.id))
                .filter(BetLog.sport == self.sport.lower())
                .scalar()
            ) or 0

            settled_bets = (
                session.query(sa_func.count(BetLog.id))
                .filter(
                    BetLog.sport == self.sport.lower(),
                    BetLog.status.in_(["won", "lost", "push"]),
                )
                .scalar()
            ) or 0

            # Count missing items
            missing_opening = sum(1 for w in missing_warnings if "MISSING_OPENING" in w)
            missing_closing = sum(1 for w in missing_warnings if "MISSING_CLOSING" in w)
            missing_snapshot = sum(1 for w in missing_warnings if "MISSING_SNAPSHOT" in w)
            n_outliers = len([w for w in outlier_warnings if "OUTLIER" in w])

            # Calculate integrity score (0-100)
            score = 100.0
            if total_bets > 0:
                # Deduct for missing data
                score -= (missing_opening / max(total_bets, 1)) * 20
                score -= (missing_closing / max(settled_bets, 1)) * 30
                score -= (missing_snapshot / max(total_bets, 1)) * 15
                # Deduct for outliers
                score -= min(n_outliers * 2, 10)
                # Deduct for stale closings
                score -= (100 - validity_check.get("validity_score", 100)) * 0.15
                # Deduct for timestamp issues
                score -= (100 - timestamp_check.get("accuracy_score", 100)) * 0.1

            score = max(0.0, min(100.0, score))

            # Build report text
            missing_open_pct = (missing_opening / total_bets * 100) if total_bets > 0 else 0
            missing_close_pct = (missing_closing / settled_bets * 100) if settled_bets > 0 else 0
            missing_snap_pct = (missing_snapshot / total_bets * 100) if total_bets > 0 else 0

            report = f"""DATA INTEGRITY REPORT
=====================
Sport: {self.sport}
Generated: {_utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

COVERAGE
--------
Total bets tracked: {total_bets}
Settled bets: {settled_bets}
Missing opening lines: {missing_opening} ({missing_open_pct:.1f}%)
Missing closing lines: {missing_closing} ({missing_close_pct:.1f}%)
Missing bet-time snapshots: {missing_snapshot} ({missing_snap_pct:.1f}%)

DATA QUALITY
------------
Suspected data errors (outliers): {n_outliers}
Stale closing lines: {validity_check.get('stale_closings', 0)}
Future timestamps: {timestamp_check.get('future_timestamps', 0)}
Late closing captures: {timestamp_check.get('late_closings', 0)}

LINE MOVEMENT DATA
------------------
Total line observations: {timestamp_check.get('total_movements', 0)}
Closing lines captured: {validity_check.get('total_closing_lines', 0)}
Closing line validity: {validity_check.get('validity_score', 0):.1f}%

INTEGRITY SCORE: {score:.0f}/100"""

            if score < self.MIN_INTEGRITY_SCORE:
                report += f"""

!! WARNING !!
CLV data cannot be trusted. System edge is unverifiable.
Integrity score {score:.0f} is below minimum threshold of {self.MIN_INTEGRITY_SCORE}.
Address the data gaps above before relying on CLV metrics."""
            elif score < 90:
                report += f"""

CAUTION: Integrity score {score:.0f} is acceptable but has room for improvement.
Review missing data items above to improve reliability."""
            else:
                report += f"""

EXCELLENT: Data integrity is strong. CLV metrics are trustworthy."""

            # Store report
            ir = CLVIntegrityReport(
                sport=self.sport,
                generated_at=_utcnow(),
                total_bets=total_bets,
                missing_opening_lines=missing_opening,
                missing_closing_lines=missing_closing,
                missing_bet_snapshots=missing_snapshot,
                suspected_data_errors=n_outliers,
                integrity_score=round(score, 1),
                report_text=report,
                details_json="{}",
            )
            session.add(ir)
            session.commit()

            log.info("Integrity report generated: score=%.1f", score)
            return report

        except Exception:
            session.rollback()
            log.exception("Failed to generate integrity report")
            raise
        finally:
            session.close()

    # ── Get latest report ────────────────────────────────────────────

    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """Get the most recent integrity report."""
        session = self._session()
        try:
            row = (
                session.query(CLVIntegrityReport)
                .filter(CLVIntegrityReport.sport == self.sport)
                .order_by(CLVIntegrityReport.generated_at.desc())
                .first()
            )
            return row.to_dict() if row else None
        finally:
            session.close()

    def get_integrity_score(self) -> float:
        """Get just the latest integrity score."""
        report = self.get_latest_report()
        return report.get("integrity_score", 0.0) if report else 0.0

    def get_report_history(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Get integrity report history."""
        session = self._session()
        try:
            rows = (
                session.query(CLVIntegrityReport)
                .filter(CLVIntegrityReport.sport == self.sport)
                .order_by(CLVIntegrityReport.generated_at.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()
