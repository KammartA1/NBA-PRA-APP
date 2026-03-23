"""
services/data_audit/report.py
==============================
Unified data quality report that aggregates all audit findings.

Produces the master DATA QUALITY AUDIT report and stores results
in the clv_integrity_reports table for historical tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from quant_system.db.schema import get_engine, get_session
from services.clv_system.models import CLVIntegrityReport, Base
from services.data_audit.timestamp_audit import TimestampAuditor
from services.data_audit.odds_audit import OddsAuditor
from services.data_audit.closing_line_audit import ClosingLineAuditor
from services.data_audit.completeness_audit import CompletenessAuditor

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DataQualityReport:
    """Master data quality audit system.

    Aggregates findings from all four audit dimensions:
      - Timestamp accuracy
      - Odds availability & validity
      - Closing line accuracy
      - Data completeness

    Produces a composite score and verdict. If any critical finding exists,
    the system's edge may be fabricated.
    """

    CRITICAL_SCORE_THRESHOLD = 80.0

    def __init__(self, sport: str = "NBA", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        self._engine = get_engine(db_path)
        Base.metadata.create_all(self._engine)

        self.timestamp_auditor = TimestampAuditor(sport=sport, db_path=db_path)
        self.odds_auditor = OddsAuditor(sport=sport, db_path=db_path)
        self.closing_auditor = ClosingLineAuditor(sport=sport, db_path=db_path)
        self.completeness_auditor = CompletenessAuditor(sport=sport, db_path=db_path)

    def _session(self):
        return get_session(self._db_path)

    def generate(self) -> str:
        """Run all audits and generate the master data quality report.

        Returns:
            Formatted report text string.

        Also stores the report in the clv_integrity_reports table.
        """
        log.info("Starting data quality audit for %s", self.sport)

        # Run all auditors
        ts_result = self.timestamp_auditor.audit()
        odds_result = self.odds_auditor.audit()
        closing_result = self.closing_auditor.audit()
        completeness_result = self.completeness_auditor.audit()

        # Composite score (weighted average of all four dimensions)
        ts_score = ts_result.get("score", 100.0)
        odds_score = odds_result.get("score", 100.0)
        closing_score = closing_result.get("score", 100.0)
        completeness_score = completeness_result.get("score", 100.0)

        composite_score = (
            ts_score * 0.20
            + odds_score * 0.25
            + closing_score * 0.30
            + completeness_score * 0.25
        )
        composite_score = round(max(0.0, min(100.0, composite_score)), 1)

        # Collect all critical findings
        critical_findings = self._extract_critical_findings(
            ts_result, odds_result, closing_result, completeness_result
        )

        # Build report text
        report = self._format_report(
            composite_score=composite_score,
            ts_score=ts_score,
            odds_score=odds_score,
            closing_score=closing_score,
            completeness_score=completeness_score,
            ts_result=ts_result,
            odds_result=odds_result,
            closing_result=closing_result,
            completeness_result=completeness_result,
            critical_findings=critical_findings,
        )

        # Store in database
        self._store_report(
            composite_score=composite_score,
            ts_result=ts_result,
            odds_result=odds_result,
            closing_result=closing_result,
            completeness_result=completeness_result,
            report_text=report,
        )

        log.info(
            "Data quality audit complete: score=%.1f (%d critical findings)",
            composite_score,
            len(critical_findings),
        )
        return report

    def generate_dict(self) -> Dict[str, Any]:
        """Run all audits and return structured results (for Streamlit)."""
        ts_result = self.timestamp_auditor.audit()
        odds_result = self.odds_auditor.audit()
        closing_result = self.closing_auditor.audit()
        completeness_result = self.completeness_auditor.audit()

        ts_score = ts_result.get("score", 100.0)
        odds_score = odds_result.get("score", 100.0)
        closing_score = closing_result.get("score", 100.0)
        completeness_score = completeness_result.get("score", 100.0)

        composite_score = round(max(0.0, min(100.0,
            ts_score * 0.20
            + odds_score * 0.25
            + closing_score * 0.30
            + completeness_score * 0.25
        )), 1)

        critical_findings = self._extract_critical_findings(
            ts_result, odds_result, closing_result, completeness_result
        )

        # Store report
        self._store_report(
            composite_score=composite_score,
            ts_result=ts_result,
            odds_result=odds_result,
            closing_result=closing_result,
            completeness_result=completeness_result,
            report_text=self._format_report(
                composite_score=composite_score,
                ts_score=ts_score,
                odds_score=odds_score,
                closing_score=closing_score,
                completeness_score=completeness_score,
                ts_result=ts_result,
                odds_result=odds_result,
                closing_result=closing_result,
                completeness_result=completeness_result,
                critical_findings=critical_findings,
            ),
        )

        return {
            "composite_score": composite_score,
            "timestamp": {
                "score": ts_score,
                **ts_result,
            },
            "odds": {
                "score": odds_score,
                **odds_result,
            },
            "closing_line": {
                "score": closing_score,
                **closing_result,
            },
            "completeness": {
                "score": completeness_score,
                **completeness_result,
            },
            "critical_findings": critical_findings,
            "is_trustworthy": composite_score >= self.CRITICAL_SCORE_THRESHOLD,
            "generated_at": _utcnow().isoformat(),
        }

    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """Get the most recent stored data quality report."""
        session = self._session()
        try:
            row = (
                session.query(CLVIntegrityReport)
                .filter(CLVIntegrityReport.sport == self.sport)
                .order_by(CLVIntegrityReport.generated_at.desc())
                .first()
            )
            if row:
                result = row.to_dict()
                try:
                    result["details"] = json.loads(row.details_json or "{}")
                except (json.JSONDecodeError, TypeError):
                    result["details"] = {}
                return result
            return None
        finally:
            session.close()

    def get_report_history(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Get historical data quality reports for trend analysis."""
        session = self._session()
        try:
            rows = (
                session.query(CLVIntegrityReport)
                .filter(CLVIntegrityReport.sport == self.sport)
                .order_by(CLVIntegrityReport.generated_at.desc())
                .limit(limit)
                .all()
            )
            results = []
            for row in rows:
                d = row.to_dict()
                try:
                    d["details"] = json.loads(row.details_json or "{}")
                except (json.JSONDecodeError, TypeError):
                    d["details"] = {}
                results.append(d)
            return results
        finally:
            session.close()

    # ── Internal helpers ──────────────────────────────────────────

    def _extract_critical_findings(
        self,
        ts_result: Dict,
        odds_result: Dict,
        closing_result: Dict,
        completeness_result: Dict,
    ) -> List[str]:
        """Extract findings that could indicate fabricated edge."""
        critical: List[str] = []

        # Timestamp critical: future timestamps or major timezone issues
        if ts_result.get("future_timestamps", 0) > 0:
            critical.append(
                f"{ts_result['future_timestamps']} timestamps are in the future — "
                f"data integrity compromised"
            )
        if ts_result.get("timezone_issues", 0) > 5:
            critical.append(
                f"{ts_result['timezone_issues']} timezone conversion errors detected — "
                f"CLV calculations may be using wrong timestamps"
            )

        # Odds critical: phantom lines or many unverified
        if odds_result.get("phantom_lines_count", 0) > 0:
            critical.append(
                f"{odds_result['phantom_lines_count']} phantom lines detected — "
                f"odds in system that may never have been offered by books"
            )
        avail_pct = odds_result.get("availability_verified_pct", 100.0)
        if avail_pct < 50:
            critical.append(
                f"Only {avail_pct:.0f}% of bet-time odds can be verified — "
                f"majority of odds data is unconfirmed"
            )

        # Closing line critical: mostly stale closes
        true_close_pct = closing_result.get("true_close_pct", 100.0)
        if true_close_pct < 70:
            critical.append(
                f"Only {true_close_pct:.0f}% of closing lines are true market closes — "
                f"CLV measured against stale data is meaningless"
            )

        # Completeness critical: massive data gaps
        bet_pct = completeness_result.get("bet_completeness_pct", 100.0)
        if bet_pct < 50:
            critical.append(
                f"Only {bet_pct:.0f}% of bets have complete data chains — "
                f"cannot reliably measure system performance"
            )
        missing_closing = completeness_result.get("missing_closing_pct", 0.0)
        if missing_closing > 30:
            critical.append(
                f"{missing_closing:.0f}% of settled bets have no closing line — "
                f"CLV cannot be calculated for these bets"
            )

        return critical

    def _format_report(
        self,
        composite_score: float,
        ts_score: float,
        odds_score: float,
        closing_score: float,
        completeness_score: float,
        ts_result: Dict,
        odds_result: Dict,
        closing_result: Dict,
        completeness_result: Dict,
        critical_findings: List[str],
    ) -> str:
        """Format the complete data quality report."""
        now = _utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        report = f"""DATA QUALITY AUDIT
==================
Sport: {self.sport}
Generated: {now}

COMPOSITE SCORE: {composite_score:.0f}/100

DIMENSION SCORES
-----------------
Timestamp accuracy:    {ts_score:.0f}/100
Odds availability:     {odds_score:.0f}/100
Closing line accuracy: {closing_score:.0f}/100
Data completeness:     {completeness_score:.0f}/100

TIMESTAMP AUDIT
---------------
Second-precision timestamps: {ts_result.get('second_precision_pct', 0):.1f}%
Impossible timestamps: {ts_result.get('impossible_timestamps', 0)}
  - Future timestamps: {ts_result.get('future_timestamps', 0)}
  - Pre-season timestamps: {ts_result.get('pre_season_timestamps', 0)}
Ingestion regularity: {ts_result.get('ingestion_regularity_pct', 0):.1f}%
  - Missed polls: {ts_result.get('missed_polls', 0)}/{ts_result.get('total_intervals', 0)}
Timezone issues: {ts_result.get('timezone_issues', 0)}
Late closing captures: {ts_result.get('late_closing_captures', 0)}

ODDS AUDIT
----------
Odds availability verified: {odds_result.get('availability_verified_pct', 0):.1f}%
  - Verified: {odds_result.get('bets_verified', 0)}/{odds_result.get('bets_checked', 0)}
Stale odds sequences: {odds_result.get('stale_lines_count', 0)}
Phantom lines: {odds_result.get('phantom_lines_count', 0)}
Unreasonable odds: {odds_result.get('unreasonable_odds_count', 0)}

CLOSING LINE AUDIT
------------------
True closing lines: {closing_result.get('true_close_pct', 0):.1f}%
  - Stale closes: {closing_result.get('stale_close_count', 0)}/{closing_result.get('total_closing_lines', 0)}
Capture consistency: {closing_result.get('capture_consistency_pct', 0):.1f}%
  - Timing std dev: {closing_result.get('capture_std_dev_sec', 0):.0f}s
Cross-source agreement: {closing_result.get('cross_source_match_pct', 0):.1f}%
Active market at close: {closing_result.get('active_market_pct', 0):.1f}%

COMPLETENESS AUDIT
------------------
Bet completeness: {completeness_result.get('bet_completeness_pct', 0):.1f}%
  - Total bets: {completeness_result.get('total_bets', 0)}
  - Settled bets: {completeness_result.get('settled_bets', 0)}
  - Complete bets: {completeness_result.get('complete_bets', 0)}
Missing data:
  - Opening lines: {completeness_result.get('missing_opening_pct', 0):.1f}%
  - Bet-time snapshots: {completeness_result.get('missing_snapshot_pct', 0):.1f}%
  - Closing lines: {completeness_result.get('missing_closing_pct', 0):.1f}%
  - CLV calculations: {completeness_result.get('missing_clv_pct', 0):.1f}%
Event coverage: {completeness_result.get('event_coverage_pct', 0):.1f}%
Line history depth: {completeness_result.get('line_history_adequate_pct', 0):.1f}% adequate"""

        # Systematic gaps
        sys_gaps = completeness_result.get("systematic_gaps", [])
        if sys_gaps:
            report += "\n\nSYSTEMATIC GAPS"
            report += "\n---------------"
            for gap in sys_gaps:
                report += f"\n  [{gap.get('type', '?')}] {gap.get('name', '?')}: {gap.get('detail', '')}"

        # Critical findings
        report += "\n\nCRITICAL FINDINGS"
        report += "\n-----------------"
        if critical_findings:
            for finding in critical_findings:
                report += f"\n  !! {finding}"
        else:
            report += "\n  None — all data quality checks passed"

        # Verdict
        report += "\n\nVERDICT"
        report += "\n-------"
        if critical_findings:
            report += (
                "\nYOUR EDGE MAY BE FABRICATED. "
                "Fix data quality before trusting any results."
            )
            report += f"\n{len(critical_findings)} critical finding(s) must be resolved."
        elif composite_score < self.CRITICAL_SCORE_THRESHOLD:
            report += (
                f"\nDATA QUALITY INSUFFICIENT (score {composite_score:.0f} < "
                f"{self.CRITICAL_SCORE_THRESHOLD:.0f}). "
                "Address issues above before relying on system metrics."
            )
        elif composite_score < 90:
            report += (
                f"\nDATA QUALITY ACCEPTABLE (score {composite_score:.0f}/100). "
                "Minor issues exist — review items above for improvement."
            )
        else:
            report += (
                f"\nDATA QUALITY EXCELLENT (score {composite_score:.0f}/100). "
                "All metrics trustworthy."
            )

        return report

    def _store_report(
        self,
        composite_score: float,
        ts_result: Dict,
        odds_result: Dict,
        closing_result: Dict,
        completeness_result: Dict,
        report_text: str,
    ) -> None:
        """Store the audit report in the database."""
        session = self._session()
        try:
            details = {
                "timestamp_audit": {
                    k: v for k, v in ts_result.items()
                    if k != "issues"
                },
                "odds_audit": {
                    k: v for k, v in odds_result.items()
                    if k not in ("issues", "stale_lines_detail", "phantom_lines_detail")
                },
                "closing_line_audit": {
                    k: v for k, v in closing_result.items()
                    if k != "issues"
                },
                "completeness_audit": {
                    k: v for k, v in completeness_result.items()
                    if k not in ("issues", "systematic_gaps", "book_coverage",
                                  "market_coverage")
                },
            }

            # Collect missing counts for the integrity report schema
            missing_opening = int(
                completeness_result.get("total_bets", 0)
                * completeness_result.get("missing_opening_pct", 0) / 100
            )
            missing_closing = int(
                completeness_result.get("settled_bets", 0)
                * completeness_result.get("missing_closing_pct", 0) / 100
            )
            missing_snapshots = int(
                completeness_result.get("total_bets", 0)
                * completeness_result.get("missing_snapshot_pct", 0) / 100
            )
            suspected_errors = (
                odds_result.get("unreasonable_odds_count", 0)
                + odds_result.get("phantom_lines_count", 0)
                + ts_result.get("impossible_timestamps", 0)
            )

            ir = CLVIntegrityReport(
                sport=self.sport,
                generated_at=_utcnow(),
                total_bets=completeness_result.get("total_bets", 0),
                missing_opening_lines=missing_opening,
                missing_closing_lines=missing_closing,
                missing_bet_snapshots=missing_snapshots,
                suspected_data_errors=suspected_errors,
                integrity_score=composite_score,
                report_text=report_text,
                details_json=json.dumps(details, default=str),
            )
            session.add(ir)
            session.commit()
            log.info("Data quality report stored: score=%.1f", composite_score)

        except Exception:
            session.rollback()
            log.exception("Failed to store data quality report")
        finally:
            session.close()
