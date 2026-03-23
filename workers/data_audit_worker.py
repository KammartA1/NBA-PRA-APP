"""
workers/data_audit_worker.py
=============================
Background worker that runs the data quality audit system.

Runs automatically after each data ingestion cycle and on a scheduled
interval. Generates a full DataQualityReport and stores the results.

Run standalone:
    python -m workers.data_audit_worker          # one-shot
    python -m workers.data_audit_worker --loop   # scheduled loop
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from workers.base import BaseWorker, standalone_main

log = logging.getLogger(__name__)


class DataAuditWorker(BaseWorker):
    """Runs the full data quality audit on a scheduled interval.

    Default interval: 30 minutes (runs after odds ingestion cycles).
    Can also be triggered on-demand via the Streamlit dashboard.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="data_audit_worker",
            interval_seconds=int(os.environ.get("AUDIT_INTERVAL", "1800")),
            max_retries=2,
            retry_delay=15.0,
            **kwargs,
        )

    def execute(self) -> Dict[str, Any]:
        from services.data_audit.report import DataQualityReport

        self.logger.info("Starting data quality audit")

        report = DataQualityReport(sport="NBA")
        result = report.generate_dict()

        composite_score = result.get("composite_score", 0)
        is_trustworthy = result.get("is_trustworthy", False)
        critical_count = len(result.get("critical_findings", []))

        self.logger.info(
            "Data quality audit complete: score=%.1f, trustworthy=%s, "
            "critical_findings=%d",
            composite_score,
            is_trustworthy,
            critical_count,
        )

        if not is_trustworthy:
            self.logger.warning(
                "DATA QUALITY BELOW THRESHOLD (%.1f < 80). "
                "Edge may be fabricated!",
                composite_score,
            )

        if critical_count > 0:
            self.logger.warning(
                "CRITICAL FINDINGS (%d):", critical_count
            )
            for finding in result.get("critical_findings", []):
                self.logger.warning("  !! %s", finding)

        return {
            "ok": True,
            "composite_score": composite_score,
            "is_trustworthy": is_trustworthy,
            "critical_findings": critical_count,
            "timestamp_score": result.get("timestamp", {}).get("score", 0),
            "odds_score": result.get("odds", {}).get("score", 0),
            "closing_score": result.get("closing_line", {}).get("score", 0),
            "completeness_score": result.get("completeness", {}).get("score", 0),
        }


def run_post_ingestion_audit() -> Dict[str, Any]:
    """Lightweight audit hook to run after each data ingestion cycle.

    This is called from the odds worker after storing new lines.
    Runs the full audit but catches and logs errors so it never
    blocks the ingestion pipeline.
    """
    try:
        from services.data_audit.report import DataQualityReport

        report = DataQualityReport(sport="NBA")
        result = report.generate_dict()

        score = result.get("composite_score", 0)
        log.info("Post-ingestion audit: score=%.1f", score)

        if not result.get("is_trustworthy", True):
            log.warning(
                "POST-INGESTION AUDIT WARNING: score=%.1f — "
                "data quality below threshold", score
            )

        return {
            "ok": True,
            "composite_score": score,
            "critical_findings": len(result.get("critical_findings", [])),
        }

    except Exception as exc:
        log.warning(
            "Post-ingestion audit failed (non-blocking): %s", exc
        )
        return {"ok": False, "error": str(exc)}


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    standalone_main(DataAuditWorker)
