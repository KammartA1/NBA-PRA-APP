"""
services/data_audit/
====================
Continuous data quality audit system for the quant engine.

Validates every dimension of data flowing through the system:
  - Timestamp accuracy and timezone correctness
  - Odds availability and staleness detection
  - Closing line validity and independent verification
  - Data completeness across events, bets, and line history

Produces a unified DataQualityReport with a composite score.
If any critical finding exists, edge may be fabricated.
"""

from services.data_audit.timestamp_audit import TimestampAuditor
from services.data_audit.odds_audit import OddsAuditor
from services.data_audit.closing_line_audit import ClosingLineAuditor
from services.data_audit.completeness_audit import CompletenessAuditor
from services.data_audit.report import DataQualityReport

__all__ = [
    "TimestampAuditor",
    "OddsAuditor",
    "ClosingLineAuditor",
    "CompletenessAuditor",
    "DataQualityReport",
]
