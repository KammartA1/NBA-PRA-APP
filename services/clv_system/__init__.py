"""
services/clv_system
====================
Fully autonomous Closing Line Value (CLV) tracking system.
Zero manual input. If CLV cannot be trusted, the entire system is invalid.

Modules:
    models.py           — Database models for CLV-specific tables
    odds_ingestion.py   — Continuous odds capture from multiple sources
    line_storage.py     — Time-series storage of all line movements
    snapshot.py         — Automatic bet-time price snapshot
    closing_capture.py  — Automatic closing line capture at event start
    clv_calculator.py   — CLV computation (cents + probability)
    integrity.py        — Data quality validation
"""

from services.clv_system.clv_calculator import CLVCalculator
from services.clv_system.integrity import CLVIntegrity
from services.clv_system.odds_ingestion import CLVOddsIngestion
from services.clv_system.line_storage import LineStorage
from services.clv_system.snapshot import BetTimeSnapshotService
from services.clv_system.closing_capture import ClosingLineCapture

__all__ = [
    "CLVCalculator",
    "CLVIntegrity",
    "CLVOddsIngestion",
    "LineStorage",
    "BetTimeSnapshotService",
    "ClosingLineCapture",
]
