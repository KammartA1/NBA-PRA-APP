"""
Quant System v1.0 — Production-Grade Automated Betting Infrastructure

Shared across NBA and Golf engines. Sport-agnostic core with
sport-specific adapters.

Modules:
    core/       — CLV tracking, edge validation, calibration, bet logging
    risk/       — Adaptive Kelly, bankroll management, failure protection
    backtest/   — Walk-forward validation, feature ablation, MC simulation
    market/     — Line tracking, sharp detection, steam moves
    learning/   — Model drift, feature monitoring, auto-reweighting
    dashboard/  — Reporting, P&L curves, calibration plots
    db/         — SQLAlchemy schema for all quant system tables
"""

__version__ = "1.0.0"
