"""
services/capital/
=================
Capital management and portfolio optimization engine.

Combines Kelly criterion bet sizing, risk-adjusted performance metrics,
portfolio correlation management, and a top-level optimizer that enforces
constraints across all dimensions.
"""

from services.capital.kelly import KellyCalculator
from services.capital.risk_adjusted import RiskMetrics
from services.capital.portfolio import PortfolioManager
from services.capital.optimizer import CapitalOptimizer

__all__ = [
    "KellyCalculator",
    "RiskMetrics",
    "PortfolioManager",
    "CapitalOptimizer",
]
