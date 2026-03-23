"""
services/execution/
====================
Execution Layer Realism — models the gap between theoretical edge and
what you actually capture after slippage, limits, latency, and rejections.

Most bettors model edge but ignore execution costs. This module quantifies
every friction that erodes profit between signal generation and bet settlement.
"""

from services.execution.slippage_model import SlippageModel
from services.execution.limit_model import LimitModel
from services.execution.latency_model import LatencyModel
from services.execution.rejection_model import RejectionModel
from services.execution.reality_simulator import ExecutionSimulator, ExecutionReport

__all__ = [
    "SlippageModel",
    "LimitModel",
    "LatencyModel",
    "RejectionModel",
    "ExecutionSimulator",
    "ExecutionReport",
]
