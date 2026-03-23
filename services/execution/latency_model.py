"""
services/execution/latency_model.py
====================================
Models the time between signal generation and bet placement, and the
probability that the line is still available at execution time.

If manual betting: latency = minutes (edge may be gone).
If automated: latency = seconds.
Model: P(line still available) as function of latency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyObservation:
    """A single observation of signal-to-execution latency."""
    bet_id: str
    signal_time: datetime
    bet_time: datetime
    latency_seconds: float
    line_still_available: bool      # Was the signal line still available?
    line_change_cents: float        # How much did the line move?
    market_type: str = "points"
    sportsbook: str = "unknown"


@dataclass
class LatencyResult:
    """Output of the latency model for a given latency."""
    latency_seconds: float
    p_line_available: float          # P(original line still available)
    p_line_within_1_cent: float      # P(line within 0.5 pts of original)
    p_line_gone: float               # P(line moved > 1 point)
    expected_line_decay_cents: float  # Expected adverse move due to latency
    edge_retention_pct: float        # % of original edge retained


@dataclass
class LatencyProfile:
    """Aggregate latency statistics."""
    n_observations: int
    mean_latency_seconds: float
    median_latency_seconds: float
    p90_latency_seconds: float
    p_line_available_overall: float
    availability_by_latency_bucket: dict
    edge_decay_curve: List[dict]     # [{latency_s, edge_retained_pct}, ...]
    execution_mode: str              # "manual", "semi_auto", "automated"


class LatencyModel:
    """Models the relationship between execution latency and line availability.

    Core insight: edge decays with time. A signal that fires 5 minutes before
    you can bet is worth far less than one you can execute in 5 seconds.

    The model fits an exponential decay: P(available) = exp(-lambda * t)
    where lambda depends on market liquidity and time of day.
    """

    # Decay rate by market liquidity tier
    DECAY_RATES = {
        "high_liquidity": 0.005,     # Points, PRA — slow decay
        "medium_liquidity": 0.010,   # Rebounds, assists
        "low_liquidity": 0.020,      # Blocks, steals, exotic
    }

    MARKET_LIQUIDITY = {
        "points": "high_liquidity",
        "pra": "high_liquidity",
        "rebounds": "medium_liquidity",
        "assists": "medium_liquidity",
        "threes": "medium_liquidity",
        "blocks": "low_liquidity",
        "steals": "low_liquidity",
        "turnovers": "low_liquidity",
        "fantasy_score": "low_liquidity",
        "double_double": "low_liquidity",
    }

    # Execution mode detection thresholds
    MODE_THRESHOLDS = {
        "automated": 15,        # < 15s median = automated
        "semi_auto": 120,       # < 2min median = semi-automated
        "manual": float("inf"), # > 2min = manual
    }

    def __init__(self, sport: str = "nba"):
        self.sport = sport
        self._observations: List[LatencyObservation] = []
        self._fitted_lambda: float = 0.01  # Default decay rate
        self._fitted = False

    def load_observations(self, observations: List[LatencyObservation]) -> None:
        """Load historical latency observations."""
        self._observations = observations
        if observations:
            self._fit()
        log.info("LatencyModel loaded %d observations", len(observations))

    def load_from_bets(self, bets: list) -> None:
        """Load from raw bet records."""
        observations = []
        for b in bets:
            if hasattr(b, "signal_generated_at"):
                sig_time = b.signal_generated_at
                bet_time = b.timestamp
                sig_line = b.signal_line
                bet_line = b.bet_line
                market = getattr(b, "market_type", "points")
                book = getattr(b, "sportsbook", "unknown")
                bid = getattr(b, "bet_id", "")
            elif isinstance(b, dict):
                sig_time = b.get("signal_generated_at")
                bet_time = b.get("timestamp")
                sig_line = b.get("signal_line", 0)
                bet_line = b.get("bet_line", 0)
                market = b.get("market_type", "points")
                book = b.get("sportsbook", "unknown")
                bid = b.get("bet_id", "")
            else:
                continue

            if not sig_time or not bet_time:
                continue

            latency = max(0, (bet_time - sig_time).total_seconds())
            line_change = abs(bet_line - sig_line)
            still_available = line_change < 0.5  # Within half a point

            observations.append(LatencyObservation(
                bet_id=str(bid),
                signal_time=sig_time,
                bet_time=bet_time,
                latency_seconds=latency,
                line_still_available=still_available,
                line_change_cents=line_change,
                market_type=market,
                sportsbook=book,
            ))
        self.load_observations(observations)

    def _fit(self) -> None:
        """Fit exponential decay rate from observations."""
        if len(self._observations) < 3:
            self._fitted = True
            return

        # MLE for exponential: lambda = n_unavailable / sum(latencies)
        latencies = np.array([o.latency_seconds for o in self._observations])
        unavailable = np.array([not o.line_still_available for o in self._observations])

        total_latency = np.sum(latencies)
        n_unavail = np.sum(unavailable)

        if total_latency > 0 and n_unavail > 0:
            self._fitted_lambda = float(n_unavail / total_latency)
        else:
            self._fitted_lambda = 0.005  # Conservative default

        # Clamp to reasonable range
        self._fitted_lambda = max(0.0001, min(0.1, self._fitted_lambda))
        self._fitted = True
        log.info("LatencyModel fitted: lambda=%.6f", self._fitted_lambda)

    def _get_decay_rate(self, market_type: str) -> float:
        """Get the appropriate decay rate for a market type."""
        if self._fitted and self._observations:
            return self._fitted_lambda

        mt = market_type.lower().replace(" ", "_")
        tier = "medium_liquidity"
        for key, val in self.MARKET_LIQUIDITY.items():
            if key in mt:
                tier = val
                break
        return self.DECAY_RATES[tier]

    def predict(
        self,
        latency_seconds: float,
        market_type: str = "points",
        edge_cents: float = 2.0,
    ) -> LatencyResult:
        """Predict line availability and edge decay for a given latency."""
        lam = self._get_decay_rate(market_type)

        p_available = math.exp(-lam * latency_seconds)
        p_within_1 = math.exp(-lam * latency_seconds * 0.5)
        p_gone = 1.0 - p_within_1

        # Expected line decay
        expected_decay = edge_cents * (1.0 - p_available)

        # Edge retention
        edge_retained = p_available * 100.0

        return LatencyResult(
            latency_seconds=latency_seconds,
            p_line_available=p_available,
            p_line_within_1_cent=p_within_1,
            p_line_gone=p_gone,
            expected_line_decay_cents=expected_decay,
            edge_retention_pct=edge_retained,
        )

    def generate_decay_curve(
        self,
        market_type: str = "points",
        max_seconds: int = 600,
        n_points: int = 50,
    ) -> List[dict]:
        """Generate edge decay curve over time."""
        curve = []
        for t in np.linspace(0, max_seconds, n_points):
            result = self.predict(t, market_type)
            curve.append({
                "latency_seconds": float(t),
                "p_line_available": result.p_line_available,
                "edge_retention_pct": result.edge_retention_pct,
            })
        return curve

    def _detect_execution_mode(self) -> str:
        """Detect whether execution is manual, semi-auto, or automated."""
        if not self._observations:
            return "manual"

        latencies = [o.latency_seconds for o in self._observations]
        median = float(np.median(latencies))

        for mode, threshold in self.MODE_THRESHOLDS.items():
            if median < threshold:
                return mode
        return "manual"

    def profile(self) -> LatencyProfile:
        """Generate aggregate latency profile."""
        if not self._observations:
            return LatencyProfile(
                n_observations=0,
                mean_latency_seconds=0,
                median_latency_seconds=0,
                p90_latency_seconds=0,
                p_line_available_overall=0,
                availability_by_latency_bucket={},
                edge_decay_curve=[],
                execution_mode="manual",
            )

        latencies = np.array([o.latency_seconds for o in self._observations])
        available = np.array([o.line_still_available for o in self._observations])

        # Availability by latency bucket
        buckets = [
            (0, 10, "<10s"),
            (10, 30, "10-30s"),
            (30, 60, "30-60s"),
            (60, 120, "1-2min"),
            (120, 300, "2-5min"),
            (300, 600, "5-10min"),
            (600, float("inf"), ">10min"),
        ]

        avail_by_bucket = {}
        for lo, hi, label in buckets:
            mask = (latencies >= lo) & (latencies < hi)
            n = int(np.sum(mask))
            if n > 0:
                avail_by_bucket[label] = {
                    "n": n,
                    "p_available": float(np.mean(available[mask])),
                    "mean_latency": float(np.mean(latencies[mask])),
                }

        return LatencyProfile(
            n_observations=len(self._observations),
            mean_latency_seconds=float(np.mean(latencies)),
            median_latency_seconds=float(np.median(latencies)),
            p90_latency_seconds=float(np.percentile(latencies, 90)),
            p_line_available_overall=float(np.mean(available)),
            availability_by_latency_bucket=avail_by_bucket,
            edge_decay_curve=self.generate_decay_curve(),
            execution_mode=self._detect_execution_mode(),
        )
