"""
services/market_reaction/edge_decay.py
=======================================
Models edge erosion over time from multiple sources:
  - Market impact: your own bets move the line
  - Book copying: one book's limits propagate to others
  - Competition: other sharps discover the same signal
  - Adaptation: books improve their own models
  - Information diffusion: private info becomes public

Each edge source has its own decay half-life. A pace/tempo edge decays
slowly (structural). A news-based edge decays in hours. A line stale
edge decays in minutes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class EdgeSource:
    """Definition of a single edge source with decay characteristics."""
    name: str
    category: str                    # structural, informational, execution, model
    initial_edge_pct: float          # Starting edge in %
    half_life_days: float            # Time for edge to halve
    market_impact_sensitivity: float  # How much own bets erode this edge (0-1)
    competition_sensitivity: float    # How fast competitors find this (0-1)
    adaptation_sensitivity: float     # How fast books adapt to this (0-1)
    min_edge_floor: float = 0.0      # Edge never decays below this (structural minimum)
    is_renewable: bool = False        # Can this edge be refreshed with new data?
    renewal_rate: float = 0.0        # If renewable, how much edge is restored per day


# Pre-configured edge sources for NBA prop betting
DEFAULT_EDGE_SOURCES: Dict[str, EdgeSource] = {
    "pace_differential": EdgeSource(
        name="pace_differential",
        category="structural",
        initial_edge_pct=2.0,
        half_life_days=180.0,
        market_impact_sensitivity=0.05,
        competition_sensitivity=0.2,
        adaptation_sensitivity=0.15,
        min_edge_floor=0.3,
        is_renewable=True,
        renewal_rate=0.005,
    ),
    "rest_effects": EdgeSource(
        name="rest_effects",
        category="structural",
        initial_edge_pct=1.5,
        half_life_days=365.0,
        market_impact_sensitivity=0.02,
        competition_sensitivity=0.3,
        adaptation_sensitivity=0.1,
        min_edge_floor=0.2,
        is_renewable=True,
        renewal_rate=0.003,
    ),
    "lineup_effects": EdgeSource(
        name="lineup_effects",
        category="informational",
        initial_edge_pct=3.0,
        half_life_days=30.0,
        market_impact_sensitivity=0.15,
        competition_sensitivity=0.5,
        adaptation_sensitivity=0.3,
        min_edge_floor=0.5,
        is_renewable=True,
        renewal_rate=0.02,
    ),
    "referee_tendencies": EdgeSource(
        name="referee_tendencies",
        category="structural",
        initial_edge_pct=1.0,
        half_life_days=270.0,
        market_impact_sensitivity=0.03,
        competition_sensitivity=0.15,
        adaptation_sensitivity=0.05,
        min_edge_floor=0.1,
        is_renewable=True,
        renewal_rate=0.002,
    ),
    "usage_redistribution": EdgeSource(
        name="usage_redistribution",
        category="informational",
        initial_edge_pct=2.5,
        half_life_days=45.0,
        market_impact_sensitivity=0.1,
        competition_sensitivity=0.4,
        adaptation_sensitivity=0.25,
        min_edge_floor=0.3,
        is_renewable=True,
        renewal_rate=0.015,
    ),
    "game_script": EdgeSource(
        name="game_script",
        category="model",
        initial_edge_pct=1.8,
        half_life_days=120.0,
        market_impact_sensitivity=0.08,
        competition_sensitivity=0.35,
        adaptation_sensitivity=0.2,
        min_edge_floor=0.2,
        is_renewable=True,
        renewal_rate=0.008,
    ),
    "defensive_matchup": EdgeSource(
        name="defensive_matchup",
        category="informational",
        initial_edge_pct=2.0,
        half_life_days=60.0,
        market_impact_sensitivity=0.12,
        competition_sensitivity=0.45,
        adaptation_sensitivity=0.3,
        min_edge_floor=0.2,
        is_renewable=True,
        renewal_rate=0.01,
    ),
    "recency_weighting": EdgeSource(
        name="recency_weighting",
        category="model",
        initial_edge_pct=1.2,
        half_life_days=90.0,
        market_impact_sensitivity=0.05,
        competition_sensitivity=0.25,
        adaptation_sensitivity=0.15,
        min_edge_floor=0.1,
        is_renewable=True,
        renewal_rate=0.005,
    ),
    "minutes_distribution": EdgeSource(
        name="minutes_distribution",
        category="informational",
        initial_edge_pct=2.2,
        half_life_days=40.0,
        market_impact_sensitivity=0.1,
        competition_sensitivity=0.4,
        adaptation_sensitivity=0.25,
        min_edge_floor=0.3,
        is_renewable=True,
        renewal_rate=0.012,
    ),
    "home_away": EdgeSource(
        name="home_away",
        category="structural",
        initial_edge_pct=0.8,
        half_life_days=365.0,
        market_impact_sensitivity=0.01,
        competition_sensitivity=0.5,
        adaptation_sensitivity=0.1,
        min_edge_floor=0.1,
        is_renewable=False,
        renewal_rate=0.0,
    ),
    "stale_line": EdgeSource(
        name="stale_line",
        category="execution",
        initial_edge_pct=4.0,
        half_life_days=14.0,
        market_impact_sensitivity=0.3,
        competition_sensitivity=0.7,
        adaptation_sensitivity=0.5,
        min_edge_floor=0.0,
        is_renewable=False,
        renewal_rate=0.0,
    ),
    "closing_line_inefficiency": EdgeSource(
        name="closing_line_inefficiency",
        category="execution",
        initial_edge_pct=3.0,
        half_life_days=21.0,
        market_impact_sensitivity=0.25,
        competition_sensitivity=0.6,
        adaptation_sensitivity=0.4,
        min_edge_floor=0.0,
        is_renewable=False,
        renewal_rate=0.0,
    ),
}


@dataclass
class DecayResult:
    """Decay analysis for a single edge source."""
    source_name: str
    category: str
    initial_edge_pct: float
    current_edge_pct: float
    days_elapsed: float
    half_life_days: float
    pct_remaining: float
    total_decay_rate: float        # Daily decay rate (combined)
    days_to_zero: float            # Estimated days until edge hits floor
    edge_floor: float
    is_renewable: bool
    steady_state_edge: float       # Long-run edge after decay + renewal balance


class EdgeDecayModel:
    """Models edge erosion over time for each edge source.

    The decay follows an exponential model with multiple decay channels:
      total_decay_rate = base_rate + market_impact + competition + adaptation

    Where base_rate = ln(2) / half_life_days

    For renewable edges, there's a renewal term that partially offsets decay,
    reaching a steady state where decay = renewal.
    """

    def __init__(
        self,
        edge_sources: Dict[str, EdgeSource] | None = None,
        bet_volume_per_day: float = 3.0,
        market_share: float = 0.001,  # What fraction of market volume is ours
    ):
        self.edge_sources = edge_sources or dict(DEFAULT_EDGE_SOURCES)
        self.bet_volume_per_day = bet_volume_per_day
        self.market_share = market_share

    def decay_single(
        self,
        source: EdgeSource,
        days_elapsed: float,
        bet_volume_per_day: float | None = None,
    ) -> DecayResult:
        """Compute edge decay for a single source after N days.

        The decay model:
          edge(t) = max(initial * exp(-total_rate * t) + renewal_accumulation, floor)

        Where:
          total_rate = base_rate * (1 + market_impact + competition + adaptation)
          renewal_accumulation = renewal_rate * (1 - exp(-total_rate * t)) / total_rate
        """
        vol = bet_volume_per_day or self.bet_volume_per_day

        # Base exponential decay rate
        base_rate = np.log(2) / max(source.half_life_days, 1.0)

        # Additional decay channels
        market_impact = source.market_impact_sensitivity * self.market_share * vol
        competition = source.competition_sensitivity * 0.01  # 1% per day base rate
        adaptation = source.adaptation_sensitivity * 0.005   # 0.5% per day base rate

        total_rate = base_rate * (1.0 + market_impact + competition + adaptation)

        # Pure decay component
        decay_factor = np.exp(-total_rate * days_elapsed)
        decayed_edge = source.initial_edge_pct * decay_factor

        # Renewal component (if applicable)
        if source.is_renewable and source.renewal_rate > 0 and total_rate > 0:
            renewal_accumulation = (
                source.renewal_rate * (1.0 - np.exp(-total_rate * days_elapsed)) / total_rate
            )
            # Steady state: when decay = renewal
            steady_state = source.renewal_rate / total_rate
        else:
            renewal_accumulation = 0.0
            steady_state = 0.0

        current_edge = max(decayed_edge + renewal_accumulation, source.min_edge_floor)
        steady_state = max(steady_state, source.min_edge_floor)

        # Time to effective zero (within 10% of floor)
        target = source.min_edge_floor + 0.1 * (source.initial_edge_pct - source.min_edge_floor)
        if total_rate > 0 and source.initial_edge_pct > target:
            ratio = max((target - renewal_accumulation) / source.initial_edge_pct, 1e-10)
            if ratio > 0:
                days_to_zero = -np.log(ratio) / total_rate
            else:
                days_to_zero = float("inf")
        else:
            days_to_zero = float("inf")

        pct_remaining = (current_edge / max(source.initial_edge_pct, 1e-10)) * 100

        return DecayResult(
            source_name=source.name,
            category=source.category,
            initial_edge_pct=round(source.initial_edge_pct, 3),
            current_edge_pct=round(current_edge, 3),
            days_elapsed=days_elapsed,
            half_life_days=source.half_life_days,
            pct_remaining=round(pct_remaining, 1),
            total_decay_rate=round(total_rate, 6),
            days_to_zero=round(min(days_to_zero, 9999), 1),
            edge_floor=source.min_edge_floor,
            is_renewable=source.is_renewable,
            steady_state_edge=round(steady_state, 3),
        )

    def decay_all(self, days_elapsed: float) -> Dict[str, DecayResult]:
        """Compute decay for all edge sources."""
        return {
            name: self.decay_single(source, days_elapsed)
            for name, source in self.edge_sources.items()
        }

    def total_edge_over_time(
        self,
        days: int = 365,
        resolution_days: int = 1,
    ) -> Dict[str, Any]:
        """Compute total portfolio edge trajectory over time.

        Returns day-by-day total edge, per-source edges, and key milestones.
        """
        time_points = list(range(0, days + 1, resolution_days))
        total_edges = []
        source_edges: Dict[str, List[float]] = {name: [] for name in self.edge_sources}

        for day in time_points:
            day_total = 0.0
            for name, source in self.edge_sources.items():
                result = self.decay_single(source, float(day))
                source_edges[name].append(result.current_edge_pct)
                day_total += result.current_edge_pct
            total_edges.append(round(day_total, 3))

        # Find milestones
        initial_total = total_edges[0] if total_edges else 0.0
        half_edge_day = next(
            (time_points[i] for i, e in enumerate(total_edges) if e <= initial_total / 2),
            days,
        )

        # Profitable threshold (assume 2% total edge is minimum viable)
        min_viable = 2.0
        viable_until = next(
            (time_points[i] for i, e in enumerate(total_edges) if e < min_viable),
            days,
        )

        # Steady state (last 10% of trajectory average)
        if len(total_edges) > 10:
            steady_state_approx = float(np.mean(total_edges[-len(total_edges) // 10:]))
        else:
            steady_state_approx = total_edges[-1] if total_edges else 0.0

        return {
            "time_points": time_points,
            "total_edge": total_edges,
            "source_edges": {k: [round(v, 3) for v in vals] for k, vals in source_edges.items()},
            "initial_total_edge": round(initial_total, 2),
            "half_life_day": half_edge_day,
            "viable_until_day": viable_until,
            "steady_state_edge": round(steady_state_approx, 2),
            "milestones": {
                "50pct_edge_remaining": half_edge_day,
                "below_2pct_total": viable_until,
                "steady_state_reached": next(
                    (time_points[i] for i in range(len(total_edges) - 1)
                     if abs(total_edges[i] - steady_state_approx) < 0.1),
                    days,
                ),
            },
        }

    def edge_composition_at(self, days_elapsed: float) -> Dict[str, Any]:
        """Show edge composition at a specific point in time.

        Returns breakdown by category and source with contribution percentages.
        """
        results = self.decay_all(days_elapsed)
        total = sum(r.current_edge_pct for r in results.values())

        by_source = []
        by_category: Dict[str, float] = {}

        for name, result in sorted(results.items(), key=lambda x: x[1].current_edge_pct, reverse=True):
            pct_of_total = (result.current_edge_pct / max(total, 1e-10)) * 100
            by_source.append({
                "source": name,
                "category": result.category,
                "edge_pct": result.current_edge_pct,
                "pct_of_total": round(pct_of_total, 1),
                "pct_remaining_of_initial": result.pct_remaining,
                "steady_state": result.steady_state_edge,
            })
            by_category[result.category] = by_category.get(result.category, 0) + result.current_edge_pct

        category_breakdown = [
            {"category": cat, "edge_pct": round(edge, 3),
             "pct_of_total": round((edge / max(total, 1e-10)) * 100, 1)}
            for cat, edge in sorted(by_category.items(), key=lambda x: x[1], reverse=True)
        ]

        return {
            "days_elapsed": days_elapsed,
            "total_edge_pct": round(total, 3),
            "by_source": by_source,
            "by_category": category_breakdown,
            "n_active_sources": sum(1 for r in results.values() if r.current_edge_pct > 0.05),
            "n_depleted_sources": sum(1 for r in results.values() if r.current_edge_pct <= 0.05),
        }
