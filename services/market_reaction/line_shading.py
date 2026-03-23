"""
services/market_reaction/line_shading.py
=========================================
Models price shading — how books adjust lines specifically against
identified sharp bettors.

Once a book flags you as sharp, they don't just limit your bet size.
They also shade the line against you: if the true line is 25.5, they
show you 25.0 (for overs) or 26.0 (for unders). This eats into your
edge without requiring a formal limit.

This module models:
  - Shading magnitude as a function of sharp score
  - Edge erosion from shading
  - Detection of shading from historical bet data
  - Net effective edge after shading adjustments
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)


@dataclass
class ShadingEstimate:
    """Estimated shading applied to a bettor."""
    avg_shading_cents: float        # Average line shading in cents/points
    shading_std: float              # Std dev of shading
    shading_pct_of_edge: float      # What % of edge does shading consume?
    effective_edge_pct: float       # Edge after shading
    confidence: float               # Confidence in estimate (0-1)
    n_observations: int             # Data points used
    detection_method: str           # How we detected the shading
    is_significant: bool            # Statistically significant?
    p_value: float                  # p-value for shading being > 0


@dataclass
class ShadingProfile:
    """Shading parameters for simulation purposes."""
    base_shading_cents: float = 0.0
    sharp_score_coefficient: float = 0.02   # Cents of shading per sharp score point
    max_shading_cents: float = 3.0
    shading_ramp_bets: int = 100            # Bets over which shading ramps up
    market_specific_multipliers: Dict[str, float] = None

    def __post_init__(self):
        if self.market_specific_multipliers is None:
            self.market_specific_multipliers = {
                "player_points": 1.0,
                "player_rebounds": 1.2,     # Thinner markets get more shading
                "player_assists": 1.2,
                "player_threes": 1.5,
                "player_points_rebounds_assists": 0.8,  # Combo markets, less shading
            }


class LineShadingModel:
    """Models sportsbook line shading against sharp bettors.

    Estimates how much edge is being consumed by adverse line presentation
    and projects future shading levels.
    """

    def __init__(self, profile: ShadingProfile | None = None):
        self.profile = profile or ShadingProfile()

    def estimate_shading(
        self,
        bet_lines: np.ndarray,
        market_lines: np.ndarray,
        closing_lines: np.ndarray,
        directions: List[str],
    ) -> ShadingEstimate:
        """Detect line shading from historical data.

        Compares the lines offered to this bettor vs the market consensus
        at the time of bet. Systematic deviation = shading.

        Args:
            bet_lines: Lines shown to this bettor.
            market_lines: Market consensus lines at bet time.
            closing_lines: Closing lines for reference.
            directions: "over" or "under" for each bet.
        """
        n = len(bet_lines)
        if n < 10:
            return ShadingEstimate(
                avg_shading_cents=0.0,
                shading_std=0.0,
                shading_pct_of_edge=0.0,
                effective_edge_pct=0.0,
                confidence=0.0,
                n_observations=n,
                detection_method="insufficient_data",
                is_significant=False,
                p_value=1.0,
            )

        # Compute shading: for overs, shading = market_line - bet_line (positive = shaded against)
        # For unders, shading = bet_line - market_line
        shading_values = np.zeros(n)
        for i in range(n):
            if directions[i].lower() == "over":
                shading_values[i] = market_lines[i] - bet_lines[i]
            else:
                shading_values[i] = bet_lines[i] - market_lines[i]

        avg_shading = float(np.mean(shading_values))
        std_shading = float(np.std(shading_values, ddof=1)) if n > 1 else 0.0

        # Statistical test: is shading significantly > 0?
        if std_shading > 0 and n >= 20:
            t_stat, p_value = sp_stats.ttest_1samp(shading_values, 0.0)
            # One-sided test: is shading positive?
            p_one_sided = p_value / 2 if t_stat > 0 else 1.0 - p_value / 2
            is_significant = p_one_sided < 0.05
        else:
            p_one_sided = 1.0
            is_significant = False

        # Estimate edge erosion
        # CLV as proxy for true edge
        clv_values = np.zeros(n)
        for i in range(n):
            if directions[i].lower() == "over":
                clv_values[i] = closing_lines[i] - bet_lines[i]
            else:
                clv_values[i] = bet_lines[i] - closing_lines[i]

        avg_clv = float(np.mean(clv_values))
        # Without shading, CLV would have been higher
        unshaded_clv = avg_clv + avg_shading
        shading_pct = (avg_shading / max(abs(unshaded_clv), 0.01)) * 100 if unshaded_clv != 0 else 0.0

        # Confidence based on sample size and effect size
        if n >= 100 and is_significant:
            confidence = 0.9
        elif n >= 50 and is_significant:
            confidence = 0.7
        elif n >= 20:
            confidence = 0.4
        else:
            confidence = 0.2

        return ShadingEstimate(
            avg_shading_cents=round(avg_shading, 3),
            shading_std=round(std_shading, 3),
            shading_pct_of_edge=round(shading_pct, 1),
            effective_edge_pct=round(max(unshaded_clv - avg_shading, 0.0), 3),
            confidence=confidence,
            n_observations=n,
            detection_method="market_consensus_comparison",
            is_significant=is_significant,
            p_value=round(p_one_sided, 4),
        )

    def predict_shading(
        self,
        sharp_score: float,
        total_bets: int,
        market_type: str = "player_points",
    ) -> Dict[str, float]:
        """Predict expected shading for a bettor with given characteristics.

        Args:
            sharp_score: Book's internal sharpness score (0-100).
            total_bets: Number of bets placed.
            market_type: Type of market being bet.

        Returns:
            Dict with predicted shading metrics.
        """
        # Base shading from sharp score
        base = self.profile.base_shading_cents
        score_component = sharp_score * self.profile.sharp_score_coefficient

        # Ramp-up: shading increases as book gets more data
        ramp = min(total_bets / max(self.profile.shading_ramp_bets, 1), 1.0)

        # Market multiplier
        market_mult = self.profile.market_specific_multipliers.get(market_type, 1.0)

        # Total predicted shading
        predicted = (base + score_component) * ramp * market_mult
        predicted = min(predicted, self.profile.max_shading_cents)

        return {
            "predicted_shading_cents": round(predicted, 3),
            "base_component": round(base, 3),
            "sharp_score_component": round(score_component * ramp, 3),
            "market_multiplier": market_mult,
            "ramp_factor": round(ramp, 3),
            "at_max": predicted >= self.profile.max_shading_cents,
        }

    def shading_trajectory(
        self,
        sharp_score: float,
        market_type: str = "player_points",
        months: int = 24,
        bets_per_month: float = 90.0,
    ) -> Dict[str, Any]:
        """Project shading over time as bet count accumulates.

        Returns month-by-month shading projections.
        """
        trajectory: List[Dict[str, float]] = []
        cumulative_bets = 0

        for month in range(1, months + 1):
            cumulative_bets += int(bets_per_month)
            pred = self.predict_shading(sharp_score, cumulative_bets, market_type)
            trajectory.append({
                "month": month,
                "cumulative_bets": cumulative_bets,
                "shading_cents": pred["predicted_shading_cents"],
                "ramp_factor": pred["ramp_factor"],
            })

        return {
            "market_type": market_type,
            "sharp_score": sharp_score,
            "trajectory": trajectory,
            "months_to_max_shading": next(
                (t["month"] for t in trajectory if t["shading_cents"] >= self.profile.max_shading_cents * 0.95),
                months,
            ),
        }

    def compute_net_edge(
        self,
        gross_edge_pct: float,
        shading_cents: float,
        avg_line: float = 25.0,
    ) -> Dict[str, float]:
        """Compute net edge after accounting for shading.

        Args:
            gross_edge_pct: True edge before shading (%).
            shading_cents: Expected shading in line-space.
            avg_line: Average line level for conversion.
        """
        # Convert shading from line-space to probability-space
        # Rough approximation: 1 point on a 25-point line ~= 4% probability shift
        shading_pct = (shading_cents / max(avg_line, 1.0)) * 100.0

        net_edge = gross_edge_pct - shading_pct
        erosion_pct = (shading_pct / max(gross_edge_pct, 0.01)) * 100

        return {
            "gross_edge_pct": round(gross_edge_pct, 2),
            "shading_cost_pct": round(shading_pct, 2),
            "net_edge_pct": round(net_edge, 2),
            "edge_erosion_pct": round(erosion_pct, 1),
            "still_profitable": net_edge > 0,
            "min_edge_needed": round(shading_pct, 2),
        }
