"""
services/capital/kelly.py
==========================
Full / fractional / uncertainty-adjusted Kelly criterion calculator.

Kelly criterion gives the optimal bet size that maximizes long-run
geometric growth. But raw Kelly is dangerous because:
  1. It assumes perfect knowledge of true probabilities
  2. It produces aggressive sizing that causes large drawdowns
  3. It doesn't account for estimation uncertainty

This module provides:
  - Full Kelly (theoretical maximum growth)
  - Fractional Kelly (configurable fraction for safety)
  - Uncertainty-adjusted Kelly (shrinks based on confidence interval width)
  - Half-Kelly and quarter-Kelly shortcuts
  - Kelly with vig adjustment
  - Simultaneous Kelly for correlated bets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Output of a Kelly calculation."""
    full_kelly_pct: float           # Optimal Kelly fraction (% of bankroll)
    fractional_kelly_pct: float     # After applying fraction
    uncertainty_adjusted_pct: float  # After uncertainty penalty
    final_stake_pct: float          # Final recommended stake %
    final_stake_dollars: float      # Dollar amount
    edge_pct: float                 # Estimated edge
    win_probability: float          # Estimated win probability
    odds_decimal: float             # Decimal odds
    kelly_fraction_used: float      # Fraction applied (e.g., 0.25)
    uncertainty_penalty: float      # Reduction from uncertainty (0-1)
    is_positive_ev: bool            # Is the bet +EV at all?
    growth_rate: float              # Expected log growth rate per bet


class KellyCalculator:
    """Kelly criterion calculator with multiple adjustment modes.

    The core Kelly formula for a binary outcome:
        f* = (p * b - q) / b
    where:
        f* = fraction of bankroll to bet
        p  = probability of winning
        q  = 1 - p (probability of losing)
        b  = net odds (profit per dollar wagered on a win)
    """

    def __init__(
        self,
        default_fraction: float = 0.25,
        max_bet_pct: float = 5.0,
        min_edge_pct: float = 2.0,
        uncertainty_scaling: float = 1.0,
    ):
        """
        Args:
            default_fraction: Default Kelly fraction (0.25 = quarter-Kelly).
            max_bet_pct: Hard cap on any single bet (% of bankroll).
            min_edge_pct: Minimum edge required to bet.
            uncertainty_scaling: How aggressively to penalize uncertainty.
        """
        self.default_fraction = default_fraction
        self.max_bet_pct = max_bet_pct
        self.min_edge_pct = min_edge_pct
        self.uncertainty_scaling = uncertainty_scaling

    def full_kelly(
        self,
        win_prob: float,
        odds_decimal: float,
    ) -> float:
        """Compute raw full Kelly fraction.

        Args:
            win_prob: Probability of winning (0-1).
            odds_decimal: Decimal odds (e.g., 1.91 for -110).

        Returns:
            Optimal Kelly fraction (can be negative for -EV bets).
        """
        if odds_decimal <= 1.0:
            return 0.0
        b = odds_decimal - 1.0  # Net odds
        q = 1.0 - win_prob
        f_star = (win_prob * b - q) / b
        return f_star

    def fractional_kelly(
        self,
        win_prob: float,
        odds_decimal: float,
        fraction: float | None = None,
    ) -> float:
        """Compute fractional Kelly.

        Args:
            win_prob: Probability of winning.
            odds_decimal: Decimal odds.
            fraction: Kelly fraction to use (default: self.default_fraction).
        """
        frac = fraction if fraction is not None else self.default_fraction
        fk = self.full_kelly(win_prob, odds_decimal)
        return max(fk * frac, 0.0)

    def uncertainty_adjusted_kelly(
        self,
        win_prob: float,
        odds_decimal: float,
        prob_std: float,
        fraction: float | None = None,
    ) -> Tuple[float, float]:
        """Compute Kelly with uncertainty adjustment.

        When our probability estimate has a wide confidence interval,
        we should bet less. This uses the lower bound of a 1-sigma
        interval as the effective probability.

        Args:
            win_prob: Point estimate of win probability.
            odds_decimal: Decimal odds.
            prob_std: Standard deviation of probability estimate.
            fraction: Kelly fraction.

        Returns:
            Tuple of (adjusted_kelly_pct, uncertainty_penalty).
        """
        frac = fraction if fraction is not None else self.default_fraction

        # Use lower bound of confidence interval
        penalty = self.uncertainty_scaling * prob_std
        effective_prob = win_prob - penalty
        effective_prob = np.clip(effective_prob, 0.01, 0.99)

        adjusted_kelly = self.full_kelly(effective_prob, odds_decimal) * frac
        adjusted_kelly = max(adjusted_kelly, 0.0)

        base_kelly = self.fractional_kelly(win_prob, odds_decimal, frac)
        uncertainty_penalty = 1.0 - (adjusted_kelly / max(base_kelly, 1e-10))
        uncertainty_penalty = np.clip(uncertainty_penalty, 0.0, 1.0)

        return adjusted_kelly, float(uncertainty_penalty)

    def compute(
        self,
        win_prob: float,
        odds_decimal: float,
        bankroll: float,
        prob_std: float = 0.0,
        fraction: float | None = None,
    ) -> KellyResult:
        """Full Kelly computation with all adjustments.

        Args:
            win_prob: Estimated win probability.
            odds_decimal: Decimal odds offered.
            bankroll: Current bankroll.
            prob_std: Uncertainty in probability estimate.
            fraction: Kelly fraction (default: self.default_fraction).

        Returns:
            KellyResult with all sizing information.
        """
        frac = fraction if fraction is not None else self.default_fraction

        # Edge calculation
        b = odds_decimal - 1.0 if odds_decimal > 1.0 else 0.0
        implied_prob = 1.0 / odds_decimal if odds_decimal > 0 else 0.5
        edge_pct = (win_prob - implied_prob) * 100.0

        is_positive_ev = edge_pct > 0

        # Full Kelly
        full_k = self.full_kelly(win_prob, odds_decimal)
        full_k_pct = max(full_k * 100, 0.0)

        # Fractional Kelly
        frac_k_pct = max(full_k * frac * 100, 0.0)

        # Uncertainty adjustment
        if prob_std > 0:
            ua_kelly, u_penalty = self.uncertainty_adjusted_kelly(
                win_prob, odds_decimal, prob_std, frac
            )
            ua_pct = max(ua_kelly * 100, 0.0)
        else:
            ua_pct = frac_k_pct
            u_penalty = 0.0

        # Apply minimum edge filter
        if edge_pct < self.min_edge_pct:
            final_pct = 0.0
        else:
            final_pct = ua_pct

        # Apply max cap
        final_pct = min(final_pct, self.max_bet_pct)

        # Dollar amount
        final_dollars = bankroll * (final_pct / 100.0)

        # Expected growth rate
        if final_pct > 0 and odds_decimal > 1.0:
            f = final_pct / 100.0
            growth = win_prob * np.log(1 + f * b) + (1 - win_prob) * np.log(1 - f)
        else:
            growth = 0.0

        return KellyResult(
            full_kelly_pct=round(full_k_pct, 4),
            fractional_kelly_pct=round(frac_k_pct, 4),
            uncertainty_adjusted_pct=round(ua_pct, 4),
            final_stake_pct=round(final_pct, 4),
            final_stake_dollars=round(final_dollars, 2),
            edge_pct=round(edge_pct, 3),
            win_probability=round(win_prob, 4),
            odds_decimal=odds_decimal,
            kelly_fraction_used=frac,
            uncertainty_penalty=round(u_penalty, 4),
            is_positive_ev=is_positive_ev,
            growth_rate=round(float(growth), 6),
        )

    def compute_batch(
        self,
        bets: List[Dict[str, float]],
        bankroll: float,
        fraction: float | None = None,
    ) -> List[KellyResult]:
        """Compute Kelly for multiple bets.

        Each bet dict should have: win_prob, odds_decimal, prob_std (optional).
        """
        results = []
        for bet in bets:
            result = self.compute(
                win_prob=bet["win_prob"],
                odds_decimal=bet["odds_decimal"],
                bankroll=bankroll,
                prob_std=bet.get("prob_std", 0.0),
                fraction=fraction,
            )
            results.append(result)
        return results

    def simultaneous_kelly(
        self,
        win_probs: np.ndarray,
        odds_decimal: np.ndarray,
        correlation_matrix: np.ndarray,
        bankroll: float,
        fraction: float | None = None,
    ) -> Dict[str, Any]:
        """Kelly sizing for simultaneous correlated bets.

        When bets are correlated, independent Kelly oversizes the portfolio.
        This adjusts for correlation by scaling down when correlation is high.

        Uses the approximation:
          f_adj_i = f_kelly_i / (1 + sum_j(rho_ij * f_kelly_j))
        """
        frac = fraction if fraction is not None else self.default_fraction
        n = len(win_probs)

        # Individual Kelly fractions
        individual_kellys = np.array([
            max(self.full_kelly(float(win_probs[i]), float(odds_decimal[i])) * frac, 0.0)
            for i in range(n)
        ])

        # Correlation adjustment
        adjusted_kellys = np.zeros(n)
        for i in range(n):
            if individual_kellys[i] <= 0:
                continue
            corr_sum = 0.0
            for j in range(n):
                if i != j and individual_kellys[j] > 0:
                    corr_sum += abs(correlation_matrix[i, j]) * individual_kellys[j]
            denominator = 1.0 + corr_sum
            adjusted_kellys[i] = individual_kellys[i] / denominator

        # Cap total exposure
        total_exposure = np.sum(adjusted_kellys)
        if total_exposure > 0.15:  # Max 15% total exposure
            scale = 0.15 / total_exposure
            adjusted_kellys *= scale

        return {
            "individual_kelly_pcts": [round(k * 100, 4) for k in individual_kellys],
            "adjusted_kelly_pcts": [round(k * 100, 4) for k in adjusted_kellys],
            "dollar_amounts": [round(k * bankroll, 2) for k in adjusted_kellys],
            "total_exposure_pct": round(float(np.sum(adjusted_kellys)) * 100, 2),
            "correlation_reduction_pct": round(
                (1.0 - float(np.sum(adjusted_kellys)) / max(float(np.sum(individual_kellys)), 1e-10)) * 100, 1
            ),
        }

    def kelly_from_american_odds(
        self,
        win_prob: float,
        american_odds: int,
        bankroll: float,
        prob_std: float = 0.0,
        fraction: float | None = None,
    ) -> KellyResult:
        """Convenience method taking American odds instead of decimal."""
        if american_odds > 0:
            decimal = 1.0 + american_odds / 100.0
        elif american_odds < 0:
            decimal = 1.0 + 100.0 / abs(american_odds)
        else:
            decimal = 2.0
        return self.compute(win_prob, decimal, bankroll, prob_std, fraction)
