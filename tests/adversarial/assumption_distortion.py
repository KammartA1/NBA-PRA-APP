"""
tests/adversarial/assumption_distortion.py
============================================
Distort each hardcoded assumption by +/-20% and measure impact.

Every model has implicit assumptions (vig amount, market efficiency,
correlation structure, etc.). This test systematically distorts each
one to find which assumptions the model is most sensitive to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class AssumptionTest:
    """Result of distorting a single assumption."""
    assumption_name: str
    baseline_value: float
    distorted_value: float
    distortion_pct: float
    baseline_metric: float    # ROI or Brier at baseline
    distorted_metric: float   # ROI or Brier after distortion
    impact_pct: float         # % change in metric
    sensitivity: str          # low, medium, high, critical
    description: str


# Hardcoded assumptions in the system
SYSTEM_ASSUMPTIONS = {
    "vig_pct": {
        "baseline": 4.76,        # Standard -110 vig (~4.76%)
        "description": "Sportsbook vig percentage on standard bets",
        "metric": "roi",
    },
    "market_efficiency": {
        "baseline": 0.95,        # Market is ~95% efficient
        "description": "How efficient the market is at pricing (0-1)",
        "metric": "edge",
    },
    "kelly_fraction": {
        "baseline": 0.25,        # Quarter-Kelly
        "description": "Fraction of Kelly criterion used for bet sizing",
        "metric": "growth",
    },
    "min_edge_threshold": {
        "baseline": 2.0,         # 2% minimum edge to bet
        "description": "Minimum edge percentage required to place a bet",
        "metric": "filter",
    },
    "slippage_cents": {
        "baseline": 0.5,         # 0.5 cents average slippage
        "description": "Average line slippage between signal and execution",
        "metric": "execution_cost",
    },
    "closing_line_accuracy": {
        "baseline": 0.98,        # Closing line is 98% accurate
        "description": "How accurate our closing line capture is",
        "metric": "clv_reliability",
    },
    "same_game_correlation": {
        "baseline": 0.35,        # 35% correlation between same-game bets
        "description": "Assumed correlation between bets in same game",
        "metric": "portfolio_risk",
    },
    "max_drawdown_limit": {
        "baseline": 25.0,        # 25% max drawdown before halt
        "description": "Maximum allowed drawdown before system halts",
        "metric": "risk_tolerance",
    },
    "win_probability_calibration": {
        "baseline": 1.0,         # Calibration factor (1.0 = perfectly calibrated)
        "description": "Multiplicative calibration factor for win probabilities",
        "metric": "calibration",
    },
    "execution_speed_seconds": {
        "baseline": 30.0,        # 30 seconds from signal to bet
        "description": "Time between signal generation and bet placement",
        "metric": "execution_cost",
    },
}


class AssumptionDistortionTest:
    """Distort system assumptions and measure impact."""

    def __init__(self, distortion_pcts: List[float] | None = None):
        """
        Args:
            distortion_pcts: List of distortion percentages to test.
                Default: [-20, -10, +10, +20]
        """
        self.distortion_pcts = distortion_pcts or [-20.0, -10.0, 10.0, 20.0]

    def run(self) -> Dict[str, Any]:
        """Run distortion tests on all assumptions."""
        results: List[Dict[str, Any]] = []
        sensitivity_summary: Dict[str, str] = {}

        for name, config in SYSTEM_ASSUMPTIONS.items():
            assumption_results = self._test_assumption(name, config)
            results.extend(assumption_results)

            # Determine sensitivity (max impact across distortions)
            max_impact = max(abs(r["impact_pct"]) for r in assumption_results)
            if max_impact > 50:
                sensitivity = "critical"
            elif max_impact > 20:
                sensitivity = "high"
            elif max_impact > 5:
                sensitivity = "medium"
            else:
                sensitivity = "low"
            sensitivity_summary[name] = sensitivity

        # Overall assessment
        critical_count = sum(1 for s in sensitivity_summary.values() if s == "critical")
        high_count = sum(1 for s in sensitivity_summary.values() if s == "high")

        return {
            "status": "completed",
            "n_assumptions_tested": len(SYSTEM_ASSUMPTIONS),
            "distortion_levels": self.distortion_pcts,
            "results": results,
            "sensitivity_summary": sensitivity_summary,
            "critical_assumptions": [
                k for k, v in sensitivity_summary.items() if v == "critical"
            ],
            "high_sensitivity_assumptions": [
                k for k, v in sensitivity_summary.items() if v == "high"
            ],
            "overall_pass": critical_count == 0,
            "verdict": "PASS" if critical_count == 0 else "FAIL",
            "interpretation": (
                f"{critical_count} CRITICAL assumptions found. "
                f"{high_count} high-sensitivity assumptions. "
                + ("System is robust to assumption errors." if critical_count == 0
                   else "System is FRAGILE to certain assumptions. Review critical items.")
            ),
        }

    def _test_assumption(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Test distortions for a single assumption."""
        baseline = config["baseline"]
        description = config["description"]
        metric_type = config["metric"]

        results = []
        for dist_pct in self.distortion_pcts:
            distorted = baseline * (1 + dist_pct / 100.0)

            # Compute impact based on metric type
            impact = self._compute_impact(name, baseline, distorted, metric_type)

            # Classify sensitivity
            abs_impact = abs(impact)
            if abs_impact > 50:
                sensitivity = "critical"
            elif abs_impact > 20:
                sensitivity = "high"
            elif abs_impact > 5:
                sensitivity = "medium"
            else:
                sensitivity = "low"

            results.append({
                "assumption": name,
                "description": description,
                "baseline_value": round(baseline, 4),
                "distorted_value": round(distorted, 4),
                "distortion_pct": dist_pct,
                "impact_pct": round(impact, 2),
                "sensitivity": sensitivity,
                "metric_type": metric_type,
            })

        return results

    def _compute_impact(
        self,
        name: str,
        baseline: float,
        distorted: float,
        metric_type: str,
    ) -> float:
        """Compute the impact of distorting an assumption.

        Uses analytical models for each assumption type rather than
        requiring a full simulation.
        """
        if name == "vig_pct":
            # Higher vig directly reduces ROI
            # ROI impact = -(distorted_vig - baseline_vig) / baseline_edge * 100
            assumed_edge = 4.0  # 4% assumed edge
            impact = -(distorted - baseline) / assumed_edge * 100
            return impact

        elif name == "market_efficiency":
            # Less efficient market = more exploitable edge
            # More efficient = less edge available
            baseline_inefficiency = 1 - baseline
            distorted_inefficiency = 1 - distorted
            if baseline_inefficiency > 0:
                impact = (distorted_inefficiency - baseline_inefficiency) / baseline_inefficiency * 100
            else:
                impact = 0.0
            return impact

        elif name == "kelly_fraction":
            # Kelly fraction affects growth rate
            # f=0.25 → growth proportional to edge^2 * fraction
            growth_baseline = baseline * (1 - baseline)
            growth_distorted = distorted * (1 - distorted)
            if growth_baseline > 0:
                impact = (growth_distorted - growth_baseline) / growth_baseline * 100
            else:
                impact = 0.0
            return impact

        elif name == "min_edge_threshold":
            # Higher threshold = fewer bets taken = potentially higher quality but lower volume
            # Assume 30% of bets are between 2-3% edge
            if distorted > baseline:
                lost_bets_pct = (distorted - baseline) / baseline * 30
                impact = -lost_bets_pct  # Negative = fewer bets
            else:
                gained_bets_pct = (baseline - distorted) / baseline * 30
                impact = gained_bets_pct * 0.5  # More bets but lower quality
            return impact

        elif name == "slippage_cents":
            # More slippage directly reduces edge
            assumed_clv = 1.5  # Average CLV in cents
            slippage_cost_change = distorted - baseline
            impact = -(slippage_cost_change / assumed_clv) * 100
            return impact

        elif name == "closing_line_accuracy":
            # Lower accuracy means CLV measurements are unreliable
            accuracy_change = distorted - baseline
            impact = accuracy_change * 200  # 1% accuracy = ~2% impact on measured CLV
            return impact

        elif name == "same_game_correlation":
            # Higher correlation = more portfolio risk
            # Portfolio variance scales with correlation
            var_baseline = 1 + baseline * 5  # Approximate for 5 correlated bets
            var_distorted = 1 + distorted * 5
            impact = (var_distorted - var_baseline) / var_baseline * 100
            return -impact  # More variance = worse

        elif name == "max_drawdown_limit":
            # Tighter limit = stopped out more often
            # Looser limit = more risk tolerance
            if distorted < baseline:
                # Tighter: more likely to halt
                impact = -(baseline - distorted) / baseline * 50
            else:
                # Looser: less likely to halt but more risk
                impact = (distorted - baseline) / baseline * 20
            return impact

        elif name == "win_probability_calibration":
            # Miscalibration directly affects bet quality
            miscalibration = abs(distorted - 1.0)
            impact = -miscalibration * 100  # Each 1% miscalibration = 1% ROI impact
            return impact

        elif name == "execution_speed_seconds":
            # Slower execution = more slippage
            speed_change = distorted - baseline
            # Each 10 seconds delay = ~0.1 cents more slippage
            additional_slippage = speed_change / 10 * 0.1
            impact = -(additional_slippage / 1.5) * 100  # Relative to average CLV
            return impact

        else:
            return 0.0
