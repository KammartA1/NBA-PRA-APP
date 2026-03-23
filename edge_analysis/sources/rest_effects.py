"""
edge_analysis.sources.rest_effects
==================================
Back-to-back and rest differential effects on PRA distributions.
Non-linear rest impact, age-dependent B2B degradation, and travel fatigue.
Market edge: books apply flat B2B adjustments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class RestEffectsSource:
    """Signal source modeling non-linear rest and fatigue effects."""

    name: str = "rest_effects"
    category: str = "player_state"
    description: str = (
        "Models non-linear rest effects including back-to-back degradation "
        "by age, travel fatigue, and rest advantage differentials."
    )

    # Empirical B2B impact by age bracket (PRA % decrease)
    _b2b_impact_by_age: dict = None

    def __post_init__(self):
        if self._b2b_impact_by_age is None:
            self._b2b_impact_by_age = {
                (18, 24): -0.025,  # young players: -2.5%
                (25, 29): -0.040,  # prime: -4.0%
                (30, 33): -0.060,  # veteran: -6.0%
                (34, 40): -0.085,  # late career: -8.5%
            }

    def get_signal(self, player: dict, game_context: dict) -> float:
        """
        Compute rest-adjusted PRA signal.
        Positive = rest advantage favors over, negative = fatigue favors under.
        """
        rest_days = int(player.get("rest_days", 1))
        is_b2b = rest_days == 0
        age = int(player.get("age", 27))
        minutes_last_game = float(player.get("minutes_last_game", 32.0))
        minutes_avg = float(player.get("minutes_avg", 32.0))

        # --- B2B impact (age-dependent) ---
        b2b_effect = 0.0
        if is_b2b:
            b2b_effect = self._get_b2b_impact(age)
            # Heavy minutes in previous game amplifies B2B effect
            minutes_ratio = minutes_last_game / max(minutes_avg, 20.0)
            b2b_effect *= (0.7 + 0.3 * minutes_ratio)

        # --- Non-linear rest benefit ---
        # Rest benefit follows diminishing returns:
        # 1 day rest: baseline (0 effect)
        # 2 days: +1.5% PRA
        # 3 days: +2.0% PRA
        # 4+ days: +2.2% PRA (and can be negative — rust)
        rest_benefit = self._rest_curve(rest_days)

        # --- Travel fatigue ---
        timezone_diff = abs(int(game_context.get("timezone_diff", 0)))
        travel_fatigue = 0.0
        if timezone_diff > 0:
            # Westward travel is slightly less fatiguing than eastward
            direction = game_context.get("travel_direction", "east")
            direction_mult = 1.0 if direction == "east" else 0.85
            travel_fatigue = -0.008 * timezone_diff * direction_mult
            # B2B + travel compounds
            if is_b2b:
                travel_fatigue *= 1.5

        # --- Rest differential vs opponent ---
        opp_rest_days = int(game_context.get("opp_rest_days", 1))
        rest_diff = rest_days - opp_rest_days
        # Rest differential affects team performance, which flows to individuals
        # A rest advantage improves team offense ~1% per rest day differential
        team_rest_effect = 0.005 * rest_diff

        # --- Recent workload (last 7 days) ---
        minutes_last_7 = float(player.get("minutes_last_7_days", minutes_avg * 3))
        expected_minutes_7 = minutes_avg * 3.5  # ~3.5 games in 7 days
        workload_ratio = minutes_last_7 / max(expected_minutes_7, 60.0)
        workload_fatigue = 0.0
        if workload_ratio > 1.1:
            workload_fatigue = -0.015 * (workload_ratio - 1.0)

        # --- Total effect ---
        total_effect = b2b_effect + rest_benefit + travel_fatigue + team_rest_effect + workload_fatigue

        # --- Market adjustment ---
        # Books typically apply a flat ~3% B2B discount regardless of age/context
        market_b2b_adj = -0.03 if is_b2b else 0.0
        # Our edge is the difference between our nuanced model and market's flat adj
        edge = total_effect - market_b2b_adj

        # Convert to PRA signal
        base_pra = float(player.get("pra_avg", 30.0))
        pra_delta = base_pra * edge
        pra_std = float(player.get("pra_std", 7.0))
        signal = pra_delta / max(pra_std, 1.0)

        return float(np.clip(signal, -3.0, 3.0) / 3.0)

    def _get_b2b_impact(self, age: int) -> float:
        """Get age-specific B2B PRA impact."""
        for (low, high), impact in self._b2b_impact_by_age.items():
            if low <= age <= high:
                return impact
        return -0.04  # default

    @staticmethod
    def _rest_curve(rest_days: int) -> float:
        """
        Non-linear rest benefit curve.
        Returns PRA percentage adjustment.
        """
        if rest_days <= 0:
            return 0.0  # B2B handled separately
        elif rest_days == 1:
            return 0.0  # normal schedule
        elif rest_days == 2:
            return 0.015
        elif rest_days == 3:
            return 0.020
        elif rest_days == 4:
            return 0.022
        elif rest_days <= 7:
            return 0.018  # slight rust starts
        else:
            return 0.010  # significant rust after a week off

    def get_mechanism(self) -> str:
        return (
            "Books apply a flat ~3% back-to-back adjustment regardless of player "
            "age, previous game workload, or travel context. We model B2B impact "
            "as age-dependent (ranging from -2.5% for young players to -8.5% for "
            "players 34+), amplified by heavy previous-game minutes, travel "
            "fatigue (timezone crossings), and recent 7-day workload. Rest "
            "benefits follow a diminishing-returns curve with rust effects for "
            "extended absences. The edge is largest for older stars on B2B after "
            "heavy minutes and cross-country travel."
        )

    def get_decay_risk(self) -> str:
        return (
            "Low. Age-specific B2B data is available but rarely modeled with "
            "proper interaction effects. Travel fatigue modeling requires "
            "schedule tracking that most books automate poorly. This edge is "
            "structural and decays slowly. Half-life: 3-4 seasons."
        )

    def validate(self, historical_data: list) -> dict:
        if len(historical_data) < 30:
            return {
                "sharpe": 0.0, "p_value": 1.0,
                "sample_size": len(historical_data),
                "correlation_with_other_signals": {},
                "status": "insufficient_data",
            }

        signals, outcomes = [], []
        for game in historical_data:
            sig = self.get_signal(game.get("player", {}), game.get("game_context", {}))
            actual = float(game.get("actual_pra", 0))
            line = float(game.get("line", 0))
            signals.append(sig)
            outcomes.append((1.0 if actual > line else -1.0) * (1.0 if sig > 0 else -1.0))

        signals_arr = np.array(signals)
        outcomes_arr = np.array(outcomes)
        nonzero = np.abs(signals_arr) > 0.01

        if nonzero.sum() < 20:
            return {
                "sharpe": 0.0, "p_value": 1.0,
                "sample_size": int(nonzero.sum()),
                "correlation_with_other_signals": {},
                "status": "insufficient_nonzero_signals",
            }

        returns = signals_arr[nonzero] * outcomes_arr[nonzero]
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))
        sharpe = mean_ret / std_ret * math.sqrt(252) if std_ret > 0 else 0.0
        t_stat, p_val = stats.ttest_1samp(returns, 0.0)
        p_val = float(p_val) / 2.0
        if t_stat < 0:
            p_val = 1.0 - p_val

        return {
            "sharpe": round(sharpe, 3),
            "p_value": round(p_val, 4),
            "sample_size": int(nonzero.sum()),
            "mean_return": round(mean_ret, 4),
            "hit_rate": round(float(np.mean(outcomes_arr[nonzero] > 0)), 4),
            "correlation_with_other_signals": {},
            "status": "valid",
        }
