"""
edge_analysis.sources.pace_differential
=======================================
Team pace matchup creates over/under opportunities.  Model expected possessions
from both teams' pace and the differential impact on each stat category.
Market edge: books use season-average pace, not matchup-specific pace.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


# League-wide pace constants (2023-24 season calibrated)
_LEAGUE_AVG_PACE: float = 99.5
_PACE_STD: float = 3.2


@dataclass
class PaceDifferentialSource:
    """Signal source based on matchup-specific pace differentials."""

    name: str = "pace_differential"
    category: str = "game_environment"
    description: str = (
        "Models expected possessions from team pace matchup, with differential "
        "impact per stat category. Captures pace-up and pace-down effects."
    )

    def get_signal(self, player: dict, game_context: dict) -> float:
        """
        Compute pace-adjusted PRA signal.
        Positive = pace environment favors over, negative = favors under.
        """
        team_pace = float(game_context.get("team_pace", _LEAGUE_AVG_PACE))
        opp_pace = float(game_context.get("opp_pace", _LEAGUE_AVG_PACE))

        # --- Expected game pace (harmonic-style blend) ---
        # NBA game pace is approximately the average of the two teams' paces,
        # weighted toward the home team slightly
        is_home = bool(game_context.get("is_home", True))
        home_weight = 0.52 if is_home else 0.48
        expected_pace = home_weight * team_pace + (1 - home_weight) * opp_pace

        # Adjust for referee tendency if available
        ref_pace_adj = float(game_context.get("ref_pace_adjustment", 0.0))
        expected_pace += ref_pace_adj

        # --- Pace delta from season average ---
        season_pace = float(player.get("team_season_pace", _LEAGUE_AVG_PACE))
        pace_delta = expected_pace - season_pace
        pace_delta_pct = pace_delta / max(season_pace, 80.0)

        # --- Stat-specific pace sensitivity ---
        # Points scale nearly linearly with pace
        # Rebounds scale sub-linearly (more misses but also more fast breaks)
        # Assists scale with pace but less directly
        stat_type = player.get("stat_type", "pra")
        sensitivity = self._get_stat_sensitivity(stat_type)

        # --- Player-specific pace response ---
        # Players with higher usage benefit more from pace-up
        usage = float(player.get("usage_rate", 0.20))
        usage_multiplier = 1.0 + (usage - 0.20) * 2.0  # higher usage = more pace effect

        # --- Expected PRA delta ---
        base_pra = float(player.get("pra_avg", 30.0))
        pra_delta = base_pra * pace_delta_pct * sensitivity * usage_multiplier

        # Normalize
        pra_std = float(player.get("pra_std", 7.0))
        signal = pra_delta / max(pra_std, 1.0)

        return float(np.clip(signal, -3.0, 3.0) / 3.0)

    @staticmethod
    def _get_stat_sensitivity(stat_type: str) -> float:
        """
        How sensitive each stat category is to pace changes.
        1.0 = linear scaling with pace.
        """
        sensitivities = {
            "points": 0.90,
            "rebounds": 0.55,
            "assists": 0.70,
            "pra": 0.75,
            "pr": 0.72,
            "pa": 0.80,
            "ra": 0.62,
            "threes": 0.85,
            "steals": 0.60,
            "blocks": 0.40,
            "turnovers": 0.80,
            "fantasy": 0.78,
        }
        return sensitivities.get(stat_type.lower(), 0.75)

    def get_mechanism(self) -> str:
        return (
            "Books set lines using season-average pace context. When two teams "
            "with extreme pace tendencies meet, the expected possessions deviate "
            "significantly from the season average. A pace-up matchup (e.g., "
            "two top-5 pace teams) can add 5+ possessions, boosting PRA by 3-5%. "
            "We model this matchup-specific pace with stat-specific sensitivity "
            "(points scale ~0.9x with pace, rebounds ~0.55x) and player usage "
            "interaction. The edge compounds when the market uses a generic "
            "season-level adjustment."
        )

    def get_decay_risk(self) -> str:
        return (
            "Medium-High. Pace data is publicly available, and sophisticated "
            "models already incorporate it. The edge is in the stat-specific "
            "sensitivity modeling and the usage interaction, not raw pace data. "
            "Expect 50% decay over 1-2 seasons."
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
