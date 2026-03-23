"""
edge_analysis.sources.usage_redistribution
===========================================
Model how usage redistributes when key players are absent.  Track historical
usage rates with/without each teammate to compute the usage boost.
Market edge: books adjust slowly to lineup news.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class UsageRedistributionSource:
    """Signal source based on usage rate redistribution when teammates are out."""

    name: str = "usage_redistribution"
    category: str = "opportunity"
    description: str = (
        "Models how usage rate and shot attempts redistribute when key "
        "teammates are absent, using historical with/without splits."
    )

    # League-average usage rate
    _league_avg_usage: float = 0.20
    # Typical redistribution elasticity
    _redistribution_elasticity: float = 0.65

    def get_signal(self, player: dict, game_context: dict) -> float:
        """
        Compute usage redistribution signal.
        Positive = expected usage boost (over signal), negative = usage drain.
        """
        base_usage = float(player.get("usage_rate", self._league_avg_usage))
        teammates_out = game_context.get("teammates_out", [])
        teammates_in = game_context.get("teammates_in", [])

        if not teammates_out and not teammates_in:
            return 0.0

        # --- Compute usage pool freed by absent players ---
        freed_usage = 0.0
        for tm in teammates_out:
            tm_usage = float(tm.get("usage_rate", self._league_avg_usage))
            tm_minutes_share = float(tm.get("minutes_share", 0.5))
            freed_usage += tm_usage * tm_minutes_share

        # --- Compute usage taken by newly inserted players ---
        taken_usage = 0.0
        for tm in teammates_in:
            tm_usage = float(tm.get("usage_rate", self._league_avg_usage * 0.7))
            tm_minutes_share = float(tm.get("minutes_share", 0.3))
            taken_usage += tm_usage * tm_minutes_share

        net_freed = freed_usage - taken_usage

        # --- Player's share of redistributed usage ---
        # Higher usage players absorb more of the freed usage
        player_usage_rank = float(player.get("usage_rank_on_team", 3))
        # Rank-based absorption: rank 1 gets ~35%, rank 2 ~25%, rank 3 ~15%, etc.
        absorption_rate = max(0.05, 0.45 - 0.10 * player_usage_rank)

        # Historical with/without data (if available)
        usage_with = player.get("usage_with_absent", None)
        usage_without = player.get("usage_without_absent", None)
        if usage_with is not None and usage_without is not None:
            historical_boost = float(usage_with) - float(usage_without)
            n_games_with = int(player.get("games_without_teammate", 5))
            # Bayesian blend: trust historical data more with more games
            blend_weight = min(1.0, n_games_with / 20.0)
            model_boost = net_freed * absorption_rate * self._redistribution_elasticity
            usage_boost = blend_weight * historical_boost + (1 - blend_weight) * model_boost
        else:
            usage_boost = net_freed * absorption_rate * self._redistribution_elasticity

        # --- Convert usage change to PRA signal ---
        # Usage rate change * possessions * scoring efficiency
        expected_possessions = float(game_context.get("expected_possessions", 98.0))
        player_minutes_share = float(player.get("minutes_avg", 32.0)) / 48.0
        player_possessions = expected_possessions * player_minutes_share

        # Points per usage possession (league avg ~1.0 PPP)
        ppp = float(player.get("points_per_usage", 1.05))
        # Usage possessions also generate assists and rebounds indirectly
        pra_per_usage = ppp + 0.35  # ~0.35 combined reb+ast per usage

        pra_delta = usage_boost * player_possessions * pra_per_usage

        # Normalize by typical PRA std (~6-8 for most players)
        pra_std = float(player.get("pra_std", 7.0))
        signal = pra_delta / max(pra_std, 1.0)

        return float(np.clip(signal, -3.0, 3.0) / 3.0)

    def get_mechanism(self) -> str:
        return (
            "When key players are absent, their usage must redistribute. Books "
            "adjust lines but typically apply a flat discount based on team-level "
            "impact. We model player-specific usage absorption based on historical "
            "with/without splits, Bayesian-blended with a structural model of "
            "usage redistribution elasticity. The timing edge is critical: we can "
            "react to lineup news faster than the market adjusts."
        )

    def get_decay_risk(self) -> str:
        return (
            "Low-Medium. This edge is partially structural (books will always be "
            "slow to adjust to late-breaking lineup news). However, as more "
            "bettors use real-time lineup data, the window shrinks. Current "
            "estimated window: 15-45 minutes after lineup confirmation."
        )

    def validate(self, historical_data: list) -> dict:
        """Validate on historical games where key players were absent."""
        if len(historical_data) < 30:
            return {
                "sharpe": 0.0,
                "p_value": 1.0,
                "sample_size": len(historical_data),
                "correlation_with_other_signals": {},
                "status": "insufficient_data",
            }

        signals = []
        outcomes = []
        for game in historical_data:
            player = game.get("player", {})
            context = game.get("game_context", {})
            actual = float(game.get("actual_pra", 0))
            line = float(game.get("line", 0))

            sig = self.get_signal(player, context)
            signals.append(sig)
            actual_dir = 1.0 if actual > line else -1.0
            sig_dir = 1.0 if sig > 0 else -1.0
            outcomes.append(actual_dir * sig_dir)

        signals_arr = np.array(signals)
        outcomes_arr = np.array(outcomes)
        nonzero = np.abs(signals_arr) > 0.01

        if nonzero.sum() < 20:
            return {
                "sharpe": 0.0,
                "p_value": 1.0,
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
