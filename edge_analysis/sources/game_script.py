"""
edge_analysis.sources.game_script
=================================
Blowout probability -> garbage time minutes -> PRA impact.
Model expected game flow from spread and track player usage
in different score states.
Market edge: books don't model game script distribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class GameScriptSource:
    """Signal source based on game script / blowout probability modeling."""

    name: str = "game_script"
    category: str = "game_environment"
    description: str = (
        "Models how game script (blowout vs close game) affects minutes "
        "distribution, usage, and PRA. Uses spread-based blowout probability "
        "and player-specific score-state usage profiles."
    )

    # Score-state thresholds
    _close_game_threshold: float = 5.0   # within 5 points
    _moderate_lead_threshold: float = 12.0
    _blowout_threshold: float = 20.0

    def get_signal(self, player: dict, game_context: dict) -> float:
        """
        Compute game-script adjusted PRA signal.
        Positive = game script favors over, negative = favors under.
        """
        spread = float(game_context.get("spread", 0.0))
        total = float(game_context.get("total", 220.0))
        is_home = bool(game_context.get("is_home", True))

        # --- Game flow distribution ---
        # Model the distribution of score differentials throughout the game
        # using spread as the mean and historical variance
        p_close, p_moderate, p_blowout_fav, p_blowout_dog = self._game_flow_probs(spread)

        # --- Player minutes by score state ---
        is_starter = bool(player.get("is_starter", True))
        minutes_avg = float(player.get("minutes_avg", 32.0))
        is_favorite = (spread < 0 and is_home) or (spread > 0 and not is_home)

        close_minutes = minutes_avg * 1.05      # slight boost in close games (crunch time)
        moderate_minutes = minutes_avg * 0.98    # normal-ish
        blowout_fav_minutes = minutes_avg * (0.72 if is_starter else 1.15)
        blowout_dog_minutes = minutes_avg * (0.78 if is_starter else 1.10)

        if is_favorite:
            expected_minutes = (
                p_close * close_minutes
                + p_moderate * moderate_minutes
                + p_blowout_fav * blowout_fav_minutes
                + p_blowout_dog * blowout_dog_minutes
            )
        else:
            expected_minutes = (
                p_close * close_minutes
                + p_moderate * moderate_minutes
                + p_blowout_fav * blowout_dog_minutes  # they're losing big
                + p_blowout_dog * blowout_fav_minutes  # they're winning big (less likely)
            )

        # --- Usage by score state ---
        # Stars get more usage in close games, less in blowouts
        base_usage = float(player.get("usage_rate", 0.20))
        close_usage = base_usage * 1.08
        moderate_usage = base_usage * 1.0
        blowout_usage = base_usage * 0.85

        if is_starter:
            expected_usage = (
                p_close * close_usage
                + p_moderate * moderate_usage
                + (p_blowout_fav + p_blowout_dog) * blowout_usage
            )
        else:
            # Bench players get MORE usage in blowouts
            expected_usage = (
                p_close * base_usage * 0.90
                + p_moderate * base_usage
                + (p_blowout_fav + p_blowout_dog) * base_usage * 1.15
            )

        # --- PRA impact ---
        expected_possessions = float(game_context.get("expected_possessions", 98.0))
        player_possessions = expected_possessions * (expected_minutes / 48.0)

        pra_per_possession = float(player.get("pra_per_possession", 0.55))
        expected_pra = player_possessions * expected_usage * pra_per_possession / base_usage

        # Market PRA (assumes average minutes and usage)
        market_possessions = expected_possessions * (minutes_avg / 48.0)
        market_pra = market_possessions * pra_per_possession

        pra_delta = expected_pra - market_pra

        pra_std = float(player.get("pra_std", 7.0))
        signal = pra_delta / max(pra_std, 1.0)

        return float(np.clip(signal, -3.0, 3.0) / 3.0)

    def _game_flow_probs(self, spread: float) -> tuple[float, float, float, float]:
        """
        Compute probabilities of different game flow scenarios.
        Returns: (p_close, p_moderate, p_blowout_favorite, p_blowout_dog)
        """
        abs_spread = abs(spread)

        # Standard deviation of final margin ~ 11 points (NBA historical)
        margin_std = 11.0

        # Distribution of final margin centered on spread
        # P(close) = P(|margin| < 5)
        p_close = float(
            stats.norm.cdf(self._close_game_threshold, abs_spread, margin_std)
            - stats.norm.cdf(-self._close_game_threshold, abs_spread, margin_std)
        )

        # P(moderate) = P(5 < |margin| < 20) approximately
        p_moderate = float(
            stats.norm.cdf(self._blowout_threshold, abs_spread, margin_std)
            - stats.norm.cdf(self._close_game_threshold, abs_spread, margin_std)
            + stats.norm.cdf(-self._close_game_threshold, abs_spread, margin_std)
            - stats.norm.cdf(-self._blowout_threshold, abs_spread, margin_std)
        )

        # P(blowout by favorite) = P(margin > 20)
        p_blowout_fav = float(
            1.0 - stats.norm.cdf(self._blowout_threshold, abs_spread, margin_std)
        )

        # P(blowout by dog) = P(margin < -20)
        p_blowout_dog = float(
            stats.norm.cdf(-self._blowout_threshold, abs_spread, margin_std)
        )

        # Normalize to ensure probabilities sum to 1
        total = p_close + p_moderate + p_blowout_fav + p_blowout_dog
        if total > 0:
            p_close /= total
            p_moderate /= total
            p_blowout_fav /= total
            p_blowout_dog /= total
        else:
            p_close, p_moderate = 0.5, 0.35
            p_blowout_fav, p_blowout_dog = 0.10, 0.05

        return p_close, p_moderate, p_blowout_fav, p_blowout_dog

    def get_mechanism(self) -> str:
        return (
            "Books set PRA lines assuming average game flow. But expected game "
            "flow is heavily influenced by the spread. A 12-point favorite has "
            "~25% blowout probability, meaning starters lose 4-8 minutes to "
            "garbage time. We model the full distribution of score differentials "
            "and compute expected minutes and usage by score state. Bench players "
            "on heavy favorites get a boost; starters on heavy favorites get a "
            "hit. The effect is asymmetric and not captured by season averages."
        )

    def get_decay_risk(self) -> str:
        return (
            "Medium. Game script modeling is conceptually simple but rarely "
            "implemented in prop markets. The edge persists because books focus "
            "on game totals and sides, not on the flow-through to player props. "
            "Half-life: 2-3 seasons."
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
