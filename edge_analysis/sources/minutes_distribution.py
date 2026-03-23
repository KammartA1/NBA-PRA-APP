"""
edge_analysis.sources.minutes_distribution
==========================================
Full distribution modeling for player minutes — beta distribution for minutes
share, incorporating foul trouble probability, blowout probability, and
rotation changes.  Market edge: books use season averages for minutes;
we model game-specific distributions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class MinutesDistributionSource:
    """Signal source based on full minutes-share distribution modeling."""

    name: str = "minutes_distribution"
    category: str = "playing_time"
    description: str = (
        "Models the full probability distribution of player minutes using "
        "beta-distributed minutes share, foul trouble probabilities, blowout "
        "risk, and rotation adjustments."
    )

    # Beta distribution prior parameters (league-wide starters)
    _prior_alpha: float = 8.0
    _prior_beta: float = 4.0

    # Foul trouble constants
    _foul_trouble_threshold: int = 3  # fouls by halftime
    _avg_foul_rate_per_min: float = 0.083  # ~3 fouls per 36 min

    # Blowout threshold (spread)
    _blowout_spread_threshold: float = 10.0

    def get_signal(self, player: dict, game_context: dict) -> float:
        """
        Compute directional signal for PRA based on minutes distribution.

        Returns positive if expected minutes > market-implied, negative if below.
        Magnitude reflects confidence (0 to ~3 standard deviations).
        """
        season_mpg = float(player.get("minutes_avg", 32.0))
        minutes_std = float(player.get("minutes_std", 4.0))
        game_total_minutes = 48.0

        # --- Beta distribution for minutes share ---
        observed_share = season_mpg / game_total_minutes
        n_games = int(player.get("games_played", 40))
        alpha, beta_param = self._fit_beta(observed_share, n_games)

        # --- Foul trouble adjustment ---
        foul_rate = float(player.get("fouls_per_min", self._avg_foul_rate_per_min))
        p_foul_trouble = self._foul_trouble_probability(foul_rate, season_mpg)

        # --- Blowout adjustment ---
        spread = float(game_context.get("spread", 0.0))
        p_blowout = self._blowout_probability(spread)
        is_favorite = spread < 0
        player_is_starter = player.get("is_starter", True)

        # Blowout reduces starter minutes, increases bench minutes
        blowout_minutes_delta = 0.0
        if p_blowout > 0.15:
            if player_is_starter and is_favorite:
                blowout_minutes_delta = -p_blowout * 6.0  # lose up to 6 min
            elif not player_is_starter and is_favorite:
                blowout_minutes_delta = p_blowout * 4.0   # gain up to 4 min
            elif player_is_starter and not is_favorite:
                blowout_minutes_delta = -p_blowout * 4.0
            else:
                blowout_minutes_delta = p_blowout * 2.0

        # --- Rotation changes ---
        rotation_delta = self._rotation_adjustment(player, game_context)

        # --- Expected minutes from our model ---
        base_expected = stats.beta.mean(alpha, beta_param) * game_total_minutes
        foul_penalty = p_foul_trouble * 5.0  # avg 5 min lost when in foul trouble
        model_expected = base_expected - foul_penalty + blowout_minutes_delta + rotation_delta

        # --- Market implied minutes (use season average as proxy) ---
        market_minutes = season_mpg

        # --- Signal: standardized difference ---
        if minutes_std < 1.0:
            minutes_std = 1.0
        signal = (model_expected - market_minutes) / minutes_std

        # Compute distribution-based probability adjustments
        pra_per_min = float(player.get("pra_per_min", 1.1))
        pra_delta = (model_expected - market_minutes) * pra_per_min

        # Scale signal to be interpretable (-1 to 1 typical range)
        signal = np.clip(signal, -3.0, 3.0) / 3.0

        return float(signal)

    def _fit_beta(self, observed_share: float, n_games: int) -> tuple[float, float]:
        """
        Fit beta distribution using Bayesian updating.
        Prior: league average for position. Posterior updated with observed data.
        """
        observed_share = np.clip(observed_share, 0.05, 0.95)
        # Method of moments with prior
        prior_mean = self._prior_alpha / (self._prior_alpha + self._prior_beta)
        prior_strength = self._prior_alpha + self._prior_beta

        # Posterior parameters (conjugate update for beta-binomial)
        effective_n = min(n_games, 82)  # cap at full season
        weight = effective_n / (effective_n + prior_strength)
        posterior_mean = weight * observed_share + (1 - weight) * prior_mean

        # Posterior concentration increases with sample size
        posterior_concentration = prior_strength + effective_n * 0.5
        alpha = posterior_mean * posterior_concentration
        beta_param = (1 - posterior_mean) * posterior_concentration

        return max(alpha, 1.01), max(beta_param, 1.01)

    def _foul_trouble_probability(self, foul_rate: float, minutes: float) -> float:
        """
        Probability of accumulating >= threshold fouls by halftime.
        Uses Poisson model for foul accumulation.
        """
        half_minutes = minutes * 0.52  # slightly more first-half minutes
        expected_fouls = foul_rate * half_minutes
        if expected_fouls <= 0:
            return 0.0
        # P(X >= threshold) = 1 - P(X < threshold) using Poisson CDF
        p_trouble = 1.0 - stats.poisson.cdf(
            self._foul_trouble_threshold - 1, expected_fouls
        )
        return float(np.clip(p_trouble, 0.0, 1.0))

    def _blowout_probability(self, spread: float) -> float:
        """
        Probability of blowout (20+ point lead) given the spread.
        Uses logistic model calibrated to historical NBA data.
        """
        abs_spread = abs(spread)
        # Logistic function: steeper curve for larger spreads
        # At spread=0, p_blowout ~ 0.12 (historical base rate)
        # At spread=10, p_blowout ~ 0.35
        # At spread=15, p_blowout ~ 0.55
        logit = -2.0 + 0.15 * abs_spread
        p_blowout = 1.0 / (1.0 + math.exp(-logit))
        return float(p_blowout)

    def _rotation_adjustment(self, player: dict, game_context: dict) -> float:
        """
        Adjust minutes for known rotation changes (injuries, rest days).
        """
        delta = 0.0
        teammates_out = game_context.get("teammates_out", [])
        if not teammates_out:
            return 0.0

        player_position = player.get("position", "G")
        for teammate in teammates_out:
            tm_position = teammate.get("position", "")
            tm_minutes = float(teammate.get("minutes_avg", 0.0))
            # Same position gets more of the minutes
            if tm_position == player_position:
                delta += tm_minutes * 0.35  # absorb ~35% of teammate minutes
            else:
                delta += tm_minutes * 0.08  # small spillover
        return float(np.clip(delta, -8.0, 12.0))

    def get_mechanism(self) -> str:
        return (
            "Books set lines based on season-average minutes. We model the full "
            "distribution of minutes share using a beta distribution, adjusting for "
            "game-specific foul trouble probability (Poisson model), blowout "
            "probability (logistic model from spread), and rotation changes from "
            "injuries. This captures the asymmetric risk that books miss — e.g., "
            "a starter on a heavy favorite has significant downside minutes risk "
            "from garbage time that season averages obscure."
        )

    def get_decay_risk(self) -> str:
        return (
            "Medium. As books incorporate more granular minutes models, this edge "
            "decays. Foul trouble and blowout adjustments remain less efficient. "
            "Half-life estimated at 2-3 seasons before significant market adaptation."
        )

    def validate(self, historical_data: list) -> dict:
        """
        Validate signal quality on historical data.
        Each entry: {player, game_context, actual_pra, line}
        """
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
            # Outcome: +1 if signal direction was correct, -1 otherwise
            actual_direction = 1.0 if actual > line else -1.0
            signal_direction = 1.0 if sig > 0 else -1.0
            outcomes.append(actual_direction * signal_direction)

        signals_arr = np.array(signals)
        outcomes_arr = np.array(outcomes)

        # Filter out zero signals
        nonzero = np.abs(signals_arr) > 0.01
        if nonzero.sum() < 20:
            return {
                "sharpe": 0.0,
                "p_value": 1.0,
                "sample_size": int(nonzero.sum()),
                "correlation_with_other_signals": {},
                "status": "insufficient_nonzero_signals",
            }

        filtered_outcomes = outcomes_arr[nonzero]
        filtered_signals = signals_arr[nonzero]

        # Weighted returns: signal magnitude * outcome
        returns = filtered_signals * filtered_outcomes
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1))

        sharpe = mean_return / std_return * math.sqrt(252) if std_return > 0 else 0.0

        # One-sample t-test: is mean return significantly > 0?
        t_stat, p_value = stats.ttest_1samp(returns, 0.0)
        p_value = float(p_value) / 2.0  # one-tailed
        if t_stat < 0:
            p_value = 1.0 - p_value

        return {
            "sharpe": round(float(sharpe), 3),
            "p_value": round(float(p_value), 4),
            "sample_size": int(nonzero.sum()),
            "mean_return": round(mean_return, 4),
            "hit_rate": round(float(np.mean(filtered_outcomes > 0)), 4),
            "correlation_with_other_signals": {},
            "status": "valid",
        }
