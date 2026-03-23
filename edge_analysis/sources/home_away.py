"""
edge_analysis.sources.home_away
===============================
Home/away PRA splits with Bayesian shrinkage toward league average.
Track venue-specific effects for extreme arenas (altitude, noise).
Market edge: books underweight home court for role players.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


# League-wide home court PRA advantage (~1.5% across all players)
_LEAGUE_HOME_BOOST: float = 0.015
_LEAGUE_HOME_STD: float = 0.04


@dataclass
class HomeAwaySplitsSource:
    """Signal source based on home/away PRA splits with Bayesian shrinkage."""

    name: str = "home_away"
    category: str = "game_environment"
    description: str = (
        "Models player-specific home/away PRA splits with Bayesian shrinkage "
        "toward league average, including venue-specific effects."
    )

    # Venue-specific adjustments (altitude, crowd noise, etc.)
    _venue_effects: dict = None

    def __post_init__(self):
        if self._venue_effects is None:
            self._venue_effects = {
                "DEN": 0.008,   # altitude effect (Denver)
                "UTA": 0.005,   # altitude effect (Salt Lake City)
                "MIA": 0.003,   # heat/humidity
                "GSW": 0.002,   # loud arena
                "BOS": 0.002,   # loud arena
                "PHX": 0.001,
                "MIL": 0.001,
            }

    def get_signal(self, player: dict, game_context: dict) -> float:
        """
        Compute home/away adjusted PRA signal.
        Positive = environment favors over, negative = favors under.
        """
        is_home = bool(game_context.get("is_home", True))

        # --- Player-specific home/away split ---
        home_pra = float(player.get("home_pra_avg", 0))
        away_pra = float(player.get("away_pra_avg", 0))
        home_games = int(player.get("home_games", 0))
        away_games = int(player.get("away_games", 0))
        overall_pra = float(player.get("pra_avg", 30.0))

        if home_games == 0 or away_games == 0:
            # No split data — use league average
            raw_split = _LEAGUE_HOME_BOOST
        else:
            raw_split = (home_pra - away_pra) / max(overall_pra, 1.0)

        # --- Bayesian shrinkage ---
        # Shrink player split toward league average based on sample size
        total_games = home_games + away_games
        shrinkage = self._compute_shrinkage(total_games, raw_split)

        # --- Venue-specific effect ---
        venue_team = game_context.get("home_team", "")
        venue_boost = 0.0
        if not is_home:
            # Playing away at a tough venue hurts
            venue_boost = -self._venue_effects.get(venue_team, 0.0)
        else:
            venue_boost = self._venue_effects.get(venue_team, 0.0)

        # --- Compute signal ---
        if is_home:
            expected_pra_adj = overall_pra * (shrinkage + venue_boost)
        else:
            expected_pra_adj = overall_pra * (-shrinkage + venue_boost)

        # Market typically uses a generic home/away factor
        # Our edge: player-specific shrunk estimates + venue effects
        market_adj = overall_pra * (_LEAGUE_HOME_BOOST if is_home else -_LEAGUE_HOME_BOOST)
        edge_pra = expected_pra_adj - market_adj

        pra_std = float(player.get("pra_std", 7.0))
        signal = edge_pra / max(pra_std, 1.0)

        return float(np.clip(signal, -3.0, 3.0) / 3.0)

    def _compute_shrinkage(self, n_games: int, raw_split: float) -> float:
        """
        Bayesian shrinkage of home/away split toward league average.
        Uses empirical Bayes approach: shrinkage = n / (n + k)
        where k is the prior strength.
        """
        prior_mean = _LEAGUE_HOME_BOOST
        prior_strength = 40.0  # ~40 games before we trust player-specific split

        weight = n_games / (n_games + prior_strength)
        shrunk = weight * raw_split + (1 - weight) * prior_mean

        return float(shrunk)

    def get_mechanism(self) -> str:
        return (
            "Books apply a generic home court adjustment (~1.5% PRA boost) to "
            "all players uniformly. In reality, home/away splits vary dramatically "
            "by player (role players show larger home boosts than stars) and venue "
            "(Denver altitude, Boston crowd noise). We use Bayesian shrinkage to "
            "estimate player-specific splits, avoiding small-sample overfitting "
            "while still capturing genuine individual effects. The edge is largest "
            "for role players with strong home splits and for games in extreme venues."
        )

    def get_decay_risk(self) -> str:
        return (
            "Medium. Home/away split data is publicly available. The edge is in "
            "the Bayesian methodology (avoiding overfitting) and venue-specific "
            "modeling. As sports analytics matures, this edge erodes. "
            "Half-life: 2 seasons."
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
