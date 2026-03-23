"""
edge_analysis.sources.recency_weighting
=======================================
Optimal lookback window per player with exponential decay weighting.
Detect regime changes (role change, injury return, trade).
Market edge: books use fixed windows; we optimize per player.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


@dataclass
class RecencyWeightingSource:
    """Signal source based on optimal recency weighting and regime detection."""

    name: str = "recency_weighting"
    category: str = "calibration"
    description: str = (
        "Optimizes the lookback window per player using exponential decay "
        "weighting. Detects regime changes (role changes, injury returns, "
        "trades) and adjusts the effective sample accordingly."
    )

    # Default half-life in games for exponential decay
    _default_half_life: int = 12
    # Regime detection parameters
    _regime_change_z_threshold: float = 2.0
    _min_regime_games: int = 5

    def get_signal(self, player: dict, game_context: dict) -> float:
        """
        Compute recency-weighted PRA signal.
        Positive = recent trend favors over, negative = favors under.
        """
        recent_pra = player.get("recent_pra_history", [])
        if len(recent_pra) < 5:
            return 0.0

        pra_arr = np.array(recent_pra, dtype=float)
        season_avg = float(player.get("pra_avg", np.mean(pra_arr)))

        # --- Detect regime change ---
        regime_idx = self._detect_regime_change(pra_arr)

        # --- Optimize half-life ---
        if regime_idx is not None and regime_idx < len(pra_arr) - self._min_regime_games:
            # Only use post-regime data
            effective_data = pra_arr[regime_idx:]
            optimal_half_life = self._optimize_half_life(effective_data)
        else:
            effective_data = pra_arr
            optimal_half_life = self._optimize_half_life(pra_arr)

        # --- Compute exponentially weighted average ---
        n = len(effective_data)
        decay_rate = math.log(2) / max(optimal_half_life, 1.0)
        weights = np.array([math.exp(-decay_rate * i) for i in range(n)])
        weights = weights[::-1]  # most recent game gets highest weight
        weights /= weights.sum()

        weighted_avg = float(np.dot(weights, effective_data))

        # --- Compare to season average (market proxy) ---
        # Books typically use a ~20-game simple average
        market_window = min(20, len(pra_arr))
        market_avg = float(np.mean(pra_arr[-market_window:]))

        # Our edge: the difference between optimally weighted and market average
        pra_delta = weighted_avg - market_avg

        # --- Confidence adjustment ---
        # If regime change detected, boost confidence (market is stale)
        confidence = 1.0
        if regime_idx is not None:
            games_since_regime = len(pra_arr) - regime_idx
            if games_since_regime >= self._min_regime_games:
                confidence = 1.3  # boost signal when regime detected
            else:
                confidence = 0.5  # too few games post-regime, reduce confidence

        # --- Trend strength ---
        # Is the recent trajectory continuing?
        if len(effective_data) >= 5:
            recent_5 = effective_data[-5:]
            trend_slope = np.polyfit(range(5), recent_5, 1)[0]
            # Add small trend component
            trend_signal = trend_slope * 0.3  # ~0.3 games worth of trend
        else:
            trend_signal = 0.0

        pra_delta_adj = (pra_delta + trend_signal) * confidence

        pra_std = float(player.get("pra_std", 7.0))
        signal = pra_delta_adj / max(pra_std, 1.0)

        return float(np.clip(signal, -3.0, 3.0) / 3.0)

    def _detect_regime_change(self, pra_arr: np.ndarray) -> int | None:
        """
        Detect the most recent regime change using CUSUM-style detection.
        Returns the index of the regime change, or None if none detected.
        """
        if len(pra_arr) < 10:
            return None

        overall_mean = np.mean(pra_arr)
        overall_std = np.std(pra_arr, ddof=1)
        if overall_std < 0.5:
            return None

        # Scan from recent to old, looking for a structural break
        best_break_idx = None
        best_break_score = 0.0

        for i in range(self._min_regime_games, len(pra_arr) - self._min_regime_games):
            pre_mean = np.mean(pra_arr[:i])
            post_mean = np.mean(pra_arr[i:])

            # Two-sample t-statistic
            n1, n2 = i, len(pra_arr) - i
            pooled_var = (
                (n1 - 1) * np.var(pra_arr[:i], ddof=1)
                + (n2 - 1) * np.var(pra_arr[i:], ddof=1)
            ) / (n1 + n2 - 2)

            if pooled_var <= 0:
                continue

            se = math.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))
            if se < 0.01:
                continue

            t_stat = abs(post_mean - pre_mean) / se

            if t_stat > self._regime_change_z_threshold and t_stat > best_break_score:
                best_break_score = t_stat
                best_break_idx = i

        return best_break_idx

    def _optimize_half_life(self, data: np.ndarray) -> float:
        """
        Find optimal half-life by minimizing weighted prediction error.
        Uses leave-one-out cross-validation with exponential weights.
        """
        if len(data) < 5:
            return float(self._default_half_life)

        def cv_error(half_life: float) -> float:
            if half_life < 1:
                return 1e10
            decay = math.log(2) / half_life
            errors = []
            for t in range(2, len(data)):
                weights = np.array([math.exp(-decay * (t - 1 - j)) for j in range(t)])
                weights /= weights.sum()
                pred = float(np.dot(weights, data[:t]))
                errors.append((data[t] - pred) ** 2)
            return float(np.mean(errors)) if errors else 1e10

        result = minimize_scalar(cv_error, bounds=(3.0, 30.0), method="bounded")
        return float(result.x) if result.success else float(self._default_half_life)

    def get_mechanism(self) -> str:
        return (
            "Books use fixed lookback windows (typically 15-20 games, simple "
            "average) to set PRA lines. We optimize the lookback window per "
            "player using cross-validated exponential decay, finding each "
            "player's optimal half-life. Critically, we detect regime changes "
            "(role changes from trades, injury returns, rotation shifts) and "
            "truncate the lookback at the change point. When a player returns "
            "from injury with a minutes restriction, or joins a new team after "
            "a trade, the book's 20-game average includes irrelevant data. "
            "Our regime-aware weighting adapts within 5 games."
        )

    def get_decay_risk(self) -> str:
        return (
            "Low-Medium. Regime detection is non-trivial to implement well. "
            "The edge is most valuable around trade deadlines, injury returns, "
            "and lineup changes. Books will eventually adopt adaptive windows "
            "but the implementation complexity keeps this edge alive. "
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
