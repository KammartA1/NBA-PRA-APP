"""
edge_analysis.sources.defensive_matchup
=======================================
Opposing team's defensive rating vs position.  Position-specific defensive
impact and suppression of specific stat categories.
Market edge: books use team-level defense, not position-specific.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


# League average defensive ratings by position (points allowed per 100 possessions)
_LEAGUE_AVG_DEF_RATING: float = 112.0
_LEAGUE_AVG_POS_DEF = {
    "PG": 112.5,
    "SG": 113.0,
    "SF": 112.0,
    "PF": 111.5,
    "C":  110.5,
    "G":  112.8,
    "F":  111.8,
}


@dataclass
class DefensiveMatchupSource:
    """Signal source based on position-specific defensive matchup analysis."""

    name: str = "defensive_matchup"
    category: str = "matchup"
    description: str = (
        "Models how opposing team's position-specific defense impacts player "
        "PRA. Tracks which defenses suppress which stat categories and "
        "models individual defender matchup effects."
    )

    def get_signal(self, player: dict, game_context: dict) -> float:
        """
        Compute defensive matchup signal.
        Positive = weak defense favors over, negative = strong defense favors under.
        """
        position = player.get("position", "G")
        opp_defense = game_context.get("opp_defense", {})

        if not opp_defense:
            return 0.0

        # --- Team-level defensive rating ---
        opp_def_rating = float(opp_defense.get("def_rating", _LEAGUE_AVG_DEF_RATING))
        team_def_delta = (opp_def_rating - _LEAGUE_AVG_DEF_RATING) / _LEAGUE_AVG_DEF_RATING

        # --- Position-specific defense ---
        pos_key = f"vs_{position.lower()}_rating"
        league_pos_avg = _LEAGUE_AVG_POS_DEF.get(position, _LEAGUE_AVG_DEF_RATING)

        opp_pos_def = float(opp_defense.get(pos_key, league_pos_avg))
        pos_def_delta = (opp_pos_def - league_pos_avg) / max(league_pos_avg, 90.0)

        # --- Stat-specific suppression ---
        stat_type = player.get("stat_type", "pra")
        stat_suppression = self._get_stat_suppression(opp_defense, stat_type, position)

        # --- Individual matchup (primary defender) ---
        primary_defender = game_context.get("primary_defender", {})
        individual_effect = 0.0
        if primary_defender:
            def_impact = float(primary_defender.get("matchup_difficulty", 0.0))
            def_games = int(primary_defender.get("matchup_games", 0))
            # Bayesian shrinkage
            reliability = min(1.0, def_games / 15.0)
            individual_effect = def_impact * reliability

        # --- Combine effects ---
        # Position-specific is more informative than team-level
        combined_def_effect = (
            0.30 * team_def_delta
            + 0.40 * pos_def_delta
            + 0.20 * stat_suppression
            + 0.10 * individual_effect
        )

        # Convert to PRA impact
        base_pra = float(player.get("pra_avg", 30.0))
        # Higher def rating allowed = more PRA for our player
        pra_delta = base_pra * combined_def_effect

        # --- Market adjustment ---
        # Books use team-level defense only
        market_def_adj = base_pra * team_def_delta * 0.5  # books capture ~50% of team effect
        edge = pra_delta - market_def_adj

        pra_std = float(player.get("pra_std", 7.0))
        signal = edge / max(pra_std, 1.0)

        return float(np.clip(signal, -3.0, 3.0) / 3.0)

    @staticmethod
    def _get_stat_suppression(
        opp_defense: dict, stat_type: str, position: str
    ) -> float:
        """
        Compute stat-specific defensive suppression.
        Some defenses are elite at preventing specific stat categories.
        """
        # Look up stat-specific defense
        stat_keys = {
            "points": "pts_allowed_vs_pos",
            "rebounds": "reb_allowed_vs_pos",
            "assists": "ast_allowed_vs_pos",
            "pra": "pra_allowed_vs_pos",
            "threes": "fg3_allowed_vs_pos",
            "steals": "stl_allowed_vs_pos",
            "blocks": "blk_allowed_vs_pos",
        }
        key = stat_keys.get(stat_type.lower(), "pra_allowed_vs_pos")
        full_key = f"{key}_{position.lower()}"

        allowed = float(opp_defense.get(full_key, 0))
        if allowed == 0:
            return 0.0

        # Compare to league average allowed for this stat/position
        league_avg = float(opp_defense.get(f"league_avg_{key}_{position.lower()}", allowed))
        if league_avg == 0:
            return 0.0

        return (allowed - league_avg) / max(league_avg, 1.0)

    def get_mechanism(self) -> str:
        return (
            "Books adjust lines using team-level defensive rating, capturing "
            "about 50% of the true defensive impact. We add position-specific "
            "defense (a team may be elite vs guards but weak vs bigs), stat-"
            "specific suppression (some defenses prevent assists but allow "
            "points), and individual defender matchup history. The edge is "
            "largest when the team-level and position-specific defenses diverge "
            "significantly — e.g., a team ranked 5th in overall defense but "
            "25th in defense vs centers."
        )

    def get_decay_risk(self) -> str:
        return (
            "Medium. Position-specific defensive data is becoming more available "
            "via NBA API. The individual matchup component has a longer half-life "
            "as it requires granular tracking data. Overall half-life: 2-3 seasons."
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
