"""
edge_analysis.edge_sources
==========================
Master catalog of ALL signals in the codebase (~53 signals across simulation,
quant_system, and services).  Documents mechanism, data advantage, decay risk,
and independence for each signal.  Computes pairwise correlation matrix and
flags non-independent pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class SignalEntry:
    """Metadata for a single signal in the system."""
    name: str
    module: str
    category: str
    mechanism: str
    data_advantage: str
    decay_risk: str
    independence_notes: str
    signal_id: int = 0


# ---------------------------------------------------------------------------
# Complete signal catalog — every signal in the codebase
# ---------------------------------------------------------------------------
_SIGNAL_CATALOG: list[dict] = [
    # ── simulation/ signals (Section 3) ──────────────────────────────────
    {
        "name": "possession_sim_points",
        "module": "simulation.possession",
        "category": "simulation",
        "mechanism": "Possession-level Monte Carlo produces full scoring distributions, not point estimates. Market uses season averages.",
        "data_advantage": "Custom possession model with shot selection, turnover, and foul probabilities per possession type.",
        "decay_risk": "Medium — as analytical models become more common, edge narrows.",
        "independence_notes": "Correlated with pace_differential (r~0.4), game_script (r~0.3).",
    },
    {
        "name": "possession_sim_rebounds",
        "module": "simulation.possession",
        "category": "simulation",
        "mechanism": "Rebound probability modeled per-possession based on shot type, distance, and player positioning.",
        "data_advantage": "Shot-type granularity for offensive vs defensive rebound probability.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with defensive_matchup (r~0.35).",
    },
    {
        "name": "possession_sim_assists",
        "module": "simulation.possession",
        "category": "simulation",
        "mechanism": "Assist probability per possession based on usage, pass tendency, and teammate shooting.",
        "data_advantage": "Lineup-aware assist modeling.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with usage_redistribution (r~0.45).",
    },
    {
        "name": "fatigue_impact",
        "module": "simulation.fatigue_model",
        "category": "player_state",
        "mechanism": "In-game fatigue curve reduces efficiency as minutes accumulate. Market ignores intra-game fatigue.",
        "data_advantage": "Player-specific fatigue curves calibrated from second-half efficiency drops.",
        "decay_risk": "Low — fatigue is fundamental and hard to model without possession-level simulation.",
        "independence_notes": "Correlated with rest_effects (r~0.55), minutes_distribution (r~0.40).",
    },
    {
        "name": "foul_trouble_minutes",
        "module": "simulation.foul_model",
        "category": "playing_time",
        "mechanism": "Foul accumulation modeled via Poisson process; early fouls reduce second-half minutes.",
        "data_advantage": "Per-player foul rates combined with ref tendency data.",
        "decay_risk": "Low — structural edge in modeling foul probability distributions.",
        "independence_notes": "Correlated with referee_tendencies (r~0.50), minutes_distribution (r~0.60).",
    },
    {
        "name": "blowout_minutes_risk",
        "module": "simulation.blowout_model",
        "category": "game_environment",
        "mechanism": "Blowout probability from spread creates asymmetric minutes risk for starters.",
        "data_advantage": "Spread-calibrated blowout probability with position-specific impact.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with game_script (r~0.75), minutes_distribution (r~0.55).",
    },
    {
        "name": "lineup_rotation_minutes",
        "module": "simulation.lineup_manager",
        "category": "playing_time",
        "mechanism": "Rotation modeling predicts minutes distribution across lineup combinations.",
        "data_advantage": "Coach-specific rotation pattern tracking.",
        "decay_risk": "Medium — rotations are somewhat predictable.",
        "independence_notes": "Correlated with minutes_distribution (r~0.65), lineup_effects (r~0.50).",
    },
    {
        "name": "player_state_momentum",
        "module": "simulation.player_state",
        "category": "player_state",
        "mechanism": "In-game player state (hot/cold streaks, confidence, rhythm) affects shot selection and efficiency.",
        "data_advantage": "Real-time state tracking during simulation.",
        "decay_risk": "Low — psychological state is inherently noisy and hard to price.",
        "independence_notes": "Correlated with recency_weighting (r~0.40).",
    },
    {
        "name": "game_engine_total_sim",
        "module": "simulation.game_engine",
        "category": "simulation",
        "mechanism": "Full game simulation combining all sub-models produces PRA distribution.",
        "data_advantage": "Integration of all simulation sub-models.",
        "decay_risk": "Medium.",
        "independence_notes": "Composite signal — correlated with all simulation sub-signals.",
    },
    {
        "name": "team_state_offense",
        "module": "simulation.team_state",
        "category": "team_environment",
        "mechanism": "Team offensive state (pace, spacing, ball movement) affects individual PRA.",
        "data_advantage": "Team-level offensive modeling beyond individual stats.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with pace_differential (r~0.50), lineup_effects (r~0.45).",
    },
    # ── quant_system/ signals (Section 9) ─────────────────────────────────
    {
        "name": "adaptive_kelly_sizing",
        "module": "quant_system.risk",
        "category": "sizing",
        "mechanism": "Kelly criterion with Bayesian edge estimation sizes bets optimally.",
        "data_advantage": "Adaptive Kelly with edge uncertainty and correlation adjustments.",
        "decay_risk": "Low — sizing methodology, not a signal per se.",
        "independence_notes": "Meta-signal — depends on edge quality of input signals.",
    },
    {
        "name": "clv_feedback_edge",
        "module": "quant_system.learning",
        "category": "calibration",
        "mechanism": "Closing line value feedback identifies persistent edge vs noise.",
        "data_advantage": "Historical CLV tracking validates whether signals beat the close.",
        "decay_risk": "Low — CLV is the ultimate validation metric.",
        "independence_notes": "Independent of all signal sources — it measures their quality.",
    },
    {
        "name": "market_odds_movement",
        "module": "quant_system.market",
        "category": "market",
        "mechanism": "Steam moves and reverse line movement indicate sharp money.",
        "data_advantage": "Real-time odds monitoring across multiple books.",
        "decay_risk": "Medium — market efficiency increases over time.",
        "independence_notes": "Partially correlated with all signals via market pricing.",
    },
    {
        "name": "bankroll_risk_state",
        "module": "quant_system.risk",
        "category": "risk",
        "mechanism": "System state (active/reduced/suspended) based on drawdown protects capital.",
        "data_advantage": "Real-time P&L tracking with adaptive thresholds.",
        "decay_risk": "N/A — risk management, not a signal.",
        "independence_notes": "Independent — meta-signal for position sizing.",
    },
    {
        "name": "backtest_regime_filter",
        "module": "quant_system.backtest",
        "category": "calibration",
        "mechanism": "Historical backtest identifies which market regimes favor which signals.",
        "data_advantage": "Multi-season backtest data with regime classification.",
        "decay_risk": "Medium — regimes shift.",
        "independence_notes": "Correlated with recency_weighting (r~0.30).",
    },
    {
        "name": "engine_composite_edge",
        "module": "quant_system.engine",
        "category": "composite",
        "mechanism": "Quant engine combines all signals with optimized weights.",
        "data_advantage": "Ensemble approach with dynamic weight adjustment.",
        "decay_risk": "Depends on component signals.",
        "independence_notes": "Composite — correlated with all component signals.",
    },
    {
        "name": "market_implied_prob",
        "module": "quant_system.market",
        "category": "market",
        "mechanism": "Extract true probability from market odds after removing vig.",
        "data_advantage": "Multi-book odds comparison for vig removal.",
        "decay_risk": "Low — fundamental market data.",
        "independence_notes": "Independent pricing signal.",
    },
    {
        "name": "correlation_risk_adj",
        "module": "quant_system.risk",
        "category": "risk",
        "mechanism": "Adjust position sizes for correlated bets (same game, same team).",
        "data_advantage": "Pairwise bet correlation tracking.",
        "decay_risk": "N/A — risk management.",
        "independence_notes": "Independent meta-signal.",
    },
    # ── services/clv_system/ signals (Section 5) ─────────────────────────
    {
        "name": "clv_cents_signal",
        "module": "services.clv_system",
        "category": "calibration",
        "mechanism": "Average CLV in cents measures persistent edge over closing line.",
        "data_advantage": "Automated closing line capture and comparison.",
        "decay_risk": "Low — CLV is the gold standard for edge validation.",
        "independence_notes": "Independent validation metric.",
    },
    {
        "name": "clv_beat_close_pct",
        "module": "services.clv_system",
        "category": "calibration",
        "mechanism": "Percentage of bets that beat the closing line.",
        "data_advantage": "Statistical significance testing on beat-close rate.",
        "decay_risk": "Low.",
        "independence_notes": "Highly correlated with clv_cents_signal (r~0.85).",
    },
    {
        "name": "clv_by_signal_source",
        "module": "services.clv_system",
        "category": "calibration",
        "mechanism": "CLV breakdown by which signal source drove the bet.",
        "data_advantage": "Attribution analysis identifies which sources produce real CLV.",
        "decay_risk": "Low.",
        "independence_notes": "Correlated with clv_cents_signal (r~0.70).",
    },
    {
        "name": "clv_trend_momentum",
        "module": "services.clv_system",
        "category": "calibration",
        "mechanism": "Trending CLV (improving or declining) indicates edge health.",
        "data_advantage": "Rolling window CLV with statistical trend detection.",
        "decay_risk": "Low.",
        "independence_notes": "Correlated with clv_cents_signal (r~0.60).",
    },
    # ── services/data_audit/ signals (Section 6) ─────────────────────────
    {
        "name": "data_freshness_score",
        "module": "services.data_audit",
        "category": "data_quality",
        "mechanism": "Stale data detection — flags when data feeds are delayed.",
        "data_advantage": "Continuous data pipeline monitoring.",
        "decay_risk": "N/A — infrastructure signal.",
        "independence_notes": "Independent quality metric.",
    },
    {
        "name": "data_completeness_score",
        "module": "services.data_audit",
        "category": "data_quality",
        "mechanism": "Missing data detection across all data sources.",
        "data_advantage": "Automated completeness checks.",
        "decay_risk": "N/A.",
        "independence_notes": "Independent quality metric.",
    },
    {
        "name": "outlier_detection_flag",
        "module": "services.data_audit",
        "category": "data_quality",
        "mechanism": "Statistical outlier detection in incoming data.",
        "data_advantage": "IQR and z-score based outlier flagging.",
        "decay_risk": "N/A.",
        "independence_notes": "Independent quality metric.",
    },
    {
        "name": "cross_source_consistency",
        "module": "services.data_audit",
        "category": "data_quality",
        "mechanism": "Cross-reference data between sources to detect errors.",
        "data_advantage": "Multi-source validation pipeline.",
        "decay_risk": "N/A.",
        "independence_notes": "Independent quality metric.",
    },
    # ── Defensive matchup signals ─────────────────────────────────────────
    {
        "name": "opp_pace_ratio",
        "module": "matchup.pace",
        "category": "matchup",
        "mechanism": "Opponent pace ratio vs league average creates possession differential.",
        "data_advantage": "Matchup-specific pace modeling.",
        "decay_risk": "Medium-High — pace data is public.",
        "independence_notes": "Correlated with pace_differential (r~0.80).",
    },
    {
        "name": "opp_def_rating",
        "module": "matchup.defense",
        "category": "matchup",
        "mechanism": "Opponent defensive rating predicts scoring difficulty.",
        "data_advantage": "Position-specific defensive ratings.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with defensive_matchup (r~0.75).",
    },
    {
        "name": "opp_reb_rate",
        "module": "matchup.rebounding",
        "category": "matchup",
        "mechanism": "Opponent rebounding rate affects player rebound opportunities.",
        "data_advantage": "Position-specific rebound rate tracking.",
        "decay_risk": "Medium.",
        "independence_notes": "Partially independent — specialized stat.",
    },
    {
        "name": "opp_ast_rate",
        "module": "matchup.assists",
        "category": "matchup",
        "mechanism": "Opponent assist rate allowed indicates defensive cohesion.",
        "data_advantage": "Assist-rate-allowed by lineup context.",
        "decay_risk": "Medium.",
        "independence_notes": "Partially independent — specialized stat.",
    },
    # ── Player state signals ──────────────────────────────────────────────
    {
        "name": "minutes_trend",
        "module": "player.state",
        "category": "player_state",
        "mechanism": "Recent minutes trend (increasing/decreasing role).",
        "data_advantage": "Regime detection for minutes changes.",
        "decay_risk": "Low-Medium.",
        "independence_notes": "Correlated with recency_weighting (r~0.55), minutes_distribution (r~0.50).",
    },
    {
        "name": "rest_days_signal",
        "module": "player.state",
        "category": "player_state",
        "mechanism": "Days of rest before game affects performance.",
        "data_advantage": "Age-specific rest curves.",
        "decay_risk": "Low.",
        "independence_notes": "Correlated with rest_effects (r~0.90).",
    },
    {
        "name": "b2b_flag",
        "module": "player.state",
        "category": "player_state",
        "mechanism": "Back-to-back indicator.",
        "data_advantage": "Interaction with age, minutes, travel.",
        "decay_risk": "Low.",
        "independence_notes": "Correlated with rest_effects (r~0.85).",
    },
    {
        "name": "age_factor",
        "module": "player.state",
        "category": "player_state",
        "mechanism": "Age affects injury risk, fatigue recovery, and performance ceiling.",
        "data_advantage": "Age-interaction models for B2B and workload.",
        "decay_risk": "Low.",
        "independence_notes": "Correlated with rest_effects (r~0.45).",
    },
    {
        "name": "foul_trouble_risk",
        "module": "player.state",
        "category": "player_state",
        "mechanism": "Historical foul rate predicts foul trouble probability.",
        "data_advantage": "Ref-adjusted foul probability.",
        "decay_risk": "Low.",
        "independence_notes": "Correlated with referee_tendencies (r~0.55), foul_trouble_minutes (r~0.70).",
    },
    # ── Usage & opportunity signals ───────────────────────────────────────
    {
        "name": "usage_rate",
        "module": "player.usage",
        "category": "opportunity",
        "mechanism": "Player usage rate determines volume of scoring opportunities.",
        "data_advantage": "Lineup-context-aware usage tracking.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with usage_redistribution (r~0.60).",
    },
    {
        "name": "usage_volatility",
        "module": "player.usage",
        "category": "opportunity",
        "mechanism": "Volatility in usage rate indicates uncertainty in role.",
        "data_advantage": "Game-to-game usage variance tracking.",
        "decay_risk": "Medium.",
        "independence_notes": "Partially independent.",
    },
    {
        "name": "shot_attempts_trend",
        "module": "player.usage",
        "category": "opportunity",
        "mechanism": "Trending shot volume indicates role change.",
        "data_advantage": "CUSUM change detection on FGA.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with usage_rate (r~0.70), recency_weighting (r~0.45).",
    },
    # ── Shooting performance signals ──────────────────────────────────────
    {
        "name": "fg_pct_vs_expected",
        "module": "player.shooting",
        "category": "shooting",
        "mechanism": "FG% vs expected FG% indicates shooting luck/skill.",
        "data_advantage": "Shot-quality-adjusted expected FG%.",
        "decay_risk": "Medium — regresses quickly.",
        "independence_notes": "Partially independent — shooting-specific.",
    },
    {
        "name": "three_pt_regression",
        "module": "player.shooting",
        "category": "shooting",
        "mechanism": "3PT% mean-reverts faster than other stats — exploit over/under-shooting.",
        "data_advantage": "Bayesian 3PT% estimation with career prior.",
        "decay_risk": "Low — 3PT variance is fundamental.",
        "independence_notes": "Independent from most non-shooting signals.",
    },
    {
        "name": "ft_rate_matchup",
        "module": "player.shooting",
        "category": "shooting",
        "mechanism": "Free throw rate varies by matchup and referee.",
        "data_advantage": "Ref-adjusted FT rate modeling.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with referee_tendencies (r~0.50).",
    },
    # ── Team environment signals ──────────────────────────────────────────
    {
        "name": "team_pace",
        "module": "team.environment",
        "category": "team_environment",
        "mechanism": "Team pace determines possession count and stat opportunities.",
        "data_advantage": "Matchup-specific pace modeling.",
        "decay_risk": "Medium-High.",
        "independence_notes": "Correlated with pace_differential (r~0.85).",
    },
    {
        "name": "team_off_rating",
        "module": "team.environment",
        "category": "team_environment",
        "mechanism": "Team offensive rating affects individual scoring efficiency.",
        "data_advantage": "Lineup-specific offensive ratings.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with lineup_effects (r~0.45).",
    },
    {
        "name": "team_def_rating",
        "module": "team.environment",
        "category": "team_environment",
        "mechanism": "Own team defense affects game flow (close vs blowout).",
        "data_advantage": "Game-flow modeling from team defense.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with game_script (r~0.35).",
    },
    {
        "name": "team_injury_impact",
        "module": "team.environment",
        "category": "team_environment",
        "mechanism": "Teammate injuries change role, usage, and minutes for remaining players.",
        "data_advantage": "Historical with/without impact modeling.",
        "decay_risk": "Low-Medium.",
        "independence_notes": "Correlated with usage_redistribution (r~0.70), lineup_effects (r~0.55).",
    },
    # ── Game context signals ──────────────────────────────────────────────
    {
        "name": "spread_implied_total",
        "module": "game.context",
        "category": "game_environment",
        "mechanism": "Spread and total imply team-specific scoring expectations.",
        "data_advantage": "Decomposition of total into team totals using spread.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with game_script (r~0.65), pace_differential (r~0.40).",
    },
    {
        "name": "game_importance",
        "module": "game.context",
        "category": "game_environment",
        "mechanism": "Playoff implications affect player effort and minutes.",
        "data_advantage": "Playoff probability modeling and rest-day prediction.",
        "decay_risk": "Low — late-season dynamics are complex.",
        "independence_notes": "Partially independent — seasonal timing signal.",
    },
    {
        "name": "schedule_density",
        "module": "game.context",
        "category": "game_environment",
        "mechanism": "Dense schedules (4 games in 5 nights) accumulate fatigue.",
        "data_advantage": "Rolling workload tracking.",
        "decay_risk": "Low.",
        "independence_notes": "Correlated with rest_effects (r~0.60), fatigue_impact (r~0.55).",
    },
    # ── Calibration & edge signals ────────────────────────────────────────
    {
        "name": "model_calibration_score",
        "module": "calibration",
        "category": "calibration",
        "mechanism": "How well-calibrated is our model probability vs actual outcomes.",
        "data_advantage": "Continuous calibration monitoring with Brier score.",
        "decay_risk": "N/A — meta-signal.",
        "independence_notes": "Independent quality metric.",
    },
    {
        "name": "edge_confidence",
        "module": "calibration",
        "category": "calibration",
        "mechanism": "Confidence interval on estimated edge.",
        "data_advantage": "Bayesian edge estimation with uncertainty.",
        "decay_risk": "N/A — meta-signal.",
        "independence_notes": "Depends on all component signals.",
    },
    # ── Market signals ────────────────────────────────────────────────────
    {
        "name": "opening_line_value",
        "module": "market",
        "category": "market",
        "mechanism": "Opening line vs our projection identifies early value.",
        "data_advantage": "Speed of line capture at market open.",
        "decay_risk": "Medium — market efficiency at open is increasing.",
        "independence_notes": "Correlated with market_odds_movement (r~0.40).",
    },
    {
        "name": "line_movement_direction",
        "module": "market",
        "category": "market",
        "mechanism": "Direction and magnitude of line movement signals sharp action.",
        "data_advantage": "Multi-book line tracking.",
        "decay_risk": "Medium.",
        "independence_notes": "Correlated with market_odds_movement (r~0.75).",
    },
    {
        "name": "cross_book_discrepancy",
        "module": "market",
        "category": "market",
        "mechanism": "Discrepancies between books indicate soft lines.",
        "data_advantage": "Real-time multi-book comparison.",
        "decay_risk": "Low-Medium — arbitrage-adjacent.",
        "independence_notes": "Partially independent — market-structure signal.",
    },
]


class EdgeSourceCatalog:
    """
    Master catalog of all signals in the codebase.
    Computes pairwise correlation matrix and flags non-independent pairs.
    """

    def __init__(self):
        self._catalog = [SignalEntry(signal_id=i, **s) for i, s in enumerate(_SIGNAL_CATALOG)]
        self._correlation_matrix: np.ndarray | None = None
        self._correlation_labels: list[str] = []

    @property
    def signals(self) -> list[SignalEntry]:
        return list(self._catalog)

    @property
    def n_signals(self) -> int:
        return len(self._catalog)

    def get_by_category(self, category: str) -> list[SignalEntry]:
        return [s for s in self._catalog if s.category == category]

    def get_by_module(self, module: str) -> list[SignalEntry]:
        return [s for s in self._catalog if module in s.module]

    def get_categories(self) -> list[str]:
        return sorted(set(s.category for s in self._catalog))

    def compute_correlation_matrix(
        self, signal_values: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute pairwise Pearson correlation matrix for all signals
        that have historical value arrays.

        Parameters
        ----------
        signal_values : dict mapping signal name -> 1D array of historical values.
                        All arrays must have the same length.

        Returns
        -------
        (correlation_matrix, signal_names)
        """
        names = sorted(signal_values.keys())
        if len(names) < 2:
            self._correlation_matrix = np.array([[1.0]])
            self._correlation_labels = names
            return self._correlation_matrix, self._correlation_labels

        # Build matrix
        n = len(names)
        first_key = names[0]
        m = len(signal_values[first_key])
        data = np.zeros((m, n))
        for j, name in enumerate(names):
            arr = np.array(signal_values[name], dtype=float)
            if len(arr) != m:
                raise ValueError(
                    f"Signal '{name}' has {len(arr)} values, expected {m}"
                )
            data[:, j] = arr

        # Pearson correlation
        corr = np.corrcoef(data, rowvar=False)
        # Handle NaN from constant columns
        corr = np.nan_to_num(corr, nan=0.0)

        self._correlation_matrix = corr
        self._correlation_labels = names
        return corr, names

    def get_non_independent_pairs(
        self, threshold: float = 0.5
    ) -> list[tuple[str, str, float]]:
        """
        Return pairs of signals with |correlation| > threshold.
        Must call compute_correlation_matrix() first.
        """
        if self._correlation_matrix is None:
            return []

        pairs = []
        n = len(self._correlation_labels)
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(self._correlation_matrix[i, j])
                if corr > threshold:
                    pairs.append((
                        self._correlation_labels[i],
                        self._correlation_labels[j],
                        round(float(self._correlation_matrix[i, j]), 3),
                    ))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    def generate_synthetic_correlation_matrix(self) -> tuple[np.ndarray, list[str]]:
        """
        Generate a correlation matrix from the documented correlations
        in the signal catalog (for visualization when historical data
        is not yet available).
        """
        import re

        names = [s.name for s in self._catalog]
        n = len(names)
        corr = np.eye(n)
        name_to_idx = {name: i for i, name in enumerate(names)}

        # Parse documented correlations from independence_notes
        corr_pattern = re.compile(r"(\w+)\s*\(r~([\d.]+)\)")

        for sig in self._catalog:
            i = name_to_idx[sig.name]
            matches = corr_pattern.findall(sig.independence_notes)
            for target_name, corr_val in matches:
                corr_val = float(corr_val)
                # Find the target signal index
                if target_name in name_to_idx:
                    j = name_to_idx[target_name]
                    corr[i, j] = corr_val
                    corr[j, i] = corr_val

        self._correlation_matrix = corr
        self._correlation_labels = names
        return corr, names

    def get_summary_table(self) -> list[dict]:
        """Return a summary table suitable for DataFrame display."""
        return [
            {
                "signal": s.name,
                "module": s.module,
                "category": s.category,
                "mechanism": s.mechanism[:100] + "..." if len(s.mechanism) > 100 else s.mechanism,
                "decay_risk": s.decay_risk.split("—")[0].strip() if "—" in s.decay_risk else s.decay_risk.split(".")[0].strip(),
            }
            for s in self._catalog
        ]
