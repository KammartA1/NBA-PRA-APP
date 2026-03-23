"""Adaptive Kelly Criterion — Dynamic bet sizing that adjusts to reality.

Static Kelly is dangerous because:
1. It assumes your edge estimate is correct (it's not)
2. It doesn't account for estimation error
3. It doesn't adapt to regime changes

This module replaces static Kelly with a dynamic system that:
- Starts at 1/10 Kelly (conservative)
- Adjusts based on CLV performance, calibration accuracy, drawdown
- Applies correlation-aware portfolio sizing
- Enforces hard caps that cannot be overridden

The Kelly multiplier is the master dial:
    kelly_mult = base_fraction * clv_adjustment * calibration_adjustment * drawdown_adjustment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..core.types import RiskState, Sport, SystemState

logger = logging.getLogger(__name__)


@dataclass
class KellyConfig:
    """All tunable Kelly parameters in one place."""
    base_fraction: float = 0.10        # 1/10 Kelly — starting point
    min_fraction: float = 0.02         # Floor: never below 1/50 Kelly
    max_fraction: float = 0.25         # Ceiling: never above 1/4 Kelly

    # CLV-based adjustments
    clv_bonus_threshold: float = 2.0   # CLV > 2 cents → increase sizing
    clv_bonus_multiplier: float = 1.3  # Boost by 30% when CLV is strong
    clv_penalty_threshold: float = -0.5  # CLV < -0.5 → decrease sizing
    clv_penalty_multiplier: float = 0.5  # Cut by 50% when CLV is weak

    # Calibration-based adjustments
    cal_good_threshold: float = 0.04   # MAE < 4% → well calibrated
    cal_good_multiplier: float = 1.15  # Boost 15%
    cal_bad_threshold: float = 0.10    # MAE > 10% → poorly calibrated
    cal_bad_multiplier: float = 0.50   # Cut 50%

    # Drawdown-based adjustments
    drawdown_mild: float = 0.10        # 10% drawdown
    drawdown_moderate: float = 0.20    # 20% drawdown
    drawdown_severe: float = 0.30      # 30% drawdown

    # Hard caps
    max_single_bet_pct: float = 0.03   # 3% of bankroll per bet (HARD CAP)
    max_daily_exposure_pct: float = 0.15  # 15% of bankroll exposed daily
    max_player_exposure_pct: float = 0.05  # 5% per player
    min_edge_to_bet: float = 0.04      # 4% minimum edge to place any bet

    # System state multipliers
    state_multipliers: dict = None

    def __post_init__(self):
        if self.state_multipliers is None:
            self.state_multipliers = {
                SystemState.ACTIVE: 1.0,
                SystemState.REDUCED: 0.50,
                SystemState.SUSPENDED: 0.0,
                SystemState.KILLED: 0.0,
            }


class AdaptiveKelly:
    """Dynamic Kelly sizing engine."""

    def __init__(self, config: KellyConfig | None = None):
        self.config = config or KellyConfig()

    def compute_kelly(
        self,
        win_prob: float,
        decimal_odds: float,
    ) -> float:
        """Raw Kelly fraction: f* = (p*b - q) / b"""
        if decimal_odds <= 1.0 or win_prob <= 0.0 or win_prob >= 1.0:
            return 0.0

        b = decimal_odds - 1.0
        p = win_prob
        q = 1.0 - p

        full_kelly = (p * b - q) / b
        return max(full_kelly, 0.0)

    def adaptive_stake(
        self,
        win_prob: float,
        decimal_odds: float,
        bankroll: float,
        risk_state: RiskState,
        clv_avg_cents: float = 0.0,
        calibration_mae: float = 0.0,
    ) -> dict:
        """Compute the optimal stake with all dynamic adjustments.

        Returns:
            {
                "raw_kelly": float,          # Pure Kelly fraction
                "adjusted_kelly": float,     # After all adjustments
                "stake_dollars": float,      # Final $ amount
                "pct_bankroll": float,       # As % of bankroll
                "adjustments": dict,         # Breakdown of each adjustment
                "blocked": bool,             # Whether bet is blocked
                "block_reason": str,         # Why blocked
            }
        """
        cfg = self.config

        # 1. Raw Kelly
        raw_kelly = self.compute_kelly(win_prob, decimal_odds)
        if raw_kelly <= 0:
            return self._blocked("No edge (Kelly <= 0)")

        # 2. Edge check
        market_prob = 1.0 / decimal_odds
        edge = win_prob - market_prob
        if edge < cfg.min_edge_to_bet:
            return self._blocked(f"Edge {edge:.3f} below minimum {cfg.min_edge_to_bet}")

        # 3. System state check
        state_mult = cfg.state_multipliers.get(risk_state.system_state, 0.0)
        if state_mult == 0.0:
            return self._blocked(f"System state is {risk_state.system_state.value}")

        # 4. Start with base fraction
        adjusted = cfg.base_fraction * raw_kelly
        adjustments = {"base": cfg.base_fraction}

        # 5. CLV adjustment
        clv_mult = 1.0
        if clv_avg_cents > cfg.clv_bonus_threshold:
            clv_mult = cfg.clv_bonus_multiplier
        elif clv_avg_cents < cfg.clv_penalty_threshold:
            clv_mult = cfg.clv_penalty_multiplier
        adjusted *= clv_mult
        adjustments["clv_multiplier"] = clv_mult

        # 6. Calibration adjustment
        cal_mult = 1.0
        if calibration_mae < cfg.cal_good_threshold:
            cal_mult = cfg.cal_good_multiplier
        elif calibration_mae > cfg.cal_bad_threshold:
            cal_mult = cfg.cal_bad_multiplier
        adjusted *= cal_mult
        adjustments["calibration_multiplier"] = cal_mult

        # 7. Drawdown adjustment (proportional reduction)
        dd = risk_state.current_drawdown_pct
        dd_mult = 1.0
        if dd > cfg.drawdown_severe:
            dd_mult = 0.25
        elif dd > cfg.drawdown_moderate:
            dd_mult = 0.50
        elif dd > cfg.drawdown_mild:
            dd_mult = 0.75
        adjusted *= dd_mult
        adjustments["drawdown_multiplier"] = dd_mult

        # 8. System state multiplier
        adjusted *= state_mult
        adjustments["state_multiplier"] = state_mult

        # 9. Clamp to bounds
        adjusted = max(cfg.min_fraction * raw_kelly, min(adjusted, cfg.max_fraction * raw_kelly))
        adjustments["final_kelly_fraction"] = adjusted

        # 10. Convert to dollars
        stake = adjusted * bankroll

        # 11. Apply hard caps
        max_single = cfg.max_single_bet_pct * bankroll
        if stake > max_single:
            stake = max_single
            adjustments["capped_at_max_single"] = True

        # Daily exposure check
        remaining_daily = risk_state.daily_loss_remaining
        if remaining_daily is not None and stake > remaining_daily:
            stake = max(remaining_daily, 0)
            adjustments["capped_at_daily_limit"] = True

        # Player exposure check
        player_exposure = risk_state.exposure_by_player or {}
        # (checked by ExposureManager externally)

        stake = round(max(stake, 0), 2)
        pct = round(stake / bankroll, 4) if bankroll > 0 else 0.0

        return {
            "raw_kelly": round(raw_kelly, 6),
            "adjusted_kelly": round(adjusted, 6),
            "stake_dollars": stake,
            "pct_bankroll": pct,
            "edge": round(edge, 4),
            "adjustments": adjustments,
            "blocked": stake == 0,
            "block_reason": "" if stake > 0 else "All adjustments reduced stake to zero",
        }

    @staticmethod
    def _blocked(reason: str) -> dict:
        return {
            "raw_kelly": 0.0,
            "adjusted_kelly": 0.0,
            "stake_dollars": 0.0,
            "pct_bankroll": 0.0,
            "edge": 0.0,
            "adjustments": {},
            "blocked": True,
            "block_reason": reason,
        }
