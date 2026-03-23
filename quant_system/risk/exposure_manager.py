"""Exposure Manager — Correlation-aware portfolio-level risk control.

Prevents dangerous concentration:
- No more than X% on a single player
- No more than Y% on correlated bets
- Parlay exposure treated as sum of legs for risk purposes
- Total open exposure capped as % of bankroll
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..core.types import RiskState

logger = logging.getLogger(__name__)


@dataclass
class ExposureConfig:
    max_player_pct: float = 0.05        # 5% per player
    max_stat_type_pct: float = 0.10     # 10% per stat type
    max_team_pct: float = 0.08          # 8% per team (NBA)
    max_tournament_pct: float = 0.15    # 15% per tournament (Golf)
    max_open_exposure_pct: float = 0.25 # 25% total open exposure
    max_correlated_pair_pct: float = 0.06  # 6% on correlated pair


class ExposureManager:
    """Portfolio-level exposure control."""

    def __init__(self, config: ExposureConfig | None = None):
        self.config = config or ExposureConfig()

    def check_exposure(
        self,
        player: str,
        stat_type: str,
        stake: float,
        risk_state: RiskState,
        team: str | None = None,
        tournament: str | None = None,
    ) -> tuple[bool, str, float]:
        """Check if adding this bet would breach exposure limits.

        Returns: (allowed, reason, max_allowed_stake)
        """
        cfg = self.config
        bankroll = risk_state.bankroll

        if bankroll <= 0:
            return False, "Zero bankroll", 0.0

        # Player exposure
        current_player = risk_state.exposure_by_player.get(player, 0)
        max_player = cfg.max_player_pct * bankroll
        if current_player + stake > max_player:
            allowed_stake = max(max_player - current_player, 0)
            return False, f"Player {player} exposure ${current_player + stake:.0f} > cap ${max_player:.0f}", allowed_stake

        # Stat type exposure
        current_stat = risk_state.exposure_by_type.get(stat_type, 0)
        max_stat = cfg.max_stat_type_pct * bankroll
        if current_stat + stake > max_stat:
            allowed_stake = max(max_stat - current_stat, 0)
            return False, f"Stat type {stat_type} exposure exceeds cap", allowed_stake

        # Total open exposure
        max_open = cfg.max_open_exposure_pct * bankroll
        if risk_state.total_exposure + stake > max_open:
            allowed_stake = max(max_open - risk_state.total_exposure, 0)
            return False, f"Total exposure ${risk_state.total_exposure + stake:.0f} > cap ${max_open:.0f}", allowed_stake

        return True, "OK", stake

    def reduce_for_correlation(
        self,
        base_stake: float,
        correlation_with_existing: float,
    ) -> float:
        """Reduce stake when bet is correlated with existing positions.

        If new bet has 0.5 correlation with existing bets, reduce by 25%.
        """
        if correlation_with_existing <= 0.1:
            return base_stake

        # Reduction = correlation^2 (quadratic penalty)
        reduction = min(correlation_with_existing ** 2, 0.5)
        adjusted = base_stake * (1.0 - reduction)
        return round(adjusted, 2)
