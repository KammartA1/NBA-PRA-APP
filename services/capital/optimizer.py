"""
services/capital/optimizer.py
==============================
Top-level capital optimizer that combines Kelly sizing, portfolio
management, and risk constraints into a single allocation decision.

This is the entry point for sizing any bet or group of bets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from services.capital.kelly import KellyCalculator, KellyResult
from services.capital.risk_adjusted import RiskMetrics, RiskReport
from services.capital.portfolio import PortfolioManager, PendingBet, PortfolioAllocation

log = logging.getLogger(__name__)


@dataclass
class OptimizationConstraints:
    """Constraints applied to the optimizer."""
    # Kelly parameters
    kelly_fraction: float = 0.25
    min_edge_pct: float = 2.0
    max_single_bet_pct: float = 3.0

    # Portfolio limits
    max_total_exposure_pct: float = 15.0
    max_player_exposure_pct: float = 5.0
    max_game_exposure_pct: float = 10.0

    # Risk limits
    max_drawdown_to_reduce: float = 15.0   # % drawdown to start reducing
    max_drawdown_to_halt: float = 25.0     # % drawdown to halt betting
    target_daily_var_pct: float = 5.0      # Target max daily VaR

    # System state
    system_state: str = "ACTIVE"           # ACTIVE, REDUCED, SUSPENDED, KILLED
    state_multiplier: float = 1.0          # Size multiplier from system state


@dataclass
class OptimizedBet:
    """Final optimized bet sizing decision."""
    bet_id: str
    player: str
    market: str
    direction: str
    win_prob: float
    odds_decimal: float
    edge_pct: float
    raw_kelly_pct: float
    adjusted_stake_pct: float
    final_stake_dollars: float
    adjustments_applied: List[str]
    should_bet: bool
    rejection_reason: str


class CapitalOptimizer:
    """Top-level optimizer combining all capital management systems.

    Usage:
        optimizer = CapitalOptimizer(bankroll=5000)
        results = optimizer.optimize_bets(pending_bets)
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        constraints: OptimizationConstraints | None = None,
        historical_pnl: np.ndarray | None = None,
    ):
        self.bankroll = bankroll
        self.constraints = constraints or OptimizationConstraints()
        self.historical_pnl = historical_pnl

        self.kelly = KellyCalculator(
            default_fraction=self.constraints.kelly_fraction,
            max_bet_pct=self.constraints.max_single_bet_pct,
            min_edge_pct=self.constraints.min_edge_pct,
        )
        self.risk = RiskMetrics()
        self.portfolio = PortfolioManager(
            max_total_exposure=self.constraints.max_total_exposure_pct,
            max_player_exposure=self.constraints.max_player_exposure_pct,
            max_game_exposure=self.constraints.max_game_exposure_pct,
        )

        # Compute current risk state if historical data available
        self._risk_report: Optional[RiskReport] = None
        if historical_pnl is not None and len(historical_pnl) >= 10:
            self._risk_report = self.risk.compute(historical_pnl, initial_bankroll=bankroll)

    def optimize_single(
        self,
        win_prob: float,
        odds_decimal: float,
        player: str = "",
        market: str = "",
        direction: str = "over",
        prob_std: float = 0.0,
    ) -> OptimizedBet:
        """Optimize sizing for a single bet."""
        adjustments: List[str] = []

        # 1. Kelly sizing
        kelly_result = self.kelly.compute(
            win_prob=win_prob,
            odds_decimal=odds_decimal,
            bankroll=self.bankroll,
            prob_std=prob_std,
            fraction=self.constraints.kelly_fraction,
        )

        if not kelly_result.is_positive_ev:
            return OptimizedBet(
                bet_id="", player=player, market=market, direction=direction,
                win_prob=win_prob, odds_decimal=odds_decimal,
                edge_pct=kelly_result.edge_pct,
                raw_kelly_pct=kelly_result.full_kelly_pct,
                adjusted_stake_pct=0.0, final_stake_dollars=0.0,
                adjustments_applied=["negative_ev"],
                should_bet=False, rejection_reason="Negative expected value",
            )

        stake_pct = kelly_result.final_stake_pct

        if kelly_result.edge_pct < self.constraints.min_edge_pct:
            return OptimizedBet(
                bet_id="", player=player, market=market, direction=direction,
                win_prob=win_prob, odds_decimal=odds_decimal,
                edge_pct=kelly_result.edge_pct,
                raw_kelly_pct=kelly_result.full_kelly_pct,
                adjusted_stake_pct=0.0, final_stake_dollars=0.0,
                adjustments_applied=["below_min_edge"],
                should_bet=False,
                rejection_reason=f"Edge {kelly_result.edge_pct:.1f}% < minimum {self.constraints.min_edge_pct}%",
            )

        # 2. System state adjustment
        if self.constraints.system_state == "KILLED":
            return OptimizedBet(
                bet_id="", player=player, market=market, direction=direction,
                win_prob=win_prob, odds_decimal=odds_decimal,
                edge_pct=kelly_result.edge_pct,
                raw_kelly_pct=kelly_result.full_kelly_pct,
                adjusted_stake_pct=0.0, final_stake_dollars=0.0,
                adjustments_applied=["system_killed"],
                should_bet=False, rejection_reason="System is in KILLED state",
            )
        elif self.constraints.system_state == "SUSPENDED":
            return OptimizedBet(
                bet_id="", player=player, market=market, direction=direction,
                win_prob=win_prob, odds_decimal=odds_decimal,
                edge_pct=kelly_result.edge_pct,
                raw_kelly_pct=kelly_result.full_kelly_pct,
                adjusted_stake_pct=0.0, final_stake_dollars=0.0,
                adjustments_applied=["system_suspended"],
                should_bet=False, rejection_reason="System is SUSPENDED",
            )
        elif self.constraints.state_multiplier < 1.0:
            stake_pct *= self.constraints.state_multiplier
            adjustments.append(f"state_reduction_{self.constraints.state_multiplier:.0%}")

        # 3. Drawdown adjustment
        if self._risk_report and self._risk_report.current_drawdown_pct > 0:
            dd = self._risk_report.current_drawdown_pct
            if dd >= self.constraints.max_drawdown_to_halt:
                return OptimizedBet(
                    bet_id="", player=player, market=market, direction=direction,
                    win_prob=win_prob, odds_decimal=odds_decimal,
                    edge_pct=kelly_result.edge_pct,
                    raw_kelly_pct=kelly_result.full_kelly_pct,
                    adjusted_stake_pct=0.0, final_stake_dollars=0.0,
                    adjustments_applied=["drawdown_halt"],
                    should_bet=False,
                    rejection_reason=f"Drawdown {dd:.1f}% exceeds halt threshold {self.constraints.max_drawdown_to_halt}%",
                )
            elif dd >= self.constraints.max_drawdown_to_reduce:
                # Linear reduction from full to 50% as drawdown approaches halt
                reduction = 0.5 + 0.5 * (
                    1.0 - (dd - self.constraints.max_drawdown_to_reduce)
                    / (self.constraints.max_drawdown_to_halt - self.constraints.max_drawdown_to_reduce)
                )
                reduction = np.clip(reduction, 0.25, 1.0)
                stake_pct *= reduction
                adjustments.append(f"drawdown_reduction_{reduction:.0%}")

        # 4. Variance adjustment
        if self._risk_report and self._risk_report.variance_ratio > 2.0:
            var_penalty = 1.0 / self._risk_report.variance_ratio
            stake_pct *= var_penalty
            adjustments.append(f"variance_penalty_{var_penalty:.0%}")

        # 5. Apply hard cap
        stake_pct = min(stake_pct, self.constraints.max_single_bet_pct)

        final_dollars = self.bankroll * (stake_pct / 100.0)

        return OptimizedBet(
            bet_id="",
            player=player,
            market=market,
            direction=direction,
            win_prob=win_prob,
            odds_decimal=odds_decimal,
            edge_pct=kelly_result.edge_pct,
            raw_kelly_pct=kelly_result.full_kelly_pct,
            adjusted_stake_pct=round(stake_pct, 4),
            final_stake_dollars=round(final_dollars, 2),
            adjustments_applied=adjustments if adjustments else ["none"],
            should_bet=stake_pct > 0,
            rejection_reason="",
        )

    def optimize_batch(
        self,
        bets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Optimize a batch of bets with portfolio-level constraints.

        Each bet dict should have: player, market, game, direction,
        win_prob, odds_decimal, prob_std (optional).
        """
        # First pass: individual Kelly sizing
        pending: List[PendingBet] = []
        individual_results: List[OptimizedBet] = []

        for i, bet in enumerate(bets):
            result = self.optimize_single(
                win_prob=bet["win_prob"],
                odds_decimal=bet["odds_decimal"],
                player=bet.get("player", f"player_{i}"),
                market=bet.get("market", "unknown"),
                direction=bet.get("direction", "over"),
                prob_std=bet.get("prob_std", 0.0),
            )
            individual_results.append(result)

            if result.should_bet:
                pending.append(PendingBet(
                    bet_id=f"bet_{i}",
                    player=result.player,
                    market=result.market,
                    game=bet.get("game", "unknown"),
                    direction=result.direction,
                    win_prob=result.win_prob,
                    odds_decimal=result.odds_decimal,
                    kelly_stake_pct=result.adjusted_stake_pct,
                    stake_dollars=result.final_stake_dollars,
                    correlation_group=bet.get("game", "unknown"),
                ))

        # Second pass: portfolio optimization
        if pending:
            portfolio_result = self.portfolio.optimize_allocation(pending, self.bankroll)

            # Update individual results with portfolio-adjusted stakes
            portfolio_map = {b["bet_id"]: b for b in portfolio_result.bets}
            for i, result in enumerate(individual_results):
                bid = f"bet_{i}"
                if bid in portfolio_map:
                    pbet = portfolio_map[bid]
                    result.adjusted_stake_pct = pbet["adjusted_stake_pct"]
                    result.final_stake_dollars = pbet["dollar_amount"]
                    if pbet["reduction_pct"] > 5:
                        result.adjustments_applied.append(
                            f"correlation_reduction_{pbet['reduction_pct']:.0f}%"
                        )
        else:
            portfolio_result = self.portfolio.optimize_allocation([], self.bankroll)

        # Build summary
        approved = [r for r in individual_results if r.should_bet]
        rejected = [r for r in individual_results if not r.should_bet]

        return {
            "bets": [
                {
                    "player": r.player,
                    "market": r.market,
                    "direction": r.direction,
                    "edge_pct": r.edge_pct,
                    "stake_pct": r.adjusted_stake_pct,
                    "stake_dollars": r.final_stake_dollars,
                    "adjustments": r.adjustments_applied,
                    "approved": r.should_bet,
                    "rejection_reason": r.rejection_reason,
                }
                for r in individual_results
            ],
            "summary": {
                "total_bets": len(bets),
                "approved": len(approved),
                "rejected": len(rejected),
                "total_exposure_pct": portfolio_result.total_exposure_pct,
                "total_dollars_at_risk": sum(r.final_stake_dollars for r in approved),
                "portfolio_edge_pct": portfolio_result.portfolio_edge_pct,
                "diversification_ratio": portfolio_result.diversification_ratio,
                "within_limits": portfolio_result.is_within_limits,
            },
            "alerts": [
                {
                    "type": a.alert_type,
                    "entity": a.entity,
                    "exposure_pct": a.exposure_pct,
                    "limit_pct": a.limit_pct,
                    "severity": a.severity,
                    "recommendation": a.recommendation,
                }
                for a in portfolio_result.concentration_alerts
            ],
        }
