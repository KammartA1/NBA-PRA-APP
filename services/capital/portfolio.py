"""
services/capital/portfolio.py
==============================
Portfolio management for concurrent sports bets.

Handles:
  - Correlation matrix of pending bets
  - Optimal allocation with correlation adjustment
  - Concentration risk checks
  - Exposure limits by player, market, game, and time window
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)


@dataclass
class PendingBet:
    """A bet that has been sized but not yet settled."""
    bet_id: str
    player: str
    market: str
    game: str
    direction: str           # over / under
    win_prob: float
    odds_decimal: float
    kelly_stake_pct: float   # Kelly-optimal stake as % of bankroll
    stake_dollars: float
    correlation_group: str   # e.g., game_id for same-game bets


@dataclass
class ConcentrationAlert:
    """Warning about portfolio concentration risk."""
    alert_type: str          # player, market, game, total
    entity: str              # The thing that's concentrated
    exposure_pct: float      # Current exposure %
    limit_pct: float         # Maximum allowed %
    severity: str            # warning, critical
    recommendation: str


@dataclass
class PortfolioAllocation:
    """Optimized portfolio allocation result."""
    bets: List[Dict[str, Any]]
    total_exposure_pct: float
    correlation_adjustment_pct: float
    concentration_alerts: List[ConcentrationAlert]
    portfolio_edge_pct: float
    portfolio_variance: float
    diversification_ratio: float
    is_within_limits: bool


class PortfolioManager:
    """Manages portfolio of concurrent bets with correlation awareness.

    Key insight: NBA prop bets are highly correlated within the same game.
    If you bet Player A overs on points AND Player B overs on points in
    the same game, both bets are correlated through game pace/score.
    """

    # Exposure limits (% of bankroll)
    MAX_TOTAL_EXPOSURE = 15.0
    MAX_PLAYER_EXPOSURE = 5.0
    MAX_MARKET_EXPOSURE = 8.0
    MAX_GAME_EXPOSURE = 10.0
    MAX_SINGLE_BET = 3.0

    # Default correlation priors
    SAME_GAME_CORRELATION = 0.35     # Bets in same game are correlated
    SAME_PLAYER_CORRELATION = 0.60   # Same player different markets
    SAME_MARKET_CORRELATION = 0.15   # Same market different games
    CROSS_GAME_CORRELATION = 0.05    # Different games minimal correlation

    def __init__(
        self,
        max_total_exposure: float = 15.0,
        max_player_exposure: float = 5.0,
        max_game_exposure: float = 10.0,
    ):
        self.MAX_TOTAL_EXPOSURE = max_total_exposure
        self.MAX_PLAYER_EXPOSURE = max_player_exposure
        self.MAX_GAME_EXPOSURE = max_game_exposure

    def estimate_correlation_matrix(
        self,
        bets: List[PendingBet],
    ) -> np.ndarray:
        """Estimate pairwise correlation between pending bets.

        Uses structural priors since we can't observe actual correlations
        for unsettled bets.
        """
        n = len(bets)
        corr = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                rho = self._pairwise_correlation(bets[i], bets[j])
                corr[i, j] = rho
                corr[j, i] = rho

        return corr

    def _pairwise_correlation(self, a: PendingBet, b: PendingBet) -> float:
        """Estimate correlation between two bets."""
        rho = self.CROSS_GAME_CORRELATION

        # Same player (highest correlation)
        if a.player == b.player:
            rho = self.SAME_PLAYER_CORRELATION
            # Same player, same direction = very high
            if a.direction == b.direction:
                rho = 0.75
        # Same game
        elif a.game == b.game:
            rho = self.SAME_GAME_CORRELATION
            # Same game, same direction
            if a.direction == b.direction:
                rho = 0.45
        # Same market type across games
        elif a.market == b.market:
            rho = self.SAME_MARKET_CORRELATION

        return rho

    def check_concentration(
        self,
        bets: List[PendingBet],
        bankroll: float,
    ) -> List[ConcentrationAlert]:
        """Check for concentration risk in the current portfolio."""
        alerts: List[ConcentrationAlert] = []

        if not bets or bankroll <= 0:
            return alerts

        # Total exposure
        total_exp = sum(b.stake_dollars for b in bets) / bankroll * 100
        if total_exp > self.MAX_TOTAL_EXPOSURE:
            alerts.append(ConcentrationAlert(
                alert_type="total",
                entity="portfolio",
                exposure_pct=round(total_exp, 1),
                limit_pct=self.MAX_TOTAL_EXPOSURE,
                severity="critical" if total_exp > self.MAX_TOTAL_EXPOSURE * 1.5 else "warning",
                recommendation=f"Reduce total exposure from {total_exp:.1f}% to {self.MAX_TOTAL_EXPOSURE}%",
            ))

        # Player concentration
        player_exposure: Dict[str, float] = {}
        for b in bets:
            player_exposure[b.player] = player_exposure.get(b.player, 0) + b.stake_dollars
        for player, exposure in player_exposure.items():
            exp_pct = exposure / bankroll * 100
            if exp_pct > self.MAX_PLAYER_EXPOSURE:
                alerts.append(ConcentrationAlert(
                    alert_type="player",
                    entity=player,
                    exposure_pct=round(exp_pct, 1),
                    limit_pct=self.MAX_PLAYER_EXPOSURE,
                    severity="critical" if exp_pct > self.MAX_PLAYER_EXPOSURE * 2 else "warning",
                    recommendation=f"Reduce {player} exposure from {exp_pct:.1f}% to {self.MAX_PLAYER_EXPOSURE}%",
                ))

        # Game concentration
        game_exposure: Dict[str, float] = {}
        for b in bets:
            game_exposure[b.game] = game_exposure.get(b.game, 0) + b.stake_dollars
        for game, exposure in game_exposure.items():
            exp_pct = exposure / bankroll * 100
            if exp_pct > self.MAX_GAME_EXPOSURE:
                alerts.append(ConcentrationAlert(
                    alert_type="game",
                    entity=game,
                    exposure_pct=round(exp_pct, 1),
                    limit_pct=self.MAX_GAME_EXPOSURE,
                    severity="critical" if exp_pct > self.MAX_GAME_EXPOSURE * 1.5 else "warning",
                    recommendation=f"Reduce game exposure from {exp_pct:.1f}% to {self.MAX_GAME_EXPOSURE}%",
                ))

        return alerts

    def optimize_allocation(
        self,
        bets: List[PendingBet],
        bankroll: float,
    ) -> PortfolioAllocation:
        """Optimize allocation across pending bets considering correlations.

        Adjusts Kelly-optimal stakes downward when bets are correlated
        and enforces concentration limits.
        """
        if not bets or bankroll <= 0:
            return PortfolioAllocation(
                bets=[], total_exposure_pct=0.0, correlation_adjustment_pct=0.0,
                concentration_alerts=[], portfolio_edge_pct=0.0,
                portfolio_variance=0.0, diversification_ratio=1.0,
                is_within_limits=True,
            )

        n = len(bets)
        corr_matrix = self.estimate_correlation_matrix(bets)

        # Individual Kelly stakes
        kelly_stakes = np.array([b.kelly_stake_pct for b in bets])

        # Correlation adjustment
        # f_adj_i = f_i / (1 + sum_j(|rho_ij| * f_j))
        adjusted_stakes = np.zeros(n)
        for i in range(n):
            if kelly_stakes[i] <= 0:
                continue
            corr_penalty = sum(
                abs(corr_matrix[i, j]) * kelly_stakes[j]
                for j in range(n) if j != i and kelly_stakes[j] > 0
            )
            adjusted_stakes[i] = kelly_stakes[i] / (1.0 + corr_penalty)

        # Cap individual bets
        adjusted_stakes = np.minimum(adjusted_stakes, self.MAX_SINGLE_BET)

        # Cap total exposure
        total = np.sum(adjusted_stakes)
        if total > self.MAX_TOTAL_EXPOSURE:
            scale = self.MAX_TOTAL_EXPOSURE / total
            adjusted_stakes *= scale

        # Apply player limits
        player_stakes: Dict[str, List[int]] = {}
        for i, b in enumerate(bets):
            player_stakes.setdefault(b.player, []).append(i)

        for player, indices in player_stakes.items():
            player_total = sum(adjusted_stakes[i] for i in indices)
            if player_total > self.MAX_PLAYER_EXPOSURE:
                scale = self.MAX_PLAYER_EXPOSURE / player_total
                for i in indices:
                    adjusted_stakes[i] *= scale

        # Apply game limits
        game_stakes: Dict[str, List[int]] = {}
        for i, b in enumerate(bets):
            game_stakes.setdefault(b.game, []).append(i)

        for game, indices in game_stakes.items():
            game_total = sum(adjusted_stakes[i] for i in indices)
            if game_total > self.MAX_GAME_EXPOSURE:
                scale = self.MAX_GAME_EXPOSURE / game_total
                for i in indices:
                    adjusted_stakes[i] *= scale

        # Compute portfolio metrics
        dollar_stakes = adjusted_stakes / 100.0 * bankroll
        total_exposure = float(np.sum(adjusted_stakes))

        correlation_adj = (1.0 - float(np.sum(adjusted_stakes)) / max(float(np.sum(kelly_stakes)), 1e-10)) * 100

        # Portfolio edge (weighted average)
        edges = np.array([
            (b.win_prob - 1.0 / b.odds_decimal) * 100
            for b in bets
        ])
        weighted_edge = float(np.average(edges, weights=np.maximum(adjusted_stakes, 1e-10)))

        # Portfolio variance
        odds_payoff = np.array([b.odds_decimal - 1.0 for b in bets])
        win_probs = np.array([b.win_prob for b in bets])
        individual_var = win_probs * (1 - win_probs) * odds_payoff ** 2
        stake_fracs = adjusted_stakes / 100.0

        portfolio_var = 0.0
        for i in range(n):
            for j in range(n):
                cov_ij = corr_matrix[i, j] * np.sqrt(individual_var[i] * individual_var[j])
                portfolio_var += stake_fracs[i] * stake_fracs[j] * cov_ij
        portfolio_var = max(portfolio_var, 0.0)

        # Diversification ratio
        undiversified_vol = float(np.sum(stake_fracs * np.sqrt(individual_var)))
        diversified_vol = np.sqrt(portfolio_var)
        div_ratio = undiversified_vol / max(diversified_vol, 1e-10)

        # Build output
        bet_details = []
        for i, b in enumerate(bets):
            bet_details.append({
                "bet_id": b.bet_id,
                "player": b.player,
                "market": b.market,
                "game": b.game,
                "direction": b.direction,
                "kelly_stake_pct": round(kelly_stakes[i], 3),
                "adjusted_stake_pct": round(adjusted_stakes[i], 3),
                "dollar_amount": round(dollar_stakes[i], 2),
                "edge_pct": round(edges[i], 2),
                "reduction_pct": round((1 - adjusted_stakes[i] / max(kelly_stakes[i], 1e-10)) * 100, 1),
            })

        alerts = self.check_concentration(bets, bankroll)

        return PortfolioAllocation(
            bets=bet_details,
            total_exposure_pct=round(total_exposure, 2),
            correlation_adjustment_pct=round(correlation_adj, 1),
            concentration_alerts=alerts,
            portfolio_edge_pct=round(weighted_edge, 3),
            portfolio_variance=round(portfolio_var, 6),
            diversification_ratio=round(div_ratio, 3),
            is_within_limits=len([a for a in alerts if a.severity == "critical"]) == 0,
        )
