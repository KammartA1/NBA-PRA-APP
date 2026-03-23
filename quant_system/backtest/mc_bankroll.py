"""Monte Carlo Bankroll Simulator — 10,000 parallel universe simulation.

Answers the question: "Given my actual edge and variance, what's the
probability distribution of future bankroll outcomes?"

This is NOT the same as the model's Monte Carlo for probabilities.
This simulates the BANKROLL PATH over many bets to estimate:
- Probability of ruin
- Expected growth rate
- Confidence intervals on final bankroll
- Worst-case drawdown distribution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MCConfig:
    n_simulations: int = 10_000
    n_bets_per_path: int = 500         # Simulate 500 bets into the future
    initial_bankroll: float = 1000.0
    ruin_threshold: float = 50.0       # Bankroll below $50 = ruin


class BankrollSimulator:
    """Monte Carlo bankroll path simulation."""

    def __init__(self, config: MCConfig | None = None):
        self.config = config or MCConfig()

    def simulate(
        self,
        avg_edge: float,
        avg_odds_decimal: float,
        avg_stake_pct: float,
        win_rate: float,
        stake_std_pct: float = 0.005,
    ) -> dict:
        """Simulate 10,000 bankroll paths.

        Args:
            avg_edge: Average edge per bet (e.g., 0.05 for 5%)
            avg_odds_decimal: Average decimal odds (e.g., 1.91)
            avg_stake_pct: Average stake as % of bankroll (e.g., 0.02)
            win_rate: Historical win rate (e.g., 0.54)
            stake_std_pct: Standard deviation of stake % (variance in sizing)

        Returns:
            {
                "ruin_probability": float,
                "median_final": float,
                "mean_final": float,
                "p5_final": float,
                "p25_final": float,
                "p75_final": float,
                "p95_final": float,
                "max_drawdown_median": float,
                "max_drawdown_p95": float,
                "growth_rate": float,
                "paths_profitable": float,      # % of paths ending above initial
                "paths_doubled": float,          # % of paths ending 2x initial
            }
        """
        cfg = self.config
        rng = np.random.default_rng(42)

        n_sims = cfg.n_simulations
        n_bets = cfg.n_bets_per_path
        initial = cfg.initial_bankroll

        # Pre-allocate
        bankrolls = np.full((n_sims, n_bets + 1), initial)
        max_drawdowns = np.zeros(n_sims)

        payout = avg_odds_decimal - 1.0

        for step in range(n_bets):
            current = bankrolls[:, step]

            # Randomize stake sizes slightly
            stake_pcts = rng.normal(avg_stake_pct, stake_std_pct, n_sims)
            stake_pcts = np.clip(stake_pcts, 0.005, 0.05)  # Between 0.5% and 5%
            stakes = current * stake_pcts

            # Determine wins/losses (Bernoulli trials at win_rate)
            wins = rng.random(n_sims) < win_rate

            # P&L
            pnl = np.where(wins, stakes * payout, -stakes)
            bankrolls[:, step + 1] = current + pnl

            # Prevent negative bankrolls (can't bet more than you have)
            bankrolls[:, step + 1] = np.maximum(bankrolls[:, step + 1], 0)

        # Compute metrics
        final_bankrolls = bankrolls[:, -1]

        # Max drawdown per path
        for i in range(n_sims):
            path = bankrolls[i]
            peak = np.maximum.accumulate(path)
            drawdowns = (peak - path) / np.maximum(peak, 1.0)
            max_drawdowns[i] = np.max(drawdowns)

        # Growth rate
        growth_rates = np.log(np.maximum(final_bankrolls, 0.01) / initial) / n_bets
        avg_growth = float(np.mean(growth_rates))

        ruin_count = np.sum(final_bankrolls < cfg.ruin_threshold)

        return {
            "ruin_probability": round(float(ruin_count / n_sims), 4),
            "median_final": round(float(np.median(final_bankrolls)), 2),
            "mean_final": round(float(np.mean(final_bankrolls)), 2),
            "p5_final": round(float(np.percentile(final_bankrolls, 5)), 2),
            "p25_final": round(float(np.percentile(final_bankrolls, 25)), 2),
            "p75_final": round(float(np.percentile(final_bankrolls, 75)), 2),
            "p95_final": round(float(np.percentile(final_bankrolls, 95)), 2),
            "max_drawdown_median": round(float(np.median(max_drawdowns)), 4),
            "max_drawdown_p95": round(float(np.percentile(max_drawdowns, 95)), 4),
            "growth_rate": round(avg_growth, 6),
            "paths_profitable": round(float(np.mean(final_bankrolls > initial)), 4),
            "paths_doubled": round(float(np.mean(final_bankrolls > 2 * initial)), 4),
            "n_simulations": n_sims,
            "n_bets_per_path": n_bets,
        }

    def simulate_from_history(self, bet_history: list[dict]) -> dict:
        """Simulate future paths using the actual distribution of past bet outcomes.

        Instead of assuming a fixed win rate, bootstrap from real P&L data.
        This captures the true variance, skew, and tail behavior.
        """
        if len(bet_history) < 20:
            return {"error": "Need at least 20 historical bets"}

        cfg = self.config
        rng = np.random.default_rng(42)

        # Extract P&L as % of bankroll at time of bet
        pnl_pcts = []
        for b in bet_history:
            bankroll_at_bet = b.get("bankroll_after", cfg.initial_bankroll) - b.get("pnl", 0)
            if bankroll_at_bet > 0:
                pnl_pcts.append(b["pnl"] / bankroll_at_bet)

        pnl_pcts = np.array(pnl_pcts)
        n_sims = cfg.n_simulations
        n_bets = cfg.n_bets_per_path

        # Bootstrap simulation
        bankrolls = np.full((n_sims, n_bets + 1), cfg.initial_bankroll)

        for step in range(n_bets):
            # Bootstrap: randomly sample from historical P&L distribution
            sampled_pnl_pcts = rng.choice(pnl_pcts, size=n_sims, replace=True)
            pnl = bankrolls[:, step] * sampled_pnl_pcts
            bankrolls[:, step + 1] = np.maximum(bankrolls[:, step] + pnl, 0)

        final = bankrolls[:, -1]
        ruin_count = np.sum(final < cfg.ruin_threshold)

        # Max drawdowns
        max_dds = np.zeros(n_sims)
        for i in range(n_sims):
            path = bankrolls[i]
            peak = np.maximum.accumulate(path)
            dds = (peak - path) / np.maximum(peak, 1.0)
            max_dds[i] = np.max(dds)

        return {
            "ruin_probability": round(float(ruin_count / n_sims), 4),
            "median_final": round(float(np.median(final)), 2),
            "mean_final": round(float(np.mean(final)), 2),
            "p5_final": round(float(np.percentile(final, 5)), 2),
            "p95_final": round(float(np.percentile(final, 95)), 2),
            "max_drawdown_median": round(float(np.median(max_dds)), 4),
            "max_drawdown_p95": round(float(np.percentile(max_dds, 95)), 4),
            "paths_profitable": round(float(np.mean(final > cfg.initial_bankroll)), 4),
            "bootstrap_source_bets": len(bet_history),
            "historical_win_rate": round(float(np.mean(pnl_pcts > 0)), 4),
            "historical_avg_pnl_pct": round(float(np.mean(pnl_pcts)), 6),
        }
