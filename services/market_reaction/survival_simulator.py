"""
services/market_reaction/survival_simulator.py
===============================================
12-month Monte Carlo simulation that combines:
  - Progressive sportsbook limits
  - Line shading against identified sharps
  - Edge decay across all sources
  - Bankroll evolution with realistic variance
  - Multi-book rotation strategy

The central question: if you start with $X bankroll and Y% edge,
what does the bankroll distribution look like after 12 months
accounting for ALL adversarial dynamics?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.market_reaction.edge_decay import EdgeDecayModel, DEFAULT_EDGE_SOURCES
from services.market_reaction.limit_progression import (
    LimitProgressionModel,
    LimitProgressionConfig,
)
from services.market_reaction.line_shading import LineShadingModel, ShadingProfile
from services.market_reaction.book_behavior import BookBehaviorModel, BOOK_PROFILES

log = logging.getLogger(__name__)


@dataclass
class SurvivalConfig:
    """Configuration for the survival simulation."""
    initial_bankroll: float = 5000.0
    months_to_simulate: int = 12
    bets_per_day: float = 3.0
    initial_avg_edge_pct: float = 4.0
    avg_bet_size_pct: float = 2.0      # % of bankroll per bet
    odds_decimal: float = 1.91         # ~-110
    kelly_fraction: float = 0.25       # Fractional Kelly
    books: List[str] = field(default_factory=lambda: ["draftkings", "fanduel", "betmgm", "caesars"])
    n_simulations: int = 1000
    max_drawdown_kill: float = 0.50    # Kill at 50% drawdown
    random_seed: int = 42


@dataclass
class MonthSnapshot:
    """State at the end of a simulated month."""
    month: int
    bankroll: float
    peak_bankroll: float
    drawdown_pct: float
    total_bets: int
    monthly_bets: int
    monthly_pnl: float
    monthly_roi_pct: float
    effective_edge_pct: float
    avg_max_bet: float              # Average max bet across active books
    active_books: int               # Books not yet banned
    cumulative_pnl: float
    is_alive: bool                  # System hasn't been killed


@dataclass
class SurvivalResult:
    """Result of a single simulation run."""
    months: List[MonthSnapshot]
    final_bankroll: float
    peak_bankroll: float
    max_drawdown_pct: float
    total_pnl: float
    total_bets: int
    survived_months: int
    was_killed: bool
    kill_reason: str
    annualized_roi_pct: float


class SurvivalSimulator:
    """12-month survival simulator with all adversarial dynamics.

    Runs Monte Carlo simulations to estimate the distribution of outcomes
    given realistic market reactions.
    """

    def __init__(self, config: SurvivalConfig | None = None):
        self.config = config or SurvivalConfig()
        self.decay_model = EdgeDecayModel()
        self.limit_model = LimitProgressionModel()
        self.shading_model = LineShadingModel()
        self.behavior_model = BookBehaviorModel()

    def run_single(self, config: SurvivalConfig | None = None, seed: int = 0) -> SurvivalResult:
        """Run a single survival simulation."""
        cfg = config or self.config
        rng = np.random.default_rng(seed)

        bankroll = cfg.initial_bankroll
        peak_bankroll = bankroll
        total_bets = 0
        cumulative_pnl = 0.0
        months: List[MonthSnapshot] = []
        was_killed = False
        kill_reason = ""

        # Track per-book state
        book_bets: Dict[str, int] = {b: 0 for b in cfg.books}
        book_banned: Dict[str, bool] = {b: False for b in cfg.books}

        bets_per_month = int(cfg.bets_per_day * 30)

        for month in range(1, cfg.months_to_simulate + 1):
            if was_killed:
                break

            days_elapsed = month * 30.0
            monthly_pnl = 0.0
            monthly_bets = 0

            # Get current edge from decay model
            composition = self.decay_model.edge_composition_at(days_elapsed)
            current_total_edge = composition["total_edge_pct"]

            # Scale edge relative to initial
            edge_ratio = current_total_edge / max(
                self.decay_model.edge_composition_at(0.0)["total_edge_pct"], 1e-10
            )
            effective_edge = cfg.initial_avg_edge_pct * edge_ratio

            # Active books
            active_books = [b for b in cfg.books if not book_banned[b]]
            if not active_books:
                was_killed = True
                kill_reason = "all_books_banned"
                break

            # Distribute bets across active books
            bets_per_book = max(bets_per_month // len(active_books), 1)

            avg_max_bet = 0.0
            for book_name in active_books:
                book_profile = self.behavior_model.get_book_profile(book_name)

                # Get current limit stage based on bet count
                total_book_bets = book_bets[book_name]
                if total_book_bets >= book_profile.min_bets_to_ban:
                    book_banned[book_name] = True
                    continue
                elif total_book_bets >= book_profile.min_bets_to_restrict:
                    max_bet = book_profile.max_bet_restricted
                    shading_mult = 2.0
                elif total_book_bets >= book_profile.min_bets_to_flag:
                    max_bet = book_profile.max_bet_watched
                    shading_mult = 1.0
                else:
                    max_bet = book_profile.max_bet_unrestricted
                    shading_mult = 0.0

                # Shading reduces edge
                shading_cost = shading_mult * 0.5  # 0.5% per shading level
                book_edge = max(effective_edge - shading_cost, 0.0)

                # Bet sizing (fractional Kelly)
                bet_size_pct = min(
                    cfg.avg_bet_size_pct / 100.0 * bankroll,
                    max_bet,
                )
                bet_size = max(bet_size_pct, 0.0)
                avg_max_bet += max_bet

                # Simulate individual bets for this book
                for _ in range(bets_per_book):
                    if bankroll <= 0:
                        was_killed = True
                        kill_reason = "bankroll_depleted"
                        break

                    actual_bet = min(bet_size, bankroll * 0.05)  # Never more than 5%

                    # Win probability
                    fair_prob = 0.50 + (book_edge / 100.0) / 2.0  # Edge → probability
                    fair_prob = np.clip(fair_prob, 0.01, 0.99)

                    # Simulate outcome
                    won = rng.random() < fair_prob
                    if won:
                        pnl = actual_bet * (cfg.odds_decimal - 1.0)
                    else:
                        pnl = -actual_bet

                    bankroll += pnl
                    monthly_pnl += pnl
                    cumulative_pnl += pnl
                    total_bets += 1
                    monthly_bets += 1
                    book_bets[book_name] += 1

                    peak_bankroll = max(peak_bankroll, bankroll)

                if was_killed:
                    break

            # Recount active books after this month
            current_active = sum(1 for b in cfg.books if not book_banned[b])
            avg_max_bet = avg_max_bet / max(len(active_books), 1)

            # Drawdown check
            drawdown = (peak_bankroll - bankroll) / max(peak_bankroll, 1.0)
            if drawdown >= cfg.max_drawdown_kill:
                was_killed = True
                kill_reason = "max_drawdown_exceeded"

            monthly_handle = monthly_bets * (cfg.avg_bet_size_pct / 100.0 * cfg.initial_bankroll)
            monthly_roi = (monthly_pnl / max(monthly_handle, 1.0)) * 100

            months.append(MonthSnapshot(
                month=month,
                bankroll=round(bankroll, 2),
                peak_bankroll=round(peak_bankroll, 2),
                drawdown_pct=round(drawdown * 100, 1),
                total_bets=total_bets,
                monthly_bets=monthly_bets,
                monthly_pnl=round(monthly_pnl, 2),
                monthly_roi_pct=round(monthly_roi, 1),
                effective_edge_pct=round(effective_edge, 2),
                avg_max_bet=round(avg_max_bet, 2),
                active_books=current_active,
                cumulative_pnl=round(cumulative_pnl, 2),
                is_alive=not was_killed,
            ))

        survived = len([m for m in months if m.is_alive])
        total_handle = total_bets * (cfg.avg_bet_size_pct / 100.0 * cfg.initial_bankroll)
        annualized_roi = (cumulative_pnl / max(total_handle, 1.0)) * 100

        max_dd = max((m.drawdown_pct for m in months), default=0.0)

        return SurvivalResult(
            months=months,
            final_bankroll=round(bankroll, 2),
            peak_bankroll=round(peak_bankroll, 2),
            max_drawdown_pct=round(max_dd, 1),
            total_pnl=round(cumulative_pnl, 2),
            total_bets=total_bets,
            survived_months=survived,
            was_killed=was_killed,
            kill_reason=kill_reason,
            annualized_roi_pct=round(annualized_roi, 1),
        )

    def run_monte_carlo(self, config: SurvivalConfig | None = None) -> Dict[str, Any]:
        """Run full Monte Carlo survival simulation.

        Returns distribution of outcomes across N simulations.
        """
        cfg = config or self.config
        n = cfg.n_simulations

        results: List[SurvivalResult] = []
        for i in range(n):
            result = self.run_single(config=cfg, seed=cfg.random_seed + i)
            results.append(result)

        # Aggregate statistics
        final_bankrolls = np.array([r.final_bankroll for r in results])
        total_pnls = np.array([r.total_pnl for r in results])
        max_drawdowns = np.array([r.max_drawdown_pct for r in results])
        survived_months = np.array([r.survived_months for r in results])
        was_killed = np.array([r.was_killed for r in results])

        # Monthly survival curve
        survival_curve = []
        for m in range(1, cfg.months_to_simulate + 1):
            alive_count = sum(1 for r in results if r.survived_months >= m)
            survival_curve.append({
                "month": m,
                "survival_pct": round(alive_count / n * 100, 1),
                "alive_count": alive_count,
            })

        # Monthly bankroll percentiles
        monthly_percentiles = []
        for m_idx in range(cfg.months_to_simulate):
            month_bankrolls = []
            for r in results:
                if m_idx < len(r.months):
                    month_bankrolls.append(r.months[m_idx].bankroll)
            if month_bankrolls:
                arr = np.array(month_bankrolls)
                monthly_percentiles.append({
                    "month": m_idx + 1,
                    "p5": round(float(np.percentile(arr, 5)), 2),
                    "p25": round(float(np.percentile(arr, 25)), 2),
                    "p50": round(float(np.percentile(arr, 50)), 2),
                    "p75": round(float(np.percentile(arr, 75)), 2),
                    "p95": round(float(np.percentile(arr, 95)), 2),
                    "mean": round(float(np.mean(arr)), 2),
                })

        # Kill reasons distribution
        kill_reasons: Dict[str, int] = {}
        for r in results:
            if r.was_killed:
                kill_reasons[r.kill_reason] = kill_reasons.get(r.kill_reason, 0) + 1

        return {
            "config": {
                "initial_bankroll": cfg.initial_bankroll,
                "months": cfg.months_to_simulate,
                "n_simulations": n,
                "initial_edge_pct": cfg.initial_avg_edge_pct,
                "books": cfg.books,
            },
            "final_bankroll": {
                "mean": round(float(np.mean(final_bankrolls)), 2),
                "median": round(float(np.median(final_bankrolls)), 2),
                "p5": round(float(np.percentile(final_bankrolls, 5)), 2),
                "p25": round(float(np.percentile(final_bankrolls, 25)), 2),
                "p75": round(float(np.percentile(final_bankrolls, 75)), 2),
                "p95": round(float(np.percentile(final_bankrolls, 95)), 2),
                "std": round(float(np.std(final_bankrolls)), 2),
            },
            "total_pnl": {
                "mean": round(float(np.mean(total_pnls)), 2),
                "median": round(float(np.median(total_pnls)), 2),
                "p5": round(float(np.percentile(total_pnls, 5)), 2),
                "p95": round(float(np.percentile(total_pnls, 95)), 2),
                "pct_profitable": round(float(np.mean(total_pnls > 0) * 100), 1),
            },
            "max_drawdown": {
                "mean": round(float(np.mean(max_drawdowns)), 1),
                "median": round(float(np.median(max_drawdowns)), 1),
                "p95": round(float(np.percentile(max_drawdowns, 95)), 1),
            },
            "survival": {
                "pct_survived_full": round(float(np.mean(~was_killed) * 100), 1),
                "avg_months_survived": round(float(np.mean(survived_months)), 1),
                "median_months_survived": round(float(np.median(survived_months)), 1),
                "curve": survival_curve,
            },
            "kill_reasons": kill_reasons,
            "monthly_percentiles": monthly_percentiles,
            "verdict": self._generate_verdict(
                final_bankrolls, total_pnls, was_killed, cfg
            ),
        }

    def _generate_verdict(
        self,
        final_bankrolls: np.ndarray,
        total_pnls: np.ndarray,
        was_killed: np.ndarray,
        config: SurvivalConfig,
    ) -> Dict[str, Any]:
        """Generate a human-readable verdict from simulation results."""
        pct_profitable = float(np.mean(total_pnls > 0) * 100)
        pct_survived = float(np.mean(~was_killed) * 100)
        median_roi = float(np.median(total_pnls / config.initial_bankroll * 100))
        ruin_pct = float(np.mean(final_bankrolls < config.initial_bankroll * 0.1) * 100)

        if pct_profitable >= 80 and pct_survived >= 70 and median_roi > 20:
            assessment = "STRONG"
            detail = "High probability of profit with good survival rate"
        elif pct_profitable >= 60 and pct_survived >= 50:
            assessment = "VIABLE"
            detail = "Reasonable profit expectations with moderate survival risk"
        elif pct_profitable >= 40:
            assessment = "MARGINAL"
            detail = "Coin-flip profitability. Edge may not overcome market friction"
        else:
            assessment = "NOT_VIABLE"
            detail = "Insufficient edge to overcome sportsbook countermeasures"

        return {
            "assessment": assessment,
            "detail": detail,
            "pct_profitable": round(pct_profitable, 1),
            "pct_survived_full_term": round(pct_survived, 1),
            "median_roi_pct": round(median_roi, 1),
            "ruin_probability_pct": round(ruin_pct, 1),
            "deploy_recommendation": assessment in ("STRONG", "VIABLE"),
        }
