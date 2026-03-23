"""
services/capital/risk_adjusted.py
==================================
Risk-adjusted performance metrics for the betting portfolio.

Computes:
  - Sharpe ratio (daily/annualized)
  - Sortino ratio (downside deviation only)
  - Maximum drawdown and recovery time
  - Calmar ratio (return / max drawdown)
  - Value at Risk (VaR) — parametric and historical
  - Conditional VaR (CVaR / Expected Shortfall)
  - Win rate, profit factor, expectancy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)


@dataclass
class RiskReport:
    """Complete risk-adjusted performance report."""
    # Return metrics
    total_return_pct: float
    annualized_return_pct: float
    avg_daily_return_pct: float

    # Risk metrics
    daily_volatility_pct: float
    annualized_volatility_pct: float
    downside_volatility_pct: float

    # Ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float

    # Drawdown
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    current_drawdown_pct: float
    avg_drawdown_pct: float
    recovery_time_days: int

    # VaR / CVaR
    var_95_daily_pct: float          # 95% VaR (daily)
    var_99_daily_pct: float          # 99% VaR (daily)
    cvar_95_daily_pct: float         # 95% CVaR (expected shortfall)
    cvar_99_daily_pct: float

    # Betting-specific
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_per_bet: float
    n_bets: int
    n_days: int
    variance_ratio: float            # actual / expected variance


class RiskMetrics:
    """Compute risk-adjusted metrics from a P&L series."""

    def __init__(self, risk_free_rate_annual: float = 0.05):
        """
        Args:
            risk_free_rate_annual: Annual risk-free rate for Sharpe calc.
        """
        self.risk_free_rate = risk_free_rate_annual

    def compute(
        self,
        pnl_series: np.ndarray,
        bankroll_series: np.ndarray | None = None,
        stakes: np.ndarray | None = None,
        initial_bankroll: float = 1000.0,
    ) -> RiskReport:
        """Compute full risk report from P&L series.

        Args:
            pnl_series: Array of per-bet P&L values.
            bankroll_series: Optional equity curve. If None, computed from pnl.
            stakes: Optional per-bet stake amounts for ROI calculation.
            initial_bankroll: Starting bankroll if equity curve not provided.
        """
        if len(pnl_series) == 0:
            return self._empty_report()

        pnl = np.asarray(pnl_series, dtype=float)
        n_bets = len(pnl)

        # Build equity curve if not provided
        if bankroll_series is not None:
            equity = np.asarray(bankroll_series, dtype=float)
        else:
            equity = np.cumsum(pnl) + initial_bankroll

        # Returns (percentage)
        if stakes is not None and len(stakes) == len(pnl):
            stake_arr = np.asarray(stakes, dtype=float)
            returns_pct = np.where(stake_arr > 0, pnl / stake_arr * 100, 0.0)
        else:
            # Use equity-based returns
            equity_shifted = np.concatenate([[initial_bankroll], equity[:-1]])
            returns_pct = np.where(
                equity_shifted > 0,
                pnl / equity_shifted * 100,
                0.0,
            )

        # Basic return metrics
        total_return = float(np.sum(pnl))
        total_return_pct = (total_return / initial_bankroll) * 100

        # Assume ~250 trading days per year, estimate days from bet count
        bets_per_day = max(n_bets / 365.0, 0.1)  # Conservative
        n_days = max(int(n_bets / max(bets_per_day, 1)), 1)
        years = max(n_days / 365.0, 0.01)
        annualized_return = total_return_pct / years if years > 0 else 0.0

        avg_daily_return = float(np.mean(returns_pct))

        # Volatility
        daily_vol = float(np.std(returns_pct, ddof=1)) if n_bets > 1 else 0.0
        ann_vol = daily_vol * np.sqrt(min(n_bets, 365))

        # Downside deviation (only negative returns)
        negative_returns = returns_pct[returns_pct < 0]
        downside_vol = float(np.std(negative_returns, ddof=1)) if len(negative_returns) > 1 else 0.01

        # Risk-free rate per period
        rf_per_period = self.risk_free_rate / max(n_bets, 1)

        # Sharpe ratio
        excess_return = avg_daily_return - rf_per_period * 100
        sharpe = excess_return / max(daily_vol, 0.01)

        # Sortino ratio (uses downside deviation)
        sortino = excess_return / max(downside_vol, 0.01)

        # Drawdown analysis
        dd_result = self._compute_drawdown(equity)

        # Calmar ratio
        calmar = annualized_return / max(abs(dd_result["max_drawdown_pct"]), 0.01)

        # Profit factor
        gross_profit = float(np.sum(pnl[pnl > 0]))
        gross_loss = float(np.abs(np.sum(pnl[pnl < 0])))
        profit_factor = gross_profit / max(gross_loss, 0.01)

        # VaR and CVaR
        var_95 = float(np.percentile(returns_pct, 5))
        var_99 = float(np.percentile(returns_pct, 1))
        cvar_95 = float(np.mean(returns_pct[returns_pct <= var_95])) if np.any(returns_pct <= var_95) else var_95
        cvar_99 = float(np.mean(returns_pct[returns_pct <= var_99])) if np.any(returns_pct <= var_99) else var_99

        # Win rate
        wins = np.sum(pnl > 0)
        losses = np.sum(pnl < 0)
        win_rate = float(wins) / max(float(wins + losses), 1)

        avg_win = float(np.mean(pnl[pnl > 0])) if np.any(pnl > 0) else 0.0
        avg_loss = float(np.mean(pnl[pnl < 0])) if np.any(pnl < 0) else 0.0
        avg_win_pct = float(np.mean(returns_pct[returns_pct > 0])) if np.any(returns_pct > 0) else 0.0
        avg_loss_pct = float(np.mean(returns_pct[returns_pct < 0])) if np.any(returns_pct < 0) else 0.0

        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Variance ratio (actual vs expected under binomial assumption)
        if n_bets >= 10:
            expected_var = win_rate * (1 - win_rate) * np.mean(np.abs(pnl)) ** 2
            actual_var = float(np.var(pnl, ddof=1))
            variance_ratio = actual_var / max(expected_var, 1e-10)
        else:
            variance_ratio = 1.0

        return RiskReport(
            total_return_pct=round(total_return_pct, 2),
            annualized_return_pct=round(annualized_return, 2),
            avg_daily_return_pct=round(avg_daily_return, 4),
            daily_volatility_pct=round(daily_vol, 4),
            annualized_volatility_pct=round(ann_vol, 2),
            downside_volatility_pct=round(downside_vol, 4),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            profit_factor=round(profit_factor, 3),
            max_drawdown_pct=round(dd_result["max_drawdown_pct"], 2),
            max_drawdown_duration_days=dd_result["max_dd_duration"],
            current_drawdown_pct=round(dd_result["current_drawdown_pct"], 2),
            avg_drawdown_pct=round(dd_result["avg_drawdown_pct"], 2),
            recovery_time_days=dd_result["recovery_time"],
            var_95_daily_pct=round(var_95, 3),
            var_99_daily_pct=round(var_99, 3),
            cvar_95_daily_pct=round(cvar_95, 3),
            cvar_99_daily_pct=round(cvar_99, 3),
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win_pct, 3),
            avg_loss_pct=round(avg_loss_pct, 3),
            expectancy_per_bet=round(expectancy, 2),
            n_bets=n_bets,
            n_days=n_days,
            variance_ratio=round(variance_ratio, 3),
        )

    def _compute_drawdown(self, equity: np.ndarray) -> Dict[str, Any]:
        """Compute drawdown metrics from an equity curve."""
        if len(equity) == 0:
            return {
                "max_drawdown_pct": 0.0,
                "max_dd_duration": 0,
                "current_drawdown_pct": 0.0,
                "avg_drawdown_pct": 0.0,
                "recovery_time": 0,
            }

        peak = np.maximum.accumulate(equity)
        drawdown_pct = np.where(peak > 0, (peak - equity) / peak * 100, 0.0)

        max_dd = float(np.max(drawdown_pct))
        current_dd = float(drawdown_pct[-1])
        avg_dd = float(np.mean(drawdown_pct[drawdown_pct > 0])) if np.any(drawdown_pct > 0) else 0.0

        # Max drawdown duration
        in_drawdown = drawdown_pct > 0
        max_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        # Recovery time from max drawdown
        max_dd_idx = int(np.argmax(drawdown_pct))
        recovery_time = 0
        for i in range(max_dd_idx, len(equity)):
            if equity[i] >= peak[max_dd_idx]:
                recovery_time = i - max_dd_idx
                break
        else:
            recovery_time = len(equity) - max_dd_idx  # Still recovering

        return {
            "max_drawdown_pct": max_dd,
            "max_dd_duration": max_duration,
            "current_drawdown_pct": current_dd,
            "avg_drawdown_pct": avg_dd,
            "recovery_time": recovery_time,
        }

    def _empty_report(self) -> RiskReport:
        """Return a zeroed-out report when no data is available."""
        return RiskReport(
            total_return_pct=0.0, annualized_return_pct=0.0, avg_daily_return_pct=0.0,
            daily_volatility_pct=0.0, annualized_volatility_pct=0.0, downside_volatility_pct=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0, profit_factor=0.0,
            max_drawdown_pct=0.0, max_drawdown_duration_days=0, current_drawdown_pct=0.0,
            avg_drawdown_pct=0.0, recovery_time_days=0,
            var_95_daily_pct=0.0, var_99_daily_pct=0.0, cvar_95_daily_pct=0.0, cvar_99_daily_pct=0.0,
            win_rate=0.0, avg_win_pct=0.0, avg_loss_pct=0.0, expectancy_per_bet=0.0,
            n_bets=0, n_days=0, variance_ratio=1.0,
        )

    def rolling_sharpe(
        self,
        pnl_series: np.ndarray,
        window: int = 50,
    ) -> np.ndarray:
        """Compute rolling Sharpe ratio over a sliding window."""
        n = len(pnl_series)
        if n < window:
            return np.full(n, np.nan)

        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            window_pnl = pnl_series[i - window + 1:i + 1]
            mean_ret = np.mean(window_pnl)
            std_ret = np.std(window_pnl, ddof=1)
            if std_ret > 0:
                result[i] = mean_ret / std_ret
            else:
                result[i] = 0.0
        return result

    def rolling_var(
        self,
        pnl_series: np.ndarray,
        window: int = 50,
        confidence: float = 0.95,
    ) -> np.ndarray:
        """Compute rolling VaR."""
        n = len(pnl_series)
        if n < window:
            return np.full(n, np.nan)

        result = np.full(n, np.nan)
        pct = (1.0 - confidence) * 100
        for i in range(window - 1, n):
            window_pnl = pnl_series[i - window + 1:i + 1]
            result[i] = np.percentile(window_pnl, pct)
        return result
