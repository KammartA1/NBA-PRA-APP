"""Walk-Forward Validation — The only backtesting method that matters.

Standard backtesting is garbage because it implicitly leaks future data
through parameter selection. Walk-forward validation prevents this by:

1. Train on window [t-N, t]
2. Test on window [t, t+K]
3. Slide forward by K
4. Repeat until end of data

This ensures every prediction is truly out-of-sample.

Critical rules:
- NEVER use future data for any feature computation
- Odds must reflect what was actually available at bet time
- Features must be computed with only past data
- Results must include realistic execution timing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np

from ..core.types import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    train_window_days: int = 180       # 6 months training
    test_window_days: int = 30         # 1 month testing
    step_days: int = 30                # Slide by 1 month
    min_train_bets: int = 50           # Minimum bets in training window
    min_test_bets: int = 10            # Minimum bets in test window
    initial_bankroll: float = 1000.0
    kelly_fraction: float = 0.10       # 1/10 Kelly for backtest


class WalkForwardValidator:
    """Walk-forward backtesting engine.

    Usage:
        validator = WalkForwardValidator(config)
        result = validator.run(
            data=historical_data_df,          # Must have: date, player, stat_type, line, actual_result
            model_fn=your_model.predict,      # fn(train_data) -> predictions_for_test
            odds_fn=lambda row: row["odds"],  # Get odds from data
        )
    """

    def __init__(self, config: WalkForwardConfig | None = None):
        self.config = config or WalkForwardConfig()

    def run(
        self,
        data: list[dict],
        model_fn: Callable,
        odds_fn: Callable | None = None,
    ) -> BacktestResult:
        """Run walk-forward validation.

        Args:
            data: List of dicts with keys:
                - date: datetime
                - player: str
                - stat_type: str
                - line: float
                - actual_result: float
                - odds_decimal: float (optional, defaults to 1.91 for -110)
                - (any other features the model needs)
            model_fn: Callable that takes (train_data, test_data) and returns
                list of dicts with {model_prob, direction} for each test row
            odds_fn: Optional callable to extract odds from a row

        Returns:
            BacktestResult with full performance metrics
        """
        cfg = self.config

        # Sort by date
        data = sorted(data, key=lambda x: x["date"])
        if not data:
            raise ValueError("No data provided")

        start_date = data[0]["date"]
        end_date = data[-1]["date"]

        all_bets = []
        bankroll = cfg.initial_bankroll
        peak = bankroll
        monthly_returns = []

        # Walk forward
        train_start = start_date
        while True:
            train_end = train_start + timedelta(days=cfg.train_window_days)
            test_end = train_end + timedelta(days=cfg.test_window_days)

            if test_end > end_date:
                break

            # Split data
            train_data = [d for d in data if train_start <= d["date"] < train_end]
            test_data = [d for d in data if train_end <= d["date"] < test_end]

            if len(train_data) < cfg.min_train_bets or len(test_data) < cfg.min_test_bets:
                train_start += timedelta(days=cfg.step_days)
                continue

            # Get model predictions (ONLY using train data)
            try:
                predictions = model_fn(train_data, test_data)
            except Exception as e:
                logger.warning("Model failed on window %s-%s: %s", train_end, test_end, e)
                train_start += timedelta(days=cfg.step_days)
                continue

            # Simulate bets
            window_pnl = 0.0
            for i, (test_row, pred) in enumerate(zip(test_data, predictions)):
                model_prob = pred.get("model_prob", 0.5)
                direction = pred.get("direction", "over")

                # Get odds
                if odds_fn:
                    odds_decimal = odds_fn(test_row)
                else:
                    odds_decimal = test_row.get("odds_decimal", 1.909)  # -110 default

                # Kelly sizing
                market_prob = 1.0 / odds_decimal
                edge = model_prob - market_prob

                if edge < 0.04:  # Minimum edge threshold
                    continue

                b = odds_decimal - 1.0
                p = model_prob
                q = 1.0 - p
                kelly = (p * b - q) / b
                if kelly <= 0:
                    continue

                stake = bankroll * cfg.kelly_fraction * kelly
                stake = min(stake, bankroll * 0.03)  # 3% cap
                stake = round(stake, 2)

                if stake <= 0:
                    continue

                # Determine outcome
                actual = test_row["actual_result"]
                line = test_row["line"]

                if direction == "over":
                    won = actual > line
                else:
                    won = actual < line

                pnl = stake * (odds_decimal - 1.0) if won else -stake
                bankroll += pnl
                peak = max(peak, bankroll)
                window_pnl += pnl

                all_bets.append({
                    "date": test_row["date"],
                    "player": test_row.get("player", ""),
                    "model_prob": model_prob,
                    "market_prob": market_prob,
                    "edge": edge,
                    "stake": stake,
                    "pnl": pnl,
                    "won": won,
                    "bankroll_after": bankroll,
                })

            monthly_returns.append({
                "window_start": train_end.isoformat(),
                "window_end": test_end.isoformat(),
                "pnl": round(window_pnl, 2),
                "n_bets": sum(1 for b in all_bets if train_end <= b["date"] < test_end),
            })

            train_start += timedelta(days=cfg.step_days)

        # Compute metrics
        if not all_bets:
            return BacktestResult(
                start_date=start_date, end_date=end_date,
                n_bets=0, total_pnl=0.0, roi_pct=0.0, win_rate=0.0,
                avg_edge=0.0, avg_clv=0.0, max_drawdown_pct=0.0,
                sharpe_ratio=0.0, kelly_growth_rate=0.0,
                monthly_returns=[], ruin_probability=1.0,
                median_final_bankroll=cfg.initial_bankroll,
                p5_final_bankroll=0.0, p95_final_bankroll=cfg.initial_bankroll,
            )

        total_staked = sum(abs(b["stake"]) for b in all_bets)
        total_pnl = bankroll - cfg.initial_bankroll
        roi = total_pnl / total_staked if total_staked > 0 else 0.0
        win_rate = sum(1 for b in all_bets if b["won"]) / len(all_bets)
        avg_edge = float(np.mean([b["edge"] for b in all_bets]))

        # Max drawdown
        peak_val = cfg.initial_bankroll
        max_dd = 0.0
        for b in all_bets:
            peak_val = max(peak_val, b["bankroll_after"])
            dd = (peak_val - b["bankroll_after"]) / peak_val
            max_dd = max(max_dd, dd)

        # Sharpe ratio (daily returns)
        daily_pnls = [b["pnl"] for b in all_bets]
        if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
            sharpe = float(np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Kelly growth rate: E[ln(1 + f*edge)]
        growth_rates = []
        for b in all_bets:
            if b["won"]:
                g = np.log(1 + b["pnl"] / max(b["bankroll_after"] - b["pnl"], 1))
            else:
                g = np.log(max(1 + b["pnl"] / max(b["bankroll_after"] - b["pnl"], 1), 0.01))
            growth_rates.append(g)
        kelly_growth = float(np.mean(growth_rates)) if growth_rates else 0.0

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            n_bets=len(all_bets),
            total_pnl=round(total_pnl, 2),
            roi_pct=round(roi * 100, 2),
            win_rate=round(win_rate, 4),
            avg_edge=round(avg_edge, 4),
            avg_clv=0.0,  # CLV requires closing line data
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            kelly_growth_rate=round(kelly_growth, 6),
            monthly_returns=monthly_returns,
            ruin_probability=0.0,  # Set by MC simulator
            median_final_bankroll=round(bankroll, 2),
            p5_final_bankroll=0.0,
            p95_final_bankroll=0.0,
        )
