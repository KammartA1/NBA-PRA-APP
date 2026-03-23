"""
services/execution/slippage_model.py
=====================================
Models the line movement between signal generation and bet placement.

The line MOVES between when you see the signal and when you place the bet.
This module computes average slippage in cents per bet and models slippage
as a function of time_to_bet, market_liquidity, and line_movement_speed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlippageObservation:
    """A single historical slippage observation."""
    bet_id: str
    signal_line: float          # Line when signal fired
    bet_line: float             # Line at actual bet placement
    closing_line: float         # Closing line for reference
    signal_time: datetime       # When signal generated
    bet_time: datetime          # When bet placed
    market_type: str            # "points", "rebounds", etc.
    sportsbook: str = "unknown"
    # Derived
    slippage_cents: float = 0.0  # bet_line - signal_line (positive = worse)
    time_to_bet_seconds: float = 0.0


@dataclass
class SlippageResult:
    """Output of the slippage model for a single bet signal."""
    expected_slippage_cents: float    # Expected line movement against us
    slippage_std: float              # Standard deviation of slippage
    p_favorable_move: float          # P(line moved in our favor)
    p_no_move: float                 # P(line didn't move)
    p_adverse_move: float            # P(line moved against us)
    expected_execution_line: float   # signal_line + expected_slippage
    slippage_cost_pct: float         # Slippage as % of edge


@dataclass
class SlippageProfile:
    """Aggregate slippage statistics across all historical bets."""
    n_observations: int
    mean_slippage_cents: float
    median_slippage_cents: float
    std_slippage_cents: float
    p90_slippage_cents: float        # 90th percentile (worst case)
    p99_slippage_cents: float        # 99th percentile (extreme)
    mean_time_to_bet_seconds: float
    slippage_by_market: dict = field(default_factory=dict)
    slippage_by_book: dict = field(default_factory=dict)
    slippage_by_time_bucket: dict = field(default_factory=dict)
    total_slippage_cost_dollars: float = 0.0


class SlippageModel:
    """Models line slippage between signal and execution.

    Slippage = f(time_to_bet, market_liquidity, line_movement_speed)

    The model uses historical observations to fit coefficients, then
    predicts expected slippage for new signals.
    """

    # Liquidity proxy: higher-volume markets have less slippage
    LIQUIDITY_WEIGHTS = {
        "points": 1.0,
        "rebounds": 0.7,
        "assists": 0.7,
        "pra": 0.85,
        "threes": 0.6,
        "blocks": 0.4,
        "steals": 0.4,
        "turnovers": 0.5,
        "fantasy_score": 0.3,
        "double_double": 0.3,
    }

    # Time buckets for slippage analysis (seconds)
    TIME_BUCKETS = [
        (0, 10, "instant"),
        (10, 60, "fast"),
        (60, 300, "moderate"),
        (300, 900, "slow"),
        (900, 3600, "very_slow"),
        (3600, float("inf"), "stale"),
    ]

    def __init__(self, sport: str = "nba"):
        self.sport = sport
        self._observations: List[SlippageObservation] = []
        self._coefficients: dict = {
            "intercept": 0.0,
            "time_coeff": 0.0,
            "liquidity_coeff": 0.0,
            "movement_coeff": 0.0,
        }
        self._fitted = False

    def load_observations(self, observations: List[SlippageObservation]) -> None:
        """Load historical slippage observations."""
        self._observations = observations
        if observations:
            self._fit()
        log.info("SlippageModel loaded %d observations", len(observations))

    def load_from_bets(self, bets: list) -> None:
        """Load from raw bet records (dict or dataclass with required fields)."""
        observations = []
        for b in bets:
            if hasattr(b, "signal_line"):
                sig_line = b.signal_line
                bet_line = b.bet_line
                closing = b.closing_line
                sig_time = b.signal_generated_at or b.timestamp
                bet_time = b.timestamp
                market = getattr(b, "market_type", "points")
                book = getattr(b, "sportsbook", "unknown")
                bid = getattr(b, "bet_id", "")
            elif isinstance(b, dict):
                sig_line = b.get("signal_line", 0)
                bet_line = b.get("bet_line", 0)
                closing = b.get("closing_line", 0)
                sig_time = b.get("signal_generated_at") or b.get("timestamp", datetime.now())
                bet_time = b.get("timestamp", datetime.now())
                market = b.get("market_type", "points")
                book = b.get("sportsbook", "unknown")
                bid = b.get("bet_id", "")
            else:
                continue

            if sig_line == 0 or bet_line == 0:
                continue

            dt = (bet_time - sig_time).total_seconds() if sig_time and bet_time else 0
            slip = bet_line - sig_line  # positive = line moved against us

            observations.append(SlippageObservation(
                bet_id=str(bid),
                signal_line=sig_line,
                bet_line=bet_line,
                closing_line=closing,
                signal_time=sig_time,
                bet_time=bet_time,
                market_type=market,
                sportsbook=book,
                slippage_cents=slip,
                time_to_bet_seconds=max(0, dt),
            ))
        self.load_observations(observations)

    def _fit(self) -> None:
        """Fit slippage model: slippage = a + b*time + c*liquidity + d*movement_speed."""
        if len(self._observations) < 5:
            # Not enough data — use simple mean
            slippages = [o.slippage_cents for o in self._observations]
            self._coefficients["intercept"] = float(np.mean(slippages)) if slippages else 0.0
            self._fitted = True
            return

        # Build feature matrix
        X = []
        y = []
        for obs in self._observations:
            time_feat = math.log1p(obs.time_to_bet_seconds)
            liq_feat = self._get_liquidity(obs.market_type)
            # Movement speed: how fast was the line moving?
            total_move = abs(obs.closing_line - obs.signal_line)
            time_window = max(1, obs.time_to_bet_seconds)
            move_speed = total_move / time_window

            X.append([1.0, time_feat, liq_feat, move_speed])
            y.append(obs.slippage_cents)

        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)

        # OLS fit with regularization (ridge, lambda=0.01)
        try:
            lam = 0.01
            XtX = X_arr.T @ X_arr + lam * np.eye(X_arr.shape[1])
            Xty = X_arr.T @ y_arr
            coeffs = np.linalg.solve(XtX, Xty)
            self._coefficients = {
                "intercept": float(coeffs[0]),
                "time_coeff": float(coeffs[1]),
                "liquidity_coeff": float(coeffs[2]),
                "movement_coeff": float(coeffs[3]),
            }
        except np.linalg.LinAlgError:
            self._coefficients["intercept"] = float(np.mean(y_arr))

        self._fitted = True
        log.info("SlippageModel fitted: %s", self._coefficients)

    def _get_liquidity(self, market_type: str) -> float:
        """Return liquidity proxy for a market type (0-1, higher = more liquid)."""
        mt = market_type.lower().replace(" ", "_")
        for key, val in self.LIQUIDITY_WEIGHTS.items():
            if key in mt:
                return val
        return 0.5

    def predict(
        self,
        signal_line: float,
        market_type: str = "points",
        time_to_bet_seconds: float = 60.0,
        line_movement_speed: float = 0.0,
    ) -> SlippageResult:
        """Predict expected slippage for a new signal."""
        if not self._fitted or not self._observations:
            return SlippageResult(
                expected_slippage_cents=0.0,
                slippage_std=0.5,
                p_favorable_move=0.2,
                p_no_move=0.3,
                p_adverse_move=0.5,
                expected_execution_line=signal_line,
                slippage_cost_pct=0.0,
            )

        time_feat = math.log1p(time_to_bet_seconds)
        liq_feat = self._get_liquidity(market_type)
        move_feat = line_movement_speed

        expected = (
            self._coefficients["intercept"]
            + self._coefficients["time_coeff"] * time_feat
            + self._coefficients["liquidity_coeff"] * liq_feat
            + self._coefficients["movement_coeff"] * move_feat
        )

        # Compute empirical std from residuals
        slippages = np.array([o.slippage_cents for o in self._observations])
        std = float(np.std(slippages)) if len(slippages) > 1 else 0.5

        # Empirical probabilities
        n = len(slippages)
        p_favorable = float(np.sum(slippages < -0.01)) / n
        p_no_move = float(np.sum(np.abs(slippages) <= 0.01)) / n
        p_adverse = float(np.sum(slippages > 0.01)) / n

        return SlippageResult(
            expected_slippage_cents=expected,
            slippage_std=std,
            p_favorable_move=p_favorable,
            p_no_move=p_no_move,
            p_adverse_move=p_adverse,
            expected_execution_line=signal_line + expected,
            slippage_cost_pct=abs(expected) / max(abs(signal_line), 0.01) * 100,
        )

    def profile(self) -> SlippageProfile:
        """Generate aggregate slippage profile from all observations."""
        if not self._observations:
            return SlippageProfile(
                n_observations=0,
                mean_slippage_cents=0.0,
                median_slippage_cents=0.0,
                std_slippage_cents=0.0,
                p90_slippage_cents=0.0,
                p99_slippage_cents=0.0,
                mean_time_to_bet_seconds=0.0,
            )

        slips = np.array([o.slippage_cents for o in self._observations])
        times = np.array([o.time_to_bet_seconds for o in self._observations])

        # By market
        by_market: dict = {}
        for obs in self._observations:
            mt = obs.market_type
            by_market.setdefault(mt, []).append(obs.slippage_cents)
        slippage_by_market = {
            k: {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}
            for k, v in by_market.items()
        }

        # By book
        by_book: dict = {}
        for obs in self._observations:
            bk = obs.sportsbook
            by_book.setdefault(bk, []).append(obs.slippage_cents)
        slippage_by_book = {
            k: {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}
            for k, v in by_book.items()
        }

        # By time bucket
        by_time: dict = {}
        for obs in self._observations:
            for lo, hi, label in self.TIME_BUCKETS:
                if lo <= obs.time_to_bet_seconds < hi:
                    by_time.setdefault(label, []).append(obs.slippage_cents)
                    break
        slippage_by_time = {
            k: {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}
            for k, v in by_time.items()
        }

        return SlippageProfile(
            n_observations=len(self._observations),
            mean_slippage_cents=float(np.mean(slips)),
            median_slippage_cents=float(np.median(slips)),
            std_slippage_cents=float(np.std(slips)),
            p90_slippage_cents=float(np.percentile(slips, 90)),
            p99_slippage_cents=float(np.percentile(slips, 99)),
            mean_time_to_bet_seconds=float(np.mean(times)),
            slippage_by_market=slippage_by_market,
            slippage_by_book=slippage_by_book,
            slippage_by_time_bucket=slippage_by_time,
            total_slippage_cost_dollars=float(np.sum(np.abs(slips))) * 0.01,
        )
