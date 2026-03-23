"""
services/execution/reality_simulator.py
========================================
The master execution reality simulator. Chains all four models to answer:

    "What price do I ACTUALLY get? Does edge survive execution reality?"

For each signal:
    1. Apply slippage model -> actual execution price
    2. Apply limit model -> actual bet size allowed
    3. Apply latency model -> P(line still available)
    4. Apply rejection model -> P(bet actually placed)

Compare theoretical profit (perfect execution) vs realistic profit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from services.execution.slippage_model import SlippageModel, SlippageProfile
from services.execution.limit_model import LimitModel, LimitProfile
from services.execution.latency_model import LatencyModel, LatencyProfile
from services.execution.rejection_model import RejectionModel, RejectionProfile

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BetSignal:
    """A bet signal to be evaluated through the execution reality filter."""
    bet_id: str
    signal_line: float
    bet_line: float
    closing_line: float
    signal_time: Optional[datetime] = None
    bet_time: Optional[datetime] = None
    market_type: str = "points"
    sportsbook: str = "unknown"
    predicted_prob: float = 0.5
    market_prob_at_bet: float = 0.5
    stake: float = 100.0
    pnl: float = 0.0
    won: Optional[bool] = None
    odds_decimal: float = 1.909
    edge_cents: float = 0.0
    direction: str = "over"
    player: str = ""


@dataclass
class SignalExecutionResult:
    """Execution reality result for a single signal."""
    bet_id: str
    # Theoretical (perfect execution)
    theoretical_line: float
    theoretical_stake: float
    theoretical_pnl: float
    # After slippage
    execution_line: float
    slippage_cents: float
    # After limits
    allowed_stake: float
    stake_was_capped: bool
    # After latency
    p_line_available: float
    edge_retention_pct: float
    # After rejection
    p_bet_placed: float
    # Combined
    realistic_pnl: float
    execution_cost: float           # theoretical_pnl - realistic_pnl
    execution_cost_pct: float       # execution_cost / |theoretical_pnl|
    edge_survives: bool             # Does edge survive execution?


@dataclass
class ExecutionReport:
    """Full execution reality report across all signals."""
    generated_at: datetime
    sport: str
    n_signals: int
    # Profit comparison
    theoretical_total_pnl: float
    realistic_total_pnl: float
    execution_cost_total: float
    execution_cost_pct: float
    # ROI comparison
    theoretical_roi: float
    realistic_roi: float
    roi_erosion_pct: float
    # Component breakdown
    slippage_cost_total: float
    limit_cost_total: float
    latency_cost_total: float
    rejection_cost_total: float
    slippage_cost_pct: float
    limit_cost_pct: float
    latency_cost_pct: float
    rejection_cost_pct: float
    # Per-signal results
    signal_results: List[SignalExecutionResult]
    # Sub-profiles
    slippage_profile: Optional[SlippageProfile] = None
    limit_profile: Optional[LimitProfile] = None
    latency_profile: Optional[LatencyProfile] = None
    rejection_profile: Optional[RejectionProfile] = None
    # Verdict
    edge_survives: bool = False
    confidence_level: float = 0.0    # Statistical confidence that edge > 0
    verdict: str = ""
    kill_signal: bool = False        # True if edge dies after execution


class ExecutionSimulator:
    """Chains slippage, limit, latency, and rejection models to produce
    a realistic execution simulation.

    Usage:
        sim = ExecutionSimulator(sport="nba")
        sim.load_historical_bets(bets)
        report = sim.simulate_real_execution(signals)
        print(report.verdict)
    """

    def __init__(self, sport: str = "nba"):
        self.sport = sport
        self.slippage = SlippageModel(sport)
        self.limits = LimitModel(sport)
        self.latency = LatencyModel(sport)
        self.rejection = RejectionModel(sport)

    def load_historical_bets(
        self,
        bets: list,
        rejected_bets: Optional[list] = None,
    ) -> None:
        """Load historical data into all sub-models."""
        self.slippage.load_from_bets(bets)
        self.limits.load_bet_history(bets)
        self.latency.load_from_bets(bets)
        self.rejection.load_from_bets(bets, rejected_bets)
        log.info("ExecutionSimulator loaded %d bets into all models", len(bets))

    def set_manual_book_limit(self, book_name: str, max_bet: float) -> None:
        """Set a known limit for a specific sportsbook."""
        self.limits.set_manual_limit(book_name, max_bet)

    def simulate_real_execution(
        self,
        signals: List[BetSignal],
    ) -> ExecutionReport:
        """Run the full execution reality simulation.

        For each signal:
            1. Apply slippage model -> actual execution price
            2. Apply limit model -> actual bet size allowed
            3. Apply latency model -> P(line still available)
            4. Apply rejection model -> P(bet actually placed)
        """
        results: List[SignalExecutionResult] = []
        theoretical_total = 0.0
        realistic_total = 0.0
        total_staked_theoretical = 0.0
        total_staked_realistic = 0.0
        slippage_cost = 0.0
        limit_cost = 0.0
        latency_cost = 0.0
        rejection_cost = 0.0

        for sig in signals:
            # --- 1. Slippage ---
            latency_s = 0.0
            if sig.signal_time and sig.bet_time:
                latency_s = max(0, (sig.bet_time - sig.signal_time).total_seconds())

            # Line movement speed
            total_move = abs(sig.closing_line - sig.signal_line)
            move_speed = total_move / max(1, latency_s) if latency_s > 0 else 0

            slip_result = self.slippage.predict(
                signal_line=sig.signal_line,
                market_type=sig.market_type,
                time_to_bet_seconds=latency_s,
                line_movement_speed=move_speed,
            )
            execution_line = slip_result.expected_execution_line
            slip_cents = slip_result.expected_slippage_cents

            # --- 2. Limits ---
            limit_result = self.limits.apply_limit(sig.sportsbook, sig.stake)
            allowed_stake = limit_result.allowed_stake

            # --- 3. Latency ---
            lat_result = self.latency.predict(
                latency_seconds=latency_s,
                market_type=sig.market_type,
                edge_cents=sig.edge_cents,
            )

            # --- 4. Rejection ---
            rej_result = self.rejection.predict(
                sportsbook=sig.sportsbook,
                signal_edge_cents=sig.edge_cents,
                market_type=sig.market_type,
            )

            # --- Combine ---
            # Theoretical P&L (perfect execution)
            theoretical_pnl = sig.pnl

            # Realistic P&L adjustments:
            # a) Slippage reduces edge
            edge_original = sig.edge_cents
            edge_after_slip = edge_original - abs(slip_cents)

            # b) Limits reduce stake
            stake_ratio = allowed_stake / sig.stake if sig.stake > 0 else 1.0

            # c) Latency reduces probability of capturing the line
            latency_factor = lat_result.edge_retention_pct / 100.0

            # d) Rejection reduces probability of bet being placed
            placement_prob = rej_result.p_acceptance

            # Realistic PnL = theoretical * stake_ratio * latency_factor * placement_prob
            # Adjusted for slippage impact on edge
            edge_ratio = max(0, edge_after_slip / edge_original) if edge_original > 0 else 1.0
            realistic_pnl = theoretical_pnl * stake_ratio * latency_factor * placement_prob * edge_ratio

            # Cost attribution
            cost_total = theoretical_pnl - realistic_pnl
            if cost_total > 0 and theoretical_pnl != 0:
                # Attribute costs proportionally
                total_discount = 1.0 - (stake_ratio * latency_factor * placement_prob * edge_ratio)
                if total_discount > 0:
                    slip_share = (1.0 - edge_ratio) / total_discount if edge_ratio < 1 else 0
                    limit_share = (1.0 - stake_ratio) / total_discount if stake_ratio < 1 else 0
                    lat_share = (1.0 - latency_factor) / total_discount if latency_factor < 1 else 0
                    rej_share = (1.0 - placement_prob) / total_discount if placement_prob < 1 else 0
                    total_shares = slip_share + limit_share + lat_share + rej_share
                    if total_shares > 0:
                        slippage_cost += cost_total * slip_share / total_shares
                        limit_cost += cost_total * limit_share / total_shares
                        latency_cost += cost_total * lat_share / total_shares
                        rejection_cost += cost_total * rej_share / total_shares

            theoretical_total += theoretical_pnl
            realistic_total += realistic_pnl
            total_staked_theoretical += sig.stake
            total_staked_realistic += allowed_stake * placement_prob

            cost = theoretical_pnl - realistic_pnl
            cost_pct = (cost / abs(theoretical_pnl) * 100) if theoretical_pnl != 0 else 0

            results.append(SignalExecutionResult(
                bet_id=sig.bet_id,
                theoretical_line=sig.signal_line,
                theoretical_stake=sig.stake,
                theoretical_pnl=theoretical_pnl,
                execution_line=execution_line,
                slippage_cents=slip_cents,
                allowed_stake=allowed_stake,
                stake_was_capped=limit_result.is_capped,
                p_line_available=lat_result.p_line_available,
                edge_retention_pct=lat_result.edge_retention_pct,
                p_bet_placed=rej_result.p_acceptance,
                realistic_pnl=realistic_pnl,
                execution_cost=cost,
                execution_cost_pct=cost_pct,
                edge_survives=realistic_pnl > 0 if theoretical_pnl > 0 else True,
            ))

        # --- Aggregate ---
        exec_cost_total = theoretical_total - realistic_total
        exec_cost_pct = (exec_cost_total / abs(theoretical_total) * 100) if theoretical_total != 0 else 0

        theo_roi = (theoretical_total / total_staked_theoretical * 100) if total_staked_theoretical > 0 else 0
        real_roi = (realistic_total / total_staked_realistic * 100) if total_staked_realistic > 0 else 0
        roi_erosion = theo_roi - real_roi

        # Statistical significance test: is realistic ROI > 0?
        if results:
            realistic_pnls = np.array([r.realistic_pnl for r in results])
            n = len(realistic_pnls)
            if n > 1 and np.std(realistic_pnls) > 0:
                t_stat = np.mean(realistic_pnls) / (np.std(realistic_pnls) / math.sqrt(n))
                # One-sided p-value approximation
                confidence = min(0.99, max(0.0, 0.5 + 0.5 * math.erf(t_stat / math.sqrt(2))))
            else:
                confidence = 0.5
        else:
            confidence = 0.0

        edge_survives = real_roi > 0 and confidence > 0.90
        kill_signal = real_roi <= 0 or confidence < 0.75

        # Build verdict
        if edge_survives:
            verdict = (
                f"EDGE SURVIVES execution reality. "
                f"Realistic ROI {real_roi:.2f}% (theoretical {theo_roi:.2f}%). "
                f"Execution costs erode {exec_cost_pct:.1f}% of profit. "
                f"Confidence: {confidence:.1%}."
            )
        elif kill_signal:
            verdict = (
                f"KILL SIGNAL: Edge does NOT survive execution. "
                f"Realistic ROI {real_roi:.2f}% (theoretical {theo_roi:.2f}%). "
                f"Execution costs consume {exec_cost_pct:.1f}% of theoretical profit. "
                f"Confidence: {confidence:.1%}. STOP BETTING until execution improves."
            )
        else:
            verdict = (
                f"MARGINAL: Edge barely survives. "
                f"Realistic ROI {real_roi:.2f}% (theoretical {theo_roi:.2f}%). "
                f"Execution costs erode {exec_cost_pct:.1f}%. "
                f"Confidence: {confidence:.1%}. Monitor closely."
            )

        # Cost breakdown percentages
        total_cost = slippage_cost + limit_cost + latency_cost + rejection_cost
        if total_cost > 0:
            slip_pct = slippage_cost / total_cost * 100
            lim_pct = limit_cost / total_cost * 100
            lat_pct = latency_cost / total_cost * 100
            rej_pct = rejection_cost / total_cost * 100
        else:
            slip_pct = lim_pct = lat_pct = rej_pct = 25.0

        return ExecutionReport(
            generated_at=datetime.utcnow(),
            sport=self.sport,
            n_signals=len(signals),
            theoretical_total_pnl=theoretical_total,
            realistic_total_pnl=realistic_total,
            execution_cost_total=exec_cost_total,
            execution_cost_pct=exec_cost_pct,
            theoretical_roi=theo_roi,
            realistic_roi=real_roi,
            roi_erosion_pct=roi_erosion,
            slippage_cost_total=slippage_cost,
            limit_cost_total=limit_cost,
            latency_cost_total=latency_cost,
            rejection_cost_total=rejection_cost,
            slippage_cost_pct=slip_pct,
            limit_cost_pct=lim_pct,
            latency_cost_pct=lat_pct,
            rejection_cost_pct=rej_pct,
            signal_results=results,
            slippage_profile=self.slippage.profile(),
            limit_profile=self.limits.profile(),
            latency_profile=self.latency.profile(),
            rejection_profile=self.rejection.profile(),
            edge_survives=edge_survives,
            confidence_level=confidence,
            verdict=verdict,
            kill_signal=kill_signal,
        )

    def execution_adjusted_roi(self, signals: List[BetSignal]) -> float:
        """ROI after all execution costs."""
        report = self.simulate_real_execution(signals)
        return report.realistic_roi

    def edge_survives_execution(self, signals: List[BetSignal]) -> bool:
        """True only if execution-adjusted ROI > 0 with statistical significance."""
        report = self.simulate_real_execution(signals)
        return report.edge_survives

    def quick_summary(self, signals: List[BetSignal]) -> dict:
        """Return a lightweight summary dict (for embedding in other reports)."""
        report = self.simulate_real_execution(signals)
        return {
            "n_signals": report.n_signals,
            "theoretical_roi": round(report.theoretical_roi, 3),
            "realistic_roi": round(report.realistic_roi, 3),
            "execution_cost_pct": round(report.execution_cost_pct, 2),
            "edge_survives": report.edge_survives,
            "kill_signal": report.kill_signal,
            "confidence": round(report.confidence_level, 4),
            "biggest_cost": max(
                [("slippage", report.slippage_cost_pct),
                 ("limits", report.limit_cost_pct),
                 ("latency", report.latency_cost_pct),
                 ("rejection", report.rejection_cost_pct)],
                key=lambda x: x[1],
            )[0],
            "verdict": report.verdict,
        }
