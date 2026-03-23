"""
services/kill_switch.py
========================
Hard kill switch with 6 non-negotiable halt conditions.

When ANY kill condition fires, the system MUST stop placing bets.
No overrides. No exceptions. These are the circuit breakers that
prevent catastrophic loss.

Kill conditions:
  1. clv_death:               Avg CLV <= 0 over last 250 bets
  2. model_worse_than_market: Brier > market Brier over 200 bets
  3. edge_decay_detected:     CLV trend negative with p < 0.05 over 500 bets
  4. execution_destroys_edge: Execution-adjusted ROI < 0 over 200 bets
  5. max_drawdown:            Bankroll drawdown > 25% from peak
  6. variance_blowup:         Actual variance > 3x expected over 100 bets
                              (reduces to 50% Kelly instead of full halt)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from database.connection import session_scope
from database.models import Bet, SystemState

log = logging.getLogger(__name__)


@dataclass
class KillCondition:
    """Status of a single kill condition."""
    name: str
    triggered: bool
    value: float
    threshold: float
    sample_size: int
    min_sample_size: int
    description: str
    severity: str         # halt, reduce, monitor
    action: str           # What to do when triggered
    details: str = ""


@dataclass
class KillSwitchStatus:
    """Overall kill switch status."""
    is_active: bool              # True = system is cleared to bet
    conditions: List[KillCondition]
    triggered_conditions: List[str]
    halt_reason: str
    severity: str                # clear, reduced, halted
    recommended_kelly_mult: float  # 1.0 = normal, 0.5 = reduced, 0.0 = halted
    checked_at: datetime
    total_bets_analyzed: int


class KillSwitch:
    """Hard kill switch system for the quant engine.

    Usage:
        ks = KillSwitch()
        status = ks.check_all(bankroll=5000, peak_bankroll=6000)
        if not status.is_active:
            print(f"SYSTEM HALTED: {status.halt_reason}")
    """

    def __init__(self, sport: str = "NBA"):
        self.sport = sport

    def check_all(
        self,
        bankroll: float = 0.0,
        peak_bankroll: float = 0.0,
    ) -> KillSwitchStatus:
        """Run all 6 kill conditions and return overall status.

        Args:
            bankroll: Current bankroll.
            peak_bankroll: Highest bankroll ever reached.
        """
        bets = self._load_settled_bets()

        conditions = [
            self._check_clv_death(bets),
            self._check_model_worse_than_market(bets),
            self._check_edge_decay(bets),
            self._check_execution_destroys_edge(bets),
            self._check_max_drawdown(bankroll, peak_bankroll),
            self._check_variance_blowup(bets),
        ]

        triggered = [c for c in conditions if c.triggered]
        halt_conditions = [c for c in triggered if c.severity == "halt"]
        reduce_conditions = [c for c in triggered if c.severity == "reduce"]

        if halt_conditions:
            is_active = False
            severity = "halted"
            halt_reason = "; ".join(c.name + ": " + c.description for c in halt_conditions)
            kelly_mult = 0.0
        elif reduce_conditions:
            is_active = True  # Still active but reduced
            severity = "reduced"
            halt_reason = "; ".join(c.name + ": " + c.description for c in reduce_conditions)
            kelly_mult = 0.5
        else:
            is_active = True
            severity = "clear"
            halt_reason = ""
            kelly_mult = 1.0

        status = KillSwitchStatus(
            is_active=is_active,
            conditions=conditions,
            triggered_conditions=[c.name for c in triggered],
            halt_reason=halt_reason,
            severity=severity,
            recommended_kelly_mult=kelly_mult,
            checked_at=datetime.now(timezone.utc),
            total_bets_analyzed=len(bets),
        )

        # Log state change
        if not is_active:
            self._record_state_change(status, bankroll)

        return status

    def is_system_active(
        self,
        bankroll: float = 0.0,
        peak_bankroll: float = 0.0,
    ) -> bool:
        """Quick check: can we place bets right now?"""
        status = self.check_all(bankroll, peak_bankroll)
        return status.is_active

    def get_halt_reason(
        self,
        bankroll: float = 0.0,
        peak_bankroll: float = 0.0,
    ) -> str:
        """Get the reason the system is halted, or empty string if active."""
        status = self.check_all(bankroll, peak_bankroll)
        return status.halt_reason

    # ── Individual kill conditions ───────────────────────────────────

    def _check_clv_death(self, bets: List[Dict]) -> KillCondition:
        """Kill condition 1: Avg CLV <= 0 over last 250 bets.

        If you can't beat the closing line over 250 bets, you don't have edge.
        """
        min_n = 250
        window = bets[-min_n:] if len(bets) >= min_n else bets

        clv_values = self._compute_clv_array(window)
        avg_clv = float(np.mean(clv_values)) if len(clv_values) > 0 else 0.0

        triggered = len(bets) >= min_n and avg_clv <= 0.0

        return KillCondition(
            name="clv_death",
            triggered=triggered,
            value=round(avg_clv, 3),
            threshold=0.0,
            sample_size=len(window),
            min_sample_size=min_n,
            description=f"Avg CLV = {avg_clv:.3f} cents over {len(window)} bets",
            severity="halt",
            action="Stop all betting. CLV is dead.",
            details=f"CLV beat rate: {float(np.mean(clv_values > 0)):.1%}" if len(clv_values) > 0 else "",
        )

    def _check_model_worse_than_market(self, bets: List[Dict]) -> KillCondition:
        """Kill condition 2: Our Brier score > market Brier over 200 bets.

        If the market's implied probabilities are better than our model's,
        we're adding negative value.
        """
        min_n = 200
        window = bets[-min_n:] if len(bets) >= min_n else bets

        pred_probs = []
        market_probs = []
        outcomes = []

        for b in window:
            pp = b.get("predicted_prob")
            odds = b.get("odds_decimal")
            profit = b.get("profit")

            if pp is not None and odds and odds > 0 and profit is not None:
                pred_probs.append(pp)
                market_probs.append(1.0 / odds)
                outcomes.append(1.0 if profit > 0 else 0.0)

        if len(pred_probs) < 30:
            return KillCondition(
                name="model_worse_than_market",
                triggered=False,
                value=0.0,
                threshold=0.0,
                sample_size=len(pred_probs),
                min_sample_size=min_n,
                description="Insufficient data for Brier comparison",
                severity="halt",
                action="",
            )

        pred_arr = np.array(pred_probs)
        market_arr = np.array(market_probs)
        out_arr = np.array(outcomes)

        our_brier = float(np.mean((pred_arr - out_arr) ** 2))
        market_brier = float(np.mean((market_arr - out_arr) ** 2))

        triggered = len(bets) >= min_n and our_brier > market_brier

        return KillCondition(
            name="model_worse_than_market",
            triggered=triggered,
            value=round(our_brier, 6),
            threshold=round(market_brier, 6),
            sample_size=len(pred_probs),
            min_sample_size=min_n,
            description=f"Our Brier={our_brier:.4f} vs Market Brier={market_brier:.4f}",
            severity="halt",
            action="Stop betting. Model adds negative value vs market.",
            details=f"Brier difference: {our_brier - market_brier:+.4f}",
        )

    def _check_edge_decay(self, bets: List[Dict]) -> KillCondition:
        """Kill condition 3: CLV trend negative with p < 0.05 over 500 bets.

        Even if CLV is still positive, a statistically significant downtrend
        means the books are adapting and edge is dying.
        """
        min_n = 500
        window = bets[-min_n:] if len(bets) >= min_n else bets

        clv_values = self._compute_clv_array(window)

        if len(clv_values) < 30:
            return KillCondition(
                name="edge_decay_detected",
                triggered=False,
                value=0.0,
                threshold=0.05,
                sample_size=len(clv_values),
                min_sample_size=min_n,
                description="Insufficient data for trend analysis",
                severity="halt",
                action="",
            )

        x = np.arange(len(clv_values))
        slope, _, _, p_value, _ = sp_stats.linregress(x, clv_values)

        triggered = len(bets) >= min_n and slope < 0 and p_value < 0.05

        return KillCondition(
            name="edge_decay_detected",
            triggered=triggered,
            value=round(p_value, 6),
            threshold=0.05,
            sample_size=len(clv_values),
            min_sample_size=min_n,
            description=f"CLV trend slope={slope:.6f}, p={p_value:.4f}",
            severity="halt",
            action="Stop betting. Statistically significant edge decay detected.",
            details=f"Estimated CLV decline: {slope * 100:.4f} cents per 100 bets",
        )

    def _check_execution_destroys_edge(self, bets: List[Dict]) -> KillCondition:
        """Kill condition 4: Execution-adjusted ROI < 0 over 200 bets.

        Even if the model has theoretical edge, if execution costs
        (slippage, limits, rejections) eat all the profit, there's no edge.
        """
        min_n = 200
        window = bets[-min_n:] if len(bets) >= min_n else bets

        total_profit = sum(b.get("profit", 0) or 0 for b in window)
        total_stake = sum(b.get("stake", 0) or 0 for b in window)

        if total_stake <= 0:
            roi = 0.0
        else:
            roi = (total_profit / total_stake) * 100.0

        triggered = len(bets) >= min_n and roi < 0.0

        return KillCondition(
            name="execution_destroys_edge",
            triggered=triggered,
            value=round(roi, 2),
            threshold=0.0,
            sample_size=len(window),
            min_sample_size=min_n,
            description=f"Execution-adjusted ROI = {roi:.2f}% over {len(window)} bets",
            severity="halt",
            action="Stop betting. Execution costs exceed theoretical edge.",
            details=f"Total P&L: ${total_profit:.2f} on ${total_stake:.2f} wagered",
        )

    def _check_max_drawdown(
        self,
        bankroll: float,
        peak_bankroll: float,
    ) -> KillCondition:
        """Kill condition 5: Bankroll drawdown > 25% from peak."""
        if peak_bankroll <= 0:
            drawdown_pct = 0.0
        else:
            drawdown_pct = (peak_bankroll - bankroll) / peak_bankroll * 100

        triggered = drawdown_pct > 25.0

        return KillCondition(
            name="max_drawdown",
            triggered=triggered,
            value=round(drawdown_pct, 1),
            threshold=25.0,
            sample_size=0,
            min_sample_size=0,
            description=f"Current drawdown = {drawdown_pct:.1f}% (bankroll=${bankroll:.0f}, peak=${peak_bankroll:.0f})",
            severity="halt",
            action="Stop all betting. Capital preservation mode.",
            details=f"Need {((peak_bankroll / max(bankroll, 1)) - 1) * 100:.1f}% gain to recover" if bankroll > 0 else "",
        )

    def _check_variance_blowup(self, bets: List[Dict]) -> KillCondition:
        """Kill condition 6: Actual variance > 3x expected over 100 bets.

        Unlike the other conditions, this reduces to 50% Kelly instead of
        halting entirely. Variance blowup often means the model's probability
        estimates are miscalibrated, but edge may still exist.
        """
        min_n = 100
        window = bets[-min_n:] if len(bets) >= min_n else bets

        profits = np.array([b.get("profit", 0) or 0 for b in window])
        pred_probs = np.array([b.get("predicted_prob", 0.5) for b in window])

        if len(profits) < 20:
            return KillCondition(
                name="variance_blowup",
                triggered=False,
                value=1.0,
                threshold=3.0,
                sample_size=len(profits),
                min_sample_size=min_n,
                description="Insufficient data for variance analysis",
                severity="reduce",
                action="",
            )

        actual_var = float(np.var(profits, ddof=1))
        avg_stake = float(np.mean(np.abs(profits)))
        avg_prob = float(np.clip(np.mean(pred_probs), 0.01, 0.99))
        expected_var = avg_prob * (1 - avg_prob) * (avg_stake ** 2)

        ratio = actual_var / max(expected_var, 1e-10)

        triggered = len(bets) >= min_n and ratio > 3.0

        return KillCondition(
            name="variance_blowup",
            triggered=triggered,
            value=round(ratio, 2),
            threshold=3.0,
            sample_size=len(window),
            min_sample_size=min_n,
            description=f"Variance ratio = {ratio:.2f}x (actual={actual_var:.2f}, expected={expected_var:.2f})",
            severity="reduce",  # Reduce, don't halt
            action="Reduce to 50% Kelly. Investigate probability calibration.",
            details=f"Actual std=${np.sqrt(actual_var):.2f}, Expected std=${np.sqrt(expected_var):.2f}",
        )

    # ── Data loading ──────────────────────────────────────────────────

    def _load_settled_bets(self) -> List[Dict]:
        """Load all settled bets from database."""
        try:
            with session_scope() as session:
                bets = (
                    session.query(Bet)
                    .filter(Bet.sport == self.sport)
                    .filter(Bet.status == "settled")
                    .order_by(Bet.timestamp.asc())
                    .all()
                )
                return [b.to_dict() for b in bets]
        except Exception as e:
            log.warning("Failed to load bets for kill switch: %s", e)
            return []

    def _compute_clv_array(self, bets: List[Dict]) -> np.ndarray:
        """Compute CLV array from bet dicts."""
        clv = []
        for b in bets:
            bl = b.get("bet_line")
            cl = b.get("closing_line")
            direction = b.get("direction", "over")
            if bl is not None and cl is not None:
                if direction.lower() == "over":
                    clv.append(cl - bl)
                else:
                    clv.append(bl - cl)
            else:
                clv.append(0.0)
        return np.array(clv)

    def _record_state_change(self, status: KillSwitchStatus, bankroll: float) -> None:
        """Record a system state change in the database."""
        try:
            state_str = "KILLED" if status.severity == "halted" else "REDUCED"
            with session_scope() as session:
                session.add(SystemState(
                    sport=self.sport,
                    state=state_str,
                    reason=status.halt_reason[:500],
                    bankroll_at_change=bankroll,
                ))
        except Exception as e:
            log.warning("Failed to record state change: %s", e)
