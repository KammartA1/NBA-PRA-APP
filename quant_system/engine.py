"""Quant Engine — Master orchestrator for the entire betting system.

This is the single entry point. Every bet flows through here:

    1. Model generates prediction
    2. Engine checks: should we bet? (edge validation, circuit breakers)
    3. Engine sizes: how much? (adaptive Kelly, exposure limits)
    4. Engine logs: record everything (bet logger, line tracker)
    5. Engine learns: after settlement, run feedback loop

Usage:
    from quant_system.engine import QuantEngine
    from quant_system.core.types import Sport, BetType

    engine = QuantEngine(sport=Sport.GOLF, initial_bankroll=1000.0)

    # Before placing a bet
    decision = engine.evaluate_bet(
        player="Scottie Scheffler",
        bet_type=BetType.OVER,
        stat_type="birdies",
        line=3.5,
        direction="over",
        model_prob=0.62,
        model_projection=4.1,
        model_std=1.5,
        odds_american=-110,
    )

    if decision["approved"]:
        engine.place_bet(decision)  # Logs + returns bet_id

    # After game ends
    engine.settle_bet(bet_id, actual_result=5.0, closing_line=4.0)

    # Daily maintenance
    engine.run_daily_cycle()

    # Dashboard
    report = engine.dashboard_report()
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .core.types import BetType, RiskState, Sport, SystemState
from .core.bet_logger import BetLogger
from .core.clv_tracker import CLVTracker
from .core.edge_validator import EdgeValidator
from .core.calibration import CalibrationMonitor
from .risk.kelly_adaptive import AdaptiveKelly, KellyConfig
from .risk.bankroll_manager import BankrollManager
from .risk.failure_protection import FailureProtection
from .risk.exposure_manager import ExposureManager
from .market.line_tracker import LineTracker
from .market.sharp_detector import SharpMoneyDetector
from .learning.feedback_loop import FeedbackLoop
from .dashboard.reporting import QuantDashboard

logger = logging.getLogger(__name__)


class QuantEngine:
    """Master orchestrator — every bet flows through here."""

    def __init__(
        self,
        sport: Sport,
        initial_bankroll: float = 1000.0,
        kelly_config: KellyConfig | None = None,
        db_path: str | None = None,
    ):
        self.sport = sport
        self.initial_bankroll = initial_bankroll
        self._db_path = db_path

        # Initialize all subsystems
        self.bet_logger = BetLogger(sport, db_path)
        self.clv_tracker = CLVTracker(sport, db_path)
        self.edge_validator = EdgeValidator(sport, db_path)
        self.calibration = CalibrationMonitor(sport, db_path)
        self.kelly = AdaptiveKelly(kelly_config or KellyConfig())
        self.bankroll_mgr = BankrollManager(sport, initial_bankroll, db_path=db_path)
        self.failure_protection = FailureProtection(sport, db_path=db_path)
        self.exposure_mgr = ExposureManager()
        self.line_tracker = LineTracker(sport, db_path)
        self.sharp_detector = SharpMoneyDetector(sport, db_path)
        self.feedback_loop = FeedbackLoop(sport, db_path)
        self.dashboard = QuantDashboard(sport, initial_bankroll, db_path)

        logger.info("QuantEngine initialized: sport=%s, bankroll=$%.2f", sport.value, initial_bankroll)

    # ── Bet Evaluation ────────────────────────────────────────────────

    def evaluate_bet(
        self,
        player: str,
        bet_type: BetType,
        stat_type: str,
        line: float,
        direction: str,
        model_prob: float,
        model_projection: float,
        model_std: float,
        odds_american: int = -110,
        confidence_score: float = 0.0,
        engine_agreement: float = 0.0,
        features_snapshot: dict | None = None,
    ) -> dict:
        """Evaluate whether to place a bet and how much to stake.

        This is THE decision function. It checks:
        1. System state (are we allowed to bet?)
        2. Circuit breakers (any triggered?)
        3. Sharp money (does it agree?)
        4. Kelly sizing (how much?)
        5. Exposure limits (room for this bet?)

        Returns:
            {
                "approved": bool,
                "rejection_reason": str,
                "stake": float,
                "kelly_details": dict,
                "sharp_signal": dict,
                "risk_state": RiskState,
                # Pass-through for place_bet()
                "player": str,
                "bet_type": BetType,
                "stat_type": str,
                "line": float,
                "direction": str,
                "model_prob": float,
                "model_projection": float,
                "model_std": float,
                "odds_american": int,
                "confidence_score": float,
                "engine_agreement": float,
                "features_snapshot": dict,
            }
        """
        # Convert odds
        if odds_american > 0:
            odds_decimal = 1.0 + odds_american / 100.0
        elif odds_american < 0:
            odds_decimal = 1.0 + 100.0 / abs(odds_american)
        else:
            odds_decimal = 2.0

        market_prob = 1.0 / odds_decimal

        # 1. Get current risk state
        system_state = self.edge_validator.get_current_state()
        risk_state = self.bankroll_mgr.get_risk_state(system_state)

        base_result = {
            "player": player,
            "bet_type": bet_type,
            "stat_type": stat_type,
            "line": line,
            "direction": direction,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "model_projection": model_projection,
            "model_std": model_std,
            "odds_american": odds_american,
            "odds_decimal": odds_decimal,
            "confidence_score": confidence_score,
            "engine_agreement": engine_agreement,
            "features_snapshot": features_snapshot or {},
            "risk_state": risk_state,
        }

        # 2. System state check
        if system_state in (SystemState.SUSPENDED, SystemState.KILLED):
            return {**base_result, "approved": False,
                    "rejection_reason": f"System is {system_state.value}",
                    "stake": 0.0, "kelly_details": {}, "sharp_signal": {}}

        # 3. Circuit breakers
        breakers = self.failure_protection.check_all(risk_state)
        if breakers["recommended_state"] in (SystemState.SUSPENDED, SystemState.KILLED):
            return {**base_result, "approved": False,
                    "rejection_reason": f"Circuit breaker: {breakers['breakers_triggered']}",
                    "stake": 0.0, "kelly_details": {}, "sharp_signal": {}}

        # 4. Sharp money check
        sharp_signal = self.sharp_detector.model_agrees_with_sharp(player, stat_type, direction)
        sharp_mult = sharp_signal.get("confidence_multiplier", 1.0)

        # Adjust model_prob by sharp signal (mild adjustment)
        adjusted_prob = model_prob
        if sharp_mult < 0.8:
            # Sharp disagrees strongly — reduce our confidence
            adjusted_prob = model_prob * 0.95 + market_prob * 0.05

        # 5. Kelly sizing
        clv_summary = self.clv_tracker.rolling_clv(100)
        cal_report = self.calibration.compute_calibration()

        kelly_result = self.kelly.adaptive_stake(
            win_prob=adjusted_prob,
            decimal_odds=odds_decimal,
            bankroll=risk_state.bankroll,
            risk_state=risk_state,
            clv_avg_cents=clv_summary.get("avg_clv_cents", 0.0),
            calibration_mae=cal_report.get("mean_absolute_error", 0.0),
        )

        if kelly_result["blocked"]:
            return {**base_result, "approved": False,
                    "rejection_reason": kelly_result["block_reason"],
                    "stake": 0.0, "kelly_details": kelly_result, "sharp_signal": sharp_signal}

        stake = kelly_result["stake_dollars"]

        # 6. Exposure check
        allowed, reason, max_stake = self.exposure_mgr.check_exposure(
            player, stat_type, stake, risk_state
        )
        if not allowed:
            if max_stake > 0:
                stake = max_stake  # Reduce to max allowed
            else:
                return {**base_result, "approved": False,
                        "rejection_reason": reason,
                        "stake": 0.0, "kelly_details": kelly_result, "sharp_signal": sharp_signal}

        # 7. Bankroll manager final check
        can_bet, reason = self.bankroll_mgr.can_place_bet(stake, player, risk_state)
        if not can_bet:
            return {**base_result, "approved": False,
                    "rejection_reason": reason,
                    "stake": 0.0, "kelly_details": kelly_result, "sharp_signal": sharp_signal}

        # APPROVED
        return {
            **base_result,
            "approved": True,
            "rejection_reason": "",
            "stake": round(stake, 2),
            "kelly_details": kelly_result,
            "sharp_signal": sharp_signal,
        }

    # ── Bet Placement ─────────────────────────────────────────────────

    def place_bet(self, decision: dict) -> str:
        """Place a bet that was approved by evaluate_bet(). Returns bet_id."""
        if not decision.get("approved"):
            raise ValueError("Cannot place unapproved bet")

        # Record the line snapshot
        self.line_tracker.record_snapshot(
            player=decision["player"],
            stat_type=decision["stat_type"],
            source="model_bet",
            line=decision["line"],
            odds_american=decision["odds_american"],
        )

        # Log the bet
        record = self.bet_logger.log_bet(
            player=decision["player"],
            bet_type=decision["bet_type"],
            stat_type=decision["stat_type"],
            line=decision["line"],
            direction=decision["direction"],
            model_prob=decision["model_prob"],
            market_prob=decision["market_prob"],
            stake=decision["stake"],
            kelly_fraction=decision["kelly_details"].get("adjusted_kelly", 0),
            odds_american=decision["odds_american"],
            model_projection=decision["model_projection"],
            model_std=decision["model_std"],
            confidence_score=decision["confidence_score"],
            engine_agreement=decision["engine_agreement"],
            features_snapshot=decision.get("features_snapshot"),
        )

        logger.info("BET PLACED: %s | %s %s %s @ %.1f | $%.2f | edge=%.3f",
                     record.bet_id, decision["player"], decision["direction"],
                     decision["stat_type"], decision["line"], decision["stake"],
                     decision["model_prob"] - decision["market_prob"])

        return record.bet_id

    # ── Settlement ────────────────────────────────────────────────────

    def settle_bet(
        self,
        bet_id: str,
        actual_result: float,
        closing_line: Optional[float] = None,
        closing_odds: Optional[int] = None,
    ) -> dict:
        """Settle a bet and compute CLV.

        Returns:
            {
                "pnl": float,
                "clv": CLVResult or None,
                "status": str,
            }
        """
        from .core.types import BetStatus

        # Determine win/loss from the bet record
        bets = self.bet_logger.get_pending_bets()
        bet = next((b for b in bets if b["bet_id"] == bet_id), None)

        if bet is None:
            # Try settled bets (might already be settled)
            raise ValueError(f"Bet {bet_id} not found in pending bets")

        line = bet["line"]
        direction = bet["direction"]

        if direction == "over":
            won = actual_result > line
        elif direction == "under":
            won = actual_result < line
        else:
            won = actual_result > line  # Default for outrights

        status = BetStatus.WON if won else BetStatus.LOST
        if actual_result == line:
            status = BetStatus.PUSH

        pnl = self.bet_logger.settle_bet(bet_id, status, actual_result, closing_line, closing_odds)

        # Calculate CLV if closing line available
        clv_result = None
        if closing_line is not None:
            try:
                clv_result = self.clv_tracker.calculate_clv(bet_id, closing_line, closing_odds)
            except Exception:
                logger.exception("CLV calculation failed for %s", bet_id)

        return {
            "pnl": pnl,
            "clv": clv_result,
            "status": status.value,
            "won": won,
        }

    # ── Daily Cycle ───────────────────────────────────────────────────

    def run_daily_cycle(self) -> dict:
        """Run the full daily maintenance cycle.

        Call this once per day (or after each batch of settlements).
        """
        risk_state = self.bankroll_mgr.get_risk_state()
        return self.feedback_loop.run_full_cycle(
            bankroll=risk_state.bankroll,
            peak_bankroll=risk_state.peak_bankroll,
        )

    # ── Dashboard ─────────────────────────────────────────────────────

    def dashboard_report(self) -> dict:
        """Generate full dashboard report."""
        return self.dashboard.full_report()

    def health_check(self) -> str:
        """Quick one-line health status."""
        return self.dashboard.health_summary()

    # ── Line Recording (for external scraper integration) ─────────────

    def record_line(
        self,
        player: str,
        stat_type: str,
        source: str,
        line: float,
        odds_american: Optional[int] = None,
        is_opening: bool = False,
        is_closing: bool = False,
    ) -> None:
        """Record a line snapshot from external scraper."""
        self.line_tracker.record_snapshot(
            player, stat_type, source, line, odds_american, is_opening, is_closing
        )
