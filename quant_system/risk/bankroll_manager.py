"""Bankroll Manager — Tracks P&L, drawdown, and enforces limits.

This is the source of truth for bankroll state. Every bet placement
and settlement updates the bankroll through this manager.

Key principles:
- Never trust cached state — always recalculate from bet log
- Peak bankroll only moves up (one-way ratchet)
- Daily limits reset at midnight UTC
- All exposure calculations are real-time
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np

from ..db.schema import BetLog, get_session
from ..core.types import RiskState, Sport, SystemState

logger = logging.getLogger(__name__)


class BankrollManager:
    """Real-time bankroll tracking and limit enforcement."""

    def __init__(
        self,
        sport: Sport,
        initial_bankroll: float = 1000.0,
        daily_loss_limit_pct: float = 0.08,   # 8% daily loss limit
        max_open_bets: int = 15,               # Max concurrent open bets
        db_path: str | None = None,
    ):
        self.sport = sport
        self.initial_bankroll = initial_bankroll
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_open_bets = max_open_bets
        self._db_path = db_path

    def _session(self):
        return get_session(self._db_path)

    def get_risk_state(self, system_state: SystemState | None = None) -> RiskState:
        """Compute current risk state from bet history."""
        session = self._session()
        try:
            # All settled bets
            settled = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost", "push"]))
                .order_by(BetLog.timestamp.asc())
                .all()
            )

            # Pending bets
            pending = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value, status="pending")
                .all()
            )

            # Calculate bankroll from history
            total_pnl = sum(r.pnl for r in settled)
            bankroll = self.initial_bankroll + total_pnl

            # Peak bankroll (running max)
            running = self.initial_bankroll
            peak = self.initial_bankroll
            for r in settled:
                running += r.pnl
                peak = max(peak, running)

            # Drawdown
            drawdown_pct = (peak - bankroll) / peak if peak > 0 else 0.0
            max_dd = self._compute_max_drawdown(settled)

            # Daily P&L
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            daily_settled = [r for r in settled if r.settled_at and r.settled_at >= today_start]
            daily_pnl = sum(r.pnl for r in daily_settled)
            daily_bet_count = len(daily_settled) + len([
                r for r in pending if r.timestamp >= today_start
            ])

            # Exposure
            total_exposure = sum(r.stake for r in pending)
            exposure_by_player: dict[str, float] = {}
            exposure_by_type: dict[str, float] = {}
            for r in pending:
                exposure_by_player[r.player] = exposure_by_player.get(r.player, 0) + r.stake
                exposure_by_type[r.bet_type] = exposure_by_type.get(r.bet_type, 0) + r.stake

            # Daily loss limit
            daily_loss_limit = bankroll * self.daily_loss_limit_pct
            daily_loss_remaining = daily_loss_limit + daily_pnl  # pnl is negative when losing

            # Kelly multiplier based on drawdown
            kelly_mult = 1.0
            if drawdown_pct > 0.30:
                kelly_mult = 0.25
            elif drawdown_pct > 0.20:
                kelly_mult = 0.50
            elif drawdown_pct > 0.10:
                kelly_mult = 0.75

            # Max single bet
            max_single = bankroll * 0.03  # 3% hard cap

            state = RiskState(
                bankroll=round(bankroll, 2),
                peak_bankroll=round(peak, 2),
                current_drawdown_pct=round(drawdown_pct, 4),
                max_drawdown_pct=round(max_dd, 4),
                daily_pnl=round(daily_pnl, 2),
                daily_bet_count=daily_bet_count,
                system_state=system_state or SystemState.ACTIVE,
                kelly_multiplier=kelly_mult,
                total_exposure=round(total_exposure, 2),
                exposure_by_player=exposure_by_player,
                exposure_by_type=exposure_by_type,
                daily_loss_limit=round(daily_loss_limit, 2),
                daily_loss_remaining=round(max(daily_loss_remaining, 0), 2),
                max_single_bet=round(max_single, 2),
            )

            return state
        finally:
            session.close()

    def can_place_bet(self, stake: float, player: str, risk_state: RiskState | None = None) -> tuple[bool, str]:
        """Check if a bet can be placed. Returns (allowed, reason)."""
        if risk_state is None:
            risk_state = self.get_risk_state()

        # System state check
        if risk_state.system_state in (SystemState.SUSPENDED, SystemState.KILLED):
            return False, f"System is {risk_state.system_state.value}"

        # Bankroll check
        if stake > risk_state.bankroll * 0.5:
            return False, f"Stake ${stake:.2f} exceeds 50% of bankroll ${risk_state.bankroll:.2f}"

        # Daily loss limit
        if risk_state.daily_loss_remaining <= 0:
            return False, "Daily loss limit reached"

        # Max single bet
        if stake > risk_state.max_single_bet:
            return False, f"Stake ${stake:.2f} exceeds max single bet ${risk_state.max_single_bet:.2f}"

        # Player exposure
        current_player_exposure = risk_state.exposure_by_player.get(player, 0)
        max_player = risk_state.bankroll * 0.05
        if current_player_exposure + stake > max_player:
            return False, f"Player {player} exposure would exceed 5% cap"

        # Open bet count
        session = self._session()
        try:
            pending_count = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value, status="pending")
                .count()
            )
            if pending_count >= self.max_open_bets:
                return False, f"Max open bets ({self.max_open_bets}) reached"
        finally:
            session.close()

        return True, "OK"

    def _compute_max_drawdown(self, settled_bets: list) -> float:
        """Compute historical maximum drawdown."""
        if not settled_bets:
            return 0.0

        running = self.initial_bankroll
        peak = running
        max_dd = 0.0

        for r in settled_bets:
            running += r.pnl
            peak = max(peak, running)
            dd = (peak - running) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return max_dd

    def pnl_curve(self) -> list[dict]:
        """Return the full P&L curve for charting."""
        session = self._session()
        try:
            settled = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost", "push"]))
                .order_by(BetLog.timestamp.asc())
                .all()
            )

            curve = [{"bet_num": 0, "bankroll": self.initial_bankroll, "pnl": 0.0}]
            running = self.initial_bankroll
            for i, r in enumerate(settled):
                running += r.pnl
                curve.append({
                    "bet_num": i + 1,
                    "bankroll": round(running, 2),
                    "pnl": round(r.pnl, 2),
                    "cumulative_pnl": round(running - self.initial_bankroll, 2),
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "player": r.player,
                    "bet_type": r.bet_type,
                })
            return curve
        finally:
            session.close()
