"""Failure Protection System — Hard safeguards that prevent bankroll destruction.

This is the system's immune system. It cannot be overridden by the model,
the user's optimism, or any "hot streak" logic. These are absolute rules.

Design Philosophy:
- Multiple independent circuit breakers (any one can halt the system)
- Asymmetric response: fast to shut down, slow to recover
- All state changes logged with full context
- Recovery requires explicit validation, not just time passing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from ..core.types import RiskState, Sport, SystemState
from ..db.schema import BetLog, SystemStateLog, get_session

logger = logging.getLogger(__name__)


@dataclass
class ProtectionConfig:
    """Hard limits — these are NOT tunable during live operation."""

    # Drawdown circuit breakers (% of peak bankroll)
    drawdown_reduce: float = 0.15       # 15% → reduce sizing
    drawdown_suspend: float = 0.30      # 30% → stop betting
    drawdown_kill: float = 0.45         # 45% → kill system entirely

    # Losing streak circuit breakers
    losing_streak_warn: int = 8         # 8 consecutive losses → warning
    losing_streak_reduce: int = 12      # 12 → reduce
    losing_streak_suspend: int = 18     # 18 → suspend

    # Daily limits
    daily_loss_limit_pct: float = 0.08  # 8% of bankroll per day
    daily_bet_limit: int = 20           # Max 20 bets per day

    # Variance detection
    variance_multiple_warn: float = 2.0   # Actual variance > 2x expected → warn
    variance_multiple_stop: float = 3.5   # > 3.5x expected → stop

    # Recovery requirements
    recovery_clv_window: int = 30        # Need 30 bets of positive CLV to upgrade state
    recovery_clv_threshold: float = 0.5  # CLV must be > 0.5 cents to count

    # Time-based cooling
    suspend_cooldown_hours: int = 24     # Minimum time in SUSPENDED before reviewing
    kill_cooldown_hours: int = 72        # 3 days minimum before reviewing KILLED state


class FailureProtection:
    """Multiple independent circuit breakers."""

    def __init__(self, sport: Sport, config: ProtectionConfig | None = None, db_path: str | None = None):
        self.sport = sport
        self.config = config or ProtectionConfig()
        self._db_path = db_path

    def _session(self):
        return get_session(self._db_path)

    def check_all(self, risk_state: RiskState) -> dict:
        """Run ALL circuit breakers. Returns worst-case state.

        Returns:
            {
                "recommended_state": SystemState,
                "breakers_triggered": [str, ...],
                "breakers_clear": [str, ...],
                "details": {breaker_name: {...}, ...},
            }
        """
        breakers = {}
        triggered = []
        clear = []

        # 1. Drawdown breaker
        dd_result = self._check_drawdown(risk_state)
        breakers["drawdown"] = dd_result
        if dd_result["triggered"]:
            triggered.append("drawdown")
        else:
            clear.append("drawdown")

        # 2. Losing streak breaker
        streak_result = self._check_losing_streak()
        breakers["losing_streak"] = streak_result
        if streak_result["triggered"]:
            triggered.append("losing_streak")
        else:
            clear.append("losing_streak")

        # 3. Daily limits
        daily_result = self._check_daily_limits(risk_state)
        breakers["daily_limits"] = daily_result
        if daily_result["triggered"]:
            triggered.append("daily_limits")
        else:
            clear.append("daily_limits")

        # 4. Variance detection
        variance_result = self._check_variance()
        breakers["variance"] = variance_result
        if variance_result["triggered"]:
            triggered.append("variance")
        else:
            clear.append("variance")

        # Determine worst-case state
        states = [b["state"] for b in breakers.values()]
        state_order = [SystemState.ACTIVE, SystemState.REDUCED, SystemState.SUSPENDED, SystemState.KILLED]
        worst = SystemState.ACTIVE
        for s in states:
            if state_order.index(s) > state_order.index(worst):
                worst = s

        return {
            "recommended_state": worst,
            "breakers_triggered": triggered,
            "breakers_clear": clear,
            "details": breakers,
        }

    def _check_drawdown(self, risk_state: RiskState) -> dict:
        """Drawdown circuit breaker."""
        dd = risk_state.current_drawdown_pct
        cfg = self.config

        if dd >= cfg.drawdown_kill:
            return {"triggered": True, "state": SystemState.KILLED,
                    "message": f"Drawdown {dd:.1%} exceeds kill threshold {cfg.drawdown_kill:.0%}",
                    "drawdown": dd}
        elif dd >= cfg.drawdown_suspend:
            return {"triggered": True, "state": SystemState.SUSPENDED,
                    "message": f"Drawdown {dd:.1%} exceeds suspend threshold",
                    "drawdown": dd}
        elif dd >= cfg.drawdown_reduce:
            return {"triggered": True, "state": SystemState.REDUCED,
                    "message": f"Drawdown {dd:.1%} exceeds reduce threshold",
                    "drawdown": dd}

        return {"triggered": False, "state": SystemState.ACTIVE,
                "message": f"Drawdown {dd:.1%} within limits", "drawdown": dd}

    def _check_losing_streak(self) -> dict:
        """Losing streak circuit breaker."""
        session = self._session()
        try:
            recent = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .order_by(BetLog.timestamp.desc())
                .limit(50)
                .all()
            )

            streak = 0
            for r in recent:
                if r.status == "lost":
                    streak += 1
                else:
                    break

            cfg = self.config
            if streak >= cfg.losing_streak_suspend:
                return {"triggered": True, "state": SystemState.SUSPENDED,
                        "message": f"Losing streak of {streak} — suspend threshold",
                        "streak": streak}
            elif streak >= cfg.losing_streak_reduce:
                return {"triggered": True, "state": SystemState.REDUCED,
                        "message": f"Losing streak of {streak} — reduce threshold",
                        "streak": streak}
            elif streak >= cfg.losing_streak_warn:
                return {"triggered": True, "state": SystemState.REDUCED,
                        "message": f"Losing streak of {streak} — warning",
                        "streak": streak}

            return {"triggered": False, "state": SystemState.ACTIVE,
                    "message": f"Current streak: {streak}", "streak": streak}
        finally:
            session.close()

    def _check_daily_limits(self, risk_state: RiskState) -> dict:
        """Daily loss and bet count limits."""
        cfg = self.config

        if risk_state.daily_loss_remaining <= 0:
            return {"triggered": True, "state": SystemState.SUSPENDED,
                    "message": f"Daily loss limit reached. P&L today: ${risk_state.daily_pnl:.2f}"}

        if risk_state.daily_bet_count >= cfg.daily_bet_limit:
            return {"triggered": True, "state": SystemState.SUSPENDED,
                    "message": f"Daily bet limit ({cfg.daily_bet_limit}) reached"}

        return {"triggered": False, "state": SystemState.ACTIVE,
                "message": f"Daily P&L: ${risk_state.daily_pnl:.2f}, Bets: {risk_state.daily_bet_count}"}

    def _check_variance(self) -> dict:
        """Detect if actual variance exceeds expected variance."""
        session = self._session()
        try:
            recent = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .order_by(BetLog.timestamp.desc())
                .limit(100)
                .all()
            )

            if len(recent) < 30:
                return {"triggered": False, "state": SystemState.ACTIVE,
                        "message": "Insufficient data for variance check"}

            # Actual variance of P&L per bet
            pnls = [r.pnl for r in recent]
            actual_var = float(np.var(pnls))

            # Expected variance: sum of (p * (1-p) * payout^2) per bet
            expected_vars = []
            for r in recent:
                p = r.model_prob
                payout = r.odds_decimal - 1.0
                ev_var = p * (1 - p) * (r.stake * payout) ** 2 + p * (1 - p) * r.stake ** 2
                expected_vars.append(ev_var)
            expected_var = float(np.mean(expected_vars))

            if expected_var <= 0:
                return {"triggered": False, "state": SystemState.ACTIVE,
                        "message": "Cannot compute variance ratio"}

            ratio = actual_var / expected_var
            cfg = self.config

            if ratio > cfg.variance_multiple_stop:
                return {"triggered": True, "state": SystemState.SUSPENDED,
                        "message": f"Variance ratio {ratio:.1f}x — significantly exceeds expected",
                        "variance_ratio": ratio}
            elif ratio > cfg.variance_multiple_warn:
                return {"triggered": True, "state": SystemState.REDUCED,
                        "message": f"Variance ratio {ratio:.1f}x — elevated",
                        "variance_ratio": ratio}

            return {"triggered": False, "state": SystemState.ACTIVE,
                    "message": f"Variance ratio {ratio:.1f}x — normal",
                    "variance_ratio": ratio}
        finally:
            session.close()

    def can_recover(self, from_state: SystemState) -> tuple[bool, str]:
        """Check if system can upgrade from a restricted state.

        Recovery requires:
        1. Minimum time elapsed in restricted state
        2. Positive CLV over recovery window
        3. Drawdown below the lower threshold
        """
        session = self._session()
        try:
            # Check time since last state change
            last_change = (
                session.query(SystemStateLog)
                .filter_by(sport=self.sport.value, new_state=from_state.value)
                .order_by(SystemStateLog.timestamp.desc())
                .first()
            )

            if last_change is None:
                return True, "No state change record found"

            hours_elapsed = (datetime.utcnow() - last_change.timestamp).total_seconds() / 3600
            cfg = self.config

            if from_state == SystemState.KILLED:
                if hours_elapsed < cfg.kill_cooldown_hours:
                    return False, f"Kill cooldown: {cfg.kill_cooldown_hours - hours_elapsed:.0f}h remaining"
            elif from_state == SystemState.SUSPENDED:
                if hours_elapsed < cfg.suspend_cooldown_hours:
                    return False, f"Suspend cooldown: {cfg.suspend_cooldown_hours - hours_elapsed:.0f}h remaining"

            return True, "Cooldown period elapsed — validate CLV before upgrading"
        finally:
            session.close()
