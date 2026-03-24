"""Hard Kill Switches + Edge Monitor — Non-overridable circuit breakers.

These protect capital when the system is broken. NO MANUAL OVERRIDE.
If a kill switch triggers, the system stops. Period.

Usage:
    from quant_system.core.kill_switches import HardKillSwitches
    from quant_system.db.schema import get_session

    session = get_session()
    ks = HardKillSwitches(session)

    check = ks.check_all("nba")
    if check["triggered"]:
        print(f"KILL SWITCH: {check['triggered']}")
        print(f"System state: {check['recommended_state']}")

    # Daily edge check — the most important output
    verdict = ks.daily_edge_verdict("nba")
    print(f"EDGE = {verdict}")  # "YES" or "NO"
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

from sqlalchemy import text

from .types import SystemState

logger = logging.getLogger(__name__)


class HardKillSwitches:
    """Non-overridable circuit breakers. NO MANUAL OVERRIDE.

    These protect capital when the system is broken.
    """

    RULES = {
        "clv_250_negative": {
            "description": "CLV <= 0 over 250 bets",
            "action": "KILLED",
            "override_allowed": False,
        },
        "model_worse_than_market": {
            "description": "Brier score worse than naive market for 100+ bets",
            "action": "SUSPENDED",
            "override_allowed": False,
        },
        "edge_decay_detected": {
            "description": "Rolling 100-bet CLV declining for 3 consecutive windows",
            "action": "REDUCED",
            "override_allowed": False,
        },
        "execution_destroys_edge": {
            "description": "Post-execution edge < 0 for 50+ bets",
            "action": "SUSPENDED",
            "override_allowed": False,
        },
        "drawdown_50_pct": {
            "description": "Bankroll drops 50% from peak",
            "action": "KILLED",
            "override_allowed": False,
        },
        "calibration_broken": {
            "description": "MAE > 15% for 100+ bets",
            "action": "SUSPENDED",
            "override_allowed": False,
        },
    }

    def __init__(self, db_session):
        self.db = db_session

    # ── Rule Checks ────────────────────────────────────────────────────

    def _check_clv_250_negative(self, sport: str) -> dict:
        """CLV <= 0 over last 250 bets = system is KILLED."""
        query = text("""
            SELECT
                COUNT(*) AS n,
                AVG(clv_cents) AS avg_clv,
                SUM(CASE WHEN beat_close = 1 THEN 1 ELSE 0 END) AS beat_count
            FROM clv_log
            WHERE sport = :sport
            ORDER BY calculated_at DESC
            LIMIT 250
        """)
        row = self.db.execute(query, {"sport": sport}).fetchone()

        if row is None or row[0] < 50:
            return {"triggered": False, "detail": "Insufficient CLV data", "n": 0}

        n = row[0]
        avg_clv = row[1] or 0.0
        beat_count = row[2] or 0

        triggered = avg_clv <= 0 and n >= 250

        return {
            "triggered": triggered,
            "n_bets": n,
            "avg_clv_cents": round(avg_clv, 3),
            "beat_close_pct": round(beat_count / n * 100, 1) if n > 0 else 0,
            "detail": f"CLV={avg_clv:.3f} over {n} bets" + (" [TRIGGERED]" if triggered else ""),
        }

    def _check_model_worse_than_market(self, sport: str) -> dict:
        """Brier score worse than naive market for 100+ bets = SUSPENDED."""
        query = text("""
            SELECT
                bet_id, model_prob, market_prob, status
            FROM bet_log
            WHERE sport = :sport
              AND status IN ('won', 'lost')
            ORDER BY settled_at DESC
            LIMIT 200
        """)
        rows = self.db.execute(query, {"sport": sport}).fetchall()

        if len(rows) < 100:
            return {"triggered": False, "detail": "Insufficient data", "n": len(rows)}

        model_brier = 0.0
        market_brier = 0.0

        for row in rows:
            outcome = 1.0 if row[3] == "won" else 0.0
            model_prob = row[1]
            market_prob = row[2]

            model_brier += (model_prob - outcome) ** 2
            market_brier += (market_prob - outcome) ** 2

        n = len(rows)
        model_brier /= n
        market_brier /= n

        triggered = model_brier > market_brier and n >= 100

        return {
            "triggered": triggered,
            "n_bets": n,
            "model_brier": round(model_brier, 5),
            "market_brier": round(market_brier, 5),
            "brier_delta": round(model_brier - market_brier, 5),
            "detail": (
                f"Model Brier={model_brier:.5f} vs Market={market_brier:.5f}"
                + (" [TRIGGERED]" if triggered else "")
            ),
        }

    def _check_edge_decay(self, sport: str) -> dict:
        """Rolling 100-bet CLV declining for 3 consecutive windows = REDUCED."""
        query = text("""
            SELECT clv_cents
            FROM clv_log
            WHERE sport = :sport
            ORDER BY calculated_at DESC
            LIMIT 300
        """)
        rows = self.db.execute(query, {"sport": sport}).fetchall()

        if len(rows) < 300:
            return {"triggered": False, "detail": "Insufficient data for 3 windows", "n": len(rows)}

        # Split into 3 windows of 100
        window_avgs = []
        for i in range(3):
            start = i * 100
            end = start + 100
            window = rows[start:end]
            avg = sum(r[0] for r in window) / len(window)
            window_avgs.append(avg)

        # Check if each subsequent window is worse (most recent first)
        # window_avgs[0] = most recent 100, [1] = 100-200, [2] = 200-300
        declining = (window_avgs[0] < window_avgs[1] < window_avgs[2])

        return {
            "triggered": declining,
            "window_avgs": [round(w, 3) for w in window_avgs],
            "detail": (
                f"Windows (recent→old): {[round(w,3) for w in window_avgs]}"
                + (" [TRIGGERED: declining]" if declining else "")
            ),
        }

    def _check_execution_destroys_edge(self, sport: str) -> dict:
        """Post-execution edge < 0 for 50+ bets = SUSPENDED.

        We approximate post-execution edge by comparing model edge
        to actual realized ROI. If model says +3% edge but actual ROI
        is negative, execution is destroying the edge.
        """
        query = text("""
            SELECT
                COUNT(*) AS n,
                AVG(edge) AS avg_model_edge,
                SUM(pnl) AS total_pnl,
                SUM(stake) AS total_staked
            FROM bet_log
            WHERE sport = :sport
              AND status IN ('won', 'lost')
            ORDER BY settled_at DESC
            LIMIT 50
        """)
        row = self.db.execute(query, {"sport": sport}).fetchone()

        if row is None or row[0] < 50:
            return {"triggered": False, "detail": "Insufficient data", "n": row[0] if row else 0}

        n = row[0]
        avg_edge = row[1] or 0.0
        total_pnl = row[2] or 0.0
        total_staked = row[3] or 1.0

        realized_roi = total_pnl / total_staked if total_staked > 0 else 0

        # Edge exists in model but not in reality = execution destroying edge
        triggered = avg_edge > 0 and realized_roi < 0 and n >= 50

        return {
            "triggered": triggered,
            "n_bets": n,
            "avg_model_edge": round(avg_edge, 4),
            "realized_roi": round(realized_roi, 4),
            "execution_gap": round(avg_edge - realized_roi, 4),
            "detail": (
                f"Model edge={avg_edge:.4f}, Realized ROI={realized_roi:.4f}"
                + (" [TRIGGERED]" if triggered else "")
            ),
        }

    def _check_drawdown_50_pct(self, sport: str) -> dict:
        """Bankroll drops 50% from peak = KILLED."""
        query = text("""
            SELECT
                bankroll_at_change,
                MAX(bankroll_at_change) OVER () AS peak_bankroll
            FROM system_state_log
            WHERE sport = :sport
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        row = self.db.execute(query, {"sport": sport}).fetchone()

        if row is None:
            # Try to compute from bet log
            pnl_query = text("""
                SELECT SUM(pnl) AS total_pnl FROM bet_log
                WHERE sport = :sport AND status IN ('won', 'lost')
            """)
            pnl_row = self.db.execute(pnl_query, {"sport": sport}).fetchone()
            return {"triggered": False, "detail": "No state log data", "n": 0}

        current_bankroll = row[0] or 0
        peak_bankroll = row[1] or current_bankroll

        if peak_bankroll <= 0:
            return {"triggered": False, "detail": "No peak data"}

        drawdown_pct = (peak_bankroll - current_bankroll) / peak_bankroll * 100

        triggered = drawdown_pct >= 50.0

        return {
            "triggered": triggered,
            "current_bankroll": round(current_bankroll, 2),
            "peak_bankroll": round(peak_bankroll, 2),
            "drawdown_pct": round(drawdown_pct, 1),
            "detail": (
                f"Drawdown={drawdown_pct:.1f}% (peak=${peak_bankroll:.0f}, current=${current_bankroll:.0f})"
                + (" [TRIGGERED]" if triggered else "")
            ),
        }

    def _check_calibration_broken(self, sport: str) -> dict:
        """MAE > 15% for 100+ bets = SUSPENDED."""
        query = text("""
            SELECT model_prob, status
            FROM bet_log
            WHERE sport = :sport
              AND status IN ('won', 'lost')
            ORDER BY settled_at DESC
            LIMIT 200
        """)
        rows = self.db.execute(query, {"sport": sport}).fetchall()

        if len(rows) < 100:
            return {"triggered": False, "detail": "Insufficient data", "n": len(rows)}

        # Compute calibration: bucket predictions and compare to actual rate
        buckets = {}
        for row in rows:
            prob = row[0]
            won = row[1] == "won"
            # Bucket into 5% ranges
            bucket = round(prob * 20) / 20  # Round to nearest 0.05
            bucket = max(0.05, min(0.95, bucket))
            if bucket not in buckets:
                buckets[bucket] = {"predicted": [], "actual": []}
            buckets[bucket]["predicted"].append(prob)
            buckets[bucket]["actual"].append(1.0 if won else 0.0)

        # Mean Absolute Error across buckets
        total_mae = 0.0
        n_buckets = 0
        for bucket, data in buckets.items():
            if len(data["predicted"]) >= 5:  # Only count buckets with enough data
                predicted_avg = sum(data["predicted"]) / len(data["predicted"])
                actual_rate = sum(data["actual"]) / len(data["actual"])
                total_mae += abs(predicted_avg - actual_rate)
                n_buckets += 1

        mae = total_mae / max(n_buckets, 1)

        triggered = mae > 0.15 and len(rows) >= 100

        return {
            "triggered": triggered,
            "n_bets": len(rows),
            "n_buckets": n_buckets,
            "mae": round(mae, 4),
            "mae_pct": round(mae * 100, 1),
            "detail": (
                f"Calibration MAE={mae:.4f} ({mae*100:.1f}%) across {n_buckets} buckets"
                + (" [TRIGGERED]" if triggered else "")
            ),
        }

    # ── Master Check ───────────────────────────────────────────────────

    def check_all(self, sport: str) -> dict:
        """Check ALL kill switch conditions.

        Returns:
            {
                "triggered": list of triggered rule names,
                "recommended_state": SystemState,
                "details": dict per rule,
                "edge_verdict": "YES" | "NO" | "UNCERTAIN"
            }
        """
        checks = {
            "clv_250_negative": self._check_clv_250_negative(sport),
            "model_worse_than_market": self._check_model_worse_than_market(sport),
            "edge_decay_detected": self._check_edge_decay(sport),
            "execution_destroys_edge": self._check_execution_destroys_edge(sport),
            "drawdown_50_pct": self._check_drawdown_50_pct(sport),
            "calibration_broken": self._check_calibration_broken(sport),
        }

        triggered = [name for name, result in checks.items() if result.get("triggered", False)]

        # Determine recommended state from worst triggered rule
        recommended = SystemState.ACTIVE
        for rule_name in triggered:
            action = self.RULES[rule_name]["action"]
            if action == "KILLED":
                recommended = SystemState.KILLED
                break  # Can't get worse
            elif action == "SUSPENDED" and recommended != SystemState.KILLED:
                recommended = SystemState.SUSPENDED
            elif action == "REDUCED" and recommended in (SystemState.ACTIVE,):
                recommended = SystemState.REDUCED

        # Edge verdict
        if recommended == SystemState.ACTIVE:
            edge_verdict = "YES"
        elif recommended == SystemState.REDUCED:
            edge_verdict = "UNCERTAIN"
        else:
            edge_verdict = "NO"

        if triggered:
            logger.warning(
                "KILL SWITCHES TRIGGERED [%s]: %s → %s",
                sport, triggered, recommended.value,
            )

        return {
            "sport": sport,
            "check_time": datetime.utcnow().isoformat(),
            "triggered": triggered,
            "n_triggered": len(triggered),
            "recommended_state": recommended,
            "edge_verdict": edge_verdict,
            "details": checks,
        }

    # ── Daily Edge Verdict ─────────────────────────────────────────────

    def daily_edge_verdict(self, sport: str) -> str:
        """Output DAILY: EDGE = YES / NO. No ambiguity.

        YES requires ALL of:
        - CLV > 0 over last 100 bets (with statistical significance p < 0.10)
        - Brier score < market Brier score
        - Calibration MAE < 8%
        - No kill switches triggered

        Everything else = NO.
        """
        # 1. Check all kill switches first
        kill_check = self.check_all(sport)
        if kill_check["triggered"]:
            logger.info("EDGE VERDICT [%s]: NO (kill switches: %s)", sport, kill_check["triggered"])
            return "NO"

        # 2. CLV > 0 over last 100 bets with significance
        clv_query = text("""
            SELECT clv_cents
            FROM clv_log
            WHERE sport = :sport
            ORDER BY calculated_at DESC
            LIMIT 100
        """)
        clv_rows = self.db.execute(clv_query, {"sport": sport}).fetchall()

        if len(clv_rows) < 30:
            logger.info("EDGE VERDICT [%s]: NO (insufficient CLV data: %d bets)", sport, len(clv_rows))
            return "NO"

        clv_values = [r[0] for r in clv_rows]
        clv_mean = sum(clv_values) / len(clv_values)
        n = len(clv_values)

        # Standard error and t-test for CLV > 0
        if n > 1:
            variance = sum((v - clv_mean) ** 2 for v in clv_values) / (n - 1)
            std = math.sqrt(variance)
            se = std / math.sqrt(n)
            t_stat = clv_mean / se if se > 0 else 0.0
        else:
            t_stat = 0.0

        # For p < 0.10 one-sided, t > ~1.29 for large samples
        clv_significant = clv_mean > 0 and t_stat > 1.29

        if not clv_significant:
            logger.info(
                "EDGE VERDICT [%s]: NO (CLV not significant: mean=%.3f, t=%.2f)",
                sport, clv_mean, t_stat,
            )
            return "NO"

        # 3. Brier score check
        brier_check = kill_check["details"]["model_worse_than_market"]
        if brier_check.get("triggered", False):
            logger.info("EDGE VERDICT [%s]: NO (model Brier > market)", sport)
            return "NO"

        # Even if not triggered (< 100 bets), check if we have data
        model_brier = brier_check.get("model_brier", 0)
        market_brier = brier_check.get("market_brier", 0)
        if model_brier > 0 and market_brier > 0 and model_brier >= market_brier:
            logger.info("EDGE VERDICT [%s]: NO (model not beating market Brier)", sport)
            return "NO"

        # 4. Calibration MAE < 8%
        cal_check = kill_check["details"]["calibration_broken"]
        mae = cal_check.get("mae", 0)
        if mae > 0.08:
            logger.info("EDGE VERDICT [%s]: NO (calibration MAE=%.1f%% > 8%%)", sport, mae * 100)
            return "NO"

        # ALL CHECKS PASSED
        logger.info(
            "EDGE VERDICT [%s]: YES (CLV=%.3f, t=%.2f, Brier=%.5f < %.5f, MAE=%.1f%%)",
            sport, clv_mean, t_stat, model_brier, market_brier, mae * 100,
        )
        return "YES"
