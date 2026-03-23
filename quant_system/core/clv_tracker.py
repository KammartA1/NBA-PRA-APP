"""Closing Line Value (CLV) Tracker — THE most important metric.

CLV measures whether you consistently get better prices than the closing
line. A bettor with positive CLV is beating the market's final assessment.
This is the single strongest predictor of long-term profitability.

If CLV is negative over a meaningful sample, the model does NOT have edge,
regardless of short-term P&L."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from ..db.schema import BetLog, CLVLog, LineSnapshot, get_session
from .types import CLVResult, Sport

logger = logging.getLogger(__name__)


class CLVTracker:
    """Tracks closing line value for all bets."""

    def __init__(self, sport: Sport, db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path

    def _session(self):
        return get_session(self._db_path)

    # ── Line Snapshots ────────────────────────────────────────────────

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
        """Record a line snapshot for tracking movement."""
        session = self._session()
        try:
            over_prob = None
            under_prob = None
            if odds_american is not None:
                # For PrizePicks, odds are effectively -110/-110 (implied ~52.4%)
                # For books, convert from American
                if odds_american == 0:
                    over_prob = 0.5
                elif odds_american > 0:
                    over_prob = 100.0 / (odds_american + 100.0)
                else:
                    over_prob = abs(odds_american) / (abs(odds_american) + 100.0)
                under_prob = 1.0 - over_prob

            snap = LineSnapshot(
                sport=self.sport.value,
                player=player,
                stat_type=stat_type,
                source=source,
                line=line,
                odds_american=odds_american,
                over_prob_implied=over_prob,
                under_prob_implied=under_prob,
                captured_at=datetime.utcnow(),
                is_opening=is_opening,
                is_closing=is_closing,
            )
            session.add(snap)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to record line snapshot")
        finally:
            session.close()

    # ── CLV Calculation ───────────────────────────────────────────────

    def calculate_clv(self, bet_id: str, closing_line: float, closing_odds: Optional[int] = None) -> CLVResult:
        """Calculate CLV for a settled bet given the closing line.

        CLV Formula:
            For over bets: CLV > 0 if closing_line > bet_line (line moved up = market agreed)
            For under bets: CLV > 0 if closing_line < bet_line (line moved down = market agreed)

        We also calculate CLV in probability space:
            clv_raw = closing_implied_prob(opposite_side) - bet_implied_prob(opposite_side)
        """
        session = self._session()
        try:
            bet = session.query(BetLog).filter_by(bet_id=bet_id).first()
            if bet is None:
                raise ValueError(f"Bet {bet_id} not found")

            opening_line = bet.line  # Our bet line IS the opening from our perspective
            bet_line = bet.line

            # Find the actual opening line from snapshots
            opening_snap = (
                session.query(LineSnapshot)
                .filter_by(
                    sport=self.sport.value,
                    player=bet.player,
                    stat_type=bet.stat_type,
                    is_opening=True,
                )
                .order_by(LineSnapshot.captured_at.desc())
                .first()
            )
            if opening_snap:
                opening_line = opening_snap.line

            line_movement = closing_line - opening_line

            # CLV in line space (positive = we got a better number)
            if bet.direction == "over":
                # For over bets, we want the line to move UP after we bet
                # (market agreeing the number should be higher)
                clv_raw = closing_line - bet_line
                beat_close = closing_line > bet_line
            else:
                # For under bets, we want the line to move DOWN
                clv_raw = bet_line - closing_line
                beat_close = closing_line < bet_line

            # CLV in cents: how many cents per dollar did we gain?
            # Approximate: each 0.5 point of line movement ≈ 2-3% edge shift
            # More precise: use probability space
            if bet.model_std > 0:
                z_bet = (bet_line - bet.model_projection) / bet.model_std
                z_close = (closing_line - bet.model_projection) / bet.model_std
                prob_over_at_bet = 1.0 - sp_stats.norm.cdf(z_bet)
                prob_over_at_close = 1.0 - sp_stats.norm.cdf(z_close)

                if bet.direction == "over":
                    clv_cents = round((prob_over_at_bet - prob_over_at_close) * 100, 2)
                else:
                    clv_cents = round((sp_stats.norm.cdf(z_close) - sp_stats.norm.cdf(z_bet)) * 100, 2)
            else:
                clv_cents = clv_raw * 2.0  # rough approximation

            result = CLVResult(
                bet_id=bet_id,
                opening_line=opening_line,
                bet_line=bet_line,
                closing_line=closing_line,
                line_movement=round(line_movement, 3),
                clv_raw=round(clv_raw, 3),
                clv_cents=round(clv_cents, 2),
                beat_close=beat_close,
            )

            # Persist
            clv_row = CLVLog(
                bet_id=bet_id,
                sport=self.sport.value,
                opening_line=opening_line,
                bet_line=bet_line,
                closing_line=closing_line,
                line_movement=result.line_movement,
                clv_raw=result.clv_raw,
                clv_cents=result.clv_cents,
                beat_close=beat_close,
            )
            session.add(clv_row)
            session.commit()

            logger.info("CLV calculated: %s | raw=%.3f | cents=%.2f | beat_close=%s",
                        bet_id, result.clv_raw, result.clv_cents, beat_close)
            return result

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Rolling CLV Metrics ───────────────────────────────────────────

    def rolling_clv(self, window: int = 100) -> dict:
        """Calculate rolling CLV over last N bets.

        Returns:
            {
                "window": int,
                "n_bets": int,
                "avg_clv_raw": float,
                "avg_clv_cents": float,
                "beat_close_pct": float,     # % of bets that beat closing line
                "clv_positive": bool,        # Is overall CLV positive?
                "clv_t_stat": float,         # T-statistic (significance)
                "clv_p_value": float,        # P-value for CLV > 0
                "trend": str,                # "improving", "stable", "declining"
            }
        """
        session = self._session()
        try:
            rows = (
                session.query(CLVLog)
                .filter_by(sport=self.sport.value)
                .order_by(CLVLog.calculated_at.desc())
                .limit(window)
                .all()
            )

            if len(rows) < 10:
                return {
                    "window": window,
                    "n_bets": len(rows),
                    "avg_clv_raw": 0.0,
                    "avg_clv_cents": 0.0,
                    "beat_close_pct": 0.0,
                    "clv_positive": False,
                    "clv_t_stat": 0.0,
                    "clv_p_value": 1.0,
                    "trend": "insufficient_data",
                }

            clv_values = [r.clv_cents for r in rows]
            beat_close_flags = [r.beat_close for r in rows]

            avg_clv_raw = float(np.mean([r.clv_raw for r in rows]))
            avg_clv_cents = float(np.mean(clv_values))
            beat_close_pct = float(np.mean(beat_close_flags))

            # T-test: is CLV significantly > 0?
            if np.std(clv_values) > 0:
                t_stat, p_value = sp_stats.ttest_1samp(clv_values, 0.0)
                # One-sided: we care about CLV > 0
                p_value_one_sided = p_value / 2 if t_stat > 0 else 1.0 - p_value / 2
            else:
                t_stat, p_value_one_sided = 0.0, 1.0

            # Trend: compare first half vs second half
            mid = len(clv_values) // 2
            first_half_avg = float(np.mean(clv_values[:mid]))
            second_half_avg = float(np.mean(clv_values[mid:]))
            diff = second_half_avg - first_half_avg

            if diff > 0.5:
                trend = "improving"
            elif diff < -0.5:
                trend = "declining"
            else:
                trend = "stable"

            return {
                "window": window,
                "n_bets": len(rows),
                "avg_clv_raw": round(avg_clv_raw, 4),
                "avg_clv_cents": round(avg_clv_cents, 2),
                "beat_close_pct": round(beat_close_pct, 4),
                "clv_positive": avg_clv_cents > 0,
                "clv_t_stat": round(float(t_stat), 3),
                "clv_p_value": round(float(p_value_one_sided), 4),
                "trend": trend,
            }
        finally:
            session.close()

    def clv_summary(self) -> dict:
        """Full CLV summary across multiple windows."""
        return {
            "clv_50": self.rolling_clv(50),
            "clv_100": self.rolling_clv(100),
            "clv_250": self.rolling_clv(250),
            "clv_500": self.rolling_clv(500),
        }

    def clv_by_bet_type(self) -> dict:
        """CLV broken down by bet type."""
        session = self._session()
        try:
            rows = (
                session.query(CLVLog, BetLog)
                .join(BetLog, CLVLog.bet_id == BetLog.bet_id)
                .filter(CLVLog.sport == self.sport.value)
                .order_by(CLVLog.calculated_at.desc())
                .limit(500)
                .all()
            )

            by_type: dict[str, list[float]] = {}
            for clv_row, bet_row in rows:
                bt = bet_row.bet_type
                by_type.setdefault(bt, []).append(clv_row.clv_cents)

            result = {}
            for bt, vals in by_type.items():
                result[bt] = {
                    "n": len(vals),
                    "avg_clv_cents": round(float(np.mean(vals)), 2),
                    "beat_close_pct": round(sum(1 for v in vals if v > 0) / len(vals), 3),
                }
            return result
        finally:
            session.close()
