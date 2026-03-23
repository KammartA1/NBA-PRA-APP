"""
services/clv_system/clv_calculator.py
=======================================
CLV computation engine — cents, probability, and segmented analysis.

This is the core calculation module.  It computes:
  - CLV in cents (line-space)
  - CLV in probability (implied-prob space)
  - Beat-close rate
  - CLV broken down by segment (sport, market, book, day, time)
  - CLV trend over time

Works with both the quant_system.db tables (BetLog, CLVLog) and the
CLV-system-specific tables (CLVClosingLine, CLVBetSnapshot).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from quant_system.db.schema import get_engine, get_session, BetLog, CLVLog
from services.clv_system.models import CLVClosingLine, CLVBetSnapshot, Base

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability (no-vig)."""
    if odds == 0:
        return 0.5
    elif odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


class CLVCalculator:
    """Full CLV computation engine.

    Computes CLV in multiple dimensions:
      - Cents (line movement in odds space)
      - Probability (implied probability difference)
      - Beat-close rate
      - Segmented analysis
    """

    def __init__(self, sport: str = "NBA", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    # ── Core CLV calculations ────────────────────────────────────────

    def compute_clv_cents(self, bet_line: float, closing_line: float,
                          direction: str = "over") -> float:
        """CLV in cents — how many cents of line value we captured.

        For over bets: CLV = closing_line - bet_line (positive = good)
        For under bets: CLV = bet_line - closing_line (positive = good)

        Example: bet at 25.5, closed at 26.5 → CLV = 1.0 point (over bet)
        """
        if direction.lower() == "over":
            return round(closing_line - bet_line, 3)
        else:
            return round(bet_line - closing_line, 3)

    def compute_clv_probability(
        self,
        bet_implied_prob: float,
        closing_implied_prob: float,
    ) -> float:
        """CLV in probability terms.

        CLV = bet_implied_prob - closing_implied_prob
        Positive means we got a better price than closing.

        Example: bet at 52% implied, closed at 54% → 2% CLV
        """
        return round(bet_implied_prob - closing_implied_prob, 6)

    def compute_clv_full(
        self,
        bet_line: float,
        closing_line: float,
        direction: str,
        model_projection: float,
        model_std: float,
        bet_odds_american: Optional[int] = None,
        closing_odds_american: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full CLV computation with both cents and probability methods.

        Uses the normal distribution model to convert line movements into
        probability-space CLV, which is more accurate than raw line diff.

        Args:
            bet_line: Line at time of bet.
            closing_line: Closing line.
            direction: "over" or "under".
            model_projection: Model's projected value.
            model_std: Model's projected standard deviation.
            bet_odds_american: American odds at bet time (optional).
            closing_odds_american: Closing American odds (optional).

        Returns:
            Dict with clv_cents, clv_prob, beat_close, etc.
        """
        # Line-space CLV
        clv_cents = self.compute_clv_cents(bet_line, closing_line, direction)
        beat_close = clv_cents > 0

        # Probability-space CLV using normal distribution
        clv_prob = 0.0
        bet_prob = None
        close_prob = None

        if model_std > 0:
            z_bet = (bet_line - model_projection) / model_std
            z_close = (closing_line - model_projection) / model_std

            if direction.lower() == "over":
                bet_prob = 1.0 - sp_stats.norm.cdf(z_bet)
                close_prob = 1.0 - sp_stats.norm.cdf(z_close)
                # Higher prob at bet time = better price for over
                clv_prob = bet_prob - close_prob
            else:
                bet_prob = sp_stats.norm.cdf(z_bet)
                close_prob = sp_stats.norm.cdf(z_close)
                clv_prob = bet_prob - close_prob

        # Odds-based CLV (if available)
        odds_clv = None
        if bet_odds_american and closing_odds_american:
            bet_impl = _american_to_implied_prob(bet_odds_american)
            close_impl = _american_to_implied_prob(closing_odds_american)
            odds_clv = self.compute_clv_probability(bet_impl, close_impl)

        return {
            "clv_cents": round(clv_cents, 3),
            "clv_prob": round(clv_prob, 6),
            "clv_odds": round(odds_clv, 6) if odds_clv is not None else None,
            "beat_close": beat_close,
            "bet_line": bet_line,
            "closing_line": closing_line,
            "direction": direction,
            "bet_implied_prob": round(bet_prob, 6) if bet_prob is not None else None,
            "close_implied_prob": round(close_prob, 6) if close_prob is not None else None,
            "line_movement": round(closing_line - bet_line, 3),
        }

    # ── Aggregate CLV metrics ────────────────────────────────────────

    def compute_beat_close_rate(self, window: int = 100) -> float:
        """Percentage of bets where we got a better price than closing.

        Args:
            window: Number of most recent bets to analyze.

        Returns:
            Float between 0 and 1 (e.g., 0.55 = 55% beat-close rate).
        """
        session = self._session()
        try:
            rows = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport.lower())
                .order_by(CLVLog.calculated_at.desc())
                .limit(window)
                .all()
            )
            if not rows:
                return 0.0
            beats = sum(1 for r in rows if r.beat_close)
            return round(beats / len(rows), 4)
        finally:
            session.close()

    def compute_rolling_clv(self, window: int = 100) -> Dict[str, Any]:
        """Compute rolling CLV metrics over the last N bets.

        Returns:
            Dict with avg_clv_cents, avg_clv_prob, beat_close_pct,
            clv_positive, t_stat, p_value, trend.
        """
        session = self._session()
        try:
            rows = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport.lower())
                .order_by(CLVLog.calculated_at.desc())
                .limit(window)
                .all()
            )

            if len(rows) < 10:
                return {
                    "window": window,
                    "n_bets": len(rows),
                    "avg_clv_cents": 0.0,
                    "avg_clv_raw": 0.0,
                    "beat_close_pct": 0.0,
                    "clv_positive": False,
                    "t_stat": 0.0,
                    "p_value": 1.0,
                    "trend": "insufficient_data",
                }

            clv_cents = [r.clv_cents for r in rows]
            clv_raw = [r.clv_raw for r in rows]
            beat_flags = [r.beat_close for r in rows]

            avg_cents = float(np.mean(clv_cents))
            avg_raw = float(np.mean(clv_raw))
            beat_pct = float(np.mean(beat_flags))

            # T-test: is CLV significantly > 0?
            t_stat = 0.0
            p_value = 1.0
            if np.std(clv_cents) > 0:
                t, p = sp_stats.ttest_1samp(clv_cents, 0.0)
                t_stat = float(t)
                p_value = float(p / 2 if t > 0 else 1.0 - p / 2)

            # Trend: compare first half vs second half
            mid = len(clv_cents) // 2
            first_half = float(np.mean(clv_cents[:mid]))
            second_half = float(np.mean(clv_cents[mid:]))
            diff = second_half - first_half

            if diff > 0.5:
                trend = "improving"
            elif diff < -0.5:
                trend = "declining"
            else:
                trend = "stable"

            return {
                "window": window,
                "n_bets": len(rows),
                "avg_clv_cents": round(avg_cents, 3),
                "avg_clv_raw": round(avg_raw, 4),
                "beat_close_pct": round(beat_pct, 4),
                "clv_positive": avg_cents > 0,
                "t_stat": round(t_stat, 3),
                "p_value": round(p_value, 4),
                "trend": trend,
            }
        finally:
            session.close()

    def compute_clv_summary(self) -> Dict[str, Any]:
        """Full CLV summary across multiple windows."""
        return {
            "clv_50": self.compute_rolling_clv(50),
            "clv_100": self.compute_rolling_clv(100),
            "clv_250": self.compute_rolling_clv(250),
            "clv_500": self.compute_rolling_clv(500),
        }

    # ── Segmented CLV ────────────────────────────────────────────────

    def compute_clv_by_segment(
        self, segment_key: str = "stat_type", window: int = 500,
    ) -> Dict[str, Dict[str, Any]]:
        """CLV broken down by segment.

        Supported segment_key values:
          - "stat_type" (market type: Points, Rebounds, etc.)
          - "bet_type" (over/under)
          - "direction"
          - "day_of_week" (Mon-Sun)
          - "hour_of_day" (0-23)

        Returns:
            Dict mapping segment values to CLV metrics.
        """
        session = self._session()
        try:
            rows = (
                session.query(CLVLog, BetLog)
                .join(BetLog, CLVLog.bet_id == BetLog.bet_id)
                .filter(CLVLog.sport == self.sport.lower())
                .order_by(CLVLog.calculated_at.desc())
                .limit(window)
                .all()
            )

            if not rows:
                return {}

            # Group by segment
            groups: Dict[str, List[Dict[str, float]]] = defaultdict(list)
            for clv_row, bet_row in rows:
                if segment_key == "stat_type":
                    key_val = bet_row.stat_type
                elif segment_key == "bet_type":
                    key_val = bet_row.bet_type
                elif segment_key == "direction":
                    key_val = bet_row.direction
                elif segment_key == "day_of_week":
                    key_val = bet_row.timestamp.strftime("%A") if bet_row.timestamp else "Unknown"
                elif segment_key == "hour_of_day":
                    key_val = str(bet_row.timestamp.hour) if bet_row.timestamp else "Unknown"
                else:
                    key_val = getattr(bet_row, segment_key, "unknown")

                groups[key_val].append({
                    "clv_cents": clv_row.clv_cents,
                    "clv_raw": clv_row.clv_raw,
                    "beat_close": clv_row.beat_close,
                })

            # Compute per-segment metrics
            result = {}
            for seg_val, entries in groups.items():
                cents = [e["clv_cents"] for e in entries]
                raws = [e["clv_raw"] for e in entries]
                beats = [e["beat_close"] for e in entries]

                result[seg_val] = {
                    "n_bets": len(entries),
                    "avg_clv_cents": round(float(np.mean(cents)), 3),
                    "avg_clv_raw": round(float(np.mean(raws)), 4),
                    "beat_close_pct": round(float(np.mean(beats)), 4),
                    "clv_positive": float(np.mean(cents)) > 0,
                    "std_clv_cents": round(float(np.std(cents)), 3),
                }

            return result

        finally:
            session.close()

    # ── CLV trend over time ──────────────────────────────────────────

    def compute_clv_trend(
        self, window: int = 500, bucket_size: int = 25,
    ) -> List[Dict[str, Any]]:
        """Compute CLV trend over time in rolling buckets.

        Divides the last `window` bets into groups of `bucket_size` and
        computes CLV for each bucket.  Useful for plotting CLV over time.

        Returns:
            List of dicts with bucket metrics, ordered chronologically.
        """
        session = self._session()
        try:
            rows = (
                session.query(CLVLog)
                .filter(CLVLog.sport == self.sport.lower())
                .order_by(CLVLog.calculated_at.asc())
                .limit(window)
                .all()
            )

            if len(rows) < bucket_size:
                return []

            # Split into buckets
            buckets = []
            for i in range(0, len(rows) - bucket_size + 1, bucket_size):
                chunk = rows[i:i + bucket_size]
                cents = [r.clv_cents for r in chunk]
                beats = [r.beat_close for r in chunk]

                buckets.append({
                    "bucket_start": chunk[0].calculated_at.isoformat() if chunk[0].calculated_at else None,
                    "bucket_end": chunk[-1].calculated_at.isoformat() if chunk[-1].calculated_at else None,
                    "n_bets": len(chunk),
                    "avg_clv_cents": round(float(np.mean(cents)), 3),
                    "beat_close_pct": round(float(np.mean(beats)), 4),
                    "cumulative_clv": round(float(np.sum(cents)), 3),
                })

            return buckets

        finally:
            session.close()

    # ── Calculate and store CLV for a bet ────────────────────────────

    def calculate_and_store(
        self,
        bet_id: str,
        closing_line: float,
        closing_odds_american: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate CLV for a bet and store the result in clv_log.

        Looks up the bet from bet_log, computes CLV, and persists.

        Args:
            bet_id: Unique bet identifier.
            closing_line: The closing line value.
            closing_odds_american: Closing odds (optional).

        Returns:
            CLV result dict.
        """
        session = self._session()
        try:
            bet = session.query(BetLog).filter_by(bet_id=bet_id).first()
            if bet is None:
                raise ValueError(f"Bet {bet_id} not found")

            # Get opening line from CLV system
            opening_line = bet.line
            from services.clv_system.models import CLVLineMovement
            opening_snap = (
                session.query(CLVLineMovement)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.player == bet.player,
                    CLVLineMovement.market_type == bet.stat_type,
                    CLVLineMovement.is_opening == True,  # noqa: E712
                )
                .order_by(CLVLineMovement.timestamp.desc())
                .first()
            )
            if opening_snap:
                opening_line = opening_snap.line

            # Compute full CLV
            clv = self.compute_clv_full(
                bet_line=bet.line,
                closing_line=closing_line,
                direction=bet.direction,
                model_projection=bet.model_projection,
                model_std=bet.model_std,
                bet_odds_american=bet.odds_american,
                closing_odds_american=closing_odds_american,
            )

            # Persist to CLV log
            existing = session.query(CLVLog).filter_by(bet_id=bet_id).first()
            if existing:
                existing.closing_line = closing_line
                existing.clv_raw = clv["clv_cents"]
                existing.clv_cents = clv["clv_cents"]
                existing.beat_close = clv["beat_close"]
                existing.line_movement = clv["line_movement"]
            else:
                clv_row = CLVLog(
                    bet_id=bet_id,
                    sport=self.sport.lower(),
                    opening_line=opening_line,
                    bet_line=bet.line,
                    closing_line=closing_line,
                    line_movement=clv["line_movement"],
                    clv_raw=clv["clv_cents"],
                    clv_cents=clv["clv_cents"],
                    beat_close=clv["beat_close"],
                )
                session.add(clv_row)

            # Update the bet record too
            bet.closing_line = closing_line
            if closing_odds_american:
                bet.closing_odds = closing_odds_american

            session.commit()
            log.info(
                "CLV stored: bet=%s cents=%.3f beat=%s",
                bet_id, clv["clv_cents"], clv["beat_close"],
            )
            return clv

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Get all CLV records ──────────────────────────────────────────

    def get_all_clv_records(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Get all CLV records, most recent first."""
        session = self._session()
        try:
            rows = (
                session.query(CLVLog, BetLog)
                .join(BetLog, CLVLog.bet_id == BetLog.bet_id)
                .filter(CLVLog.sport == self.sport.lower())
                .order_by(CLVLog.calculated_at.desc())
                .limit(limit)
                .all()
            )

            results = []
            for clv_row, bet_row in rows:
                results.append({
                    "bet_id": clv_row.bet_id,
                    "player": bet_row.player,
                    "stat_type": bet_row.stat_type,
                    "direction": bet_row.direction,
                    "bet_line": clv_row.bet_line,
                    "closing_line": clv_row.closing_line,
                    "opening_line": clv_row.opening_line,
                    "clv_cents": clv_row.clv_cents,
                    "clv_raw": clv_row.clv_raw,
                    "beat_close": clv_row.beat_close,
                    "line_movement": clv_row.line_movement,
                    "calculated_at": clv_row.calculated_at.isoformat() if clv_row.calculated_at else None,
                    "bet_timestamp": bet_row.timestamp.isoformat() if bet_row.timestamp else None,
                    "status": bet_row.status,
                    "pnl": bet_row.pnl,
                })

            return results
        finally:
            session.close()
