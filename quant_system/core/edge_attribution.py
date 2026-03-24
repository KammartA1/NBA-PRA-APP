"""Edge Attribution Engine — Decomposes WHERE profit actually comes from.

Most bettors think they have edge. This module answers: is the edge from
prediction skill, timing (CLV), market inefficiency (soft books), or luck?

Uses counterfactual analysis: what would profit be if we bet at close?

Usage:
    from quant_system.core.edge_attribution import EdgeAttributionEngine
    from quant_system.db.schema import get_session

    session = get_session()
    engine = EdgeAttributionEngine(session)
    report = engine.daily_attribution_report("nba")
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


class EdgeAttributionEngine:
    """Decomposes realized profits into prediction edge, CLV/timing edge,
    market inefficiency, and variance (luck).

    Uses counterfactual analysis: what would profit be if bet at close?
    """

    def __init__(self, db_session):
        self.db = db_session

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _american_to_decimal(odds: int) -> float:
        if odds > 0:
            return 1.0 + odds / 100.0
        elif odds < 0:
            return 1.0 + 100.0 / abs(odds)
        return 2.0

    @staticmethod
    def _decimal_to_implied(decimal_odds: float) -> float:
        if decimal_odds <= 1.0:
            return 1.0
        return 1.0 / decimal_odds

    @staticmethod
    def _implied_to_decimal(prob: float) -> float:
        if prob <= 0.0:
            return 100.0
        return 1.0 / prob

    @staticmethod
    def _bet_pnl(stake: float, decimal_odds: float, won: bool) -> float:
        """Compute PnL for a single bet."""
        if won:
            return stake * (decimal_odds - 1.0)
        return -stake

    @staticmethod
    def _z_score(mean: float, n: int) -> float:
        """Z-score for a mean over n samples (assumes std ~ |mean| heuristic)."""
        if n < 2:
            return 0.0
        se = max(abs(mean), 0.01) / math.sqrt(n)
        return mean / se if se > 0 else 0.0

    # ── Core Attribution ───────────────────────────────────────────────

    def attribute_profits(self, sport: str, window: int = 500) -> dict:
        """Pull last N settled bets and decompose profit sources.

        For each bet:
        - prediction_pnl: PnL if model probability was applied at closing odds
        - timing_pnl: PnL difference from betting at open vs close (CLV captured)
        - market_pnl: PnL from soft book premium vs sharp book odds
        - variance_pnl: residual (total_pnl - prediction - timing - market)

        Returns breakdown with percentages and confidence intervals.
        """
        query = text("""
            SELECT
                b.bet_id,
                b.stake,
                b.odds_decimal,
                b.model_prob,
                b.market_prob,
                b.pnl,
                b.status,
                b.closing_line,
                b.closing_odds,
                b.line,
                b.direction,
                c.clv_raw,
                c.clv_cents,
                c.closing_line AS clv_closing_line,
                c.beat_close
            FROM bet_log b
            LEFT JOIN clv_log c ON b.bet_id = c.bet_id
            WHERE b.sport = :sport
              AND b.status IN ('won', 'lost')
            ORDER BY b.settled_at DESC
            LIMIT :window
        """)

        rows = self.db.execute(query, {"sport": sport, "window": window}).fetchall()

        if not rows:
            return self._empty_attribution()

        total_pnl = 0.0
        prediction_pnl = 0.0
        timing_pnl = 0.0
        market_pnl = 0.0
        n_with_clv = 0

        for row in rows:
            bet_id = row[0]
            stake = row[1]
            odds_decimal = row[2]
            model_prob = row[3]
            market_prob = row[4]
            pnl = row[5]
            status = row[6]
            closing_line = row[7]
            closing_odds = row[8]
            bet_line = row[9]
            direction = row[10]
            clv_raw = row[11]
            clv_cents = row[12]

            won = status == "won"
            total_pnl += pnl

            # --- Prediction edge ---
            # Counterfactual: what if we bet at CLOSING odds with our model prob?
            # Prediction edge = model_prob - closing_market_prob (skill at picking winners)
            if closing_odds is not None and closing_odds != 0:
                closing_decimal = self._american_to_decimal(closing_odds)
                closing_market_prob = self._decimal_to_implied(closing_decimal)
            else:
                closing_decimal = odds_decimal
                closing_market_prob = market_prob

            # Prediction PnL: what we'd make if we bet at closing line
            # with our model edge (model_prob - closing_market_prob)
            prediction_edge = model_prob - closing_market_prob
            # Expected PnL from pure prediction at closing odds
            pred_ev = stake * prediction_edge * (closing_decimal - 1.0)
            prediction_pnl += pred_ev

            # --- Timing edge (CLV) ---
            # Difference between what we got and what closing line offered
            if clv_cents is not None:
                timing_contribution = stake * (clv_cents / 100.0)
                timing_pnl += timing_contribution
                n_with_clv += 1
            else:
                # Estimate from odds movement
                if closing_odds is not None and closing_odds != 0:
                    open_ev = self._decimal_to_implied(odds_decimal)
                    close_ev = self._decimal_to_implied(closing_decimal)
                    timing_contribution = stake * (close_ev - open_ev)
                    timing_pnl += timing_contribution
                    n_with_clv += 1

            # --- Market inefficiency ---
            # Soft book premium: difference between our book's odds and sharp market
            # Approximated as market_prob deviation from true probability
            # Sharp books have tighter lines; retail books have wider spreads
            # We estimate market inefficiency as the juice gap
            vig_at_bet = max(0, market_prob - 0.5) * 2  # rough vig estimate
            soft_premium = stake * vig_at_bet * 0.02  # ~2% soft book premium estimate
            market_pnl += soft_premium

        # Variance is the residual
        variance_pnl = total_pnl - prediction_pnl - timing_pnl - market_pnl

        n = len(rows)

        # Compute percentages (handle zero total)
        abs_total = abs(total_pnl) if abs(total_pnl) > 0.01 else 1.0

        result = {
            "window": n,
            "total_pnl": round(total_pnl, 2),
            "prediction_edge_pnl": round(prediction_pnl, 2),
            "prediction_edge_pct": round(prediction_pnl / abs_total * 100, 1),
            "timing_edge_pnl": round(timing_pnl, 2),
            "timing_edge_pct": round(timing_pnl / abs_total * 100, 1),
            "market_inefficiency_pnl": round(market_pnl, 2),
            "market_inefficiency_pct": round(market_pnl / abs_total * 100, 1),
            "variance_pnl": round(variance_pnl, 2),
            "variance_pct": round(variance_pnl / abs_total * 100, 1),
            "n_bets_with_clv": n_with_clv,
            "confidence": self._attribution_confidence(n, n_with_clv),
        }

        return result

    def counterfactual_analysis(self, bet_id: str) -> dict:
        """For a single bet: what would have happened if we bet at closing line?

        Computes the pure prediction edge vs timing edge for one bet.
        """
        query = text("""
            SELECT
                b.bet_id, b.stake, b.odds_decimal, b.odds_american,
                b.model_prob, b.market_prob, b.pnl, b.status,
                b.closing_line, b.closing_odds, b.line, b.direction,
                c.clv_raw, c.clv_cents, c.beat_close
            FROM bet_log b
            LEFT JOIN clv_log c ON b.bet_id = c.bet_id
            WHERE b.bet_id = :bet_id
        """)

        row = self.db.execute(query, {"bet_id": bet_id}).fetchone()
        if row is None:
            return {"error": f"Bet {bet_id} not found"}

        stake = row[1]
        odds_decimal = row[2]
        odds_american = row[3]
        model_prob = row[4]
        market_prob = row[5]
        actual_pnl = row[6]
        status = row[7]
        closing_line = row[8]
        closing_odds = row[9]
        bet_line = row[10]
        direction = row[11]
        clv_raw = row[12]
        clv_cents = row[13]
        beat_close = row[14]

        won = status == "won"

        # What would PnL be at closing odds?
        if closing_odds is not None and closing_odds != 0:
            closing_decimal = self._american_to_decimal(closing_odds)
        else:
            closing_decimal = odds_decimal

        counterfactual_pnl = self._bet_pnl(stake, closing_decimal, won)

        # Timing edge = actual PnL - counterfactual PnL
        timing_edge_pnl = actual_pnl - counterfactual_pnl

        # Prediction edge = counterfactual PnL (what we'd make even at close)
        prediction_edge_pnl = counterfactual_pnl

        # Model edge over closing market
        closing_market_prob = self._decimal_to_implied(closing_decimal)
        pure_prediction_edge = model_prob - closing_market_prob

        return {
            "bet_id": bet_id,
            "actual_pnl": round(actual_pnl, 2),
            "counterfactual_pnl_at_close": round(counterfactual_pnl, 2),
            "prediction_edge_pnl": round(prediction_edge_pnl, 2),
            "timing_edge_pnl": round(timing_edge_pnl, 2),
            "model_prob": model_prob,
            "market_prob_at_bet": market_prob,
            "market_prob_at_close": round(closing_market_prob, 4),
            "pure_prediction_edge": round(pure_prediction_edge, 4),
            "beat_closing_line": bool(beat_close) if beat_close is not None else None,
            "clv_cents": clv_cents,
            "won": won,
            "verdict": self._single_bet_verdict(
                pure_prediction_edge, timing_edge_pnl, actual_pnl
            ),
        }

    def daily_attribution_report(self, sport: str) -> dict:
        """Daily output with full attribution breakdown and verdict.

        Returns:
            {
                "total_pnl": float,
                "prediction_edge_pnl": float,
                "prediction_edge_pct": float,
                "timing_edge_pnl": float,
                "timing_edge_pct": float,
                "market_inefficiency_pnl": float,
                "market_inefficiency_pct": float,
                "variance_pnl": float,
                "variance_pct": float,
                "verdict": "REAL_EDGE" | "CLV_DEPENDENT" | "LUCK_DRIVEN" | "NO_EDGE"
            }
        """
        # Pull today's settled bets
        today = datetime.utcnow().date()
        query = text("""
            SELECT
                b.bet_id, b.stake, b.odds_decimal, b.model_prob, b.market_prob,
                b.pnl, b.status, b.closing_odds,
                c.clv_raw, c.clv_cents
            FROM bet_log b
            LEFT JOIN clv_log c ON b.bet_id = c.bet_id
            WHERE b.sport = :sport
              AND b.status IN ('won', 'lost')
              AND DATE(b.settled_at) = :today
            ORDER BY b.settled_at DESC
        """)

        rows = self.db.execute(query, {"sport": sport, "today": str(today)}).fetchall()

        # Also get rolling 100-bet attribution for the verdict
        rolling = self.attribute_profits(sport, window=100)

        if not rows:
            # No bets settled today — return rolling data with today's zeroes
            return {
                "date": str(today),
                "total_pnl": 0.0,
                "prediction_edge_pnl": 0.0,
                "prediction_edge_pct": 0.0,
                "timing_edge_pnl": 0.0,
                "timing_edge_pct": 0.0,
                "market_inefficiency_pnl": 0.0,
                "market_inefficiency_pct": 0.0,
                "variance_pnl": 0.0,
                "variance_pct": 0.0,
                "verdict": self._compute_verdict(rolling),
                "rolling_100_attribution": rolling,
                "n_bets_today": 0,
            }

        # Compute today's attribution
        total_pnl = 0.0
        prediction_pnl = 0.0
        timing_pnl = 0.0
        market_pnl = 0.0

        for row in rows:
            stake = row[1]
            odds_decimal = row[2]
            model_prob = row[3]
            market_prob = row[4]
            pnl = row[5]
            status = row[6]
            closing_odds = row[7]
            clv_raw = row[8]
            clv_cents = row[9]

            won = status == "won"
            total_pnl += pnl

            # Closing odds
            if closing_odds is not None and closing_odds != 0:
                closing_decimal = self._american_to_decimal(closing_odds)
                closing_market_prob = self._decimal_to_implied(closing_decimal)
            else:
                closing_decimal = odds_decimal
                closing_market_prob = market_prob

            # Prediction edge
            pred_edge = model_prob - closing_market_prob
            prediction_pnl += stake * pred_edge * (closing_decimal - 1.0)

            # Timing edge
            if clv_cents is not None:
                timing_pnl += stake * (clv_cents / 100.0)
            else:
                open_prob = self._decimal_to_implied(odds_decimal)
                close_prob = self._decimal_to_implied(closing_decimal)
                timing_pnl += stake * (close_prob - open_prob)

            # Market inefficiency (soft book premium estimate)
            vig_estimate = max(0, market_prob - 0.5) * 2
            market_pnl += stake * vig_estimate * 0.02

        variance_pnl = total_pnl - prediction_pnl - timing_pnl - market_pnl
        abs_total = abs(total_pnl) if abs(total_pnl) > 0.01 else 1.0
        n = len(rows)

        return {
            "date": str(today),
            "total_pnl": round(total_pnl, 2),
            "prediction_edge_pnl": round(prediction_pnl, 2),
            "prediction_edge_pct": round(prediction_pnl / abs_total * 100, 1),
            "timing_edge_pnl": round(timing_pnl, 2),
            "timing_edge_pct": round(timing_pnl / abs_total * 100, 1),
            "market_inefficiency_pnl": round(market_pnl, 2),
            "market_inefficiency_pct": round(market_pnl / abs_total * 100, 1),
            "variance_pnl": round(variance_pnl, 2),
            "variance_pct": round(variance_pnl / abs_total * 100, 1),
            "verdict": self._compute_verdict(rolling),
            "rolling_100_attribution": rolling,
            "n_bets_today": n,
        }

    # ── Verdict Logic ──────────────────────────────────────────────────

    def _compute_verdict(self, attribution: dict) -> str:
        """Determine overall edge verdict from attribution breakdown.

        REAL_EDGE: Prediction edge is dominant source of profit.
        CLV_DEPENDENT: Profit comes mostly from timing/CLV, not prediction.
        LUCK_DRIVEN: Variance is dominant. Unsustainable.
        NO_EDGE: Total PnL is negative or near zero.
        """
        total = attribution.get("total_pnl", 0)
        pred_pct = attribution.get("prediction_edge_pct", 0)
        timing_pct = attribution.get("timing_edge_pct", 0)
        variance_pct = attribution.get("variance_pct", 0)
        n = attribution.get("window", 0)

        if n < 30:
            return "UNCERTAIN"

        if total <= 0:
            return "NO_EDGE"

        # Prediction edge is the dominant profitable source
        if pred_pct > 40 and pred_pct > variance_pct:
            return "REAL_EDGE"

        # Timing/CLV is the main profit driver
        if timing_pct > 50:
            return "CLV_DEPENDENT"

        # Variance (luck) dominates
        if abs(variance_pct) > 50:
            return "LUCK_DRIVEN"

        # Mixed / unclear
        if total > 0 and pred_pct > 20:
            return "REAL_EDGE"

        return "NO_EDGE"

    def _single_bet_verdict(self, prediction_edge: float, timing_pnl: float,
                            actual_pnl: float) -> str:
        """Verdict for a single bet's counterfactual."""
        if actual_pnl <= 0:
            return "LOSS"
        if prediction_edge > 0.02 and timing_pnl > 0:
            return "SKILL_AND_TIMING"
        if prediction_edge > 0.02:
            return "PURE_SKILL"
        if timing_pnl > 0:
            return "TIMING_ONLY"
        return "VARIANCE"

    def _attribution_confidence(self, n_total: int, n_with_clv: int) -> str:
        """How confident are we in this attribution?"""
        if n_total < 50:
            return "LOW"
        if n_with_clv < n_total * 0.5:
            return "MEDIUM"  # Missing CLV data for many bets
        if n_total < 200:
            return "MEDIUM"
        return "HIGH"

    def _empty_attribution(self) -> dict:
        return {
            "window": 0,
            "total_pnl": 0.0,
            "prediction_edge_pnl": 0.0,
            "prediction_edge_pct": 0.0,
            "timing_edge_pnl": 0.0,
            "timing_edge_pct": 0.0,
            "market_inefficiency_pnl": 0.0,
            "market_inefficiency_pct": 0.0,
            "variance_pnl": 0.0,
            "variance_pct": 0.0,
            "n_bets_with_clv": 0,
            "confidence": "NONE",
        }
