"""Sharp Money Detection — Identify when professional bettors disagree with you.

Sharp books (Pinnacle, Circa) are market-makers whose lines are set by
the weight of professional money. When sharp lines diverge from soft
books (DraftKings, FanDuel), it indicates where smart money is flowing.

Rules:
- If sharp money AGREES with your model → increase confidence
- If sharp money DISAGREES → reduce confidence or skip the bet
- Steam moves (rapid line changes across books) indicate sharp action

This is the market's immune system — respect it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np

from ..db.schema import LineSnapshot, get_session
from ..core.types import Sport

logger = logging.getLogger(__name__)

# Sharp book hierarchy (most sharp to least)
SHARP_SOURCES = ["pinnacle", "circa", "bookmaker", "betcris"]
SOFT_SOURCES = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet", "prizepicks"]


class SharpMoneyDetector:
    """Detects sharp money flow from line divergence."""

    def __init__(self, sport: Sport, db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path

    def _session(self):
        return get_session(self._db_path)

    def sharp_vs_soft(self, player: str, stat_type: str) -> dict:
        """Compare sharp book lines vs soft book lines.

        Returns:
            {
                "sharp_line": float or None,
                "soft_line": float or None,
                "divergence": float,
                "sharp_direction": str,      # "over", "under", "neutral"
                "confidence_adjustment": float, # Multiplier: 0.5-1.3
                "sources": {"sharp": [...], "soft": [...]},
            }
        """
        session = self._session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=24)
            snaps = (
                session.query(LineSnapshot)
                .filter_by(sport=self.sport.value, player=player, stat_type=stat_type)
                .filter(LineSnapshot.captured_at >= cutoff)
                .order_by(LineSnapshot.captured_at.desc())
                .all()
            )

            if not snaps:
                return {
                    "sharp_line": None, "soft_line": None, "divergence": 0.0,
                    "sharp_direction": "neutral", "confidence_adjustment": 1.0,
                    "sources": {"sharp": [], "soft": []},
                }

            # Get latest line from each source
            latest_by_source: dict[str, float] = {}
            for snap in snaps:
                src = snap.source.lower()
                if src not in latest_by_source:
                    latest_by_source[src] = snap.line

            sharp_lines = [latest_by_source[s] for s in SHARP_SOURCES if s in latest_by_source]
            soft_lines = [latest_by_source[s] for s in SOFT_SOURCES if s in latest_by_source]

            if not sharp_lines or not soft_lines:
                return {
                    "sharp_line": float(np.mean(sharp_lines)) if sharp_lines else None,
                    "soft_line": float(np.mean(soft_lines)) if soft_lines else None,
                    "divergence": 0.0,
                    "sharp_direction": "neutral",
                    "confidence_adjustment": 1.0,
                    "sources": {
                        "sharp": [s for s in SHARP_SOURCES if s in latest_by_source],
                        "soft": [s for s in SOFT_SOURCES if s in latest_by_source],
                    },
                }

            sharp_avg = float(np.mean(sharp_lines))
            soft_avg = float(np.mean(soft_lines))
            divergence = sharp_avg - soft_avg

            # Interpret direction
            if divergence > 0.5:
                sharp_dir = "over"  # Sharp has higher line → sharps expect OVER
            elif divergence < -0.5:
                sharp_dir = "under"
            else:
                sharp_dir = "neutral"

            # Confidence adjustment
            # Big divergence = sharps disagree with softs = be careful
            abs_div = abs(divergence)
            if abs_div > 1.5:
                conf_adj = 0.50  # Major disagreement — cut confidence in half
            elif abs_div > 1.0:
                conf_adj = 0.70
            elif abs_div > 0.5:
                conf_adj = 0.85
            else:
                conf_adj = 1.0  # Lines agree

            return {
                "sharp_line": round(sharp_avg, 2),
                "soft_line": round(soft_avg, 2),
                "divergence": round(divergence, 2),
                "sharp_direction": sharp_dir,
                "confidence_adjustment": conf_adj,
                "sources": {
                    "sharp": [s for s in SHARP_SOURCES if s in latest_by_source],
                    "soft": [s for s in SOFT_SOURCES if s in latest_by_source],
                },
            }
        finally:
            session.close()

    def model_agrees_with_sharp(
        self,
        player: str,
        stat_type: str,
        model_direction: str,
    ) -> dict:
        """Check if model's bet direction agrees with sharp money.

        Returns:
            {
                "agrees": bool or None,
                "confidence_multiplier": float,   # 0.5 if disagrees, 1.2 if agrees
                "message": str,
            }
        """
        sharp_info = self.sharp_vs_soft(player, stat_type)
        sharp_dir = sharp_info["sharp_direction"]

        if sharp_dir == "neutral" or sharp_info["sharp_line"] is None:
            return {
                "agrees": None,
                "confidence_multiplier": 1.0,
                "message": "No sharp signal detected",
            }

        agrees = (model_direction == sharp_dir)

        if agrees:
            mult = 1.15  # Boost — model and sharps aligned
            msg = f"Model ({model_direction}) AGREES with sharp money ({sharp_dir})"
        else:
            mult = 0.60  # Penalty — you're fighting professional bettors
            msg = f"Model ({model_direction}) DISAGREES with sharp money ({sharp_dir}) — CAUTION"
            logger.warning("Sharp disagreement: %s %s %s — model says %s, sharps say %s",
                          player, stat_type, model_direction, model_direction, sharp_dir)

        return {
            "agrees": agrees,
            "confidence_multiplier": mult,
            "message": msg,
        }
