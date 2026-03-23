"""
edge_analysis/decomposer.py
=============================
Master EdgeDecomposer — orchestrates all 5 edge components and produces
a unified attribution report.

Usage:
    decomposer = EdgeDecomposer(sport="nba")
    report = decomposer.run()
    print(report.verdict)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from edge_analysis.schemas import BetRecord, EdgeReport
from edge_analysis.predictive import compute_predictive_edge
from edge_analysis.informational import compute_informational_edge
from edge_analysis.market_inefficiency import compute_market_inefficiency_edge
from edge_analysis.execution import compute_execution_edge
from edge_analysis.structural import compute_structural_edge

log = logging.getLogger(__name__)


class EdgeDecomposer:
    """Decomposes total system performance into exactly 5 edge components.

    Components:
      1. Predictive — probability accuracy vs market
      2. Informational — signal timing vs line movement
      3. Market Inefficiency — CLV (closing line value)
      4. Execution — slippage between signal and bet
      5. Structural — correlation, Kelly sizing, diversification

    Each component is independently computed and attributed a percentage
    of total ROI. The final verdict identifies which components carry the
    weight and which are illusions.
    """

    def __init__(self, sport: str = "nba", db_path: str | None = None):
        self.sport = sport.lower()
        self._db_path = db_path

    def load_bets_from_db(self) -> List[BetRecord]:
        """Load all historical bets from the quant system database.

        Joins BetLog with CLVLog and LineSnapshot to build full BetRecord
        objects with timing, lines, and outcome data.
        """
        from quant_system.db.schema import get_session, BetLog, CLVLog, LineSnapshot

        session = get_session(self._db_path)
        try:
            # Get all settled bets
            rows = (
                session.query(BetLog)
                .filter(
                    BetLog.sport == self.sport,
                    BetLog.status.in_(["won", "lost", "push"]),
                )
                .order_by(BetLog.timestamp.asc())
                .all()
            )

            records = []
            for row in rows:
                # Look up CLV data
                clv = (
                    session.query(CLVLog)
                    .filter_by(bet_id=row.bet_id)
                    .first()
                )

                opening_line = clv.opening_line if clv else row.line
                closing_line = clv.closing_line if clv else (row.closing_line or row.line)

                # Look for line movement timing
                line_moved_at = None
                first_move = (
                    session.query(LineSnapshot)
                    .filter(
                        LineSnapshot.player == row.player,
                        LineSnapshot.stat_type == row.stat_type,
                        LineSnapshot.sport == self.sport,
                        LineSnapshot.captured_at > row.timestamp,
                        LineSnapshot.line != row.line,
                    )
                    .order_by(LineSnapshot.captured_at.asc())
                    .first()
                )
                if first_move:
                    line_moved_at = first_move.captured_at

                # Determine if won
                won = None
                if row.status == "won":
                    won = True
                elif row.status == "lost":
                    won = False
                elif row.status == "push":
                    won = None  # Exclude pushes from win/loss analysis

                # Market implied probability from closing
                market_prob_at_close = None
                if row.closing_odds:
                    market_prob_at_close = _american_to_implied(row.closing_odds)

                records.append(BetRecord(
                    bet_id=row.bet_id,
                    timestamp=row.timestamp,
                    signal_generated_at=row.timestamp,  # Approximate — signal ≈ bet time
                    player=row.player,
                    sport=self.sport,
                    market_type=row.stat_type,
                    direction=row.direction,
                    signal_line=opening_line,
                    bet_line=row.line,
                    closing_line=closing_line,
                    opening_line=opening_line,
                    predicted_prob=row.model_prob,
                    market_prob_at_bet=row.market_prob,
                    market_prob_at_close=market_prob_at_close,
                    actual_outcome=row.actual_result,
                    won=won,
                    stake=row.stake,
                    pnl=row.pnl or 0.0,
                    odds_american=row.odds_american,
                    odds_decimal=row.odds_decimal,
                    kelly_fraction=row.kelly_fraction,
                    model_projection=row.model_projection,
                    model_std=row.model_std,
                    confidence_score=row.confidence_score or 0.0,
                    line_moved_at=line_moved_at,
                ))

            log.info("Loaded %d settled bets for edge decomposition", len(records))
            return records

        finally:
            session.close()

    def run(self, bets: Optional[List[BetRecord]] = None) -> EdgeReport:
        """Execute full edge decomposition.

        If bets not provided, loads from database.

        Args:
            bets: Optional list of BetRecord objects. If None, loads from DB.

        Returns:
            EdgeReport with full 5-component attribution.
        """
        if bets is None:
            bets = self.load_bets_from_db()

        if len(bets) < 10:
            return EdgeReport(
                generated_at=datetime.now(timezone.utc),
                sport=self.sport,
                total_roi=0.0,
                total_bets=len(bets),
                total_pnl=0.0,
                predictive_pct=0.0,
                informational_pct=0.0,
                market_pct=0.0,
                execution_pct=0.0,
                structural_pct=0.0,
                verdict="Insufficient data for edge decomposition (need 10+ bets)",
            )

        # Compute total ROI
        total_pnl = sum(b.pnl for b in bets if b.pnl is not None)
        total_staked = sum(b.stake for b in bets if b.stake > 0)
        total_roi = (total_pnl / total_staked * 100.0) if total_staked > 0 else 0.0

        # Run each component
        predictive = compute_predictive_edge(bets, total_roi)
        informational = compute_informational_edge(bets, total_roi)
        market = compute_market_inefficiency_edge(bets, total_roi)
        execution = compute_execution_edge(bets, total_roi)
        structural = compute_structural_edge(bets, total_roi)

        # Normalize percentages to sum to 100%
        raw_pcts = [
            predictive.edge_pct_of_roi,
            informational.edge_pct_of_roi,
            market.edge_pct_of_roi,
            execution.edge_pct_of_roi,
            structural.edge_pct_of_roi,
        ]
        raw_total = sum(abs(p) for p in raw_pcts)
        if raw_total > 0:
            # Scale so absolute values sum to 100
            scale = 100.0 / raw_total
            pred_pct = predictive.edge_pct_of_roi * scale
            info_pct = informational.edge_pct_of_roi * scale
            mkt_pct = market.edge_pct_of_roi * scale
            exec_pct = execution.edge_pct_of_roi * scale
            struct_pct = structural.edge_pct_of_roi * scale
        else:
            pred_pct = info_pct = mkt_pct = exec_pct = struct_pct = 20.0

        # Update component results with normalized percentages
        predictive.edge_pct_of_roi = round(pred_pct, 2)
        informational.edge_pct_of_roi = round(info_pct, 2)
        market.edge_pct_of_roi = round(mkt_pct, 2)
        execution.edge_pct_of_roi = round(exec_pct, 2)
        structural.edge_pct_of_roi = round(struct_pct, 2)

        # Build calibration curve from predictive component
        cal_curve = []
        if predictive.details.get("calibration_curve"):
            from edge_analysis.schemas import CalibrationPoint
            for pt in predictive.details["calibration_curve"]:
                bounds = pt["bucket"].replace("%", "").split("-")
                cal_curve.append(CalibrationPoint(
                    bucket_lower=float(bounds[0]) / 100.0,
                    bucket_upper=float(bounds[1]) / 100.0,
                    predicted_avg=pt["predicted"],
                    actual_rate=pt["actual"],
                    n_bets=pt["n"],
                    calibration_error=pt["error"],
                ))

        # Determine the heavy lifter and illusions
        components = [
            ("Predictive", pred_pct, predictive.is_positive, predictive.is_significant),
            ("Informational", info_pct, informational.is_positive, informational.is_significant),
            ("Market Inefficiency", mkt_pct, market.is_positive, market.is_significant),
            ("Execution", exec_pct, execution.is_positive, execution.is_significant),
            ("Structural", struct_pct, structural.is_positive, structural.is_significant),
        ]

        # Heavy lifter: highest positive, significant component
        positive_sig = [(n, p) for n, p, pos, sig in components if pos and sig]
        if positive_sig:
            heavy_lifter = max(positive_sig, key=lambda x: x[1])[0]
        else:
            positive_any = [(n, p) for n, p, pos, sig in components if pos]
            heavy_lifter = max(positive_any, key=lambda x: x[1])[0] if positive_any else "None detected"

        # Illusions: components that are not positive or not significant
        illusions = [n for n, p, pos, sig in components if not pos or (abs(p) > 10 and not sig)]

        # Final verdict
        verdict = _build_final_verdict(
            total_roi, total_pnl, len(bets),
            predictive, informational, market, execution, structural,
            heavy_lifter, illusions,
        )

        return EdgeReport(
            generated_at=datetime.now(timezone.utc),
            sport=self.sport,
            total_roi=round(total_roi, 4),
            total_bets=len(bets),
            total_pnl=round(total_pnl, 2),
            predictive_pct=round(pred_pct, 2),
            informational_pct=round(info_pct, 2),
            market_pct=round(mkt_pct, 2),
            execution_pct=round(exec_pct, 2),
            structural_pct=round(struct_pct, 2),
            predictive=predictive,
            informational=informational,
            market_inefficiency=market,
            execution=execution,
            structural=structural,
            calibration_curve=cal_curve,
            brier_score=predictive.details.get("brier_model", 0.0),
            log_loss=predictive.details.get("logloss_model", 0.0),
            brier_baseline=predictive.details.get("brier_market", 0.25),
            log_loss_baseline=predictive.details.get("logloss_market", 0.693),
            verdict=verdict,
            heavy_lifter=heavy_lifter,
            illusions=illusions,
        )


def _american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _build_final_verdict(
    total_roi: float,
    total_pnl: float,
    n_bets: int,
    predictive, informational, market, execution, structural,
    heavy_lifter: str,
    illusions: List[str],
) -> str:
    """Build the definitive final verdict."""
    lines = []
    lines.append("=" * 70)
    lines.append("EDGE DECOMPOSITION VERDICT")
    lines.append("=" * 70)
    lines.append(f"Total ROI: {total_roi:+.2f}% across {n_bets} bets (P&L: ${total_pnl:+,.2f})")
    lines.append("")

    # Component summary
    lines.append("COMPONENT ATTRIBUTION:")
    components = [
        ("Predictive", predictive),
        ("Informational", informational),
        ("Market Inefficiency", market),
        ("Execution", execution),
        ("Structural", structural),
    ]
    for name, comp in components:
        status = "REAL" if comp.is_positive and comp.is_significant else (
            "POSSIBLE" if comp.is_positive else "ILLUSION"
        )
        lines.append(f"  {name:25s} {comp.edge_pct_of_roi:6.1f}%  [{status}]  p={comp.p_value:.4f}")

    lines.append("")
    lines.append(f"HEAVY LIFTER: {heavy_lifter}")
    lines.append(f"  This component is doing the real work.")

    if illusions:
        lines.append(f"ILLUSIONS: {', '.join(illusions)}")
        lines.append(f"  These components appear to contribute but lack significance.")
    else:
        lines.append("ILLUSIONS: None — all components show genuine contribution.")

    lines.append("")
    lines.append("WHICH COMPONENT IS DOING THE HEAVY LIFTING — AND WHICH ARE ILLUSIONS?")
    lines.append("-" * 70)

    if market.is_positive and market.is_significant:
        lines.append(
            "CLV is positive and significant. This is the strongest evidence of real edge. "
            "The system is consistently getting better prices than closing — this is NOT luck."
        )
    elif total_roi > 0 and not market.is_positive:
        lines.append(
            "WARNING: Positive ROI but NEGATIVE CLV. This is the classic sign of "
            "a lucky streak, not genuine edge. Profitable by luck, not by skill. "
            "Without positive CLV, this system has NO sustainable edge."
        )

    if predictive.is_positive and predictive.is_significant:
        lines.append(
            "The model produces genuinely better probabilities than the market. "
            "This is the foundation of a real quantitative edge."
        )
    elif not predictive.is_positive:
        lines.append(
            "The model's probability estimates are NOT better than implied market prices. "
            "The predictive engine needs improvement or the edge comes from elsewhere."
        )

    lines.append("=" * 70)
    return "\n".join(lines)
