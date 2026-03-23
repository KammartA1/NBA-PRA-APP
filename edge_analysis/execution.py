"""
edge_analysis/execution.py
============================
Component 4: EXECUTION EDGE — Are we getting good prices or losing to slippage?

Tracks the gap between signal price and actual execution price:
  - execution_cost = bet_line - signal_line (for over bets)
  - execution_cost = signal_line - bet_line (for under bets)
  - Positive cost = slippage (we got a worse price)
  - Negative cost = improvement (we got a better price)

Also analyzes: optimal execution timing, slippage by market type,
and total execution drag on ROI.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.schemas import BetRecord, EdgeComponentResult

log = logging.getLogger(__name__)


def _compute_execution_cost(bet: BetRecord) -> float:
    """Compute execution cost (slippage) for a single bet.

    Positive = slippage (worse price than signal).
    Negative = price improvement (better price than signal).
    """
    if bet.direction.lower() == "over":
        # For over bets: higher line = worse (need more stats to clear)
        return bet.bet_line - bet.signal_line
    else:
        # For under bets: lower line = worse (less room for stat to stay under)
        return bet.signal_line - bet.bet_line


def _compute_execution_vs_close(bet: BetRecord) -> float:
    """How much of the signal-to-close move did we capture?

    Returns fraction: 1.0 = captured all value, 0.0 = captured none.
    """
    if bet.direction.lower() == "over":
        total_move = bet.closing_line - bet.signal_line
        captured = bet.closing_line - bet.bet_line
    else:
        total_move = bet.signal_line - bet.closing_line
        captured = bet.bet_line - bet.closing_line

    if abs(total_move) < 0.01:
        return 1.0  # No move, no slippage
    return captured / total_move if total_move != 0 else 0.0


def compute_execution_edge(
    bets: List[BetRecord],
    total_roi: float,
) -> EdgeComponentResult:
    """Analyze the execution quality component of edge.

    Models the gap between signal price and actual execution price.
    Tracks: signal_line, bet_line, closing_line for every bet.
    Computes: execution_cost = bet_line - signal_line

    Args:
        bets: List of BetRecord objects.
        total_roi: System's total ROI for attribution.

    Returns:
        EdgeComponentResult with execution edge analysis.
    """
    valid = [b for b in bets if b.signal_line is not None and b.signal_line != 0]
    if len(valid) < 20:
        return EdgeComponentResult(
            name="execution",
            edge_pct_of_roi=0.0,
            absolute_value=0.0,
            p_value=1.0,
            is_significant=False,
            is_positive=False,
            sample_size=len(valid),
            verdict="Insufficient data for execution edge analysis (need 20+ bets with signal lines)",
        )

    # Compute execution costs
    costs = np.array([_compute_execution_cost(b) for b in valid])
    capture_rates = np.array([_compute_execution_vs_close(b) for b in valid])

    avg_cost = float(np.mean(costs))
    median_cost = float(np.median(costs))
    std_cost = float(np.std(costs))
    avg_capture = float(np.mean(capture_rates))
    pct_improved = float(np.mean(costs < 0))  # % of bets where we got a better price

    # Significance: is avg slippage significantly different from 0?
    t_stat, p_two = sp_stats.ttest_1samp(costs, 0.0)
    p_value = float(p_two)  # two-sided — we care about both directions

    # For execution, negative cost = positive edge (we got better prices)
    is_positive = avg_cost < 0  # negative cost means price improvement
    is_significant = p_value < 0.05

    # Slippage by market type
    segments: Dict[str, List[float]] = {}
    for b in valid:
        seg = b.market_type
        if seg not in segments:
            segments[seg] = []
        segments[seg].append(_compute_execution_cost(b))

    cost_by_market = {}
    for seg, vals in segments.items():
        arr = np.array(vals)
        cost_by_market[seg] = {
            "avg_slippage": round(float(np.mean(arr)), 3),
            "pct_improved": round(float(np.mean(arr < 0)), 4),
            "n_bets": len(vals),
        }

    # Total execution drag on P&L
    # Estimate: each cent of slippage on a $100 bet costs roughly $1
    # More precisely: convert line slippage to probability cost
    total_drag_cents = float(np.sum(costs))

    # Attribution
    if is_positive and is_significant:
        # Good execution = adds to ROI (typically 5-15%)
        exec_pct = min(15.0, pct_improved * 20.0)
    elif is_positive:
        exec_pct = min(8.0, pct_improved * 10.0)
    elif is_significant:
        # Bad execution = negative contribution
        exec_pct = max(-20.0, -abs(avg_cost) * 10.0)
    else:
        exec_pct = max(-10.0, -abs(avg_cost) * 5.0)

    # Build verdict
    verdict_parts = []
    if is_positive and is_significant:
        verdict_parts.append(
            f"POSITIVE execution edge. Avg price improvement: {abs(avg_cost):.2f} cents. "
            f"{pct_improved:.0%} of bets executed at better-than-signal price. "
            f"Capture rate: {avg_capture:.0%}. p={p_value:.4f}."
        )
    elif is_positive:
        verdict_parts.append(
            f"Slight execution advantage ({abs(avg_cost):.2f} cents avg improvement) "
            f"but NOT significant (p={p_value:.4f}). {pct_improved:.0%} improved."
        )
    elif is_significant:
        verdict_parts.append(
            f"EXECUTION DRAG detected. Avg slippage: {avg_cost:.2f} cents per bet. "
            f"Only {pct_improved:.0%} of bets got better-than-signal prices. "
            f"Total drag: {total_drag_cents:.1f} cents across {len(valid)} bets. "
            f"This is actively destroying edge."
        )
    else:
        verdict_parts.append(
            f"Neutral execution. Avg slippage: {avg_cost:.2f} cents (not significant, p={p_value:.4f}). "
            f"{pct_improved:.0%} improved. Capture rate: {avg_capture:.0%}."
        )

    # Flag worst market for execution
    worst = max(cost_by_market.items(), key=lambda x: x[1]["avg_slippage"]) if cost_by_market else None
    if worst and worst[1]["avg_slippage"] > 0.5:
        verdict_parts.append(
            f"Worst execution: {worst[0]} ({worst[1]['avg_slippage']:+.2f} cents avg slippage)."
        )

    return EdgeComponentResult(
        name="execution",
        edge_pct_of_roi=round(exec_pct, 2),
        absolute_value=round(avg_cost, 3),
        p_value=round(p_value, 4),
        is_significant=is_significant,
        is_positive=is_positive,
        sample_size=len(valid),
        details={
            "avg_slippage_cents": round(avg_cost, 3),
            "median_slippage_cents": round(median_cost, 3),
            "std_slippage_cents": round(std_cost, 3),
            "avg_capture_rate": round(avg_capture, 4),
            "pct_price_improved": round(pct_improved, 4),
            "total_drag_cents": round(total_drag_cents, 2),
            "cost_by_market": cost_by_market,
        },
        verdict=" ".join(verdict_parts),
    )
