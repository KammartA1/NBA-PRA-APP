"""
edge_analysis/informational.py
================================
Component 2: INFORMATIONAL EDGE — Do we know something before the market adjusts?

Analyzes signal timing vs line movement to determine if we're ahead of the market
or just chasing moves that already happened.

Key metric: time_delta = signal_generation_time - line_movement_time
  - Negative delta → we signaled AFTER the move (fake edge)
  - Positive delta → we signaled BEFORE the move (real informational edge)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.schemas import BetRecord, EdgeComponentResult

log = logging.getLogger(__name__)


def _compute_signal_lead_time(bet: BetRecord) -> Optional[float]:
    """Compute lead time in minutes between signal generation and line movement.

    Returns:
        Positive = signal fired before line moved (good).
        Negative = signal fired after line moved (bad).
        None = insufficient timing data.
    """
    if bet.signal_generated_at is None or bet.line_moved_at is None:
        return None
    delta = (bet.line_moved_at - bet.signal_generated_at).total_seconds() / 60.0
    return delta  # positive = we were early


def _compute_signal_to_bet_latency(bet: BetRecord) -> Optional[float]:
    """Time in minutes from signal generation to bet placement."""
    if bet.signal_generated_at is None:
        return None
    delta = (bet.timestamp - bet.signal_generated_at).total_seconds() / 60.0
    return delta


def compute_informational_edge(
    bets: List[BetRecord],
    total_roi: float,
) -> EdgeComponentResult:
    """Analyze the informational timing component of edge.

    For each bet, computes:
    1. Signal lead time: did our signal fire before or after the line moved?
    2. Signal-to-bet latency: how fast did we act on the signal?
    3. Line movement after signal: did the line continue moving our way?

    Informational edge exists if we consistently generate signals BEFORE
    the market adjusts, giving us time to capture the value.

    Args:
        bets: List of BetRecord objects.
        total_roi: System's total ROI for attribution.

    Returns:
        EdgeComponentResult with informational edge analysis.
    """
    # Compute lead times for bets that have timing data
    lead_times = []
    latencies = []
    bets_with_timing = []

    for bet in bets:
        lt = _compute_signal_lead_time(bet)
        if lt is not None:
            lead_times.append(lt)
            bets_with_timing.append(bet)
        latency = _compute_signal_to_bet_latency(bet)
        if latency is not None:
            latencies.append(latency)

    # If we have no timing data, estimate from line movement patterns
    # Use signal_line vs bet_line vs closing_line to infer timing advantage
    inferred_timing = []
    for bet in bets:
        if bet.won is None:
            continue
        # If signal_line != bet_line, the line moved between signal and bet
        signal_to_bet_move = bet.bet_line - bet.signal_line
        # If bet_line != closing_line, the line moved after we bet
        bet_to_close_move = bet.closing_line - bet.bet_line

        if bet.direction.lower() == "over":
            # For over bets: line moving UP after we bet = we were early
            post_bet_value = bet_to_close_move
        else:
            # For under bets: line moving DOWN after we bet = we were early
            post_bet_value = -bet_to_close_move

        inferred_timing.append(post_bet_value)

    if len(lead_times) < 10 and len(inferred_timing) < 20:
        # Use whatever data we have
        all_timing = lead_times if lead_times else inferred_timing
        if len(all_timing) < 10:
            return EdgeComponentResult(
                name="informational",
                edge_pct_of_roi=0.0,
                absolute_value=0.0,
                p_value=1.0,
                is_significant=False,
                is_positive=False,
                sample_size=len(all_timing),
                verdict="Insufficient timing data for informational edge analysis",
            )

    # Primary analysis: direct timing data
    has_direct_timing = len(lead_times) >= 10
    if has_direct_timing:
        lt_arr = np.array(lead_times)
        median_lead = float(np.median(lt_arr))
        mean_lead = float(np.mean(lt_arr))
        pct_ahead = float(np.mean(lt_arr > 0))

        # One-sample t-test: is mean lead time > 0?
        t_stat, p_two = sp_stats.ttest_1samp(lt_arr, 0.0)
        p_value = float(p_two / 2 if t_stat > 0 else 1.0 - p_two / 2)
        is_positive = median_lead > 0
        sample_size = len(lead_times)
    else:
        # Fallback: use inferred timing from line movements
        inf_arr = np.array(inferred_timing)
        median_lead = float(np.median(inf_arr))
        mean_lead = float(np.mean(inf_arr))
        pct_ahead = float(np.mean(inf_arr > 0))

        t_stat, p_two = sp_stats.ttest_1samp(inf_arr, 0.0)
        p_value = float(p_two / 2 if t_stat > 0 else 1.0 - p_two / 2)
        is_positive = mean_lead > 0
        sample_size = len(inferred_timing)

    is_significant = p_value < 0.05

    # Attribution: informational edge contribution to ROI
    # If we're consistently ahead of line moves, this is a real edge source
    # Scale by % of bets where we were ahead and magnitude of lead
    if is_positive and is_significant:
        # Informational edge typically accounts for 10-30% of total edge
        info_pct = min(30.0, pct_ahead * 40.0)
    elif is_positive:
        info_pct = min(15.0, pct_ahead * 20.0)
    else:
        info_pct = 0.0

    # Latency analysis
    median_latency = float(np.median(latencies)) if latencies else None
    mean_latency = float(np.mean(latencies)) if latencies else None

    # Build verdict
    timing_type = "direct signal timing" if has_direct_timing else "inferred line movement"
    verdict_parts = []

    if is_positive and is_significant:
        verdict_parts.append(
            f"REAL informational edge detected via {timing_type}. "
            f"Median lead time: {median_lead:+.1f} min. "
            f"{pct_ahead:.0%} of signals fired before line moved. p={p_value:.4f}."
        )
    elif is_positive and not is_significant:
        verdict_parts.append(
            f"Possible informational edge but NOT significant (p={p_value:.4f}). "
            f"Median lead: {median_lead:+.1f} min. {pct_ahead:.0%} ahead of market."
        )
    else:
        verdict_parts.append(
            f"NO informational edge. Signals fire AFTER line moves. "
            f"Median lead: {median_lead:+.1f} min. Only {pct_ahead:.0%} ahead. "
            f"This system is REACTING to the market, not anticipating it."
        )

    if median_latency is not None:
        verdict_parts.append(f"Signal-to-bet latency: {median_latency:.1f} min median.")

    return EdgeComponentResult(
        name="informational",
        edge_pct_of_roi=round(info_pct, 2),
        absolute_value=round(median_lead, 2),
        p_value=round(p_value, 4),
        is_significant=is_significant,
        is_positive=is_positive,
        sample_size=sample_size,
        details={
            "timing_source": timing_type,
            "median_lead_minutes": round(median_lead, 2),
            "mean_lead_minutes": round(mean_lead, 2),
            "pct_ahead_of_market": round(pct_ahead, 4),
            "t_stat": round(float(t_stat), 3),
            "median_latency_minutes": round(median_latency, 2) if median_latency else None,
            "mean_latency_minutes": round(mean_latency, 2) if mean_latency else None,
            "n_with_direct_timing": len(lead_times),
            "n_with_inferred_timing": len(inferred_timing),
        },
        verdict=" ".join(verdict_parts),
    )
