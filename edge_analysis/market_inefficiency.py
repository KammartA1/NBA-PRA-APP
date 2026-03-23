"""
edge_analysis/market_inefficiency.py
======================================
Component 3: MARKET INEFFICIENCY CAPTURE — Are we exploiting real pricing errors?

Uses the CLV system built in Section 5 to determine:
  - Average CLV across all bets
  - CLV distribution (are we consistently beating the close?)
  - CLV by market segment
  - Whether positive CLV is statistically significant

If avg CLV <= 0, there is NO market inefficiency edge. Period.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.schemas import BetRecord, EdgeComponentResult

log = logging.getLogger(__name__)


def _compute_clv_for_bet(bet: BetRecord) -> float:
    """Compute CLV in cents for a single bet."""
    if bet.direction.lower() == "over":
        return bet.closing_line - bet.bet_line
    else:
        return bet.bet_line - bet.closing_line


def _compute_clv_probability(bet: BetRecord) -> Optional[float]:
    """Compute CLV in probability space using normal distribution model."""
    if bet.model_std <= 0 or bet.model_projection == 0:
        return None
    from scipy.stats import norm
    z_bet = (bet.bet_line - bet.model_projection) / bet.model_std
    z_close = (bet.closing_line - bet.model_projection) / bet.model_std

    if bet.direction.lower() == "over":
        bet_prob = 1.0 - norm.cdf(z_bet)
        close_prob = 1.0 - norm.cdf(z_close)
    else:
        bet_prob = norm.cdf(z_bet)
        close_prob = norm.cdf(z_close)

    return bet_prob - close_prob  # positive = we got better price


def compute_market_inefficiency_edge(
    bets: List[BetRecord],
    total_roi: float,
) -> EdgeComponentResult:
    """Analyze the market inefficiency component of edge.

    This is the gold standard: CLV (Closing Line Value).
    If you consistently beat the closing line, you are exploiting
    genuine market inefficiencies. If not, your profits are luck.

    Args:
        bets: List of BetRecord objects with closing lines.
        total_roi: System's total ROI for attribution.

    Returns:
        EdgeComponentResult with market inefficiency analysis.
    """
    # Filter to bets with valid closing lines
    valid = [b for b in bets if b.closing_line is not None and b.closing_line != 0]
    if len(valid) < 20:
        return EdgeComponentResult(
            name="market_inefficiency",
            edge_pct_of_roi=0.0,
            absolute_value=0.0,
            p_value=1.0,
            is_significant=False,
            is_positive=False,
            sample_size=len(valid),
            verdict="Insufficient CLV data for market inefficiency analysis (need 20+ bets with closing lines)",
        )

    # Compute CLV for every bet
    clv_cents = np.array([_compute_clv_for_bet(b) for b in valid])
    clv_probs = []
    for b in valid:
        cp = _compute_clv_probability(b)
        if cp is not None:
            clv_probs.append(cp)
    clv_prob_arr = np.array(clv_probs) if clv_probs else np.array([])

    # Core metrics
    avg_clv = float(np.mean(clv_cents))
    median_clv = float(np.median(clv_cents))
    std_clv = float(np.std(clv_cents))
    beat_close_rate = float(np.mean(clv_cents > 0))

    # Significance test: is mean CLV > 0?
    t_stat, p_two = sp_stats.ttest_1samp(clv_cents, 0.0)
    p_value = float(p_two / 2 if t_stat > 0 else 1.0 - p_two / 2)

    is_positive = avg_clv > 0
    is_significant = p_value < 0.05

    # CLV by market segment
    segments: Dict[str, List[float]] = {}
    for b in valid:
        seg = b.market_type
        if seg not in segments:
            segments[seg] = []
        segments[seg].append(_compute_clv_for_bet(b))

    clv_by_market = {}
    for seg, vals in segments.items():
        arr = np.array(vals)
        clv_by_market[seg] = {
            "avg_clv": round(float(np.mean(arr)), 3),
            "beat_close_pct": round(float(np.mean(arr > 0)), 4),
            "n_bets": len(vals),
            "is_positive": float(np.mean(arr)) > 0,
        }

    # CLV distribution percentiles
    pcts = np.percentile(clv_cents, [5, 10, 25, 50, 75, 90, 95])

    # Attribution: market inefficiency edge share of ROI
    # CLV is the single most important metric in sports betting
    # If CLV > 0 and significant, this is the dominant edge source
    if is_positive and is_significant:
        # Strong CLV = major edge source (40-70% of total)
        market_pct = min(70.0, max(30.0, beat_close_rate * 100.0))
    elif is_positive:
        market_pct = min(25.0, beat_close_rate * 50.0)
    else:
        market_pct = 0.0

    # Probability-space CLV (if available)
    avg_clv_prob = round(float(np.mean(clv_prob_arr)), 6) if len(clv_prob_arr) > 0 else None

    # Build verdict
    verdict_parts = []
    if is_positive and is_significant:
        verdict_parts.append(
            f"REAL market inefficiency edge. Avg CLV = {avg_clv:+.2f} cents, "
            f"beat-close rate = {beat_close_rate:.1%}. p={p_value:.4f}. "
            f"This system is exploiting genuine pricing errors."
        )
    elif is_positive and not is_significant:
        verdict_parts.append(
            f"Positive CLV ({avg_clv:+.2f} cents) but NOT significant (p={p_value:.4f}). "
            f"Beat-close rate {beat_close_rate:.1%}. Could be noise — need more data."
        )
    else:
        verdict_parts.append(
            f"NO market inefficiency edge. Avg CLV = {avg_clv:+.2f} cents, "
            f"beat-close rate = {beat_close_rate:.1%}. "
            f"This system is NOT consistently beating the closing line. "
            f"Any profits are likely luck or survivorship bias."
        )

    # Flag best and worst market segments
    best_seg = max(clv_by_market.items(), key=lambda x: x[1]["avg_clv"]) if clv_by_market else None
    worst_seg = min(clv_by_market.items(), key=lambda x: x[1]["avg_clv"]) if clv_by_market else None
    if best_seg:
        verdict_parts.append(
            f"Best market: {best_seg[0]} ({best_seg[1]['avg_clv']:+.2f} cents). "
            f"Worst: {worst_seg[0]} ({worst_seg[1]['avg_clv']:+.2f} cents)."
        )

    return EdgeComponentResult(
        name="market_inefficiency",
        edge_pct_of_roi=round(market_pct, 2),
        absolute_value=round(avg_clv, 3),
        p_value=round(p_value, 4),
        is_significant=is_significant,
        is_positive=is_positive,
        sample_size=len(valid),
        details={
            "avg_clv_cents": round(avg_clv, 3),
            "median_clv_cents": round(median_clv, 3),
            "std_clv_cents": round(std_clv, 3),
            "beat_close_rate": round(beat_close_rate, 4),
            "avg_clv_probability": avg_clv_prob,
            "t_stat": round(float(t_stat), 3),
            "clv_percentiles": {
                "p5": round(float(pcts[0]), 3),
                "p10": round(float(pcts[1]), 3),
                "p25": round(float(pcts[2]), 3),
                "p50": round(float(pcts[3]), 3),
                "p75": round(float(pcts[4]), 3),
                "p90": round(float(pcts[5]), 3),
                "p95": round(float(pcts[6]), 3),
            },
            "clv_by_market": clv_by_market,
            "clv_distribution": clv_cents.tolist(),
        },
        verdict=" ".join(verdict_parts),
    )
