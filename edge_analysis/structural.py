"""
edge_analysis/structural.py
==============================
Component 5: STRUCTURAL EDGE — Portfolio construction, correlation, bankroll management.

Analyzes:
  - Bet correlation (are bets independent or clustered?)
  - Actual variance vs expected variance if bets were independent
  - Kelly criterion adherence and its impact on returns
  - Position sizing efficiency
  - Diversification across markets, players, and time
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.schemas import BetRecord, EdgeComponentResult

log = logging.getLogger(__name__)


def _compute_daily_returns(bets: List[BetRecord]) -> np.ndarray:
    """Aggregate bet P&L into daily returns."""
    daily: Dict[str, float] = defaultdict(float)
    for b in bets:
        if b.pnl is not None and b.pnl != 0:
            day_key = b.timestamp.strftime("%Y-%m-%d")
            daily[day_key] += b.pnl
    if not daily:
        return np.array([])
    return np.array([v for _, v in sorted(daily.items())])


def _compute_bet_correlation(bets: List[BetRecord]) -> Tuple[float, float]:
    """Estimate correlation between bet outcomes.

    Computes the actual variance of same-day returns vs expected variance
    if bets were independent Bernoulli trials.

    Returns:
        (correlation_estimate, variance_ratio)
        variance_ratio > 1 means bets are positively correlated (clustered)
        variance_ratio < 1 means bets are negatively correlated (diversified)
    """
    # Group bets by day
    daily_bets: Dict[str, List[BetRecord]] = defaultdict(list)
    for b in bets:
        if b.won is not None:
            day_key = b.timestamp.strftime("%Y-%m-%d")
            daily_bets[day_key].append(b)

    if len(daily_bets) < 10:
        return 0.0, 1.0

    # For each day with multiple bets, compute win correlation
    actual_variances = []
    expected_variances = []

    for day, day_bets in daily_bets.items():
        if len(day_bets) < 2:
            continue
        outcomes = np.array([1.0 if b.won else 0.0 for b in day_bets])
        n = len(outcomes)
        p = float(np.mean(outcomes))

        # Expected variance under independence: n * p * (1 - p)
        expected_var = n * p * (1.0 - p) if 0 < p < 1 else 0.0
        # Actual variance
        actual_var = float(np.var(outcomes) * n)

        if expected_var > 0:
            actual_variances.append(actual_var)
            expected_variances.append(expected_var)

    if not actual_variances:
        return 0.0, 1.0

    avg_actual = float(np.mean(actual_variances))
    avg_expected = float(np.mean(expected_variances))

    variance_ratio = avg_actual / avg_expected if avg_expected > 0 else 1.0

    # Estimate average pairwise correlation from variance ratio
    # Var(sum) = n*var + n*(n-1)*cov = n*p*(1-p) * (1 + (n-1)*rho)
    # So variance_ratio = 1 + (n-1)*rho approximately
    avg_n = float(np.mean([len(v) for v in daily_bets.values() if len(v) >= 2]))
    if avg_n > 1:
        rho = (variance_ratio - 1.0) / (avg_n - 1.0)
    else:
        rho = 0.0

    return float(np.clip(rho, -1.0, 1.0)), variance_ratio


def _kelly_analysis(bets: List[BetRecord]) -> Dict:
    """Analyze Kelly criterion adherence and its impact."""
    settled = [b for b in bets if b.won is not None and b.kelly_fraction > 0]
    if len(settled) < 10:
        return {"sufficient_data": False}

    kelly_fractions = np.array([b.kelly_fraction for b in settled])
    stakes = np.array([b.stake for b in settled])
    edges = np.array([b.predicted_prob - b.market_prob_at_bet for b in settled])
    outcomes = np.array([1.0 if b.won else 0.0 for b in settled])

    avg_kelly = float(np.mean(kelly_fractions))
    max_kelly = float(np.max(kelly_fractions))
    std_kelly = float(np.std(kelly_fractions))

    # Optimal Kelly would be edge / (odds - 1)
    # Check if actual sizing matches
    win_rate = float(np.mean(outcomes))
    avg_edge = float(np.mean(edges))

    # Compute growth rate: G = sum(log(1 + f * r_i)) / n
    # where f = fraction, r_i = return on bet i
    returns = []
    for b in settled:
        if b.won:
            r = b.odds_decimal - 1.0  # Net return per dollar
        else:
            r = -1.0
        growth = np.log(max(1e-10, 1.0 + b.kelly_fraction * r))
        returns.append(growth)

    growth_rate = float(np.mean(returns)) if returns else 0.0

    # Compare to full Kelly growth rate
    # Full Kelly: f* = edge / variance
    full_kelly_returns = []
    for b in settled:
        edge = b.predicted_prob - b.market_prob_at_bet
        if b.odds_decimal > 1:
            f_star = max(0, edge / (b.odds_decimal - 1.0))
        else:
            f_star = 0
        if b.won:
            r = b.odds_decimal - 1.0
        else:
            r = -1.0
        growth = np.log(max(1e-10, 1.0 + f_star * r))
        full_kelly_returns.append(growth)

    full_kelly_growth = float(np.mean(full_kelly_returns)) if full_kelly_returns else 0.0

    return {
        "sufficient_data": True,
        "avg_kelly_fraction": round(avg_kelly, 4),
        "max_kelly_fraction": round(max_kelly, 4),
        "std_kelly_fraction": round(std_kelly, 4),
        "actual_growth_rate": round(growth_rate, 6),
        "full_kelly_growth_rate": round(full_kelly_growth, 6),
        "kelly_efficiency": round(growth_rate / full_kelly_growth, 4) if full_kelly_growth > 0 else 0.0,
        "avg_edge": round(avg_edge, 4),
        "win_rate": round(win_rate, 4),
    }


def _diversification_analysis(bets: List[BetRecord]) -> Dict:
    """Analyze portfolio diversification across dimensions."""
    if not bets:
        return {}

    # Count by market type
    market_counts: Dict[str, int] = defaultdict(int)
    player_counts: Dict[str, int] = defaultdict(int)
    for b in bets:
        market_counts[b.market_type] += 1
        player_counts[b.player] += 1

    n_markets = len(market_counts)
    n_players = len(player_counts)
    total = len(bets)

    # Herfindahl index (concentration) — 0 = fully diversified, 1 = fully concentrated
    hhi_market = sum((c / total) ** 2 for c in market_counts.values()) if total > 0 else 1.0
    hhi_player = sum((c / total) ** 2 for c in player_counts.values()) if total > 0 else 1.0

    # Top concentration
    top_market = max(market_counts.items(), key=lambda x: x[1])
    top_player = max(player_counts.items(), key=lambda x: x[1])

    return {
        "n_unique_markets": n_markets,
        "n_unique_players": n_players,
        "hhi_market": round(hhi_market, 4),
        "hhi_player": round(hhi_player, 4),
        "top_market": {"name": top_market[0], "pct": round(top_market[1] / total, 4)},
        "top_player": {"name": top_player[0], "pct": round(top_player[1] / total, 4)},
        "market_distribution": {k: v for k, v in sorted(market_counts.items(), key=lambda x: -x[1])},
    }


def compute_structural_edge(
    bets: List[BetRecord],
    total_roi: float,
) -> EdgeComponentResult:
    """Analyze the structural (portfolio/bankroll) component of edge.

    Evaluates:
    1. Bet correlation — are bets independent or clustered?
    2. Variance ratio — actual vs expected if independent
    3. Kelly criterion adherence and growth rate
    4. Diversification across markets/players
    5. Overall portfolio construction quality

    Args:
        bets: List of BetRecord objects.
        total_roi: System's total ROI for attribution.

    Returns:
        EdgeComponentResult with structural edge analysis.
    """
    settled = [b for b in bets if b.won is not None]
    if len(settled) < 20:
        return EdgeComponentResult(
            name="structural",
            edge_pct_of_roi=0.0,
            absolute_value=0.0,
            p_value=1.0,
            is_significant=False,
            is_positive=False,
            sample_size=len(settled),
            verdict="Insufficient data for structural edge analysis (need 20+ settled bets)",
        )

    # 1. Bet correlation analysis
    rho, variance_ratio = _compute_bet_correlation(settled)

    # 2. Kelly analysis
    kelly = _kelly_analysis(settled)

    # 3. Diversification
    diversification = _diversification_analysis(settled)

    # 4. Daily return analysis
    daily_returns = _compute_daily_returns(settled)
    if len(daily_returns) >= 10:
        sharpe = (float(np.mean(daily_returns)) / float(np.std(daily_returns))
                  * np.sqrt(252) if np.std(daily_returns) > 0 else 0.0)
        max_drawdown = 0.0
        cumulative = np.cumsum(daily_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    else:
        sharpe = 0.0
        max_drawdown = 0.0

    # 5. Significance: compare actual variance to independence assumption
    # Under independence, variance_ratio should be ~1.0
    # Test if variance_ratio is significantly different from 1.0
    p_value = 0.5  # Default — correlation test needs bootstrap
    if len(daily_returns) >= 20:
        # Bootstrap test for variance ratio
        n_boot = 1000
        boot_ratios = []
        for _ in range(n_boot):
            idx = np.random.choice(len(daily_returns), size=len(daily_returns), replace=True)
            boot_var = float(np.var(daily_returns[idx]))
            orig_var = float(np.var(daily_returns))
            if orig_var > 0:
                boot_ratios.append(boot_var / orig_var)
        if boot_ratios:
            boot_arr = np.array(boot_ratios)
            p_value = float(np.mean(boot_arr > variance_ratio))

    is_positive = variance_ratio < 1.2 and (not kelly.get("sufficient_data") or kelly.get("kelly_efficiency", 0) > 0.3)
    is_significant = p_value < 0.05 or (kelly.get("sufficient_data") and kelly.get("kelly_efficiency", 0) > 0.5)

    # Attribution
    structural_pct = 0.0
    if is_positive:
        # Good structure adds 5-20% to edge
        if kelly.get("sufficient_data"):
            # Kelly efficiency drives attribution
            eff = kelly.get("kelly_efficiency", 0)
            structural_pct = min(20.0, eff * 25.0)
        else:
            structural_pct = 10.0 if variance_ratio < 1.1 else 5.0

        # Bonus for diversification
        hhi = diversification.get("hhi_market", 1.0)
        if hhi < 0.2:  # Well diversified
            structural_pct += 3.0
    else:
        # Bad structure = negative contribution
        if variance_ratio > 1.5:
            structural_pct = -10.0
        elif variance_ratio > 1.2:
            structural_pct = -5.0

    # Build verdict
    verdict_parts = []

    # Correlation verdict
    if abs(rho) < 0.05:
        verdict_parts.append(
            f"Bet independence: GOOD. Estimated pairwise correlation rho={rho:.3f}. "
            f"Variance ratio={variance_ratio:.2f} (1.0 = perfectly independent)."
        )
    elif rho > 0.05:
        verdict_parts.append(
            f"WARNING: Bets are positively correlated (rho={rho:.3f}). "
            f"Variance ratio={variance_ratio:.2f}. Portfolio has cluster risk — "
            f"wins and losses come in bunches. Diversify across markets/players."
        )
    else:
        verdict_parts.append(
            f"Bets are negatively correlated (rho={rho:.3f}). "
            f"Variance ratio={variance_ratio:.2f}. Good natural hedge in portfolio."
        )

    # Kelly verdict
    if kelly.get("sufficient_data"):
        eff = kelly.get("kelly_efficiency", 0)
        if eff > 0.7:
            verdict_parts.append(
                f"Kelly sizing: EXCELLENT. Efficiency={eff:.0%} of theoretical growth rate."
            )
        elif eff > 0.4:
            verdict_parts.append(
                f"Kelly sizing: ACCEPTABLE. Efficiency={eff:.0%}. Room for improvement."
            )
        else:
            verdict_parts.append(
                f"Kelly sizing: POOR. Efficiency={eff:.0%}. Sizing is suboptimal — "
                f"either over-betting or under-betting relative to edge."
            )

    # Diversification verdict
    hhi = diversification.get("hhi_market", 1.0)
    if hhi < 0.15:
        verdict_parts.append("Diversification: EXCELLENT across market types.")
    elif hhi < 0.30:
        verdict_parts.append("Diversification: ACCEPTABLE.")
    else:
        top = diversification.get("top_market", {})
        verdict_parts.append(
            f"Diversification: POOR. {top.get('name', '?')} accounts for "
            f"{top.get('pct', 0):.0%} of all bets. Concentration risk."
        )

    return EdgeComponentResult(
        name="structural",
        edge_pct_of_roi=round(structural_pct, 2),
        absolute_value=round(variance_ratio, 4),
        p_value=round(p_value, 4),
        is_significant=is_significant,
        is_positive=is_positive,
        sample_size=len(settled),
        details={
            "correlation_rho": round(rho, 4),
            "variance_ratio": round(variance_ratio, 4),
            "annualized_sharpe": round(sharpe, 3),
            "max_drawdown": round(max_drawdown, 2),
            "kelly": kelly,
            "diversification": diversification,
        },
        verdict=" ".join(verdict_parts),
    )
