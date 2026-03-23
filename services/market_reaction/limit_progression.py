"""
services/market_reaction/limit_progression.py
==============================================
Simulates the progression of betting limits over time as books identify
and react to a sharp bettor.

Models the timeline from unrestricted → watched → limited → banned, with
probabilistic estimates of when each transition occurs and what max bet
sizes will be available at each stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.market_reaction.book_behavior import (
    BookBehaviorModel,
    BookProfile,
    BettorProfile,
    BOOK_PROFILES,
)

log = logging.getLogger(__name__)


@dataclass
class LimitStage:
    """A single stage in the limit progression timeline."""
    stage_name: str              # unrestricted, watched, restricted, banned
    entry_bet_number: int        # Estimated bet number when this stage starts
    entry_month: float           # Estimated month when this stage starts
    max_bet: float               # Maximum bet size at this stage
    expected_duration_bets: int  # How many bets this stage lasts
    expected_duration_months: float
    effective_edge_pct: float    # Edge after shading/limits at this stage
    monthly_ev: float            # Expected monthly profit at this stage
    cumulative_ev: float         # Cumulative profit up to end of this stage


@dataclass
class LimitTimeline:
    """Full limit progression timeline for a single book."""
    book_name: str
    stages: List[LimitStage]
    total_profitable_months: float
    total_lifetime_ev: float
    months_to_full_limit: float
    months_to_ban: float
    optimal_exit_month: float    # When to stop betting this book


@dataclass
class LimitProgressionConfig:
    """Configuration for the limit progression simulation."""
    bets_per_day: float = 3.0
    avg_edge_pct: float = 4.0       # Starting true edge %
    avg_bet_size: float = 100.0     # Starting average bet size
    avg_odds_decimal: float = 1.91  # Approx -110
    shading_per_stage: float = 1.0  # Additional cents of shading per stage
    months_to_simulate: int = 24


class LimitProgressionModel:
    """Simulates limit progression across sportsbooks.

    For each book, models the expected timeline of restrictions and computes
    the total extractable value before being shut down.
    """

    def __init__(self, config: LimitProgressionConfig | None = None):
        self.config = config or LimitProgressionConfig()
        self.behavior_model = BookBehaviorModel()

    def simulate_book(
        self,
        book_name: str,
        bettor: BettorProfile | None = None,
        config: LimitProgressionConfig | None = None,
    ) -> LimitTimeline:
        """Simulate limit progression for a single book.

        Returns a LimitTimeline with all stages and profitability metrics.
        """
        cfg = config or self.config
        book = self.behavior_model.get_book_profile(book_name)
        bets_per_month = cfg.bets_per_day * 30.0

        stages: List[LimitStage] = []
        cumulative_ev = 0.0

        # Stage definitions with their characteristics
        stage_defs = [
            ("unrestricted", book.max_bet_unrestricted, book.min_bets_to_flag, 0.0),
            ("watched", book.max_bet_watched, book.min_bets_to_restrict - book.min_bets_to_flag, 1.0),
            ("restricted", book.max_bet_restricted, book.min_bets_to_ban - book.min_bets_to_restrict, 2.0),
            ("banned", book.max_bet_banned, 0, 3.0),
        ]

        current_bet = 0
        current_month = 0.0

        for stage_name, max_bet, duration_bets, shading_mult in stage_defs:
            if max_bet <= 0 and stage_name != "banned":
                continue

            # Edge erosion at this stage
            shading_cost = shading_mult * cfg.shading_per_stage
            effective_edge = max(cfg.avg_edge_pct - shading_cost, 0.0)

            # Effective bet size (capped by limit)
            effective_bet = min(cfg.avg_bet_size, max_bet) if max_bet > 0 else 0.0

            # Duration
            if stage_name == "banned":
                duration_bets = 0
                duration_months = 0.0
            elif duration_bets <= 0:
                duration_bets = int(bets_per_month * 6)  # Default 6 months
                duration_months = 6.0
            else:
                # Add randomness: book aggressiveness affects speed
                speed_mult = 0.5 + book.aggressiveness
                adjusted_duration = int(duration_bets / speed_mult)
                duration_bets = max(adjusted_duration, 1)
                duration_months = duration_bets / max(bets_per_month, 1.0)

            # Monthly EV at this stage
            if effective_bet > 0 and effective_edge > 0 and stage_name != "banned":
                ev_per_bet = effective_bet * (effective_edge / 100.0)
                monthly_ev = ev_per_bet * bets_per_month
            else:
                monthly_ev = 0.0

            stage_total_ev = monthly_ev * duration_months
            cumulative_ev += stage_total_ev

            stages.append(LimitStage(
                stage_name=stage_name,
                entry_bet_number=current_bet,
                entry_month=round(current_month, 1),
                max_bet=max_bet,
                expected_duration_bets=duration_bets,
                expected_duration_months=round(duration_months, 1),
                effective_edge_pct=round(effective_edge, 2),
                monthly_ev=round(monthly_ev, 2),
                cumulative_ev=round(cumulative_ev, 2),
            ))

            current_bet += duration_bets
            current_month += duration_months

        # Compute timeline summary
        profitable_months = sum(
            s.expected_duration_months for s in stages if s.monthly_ev > 0
        )
        months_to_restrict = sum(
            s.expected_duration_months for s in stages
            if s.stage_name in ("unrestricted", "watched")
        )
        months_to_ban = sum(
            s.expected_duration_months for s in stages
            if s.stage_name != "banned"
        )

        # Optimal exit: when marginal EV per month drops below threshold
        optimal_exit = months_to_ban
        for s in reversed(stages):
            if s.monthly_ev > 50:  # Worth more than $50/month
                optimal_exit = s.entry_month + s.expected_duration_months
                break

        return LimitTimeline(
            book_name=book_name,
            stages=stages,
            total_profitable_months=round(profitable_months, 1),
            total_lifetime_ev=round(cumulative_ev, 2),
            months_to_full_limit=round(months_to_restrict, 1),
            months_to_ban=round(months_to_ban, 1),
            optimal_exit_month=round(optimal_exit, 1),
        )

    def simulate_all_books(
        self,
        config: LimitProgressionConfig | None = None,
    ) -> Dict[str, LimitTimeline]:
        """Simulate limit progression across all known books."""
        cfg = config or self.config
        results = {}
        for name in BOOK_PROFILES:
            results[name] = self.simulate_book(name, config=cfg)
        return results

    def optimal_rotation_strategy(
        self,
        config: LimitProgressionConfig | None = None,
    ) -> Dict[str, Any]:
        """Compute the optimal book rotation strategy to maximize lifetime EV.

        Instead of hammering one book, rotate across books to delay limits.
        """
        cfg = config or self.config
        timelines = self.simulate_all_books(config=cfg)

        # Sort books by total lifetime EV (highest first)
        ranked = sorted(
            timelines.items(),
            key=lambda x: x[1].total_lifetime_ev,
            reverse=True,
        )

        # Simple rotation: distribute bets proportionally to remaining capacity
        total_ev = sum(t.total_lifetime_ev for _, t in ranked)
        total_months = max(t.months_to_ban for _, t in ranked) if ranked else 0

        rotation_plan: List[Dict[str, Any]] = []
        for name, timeline in ranked:
            if timeline.total_lifetime_ev <= 0:
                continue
            pct_allocation = timeline.total_lifetime_ev / max(total_ev, 1.0)
            rotation_plan.append({
                "book": name,
                "allocation_pct": round(pct_allocation * 100, 1),
                "lifetime_ev": timeline.total_lifetime_ev,
                "months_active": timeline.total_profitable_months,
                "optimal_exit": timeline.optimal_exit_month,
            })

        # Staggered start recommendation
        if len(rotation_plan) >= 2:
            stagger_months = total_months / len(rotation_plan)
        else:
            stagger_months = 0

        return {
            "rotation_plan": rotation_plan,
            "total_portfolio_ev": round(total_ev, 2),
            "total_active_months": round(total_months, 1),
            "recommended_stagger_months": round(stagger_months, 1),
            "strategy": (
                "rotate" if len(rotation_plan) >= 3
                else "concentrate" if len(rotation_plan) == 1
                else "split"
            ),
        }

    def monte_carlo_limits(
        self,
        book_name: str,
        n_sims: int = 1000,
        config: LimitProgressionConfig | None = None,
    ) -> Dict[str, Any]:
        """Monte Carlo simulation of limit timelines with randomized book behavior.

        Books don't act deterministically — some accounts get flagged early,
        others fly under the radar. This models the distribution of outcomes.
        """
        cfg = config or self.config
        book = self.behavior_model.get_book_profile(book_name)
        bets_per_month = cfg.bets_per_day * 30.0

        months_to_limit = []
        lifetime_evs = []
        months_to_ban_list = []

        rng = np.random.default_rng(42)

        for _ in range(n_sims):
            # Randomize book aggressiveness (some reviewers are stricter)
            agg_noise = rng.normal(0, 0.15)
            effective_agg = np.clip(book.aggressiveness + agg_noise, 0.05, 0.95)

            # Randomize detection speed
            detect_noise = rng.uniform(0.5, 1.5)
            effective_flag_bets = int(book.min_bets_to_flag * detect_noise)
            effective_restrict_bets = int(book.min_bets_to_restrict * detect_noise)
            effective_ban_bets = int(book.min_bets_to_ban * detect_noise)

            # Simulate progression
            flag_month = effective_flag_bets / max(bets_per_month, 1)
            restrict_month = effective_restrict_bets / max(bets_per_month, 1)
            ban_month = effective_ban_bets / max(bets_per_month, 1)

            # EV before restriction
            unrestricted_ev = (
                min(cfg.avg_bet_size, book.max_bet_unrestricted)
                * (cfg.avg_edge_pct / 100.0)
                * bets_per_month
                * flag_month
            )
            watched_ev = (
                min(cfg.avg_bet_size, book.max_bet_watched)
                * (max(cfg.avg_edge_pct - 1.0, 0) / 100.0)
                * bets_per_month
                * (restrict_month - flag_month)
            )
            restricted_ev = (
                min(cfg.avg_bet_size, book.max_bet_restricted)
                * (max(cfg.avg_edge_pct - 2.0, 0) / 100.0)
                * bets_per_month
                * (ban_month - restrict_month)
            )

            total_ev = unrestricted_ev + watched_ev + restricted_ev

            months_to_limit.append(restrict_month)
            months_to_ban_list.append(ban_month)
            lifetime_evs.append(total_ev)

        months_arr = np.array(months_to_limit)
        ev_arr = np.array(lifetime_evs)
        ban_arr = np.array(months_to_ban_list)

        return {
            "book": book_name,
            "n_sims": n_sims,
            "months_to_limit": {
                "mean": round(float(np.mean(months_arr)), 1),
                "median": round(float(np.median(months_arr)), 1),
                "p10": round(float(np.percentile(months_arr, 10)), 1),
                "p25": round(float(np.percentile(months_arr, 25)), 1),
                "p75": round(float(np.percentile(months_arr, 75)), 1),
                "p90": round(float(np.percentile(months_arr, 90)), 1),
            },
            "months_to_ban": {
                "mean": round(float(np.mean(ban_arr)), 1),
                "median": round(float(np.median(ban_arr)), 1),
                "p10": round(float(np.percentile(ban_arr, 10)), 1),
                "p90": round(float(np.percentile(ban_arr, 90)), 1),
            },
            "lifetime_ev": {
                "mean": round(float(np.mean(ev_arr)), 2),
                "median": round(float(np.median(ev_arr)), 2),
                "p10": round(float(np.percentile(ev_arr, 10)), 2),
                "p90": round(float(np.percentile(ev_arr, 90)), 2),
                "std": round(float(np.std(ev_arr)), 2),
            },
            "survival_distribution": {
                "months": list(range(1, 25)),
                "survival_pct": [
                    round(float(np.mean(ban_arr > m) * 100), 1) for m in range(1, 25)
                ],
            },
        }
