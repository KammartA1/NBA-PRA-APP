"""
services/market_reaction/book_behavior.py
==========================================
Models how sportsbooks track, classify, and ultimately limit winning bettors.

Books maintain internal profiles that track:
  - Lifetime win rate and CLV captured
  - Bet frequency and market diversity
  - Steam-move correlation (are you betting right before big line moves?)
  - Account age and verification level

This module simulates the book's classification algorithm and predicts
when a bettor will transition from unrestricted to limited to banned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BettorProfile:
    """The book's internal view of a bettor."""
    account_id: str = "default"
    total_bets: int = 0
    total_handle: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.50
    avg_clv_cents: float = 0.0
    clv_beat_rate: float = 0.50
    steam_correlation: float = 0.0
    market_diversity: float = 1.0    # 1.0 = all in one market, 0.0 = spread
    avg_bet_size: float = 50.0
    max_bet_size: float = 100.0
    account_age_days: int = 0
    bet_frequency_per_day: float = 0.0
    sharp_score: float = 0.0        # 0-100 internal sharpness score


@dataclass
class BookProfile:
    """Configuration for a specific sportsbook's tolerance."""
    name: str = "generic"
    # CLV thresholds for classification
    clv_yellow_threshold: float = 0.5    # Avg CLV > 0.5 cents → watched
    clv_red_threshold: float = 1.5       # Avg CLV > 1.5 cents → restricted
    clv_ban_threshold: float = 3.0       # Avg CLV > 3.0 cents → banned
    # Win rate thresholds
    win_rate_yellow: float = 0.54
    win_rate_red: float = 0.57
    win_rate_ban: float = 0.60
    # Bet count thresholds (books need sample size to act)
    min_bets_to_flag: int = 50
    min_bets_to_restrict: int = 150
    min_bets_to_ban: int = 300
    # How aggressive the book is (0-1, higher = faster to limit)
    aggressiveness: float = 0.5
    # Max allowed bet at each stage
    max_bet_unrestricted: float = 500.0
    max_bet_watched: float = 200.0
    max_bet_restricted: float = 50.0
    max_bet_banned: float = 0.0
    # Steam correlation sensitivity
    steam_sensitivity: float = 0.3


BOOK_PROFILES: Dict[str, BookProfile] = {
    "draftkings": BookProfile(
        name="draftkings",
        clv_yellow_threshold=0.8,
        clv_red_threshold=2.0,
        clv_ban_threshold=4.0,
        min_bets_to_flag=100,
        min_bets_to_restrict=250,
        min_bets_to_ban=500,
        aggressiveness=0.35,
        max_bet_unrestricted=1000.0,
        max_bet_watched=500.0,
        max_bet_restricted=100.0,
    ),
    "fanduel": BookProfile(
        name="fanduel",
        clv_yellow_threshold=0.6,
        clv_red_threshold=1.5,
        clv_ban_threshold=3.0,
        min_bets_to_flag=80,
        min_bets_to_restrict=200,
        min_bets_to_ban=400,
        aggressiveness=0.45,
        max_bet_unrestricted=750.0,
        max_bet_watched=300.0,
        max_bet_restricted=50.0,
    ),
    "betmgm": BookProfile(
        name="betmgm",
        clv_yellow_threshold=0.5,
        clv_red_threshold=1.0,
        clv_ban_threshold=2.0,
        min_bets_to_flag=50,
        min_bets_to_restrict=100,
        min_bets_to_ban=200,
        aggressiveness=0.70,
        max_bet_unrestricted=500.0,
        max_bet_watched=100.0,
        max_bet_restricted=25.0,
    ),
    "caesars": BookProfile(
        name="caesars",
        clv_yellow_threshold=0.4,
        clv_red_threshold=1.0,
        clv_ban_threshold=2.5,
        min_bets_to_flag=40,
        min_bets_to_restrict=100,
        min_bets_to_ban=200,
        aggressiveness=0.65,
        max_bet_unrestricted=500.0,
        max_bet_watched=150.0,
        max_bet_restricted=25.0,
    ),
    "pinnacle": BookProfile(
        name="pinnacle",
        clv_yellow_threshold=2.0,
        clv_red_threshold=5.0,
        clv_ban_threshold=100.0,  # Pinnacle basically never bans
        min_bets_to_flag=500,
        min_bets_to_restrict=2000,
        min_bets_to_ban=99999,
        aggressiveness=0.05,
        max_bet_unrestricted=5000.0,
        max_bet_watched=3000.0,
        max_bet_restricted=1000.0,
    ),
}


class BookBehaviorModel:
    """Simulates how a sportsbook classifies and reacts to a bettor.

    Central method: ``classify(bettor_profile, book_profile)`` returns the
    book's current classification and predicted next actions.
    """

    # Classification states
    UNRESTRICTED = "unrestricted"
    WATCHED = "watched"
    RESTRICTED = "restricted"
    BANNED = "banned"

    def __init__(self):
        self.book_profiles = dict(BOOK_PROFILES)

    def get_book_profile(self, book_name: str) -> BookProfile:
        """Get profile for a known book, or return a generic one."""
        return self.book_profiles.get(
            book_name.lower(),
            BookProfile(name=book_name.lower()),
        )

    def compute_sharp_score(self, bettor: BettorProfile) -> float:
        """Compute the book's internal sharpness score (0-100).

        Higher = sharper = more likely to be limited.
        """
        score = 0.0

        # CLV component (0-40 points)
        if bettor.avg_clv_cents > 0:
            clv_score = min(bettor.avg_clv_cents / 5.0, 1.0) * 40.0
            score += clv_score

        # Win rate component (0-25 points)
        if bettor.total_bets >= 30:
            wr_excess = max(bettor.win_rate - 0.50, 0.0)
            wr_score = min(wr_excess / 0.10, 1.0) * 25.0
            score += wr_score

        # CLV beat rate component (0-20 points)
        if bettor.total_bets >= 30:
            beat_excess = max(bettor.clv_beat_rate - 0.50, 0.0)
            beat_score = min(beat_excess / 0.20, 1.0) * 20.0
            score += beat_score

        # Steam correlation component (0-10 points)
        steam_score = min(max(bettor.steam_correlation, 0.0), 1.0) * 10.0
        score += steam_score

        # Market concentration penalty (0-5 points)
        # Betting only one market = suspicious
        conc_score = bettor.market_diversity * 5.0
        score += conc_score

        return round(min(score, 100.0), 1)

    def classify(self, bettor: BettorProfile, book: BookProfile) -> Dict[str, Any]:
        """Classify a bettor from the book's perspective.

        Returns:
            Dict with classification, max_bet, sharp_score, and reasoning.
        """
        sharp_score = self.compute_sharp_score(bettor)
        bettor.sharp_score = sharp_score

        classification = self.UNRESTRICTED
        reasons: List[str] = []

        # Not enough data yet -- stay unrestricted
        if bettor.total_bets < book.min_bets_to_flag:
            return {
                "classification": self.UNRESTRICTED,
                "sharp_score": sharp_score,
                "max_bet": book.max_bet_unrestricted,
                "reasons": ["insufficient sample size for classification"],
                "bets_to_next_review": book.min_bets_to_flag - bettor.total_bets,
                "risk_level": "low",
            }

        # Composite trigger score (0-1, higher = closer to limit)
        trigger_score = 0.0

        # CLV-based triggers
        if bettor.avg_clv_cents >= book.clv_ban_threshold:
            trigger_score = max(trigger_score, 1.0)
            reasons.append(f"CLV {bettor.avg_clv_cents:.1f}c >= ban threshold {book.clv_ban_threshold:.1f}c")
        elif bettor.avg_clv_cents >= book.clv_red_threshold:
            trigger_score = max(trigger_score, 0.7)
            reasons.append(f"CLV {bettor.avg_clv_cents:.1f}c >= restrict threshold {book.clv_red_threshold:.1f}c")
        elif bettor.avg_clv_cents >= book.clv_yellow_threshold:
            trigger_score = max(trigger_score, 0.3)
            reasons.append(f"CLV {bettor.avg_clv_cents:.1f}c >= watch threshold {book.clv_yellow_threshold:.1f}c")

        # Win rate triggers
        if bettor.win_rate >= book.win_rate_ban and bettor.total_bets >= book.min_bets_to_restrict:
            trigger_score = max(trigger_score, 0.9)
            reasons.append(f"win rate {bettor.win_rate:.1%} >= ban threshold {book.win_rate_ban:.1%}")
        elif bettor.win_rate >= book.win_rate_red and bettor.total_bets >= book.min_bets_to_flag:
            trigger_score = max(trigger_score, 0.6)
            reasons.append(f"win rate {bettor.win_rate:.1%} >= restrict threshold {book.win_rate_red:.1%}")
        elif bettor.win_rate >= book.win_rate_yellow:
            trigger_score = max(trigger_score, 0.25)
            reasons.append(f"win rate {bettor.win_rate:.1%} >= watch threshold {book.win_rate_yellow:.1%}")

        # Steam correlation
        if bettor.steam_correlation > book.steam_sensitivity:
            trigger_score += 0.15
            reasons.append(f"steam correlation {bettor.steam_correlation:.2f} > threshold {book.steam_sensitivity:.2f}")

        # Apply aggressiveness multiplier
        effective_trigger = trigger_score * (0.5 + book.aggressiveness)

        # Determine classification based on effective trigger and bet count
        if effective_trigger >= 0.8 and bettor.total_bets >= book.min_bets_to_ban:
            classification = self.BANNED
        elif effective_trigger >= 0.5 and bettor.total_bets >= book.min_bets_to_restrict:
            classification = self.RESTRICTED
        elif effective_trigger >= 0.2 and bettor.total_bets >= book.min_bets_to_flag:
            classification = self.WATCHED
        else:
            classification = self.UNRESTRICTED

        # Max bet based on classification
        max_bet_map = {
            self.UNRESTRICTED: book.max_bet_unrestricted,
            self.WATCHED: book.max_bet_watched,
            self.RESTRICTED: book.max_bet_restricted,
            self.BANNED: book.max_bet_banned,
        }
        max_bet = max_bet_map[classification]

        # Risk level
        risk_map = {
            self.UNRESTRICTED: "low",
            self.WATCHED: "medium",
            self.RESTRICTED: "high",
            self.BANNED: "critical",
        }

        # Estimate bets until next stage
        if classification == self.UNRESTRICTED:
            bets_to_next = max(book.min_bets_to_flag - bettor.total_bets, 0)
        elif classification == self.WATCHED:
            bets_to_next = max(book.min_bets_to_restrict - bettor.total_bets, 0)
        elif classification == self.RESTRICTED:
            bets_to_next = max(book.min_bets_to_ban - bettor.total_bets, 0)
        else:
            bets_to_next = 0

        return {
            "classification": classification,
            "sharp_score": sharp_score,
            "max_bet": max_bet,
            "trigger_score": round(effective_trigger, 3),
            "reasons": reasons if reasons else ["within normal parameters"],
            "bets_to_next_review": bets_to_next,
            "risk_level": risk_map[classification],
        }

    def classify_all_books(self, bettor: BettorProfile) -> Dict[str, Dict[str, Any]]:
        """Classify bettor across all known sportsbooks."""
        results = {}
        for name, profile in self.book_profiles.items():
            results[name] = self.classify(bettor, profile)
        return results

    def build_profile_from_bets(self, bets: List[Dict[str, Any]]) -> BettorProfile:
        """Build a BettorProfile from a list of bet dictionaries.

        Each bet dict should have: stake, profit, predicted_prob, closing_line,
        bet_line, direction, market, timestamp.
        """
        if not bets:
            return BettorProfile()

        total_bets = len(bets)
        total_handle = sum(b.get("stake", 0.0) for b in bets)
        total_pnl = sum(b.get("profit", 0.0) or 0.0 for b in bets)

        # Win rate
        settled = [b for b in bets if b.get("profit") is not None]
        wins = sum(1 for b in settled if (b.get("profit", 0.0) or 0.0) > 0)
        win_rate = wins / max(len(settled), 1)

        # CLV analysis
        clv_values = []
        for b in bets:
            bl = b.get("bet_line")
            cl = b.get("closing_line")
            direction = b.get("direction", "over")
            if bl is not None and cl is not None:
                if direction.lower() == "over":
                    clv = cl - bl
                else:
                    clv = bl - cl
                clv_values.append(clv)

        avg_clv = float(np.mean(clv_values)) if clv_values else 0.0
        clv_beat = sum(1 for c in clv_values if c > 0) / max(len(clv_values), 1)

        # Market diversity (entropy-based)
        markets = [b.get("market", "unknown") for b in bets]
        unique_markets = set(markets)
        if len(unique_markets) <= 1:
            market_div = 1.0
        else:
            from collections import Counter
            counts = Counter(markets)
            probs = np.array(list(counts.values()), dtype=float) / total_bets
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(unique_markets))
            market_div = 1.0 - (entropy / max(max_entropy, 1e-10))

        # Bet frequency
        timestamps = sorted([b.get("timestamp") for b in bets if b.get("timestamp")])
        if len(timestamps) >= 2:
            first = timestamps[0]
            last = timestamps[-1]
            if isinstance(first, str):
                from datetime import datetime as dt
                first = dt.fromisoformat(first)
                last = dt.fromisoformat(last)
            span_days = max((last - first).total_seconds() / 86400, 1.0)
            freq = total_bets / span_days
            age_days = int(span_days)
        else:
            freq = 0.0
            age_days = 0

        stakes = [b.get("stake", 0.0) for b in bets]

        return BettorProfile(
            total_bets=total_bets,
            total_handle=total_handle,
            total_pnl=total_pnl,
            win_rate=win_rate,
            avg_clv_cents=avg_clv,
            clv_beat_rate=clv_beat,
            steam_correlation=0.0,  # Would require line movement data
            market_diversity=market_div,
            avg_bet_size=float(np.mean(stakes)) if stakes else 0.0,
            max_bet_size=float(np.max(stakes)) if stakes else 0.0,
            account_age_days=age_days,
            bet_frequency_per_day=freq,
        )
