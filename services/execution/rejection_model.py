"""
services/execution/rejection_model.py
======================================
Models bet rejection rates and the selection bias they create.

Some bets get rejected (odds changed, market suspended). The critical insight:
rejection bias means the BEST bets are more likely to be rejected, creating
a selection bias that inflates perceived edge in track records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RejectionObservation:
    """A single observation of bet acceptance/rejection."""
    bet_id: str
    sportsbook: str
    was_rejected: bool
    rejection_reason: str           # "odds_changed", "market_suspended", "limit_exceeded", "timeout"
    signal_edge_cents: float        # Edge at time of signal
    market_type: str = "points"
    timestamp: Optional[datetime] = None
    attempted_stake: float = 0.0


@dataclass
class RejectionResult:
    """Prediction of rejection probability for a new bet."""
    p_rejection: float              # P(bet gets rejected)
    p_acceptance: float             # P(bet goes through)
    rejection_bias_factor: float    # How much sharper rejected bets are vs accepted
    effective_edge_after_bias: float  # Edge adjusted for selection bias
    expected_attempts_needed: float   # Average attempts to place this bet


@dataclass
class RejectionProfile:
    """Aggregate rejection statistics across all bets."""
    n_attempted: int
    n_rejected: int
    n_accepted: int
    overall_rejection_rate: float
    rejection_rate_by_book: Dict[str, dict]
    rejection_rate_by_reason: Dict[str, float]
    rejection_rate_by_edge_bucket: Dict[str, dict]
    # Selection bias
    mean_edge_accepted: float       # Average edge of accepted bets
    mean_edge_rejected: float       # Average edge of rejected bets
    selection_bias_ratio: float     # rejected_edge / accepted_edge
    inflated_roi_pct: float         # How much ROI is inflated by rejection bias
    true_adjusted_roi_pct: float    # ROI after correcting for rejection bias


class RejectionModel:
    """Models sportsbook rejection behavior and resulting selection bias.

    Key insight: sportsbooks don't reject bets randomly. They preferentially
    reject bets with the most edge (sharp action). This means your track record
    of PLACED bets looks better than your actual signal quality — because the
    sharper signals got rejected.

    This creates a dangerous illusion: you think your edge is X%, but the bets
    that actually went through had less edge than the ones that didn't.
    """

    REJECTION_REASONS = [
        "odds_changed",
        "market_suspended",
        "limit_exceeded",
        "timeout",
        "manual_review",
        "account_restricted",
    ]

    # Default rejection rates by book (empirical)
    DEFAULT_BOOK_REJECTION_RATES = {
        "pinnacle": 0.02,
        "betrivers": 0.08,
        "draftkings": 0.05,
        "fanduel": 0.05,
        "mgm": 0.10,
        "caesars": 0.12,
        "pointsbet": 0.15,
        "prizepicks": 0.03,
        "underdog": 0.03,
    }

    def __init__(self, sport: str = "nba"):
        self.sport = sport
        self._observations: List[RejectionObservation] = []
        self._edge_rejection_correlation: float = 0.0
        self._fitted = False

    def load_observations(self, observations: List[RejectionObservation]) -> None:
        """Load historical rejection observations."""
        self._observations = observations
        if observations:
            self._fit()
        log.info("RejectionModel loaded %d observations", len(observations))

    def load_from_bets(self, bets: list, rejected_bets: Optional[list] = None) -> None:
        """Load from bet records. rejected_bets is a list of attempted but rejected bets."""
        observations = []

        # Accepted bets
        for b in bets:
            if hasattr(b, "bet_id"):
                bid = b.bet_id
                book = getattr(b, "sportsbook", "unknown")
                edge = getattr(b, "predicted_prob", 0.5) - getattr(b, "market_prob_at_bet", 0.5)
                market = getattr(b, "market_type", "points")
                ts = getattr(b, "timestamp", None)
                stake = getattr(b, "stake", 0)
            elif isinstance(b, dict):
                bid = b.get("bet_id", "")
                book = b.get("sportsbook", "unknown")
                edge = b.get("predicted_prob", 0.5) - b.get("market_prob_at_bet", 0.5)
                market = b.get("market_type", "points")
                ts = b.get("timestamp")
                stake = b.get("stake", 0)
            else:
                continue

            observations.append(RejectionObservation(
                bet_id=str(bid),
                sportsbook=book,
                was_rejected=False,
                rejection_reason="",
                signal_edge_cents=edge * 100,
                market_type=market,
                timestamp=ts,
                attempted_stake=stake,
            ))

        # Rejected bets
        if rejected_bets:
            for b in rejected_bets:
                if hasattr(b, "bet_id"):
                    bid = b.bet_id
                    book = getattr(b, "sportsbook", "unknown")
                    edge = getattr(b, "predicted_prob", 0.5) - getattr(b, "market_prob_at_bet", 0.5)
                    market = getattr(b, "market_type", "points")
                    reason = getattr(b, "rejection_reason", "odds_changed")
                    ts = getattr(b, "timestamp", None)
                    stake = getattr(b, "stake", 0)
                elif isinstance(b, dict):
                    bid = b.get("bet_id", "")
                    book = b.get("sportsbook", "unknown")
                    edge = b.get("predicted_prob", 0.5) - b.get("market_prob_at_bet", 0.5)
                    market = b.get("market_type", "points")
                    reason = b.get("rejection_reason", "odds_changed")
                    ts = b.get("timestamp")
                    stake = b.get("stake", 0)
                else:
                    continue

                observations.append(RejectionObservation(
                    bet_id=str(bid),
                    sportsbook=book,
                    was_rejected=True,
                    rejection_reason=reason,
                    signal_edge_cents=edge * 100,
                    market_type=market,
                    timestamp=ts,
                    attempted_stake=stake,
                ))

        self.load_observations(observations)

    def _fit(self) -> None:
        """Fit rejection model — primarily compute edge-rejection correlation."""
        if len(self._observations) < 5:
            self._fitted = True
            return

        edges = np.array([o.signal_edge_cents for o in self._observations])
        rejected = np.array([1.0 if o.was_rejected else 0.0 for o in self._observations])

        # Point-biserial correlation between edge and rejection
        if np.std(edges) > 0 and np.std(rejected) > 0:
            self._edge_rejection_correlation = float(np.corrcoef(edges, rejected)[0, 1])
        else:
            self._edge_rejection_correlation = 0.0

        self._fitted = True
        log.info(
            "RejectionModel fitted: edge-rejection correlation = %.4f",
            self._edge_rejection_correlation,
        )

    def predict(
        self,
        sportsbook: str = "unknown",
        signal_edge_cents: float = 2.0,
        market_type: str = "points",
    ) -> RejectionResult:
        """Predict rejection probability for a new bet attempt."""
        book = sportsbook.lower()

        # Base rejection rate from empirical data or defaults
        if self._observations:
            book_obs = [o for o in self._observations if o.sportsbook.lower() == book]
            if book_obs:
                base_rate = sum(1 for o in book_obs if o.was_rejected) / len(book_obs)
            else:
                base_rate = self.DEFAULT_BOOK_REJECTION_RATES.get(book, 0.05)
        else:
            base_rate = self.DEFAULT_BOOK_REJECTION_RATES.get(book, 0.05)

        # Adjust for edge: higher-edge bets get rejected more
        # P(reject) = base_rate * (1 + correlation * edge_z_score)
        edge_adjustment = 1.0
        if self._edge_rejection_correlation > 0 and self._observations:
            edges = [o.signal_edge_cents for o in self._observations]
            mean_edge = float(np.mean(edges))
            std_edge = float(np.std(edges)) if len(edges) > 1 else 1.0
            if std_edge > 0:
                z = (signal_edge_cents - mean_edge) / std_edge
                edge_adjustment = 1.0 + self._edge_rejection_correlation * z

        p_rejection = min(0.95, max(0.01, base_rate * edge_adjustment))
        p_acceptance = 1.0 - p_rejection

        # Selection bias factor
        if self._observations:
            accepted = [o.signal_edge_cents for o in self._observations if not o.was_rejected]
            rejected_obs = [o.signal_edge_cents for o in self._observations if o.was_rejected]
            mean_accepted = float(np.mean(accepted)) if accepted else 0.0
            mean_rejected = float(np.mean(rejected_obs)) if rejected_obs else 0.0
            bias_factor = (mean_rejected / mean_accepted) if mean_accepted > 0 else 1.0
        else:
            bias_factor = 1.15  # Default: rejected bets are 15% sharper

        # Adjust edge for bias
        effective_edge = signal_edge_cents * p_acceptance

        # Expected attempts to get a bet through
        expected_attempts = 1.0 / p_acceptance if p_acceptance > 0 else float("inf")

        return RejectionResult(
            p_rejection=p_rejection,
            p_acceptance=p_acceptance,
            rejection_bias_factor=bias_factor,
            effective_edge_after_bias=effective_edge,
            expected_attempts_needed=expected_attempts,
        )

    def compute_selection_bias(self, track_record_roi: float) -> dict:
        """Compute how much selection bias inflates the track record ROI.

        If rejected bets are systematically sharper, the PLACED bets look
        worse than the full signal set. But conversely, if the book rejects
        the sharpest signals, your track record of placed bets may look
        BETTER than your true edge (because the bets that would have won
        most got rejected before they could be tracked as wins).
        """
        if not self._observations:
            return {
                "raw_roi": track_record_roi,
                "bias_adjustment": 0.0,
                "adjusted_roi": track_record_roi,
                "selection_bias_detected": False,
                "explanation": "No rejection data available",
            }

        accepted = [o for o in self._observations if not o.was_rejected]
        rejected = [o for o in self._observations if o.was_rejected]

        if not rejected:
            return {
                "raw_roi": track_record_roi,
                "bias_adjustment": 0.0,
                "adjusted_roi": track_record_roi,
                "selection_bias_detected": False,
                "explanation": "No rejections in history — no selection bias",
            }

        mean_edge_accepted = float(np.mean([o.signal_edge_cents for o in accepted])) if accepted else 0
        mean_edge_rejected = float(np.mean([o.signal_edge_cents for o in rejected]))
        rejection_rate = len(rejected) / len(self._observations)

        # If rejected bets had MORE edge, the track record undersells our signal
        # If rejected bets had LESS edge, the track record oversells our signal
        if mean_edge_accepted > 0:
            bias_ratio = mean_edge_rejected / mean_edge_accepted
        else:
            bias_ratio = 1.0

        # Adjustment: ROI * (1 - rejection_rate * (1 - 1/bias_ratio))
        if bias_ratio > 1:
            # Rejected bets were sharper — ROI is deflated (we're better than shown)
            adjustment = rejection_rate * (1 - 1 / bias_ratio) * track_record_roi
            adjusted_roi = track_record_roi + adjustment
            explanation = (
                f"Rejected bets were {bias_ratio:.1f}x sharper than accepted. "
                f"Track record understates true signal quality by {abs(adjustment):.2f}%."
            )
        elif bias_ratio < 1:
            # Rejected bets were duller — ROI is inflated
            adjustment = -rejection_rate * (1 - bias_ratio) * track_record_roi
            adjusted_roi = track_record_roi + adjustment
            explanation = (
                f"Rejected bets had lower edge than accepted. "
                f"Track record may overstate true edge by {abs(adjustment):.2f}%."
            )
        else:
            adjustment = 0.0
            adjusted_roi = track_record_roi
            explanation = "No significant selection bias detected."

        return {
            "raw_roi": track_record_roi,
            "bias_adjustment": adjustment,
            "adjusted_roi": adjusted_roi,
            "selection_bias_detected": abs(bias_ratio - 1.0) > 0.1,
            "mean_edge_accepted": mean_edge_accepted,
            "mean_edge_rejected": mean_edge_rejected,
            "rejection_rate": rejection_rate,
            "bias_ratio": bias_ratio,
            "explanation": explanation,
        }

    def profile(self) -> RejectionProfile:
        """Generate aggregate rejection profile."""
        if not self._observations:
            return RejectionProfile(
                n_attempted=0,
                n_rejected=0,
                n_accepted=0,
                overall_rejection_rate=0.0,
                rejection_rate_by_book={},
                rejection_rate_by_reason={},
                rejection_rate_by_edge_bucket={},
                mean_edge_accepted=0.0,
                mean_edge_rejected=0.0,
                selection_bias_ratio=1.0,
                inflated_roi_pct=0.0,
                true_adjusted_roi_pct=0.0,
            )

        rejected = [o for o in self._observations if o.was_rejected]
        accepted = [o for o in self._observations if not o.was_rejected]

        # By book
        by_book: Dict[str, dict] = {}
        for obs in self._observations:
            bk = obs.sportsbook
            by_book.setdefault(bk, {"total": 0, "rejected": 0})
            by_book[bk]["total"] += 1
            if obs.was_rejected:
                by_book[bk]["rejected"] += 1
        rate_by_book = {
            k: {"rate": v["rejected"] / v["total"], "n": v["total"]}
            for k, v in by_book.items()
        }

        # By reason
        by_reason: Dict[str, int] = {}
        for obs in rejected:
            by_reason[obs.rejection_reason] = by_reason.get(obs.rejection_reason, 0) + 1
        total_rejected = len(rejected)
        rate_by_reason = {
            k: v / total_rejected for k, v in by_reason.items()
        } if total_rejected > 0 else {}

        # By edge bucket
        edges = [o.signal_edge_cents for o in self._observations]
        if edges:
            p25, p50, p75 = np.percentile(edges, [25, 50, 75])
        else:
            p25, p50, p75 = 0, 0, 0

        edge_buckets = [
            (float("-inf"), p25, "low_edge"),
            (p25, p50, "medium_edge"),
            (p50, p75, "high_edge"),
            (p75, float("inf"), "very_high_edge"),
        ]

        rate_by_edge: Dict[str, dict] = {}
        for lo, hi, label in edge_buckets:
            bucket_obs = [o for o in self._observations if lo <= o.signal_edge_cents < hi]
            if bucket_obs:
                rate_by_edge[label] = {
                    "n": len(bucket_obs),
                    "rejection_rate": sum(1 for o in bucket_obs if o.was_rejected) / len(bucket_obs),
                    "mean_edge": float(np.mean([o.signal_edge_cents for o in bucket_obs])),
                }

        mean_accepted = float(np.mean([o.signal_edge_cents for o in accepted])) if accepted else 0
        mean_rejected = float(np.mean([o.signal_edge_cents for o in rejected])) if rejected else 0
        bias_ratio = (mean_rejected / mean_accepted) if mean_accepted > 0 else 1.0

        return RejectionProfile(
            n_attempted=len(self._observations),
            n_rejected=len(rejected),
            n_accepted=len(accepted),
            overall_rejection_rate=len(rejected) / len(self._observations),
            rejection_rate_by_book=rate_by_book,
            rejection_rate_by_reason=rate_by_reason,
            rejection_rate_by_edge_bucket=rate_by_edge,
            mean_edge_accepted=mean_accepted,
            mean_edge_rejected=mean_rejected,
            selection_bias_ratio=bias_ratio,
            inflated_roi_pct=0.0,
            true_adjusted_roi_pct=0.0,
        )
