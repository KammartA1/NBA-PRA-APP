"""Shared dataclasses for the entire quant system."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class BetStatus(enum.Enum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    VOID = "void"


class Sport(enum.Enum):
    NBA = "nba"
    GOLF = "golf"


class BetType(enum.Enum):
    OVER = "over"
    UNDER = "under"
    OUTRIGHT = "outright"
    H2H = "h2h"
    TOP5 = "top5"
    TOP10 = "top10"
    TOP20 = "top20"
    MAKE_CUT = "make_cut"
    PARLAY_POWER = "parlay_power"
    PARLAY_FLEX = "parlay_flex"


class SystemState(enum.Enum):
    """Global system state for failure protection."""
    ACTIVE = "active"               # Normal operation
    REDUCED = "reduced"             # Reduced sizing (CLV warning)
    SUSPENDED = "suspended"         # No new bets (CLV critical)
    KILLED = "killed"               # System shutdown (catastrophic)


@dataclass
class BetRecord:
    """Immutable record of every bet placed."""
    bet_id: str
    sport: Sport
    timestamp: datetime
    player: str
    bet_type: BetType
    stat_type: str                  # e.g., "points", "birdies", "fantasy_score"
    line: float                     # PrizePicks/book line
    direction: str                  # "over" or "under"
    model_prob: float               # Model's probability at time of bet
    market_prob: float              # Implied probability from market
    edge: float                     # model_prob - market_prob
    stake: float                    # Dollar amount wagered
    kelly_fraction: float           # Kelly fraction used
    odds_american: int              # American odds at time of bet
    odds_decimal: float             # Decimal odds at time of bet
    # Context
    model_projection: float         # Model's projected value
    model_std: float                # Model's projected std dev
    confidence_score: float         # 0-1 confidence rating
    engine_agreement: float         # Ensemble agreement metric
    # Post-settlement
    status: BetStatus = BetStatus.PENDING
    actual_result: Optional[float] = None
    closing_line: Optional[float] = None
    closing_odds: Optional[int] = None
    settled_at: Optional[datetime] = None
    pnl: float = 0.0
    # Metadata
    model_version: str = "1.0"
    features_used: str = ""         # JSON string of feature snapshot
    notes: str = ""


@dataclass
class CLVResult:
    """Closing line value measurement for a single bet."""
    bet_id: str
    opening_line: float
    bet_line: float
    closing_line: float
    line_movement: float            # closing - opening
    clv_raw: float                  # closing_prob - bet_prob (positive = good)
    clv_cents: float                # CLV in cents per dollar
    beat_close: bool                # Did we beat the closing line?


@dataclass
class EdgeReport:
    """Daily edge validation report."""
    report_date: datetime
    sport: Sport
    # Rolling CLV
    clv_last_50: float
    clv_last_100: float
    clv_last_250: float
    clv_last_500: float
    # Calibration
    calibration_error: float        # Mean absolute calibration error
    calibration_buckets: list       # List of CalibrationBucket
    # Model vs Market
    model_roi: float                # Actual ROI
    expected_roi: float             # Expected ROI from model probs
    # Signals
    edge_exists: bool               # Overall: does edge exist?
    system_state: SystemState       # Recommended system state
    warnings: list = field(default_factory=list)
    actions: list = field(default_factory=list)


@dataclass
class CalibrationBucket:
    """One bucket in a calibration curve."""
    prob_lower: float               # e.g., 0.50
    prob_upper: float               # e.g., 0.55
    predicted_avg: float            # Mean predicted probability
    actual_rate: float              # Actual hit rate
    n_bets: int                     # Sample size
    calibration_error: float        # |predicted - actual|
    is_overconfident: bool          # predicted > actual


@dataclass
class RiskState:
    """Current risk management state."""
    bankroll: float
    peak_bankroll: float
    current_drawdown_pct: float
    max_drawdown_pct: float
    daily_pnl: float
    daily_bet_count: int
    system_state: SystemState
    kelly_multiplier: float         # Dynamic Kelly adjustment (0.0 - 1.0)
    # Exposure
    total_exposure: float           # Sum of all pending bets
    exposure_by_player: dict        # {player: total_stake}
    exposure_by_type: dict          # {bet_type: total_stake}
    # Limits
    daily_loss_limit: float
    daily_loss_remaining: float
    max_single_bet: float


@dataclass
class BacktestResult:
    """Result from a walk-forward backtest."""
    start_date: datetime
    end_date: datetime
    n_bets: int
    total_pnl: float
    roi_pct: float
    win_rate: float
    avg_edge: float
    avg_clv: float
    max_drawdown_pct: float
    sharpe_ratio: float
    kelly_growth_rate: float
    # By period
    monthly_returns: list
    # Simulation
    ruin_probability: float         # P(bankroll < 0) over 10K paths
    median_final_bankroll: float
    p5_final_bankroll: float        # 5th percentile (worst case)
    p95_final_bankroll: float       # 95th percentile (best case)
