"""
edge_analysis.sources — All 10 independent edge signal modules.
"""
from __future__ import annotations

from edge_analysis.sources.minutes_distribution import MinutesDistributionSource
from edge_analysis.sources.usage_redistribution import UsageRedistributionSource
from edge_analysis.sources.pace_differential import PaceDifferentialSource
from edge_analysis.sources.rest_effects import RestEffectsSource
from edge_analysis.sources.home_away import HomeAwaySplitsSource
from edge_analysis.sources.referee_tendencies import RefereeTendenciesSource
from edge_analysis.sources.lineup_effects import LineupEffectsSource
from edge_analysis.sources.game_script import GameScriptSource
from edge_analysis.sources.defensive_matchup import DefensiveMatchupSource
from edge_analysis.sources.recency_weighting import RecencyWeightingSource

ALL_SOURCES = [
    MinutesDistributionSource,
    UsageRedistributionSource,
    PaceDifferentialSource,
    RestEffectsSource,
    HomeAwaySplitsSource,
    RefereeTendenciesSource,
    LineupEffectsSource,
    GameScriptSource,
    DefensiveMatchupSource,
    RecencyWeightingSource,
]

__all__ = [
    "MinutesDistributionSource",
    "UsageRedistributionSource",
    "PaceDifferentialSource",
    "RestEffectsSource",
    "HomeAwaySplitsSource",
    "RefereeTendenciesSource",
    "LineupEffectsSource",
    "GameScriptSource",
    "DefensiveMatchupSource",
    "RecencyWeightingSource",
    "ALL_SOURCES",
]
