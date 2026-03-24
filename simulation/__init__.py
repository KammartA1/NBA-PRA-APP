"""
simulation
==========
Possession-level NBA game simulator for PRA (Points + Rebounds + Assists)
distribution modelling.  Exports all key classes for external use.
"""

from simulation.config import SimulationConfig, CoachArchetype, DEFAULT_CONFIG
from simulation.player_state import PlayerProfile, PlayerState
from simulation.team_state import TeamState
from simulation.fatigue_model import FatigueModel, FatigueResult
from simulation.foul_model import FoulModel, FoulEvent
from simulation.game_script import GameScriptModel, GamePhase, ScriptDecision
from simulation.blowout_model import BlowoutModel, BlowoutAssessment
from simulation.lineup_manager import LineupManager, SubstitutionEvent
from simulation.possession import PossessionEngine, PossessionResult
from simulation.game_engine import (
    GameEngine,
    PlayerDistribution,
    SimulationOutput,
)
from simulation.validation import SimulationValidator, ValidationReport
from simulation.data_loader import SimulationDataLoader

__all__ = [
    # Config
    "SimulationConfig",
    "CoachArchetype",
    "DEFAULT_CONFIG",
    # Player / Team
    "PlayerProfile",
    "PlayerState",
    "TeamState",
    # Models
    "FatigueModel",
    "FatigueResult",
    "FoulModel",
    "FoulEvent",
    "GameScriptModel",
    "GamePhase",
    "ScriptDecision",
    "BlowoutModel",
    "BlowoutAssessment",
    "LineupManager",
    "SubstitutionEvent",
    # Possession
    "PossessionEngine",
    "PossessionResult",
    # Engine
    "GameEngine",
    "PlayerDistribution",
    "SimulationOutput",
    # Validation
    "SimulationValidator",
    "ValidationReport",
    # Data Loader
    "SimulationDataLoader",
]
