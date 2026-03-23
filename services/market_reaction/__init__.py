"""
services/market_reaction/
=========================
Models how sportsbooks react to winning bettors over time — limit
progression, line shading, edge decay, and long-term survival.

The central question: if you have real edge today, how long before the
books adapt and destroy it?
"""

from services.market_reaction.book_behavior import BookBehaviorModel
from services.market_reaction.limit_progression import LimitProgressionModel
from services.market_reaction.line_shading import LineShadingModel
from services.market_reaction.edge_decay import EdgeDecayModel
from services.market_reaction.survival_simulator import SurvivalSimulator

__all__ = [
    "BookBehaviorModel",
    "LimitProgressionModel",
    "LineShadingModel",
    "EdgeDecayModel",
    "SurvivalSimulator",
]
