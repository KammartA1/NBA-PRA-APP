"""
edge_analysis — Edge source identification, validation, and independence testing.

Provides the full edge analysis pipeline:
  - 10 independent signal sources (edge_analysis.sources.*)
  - Signal registry with independence matrix (source_registry)
  - Master signal catalog with correlation analysis (edge_sources)
"""
from __future__ import annotations

from edge_analysis.source_registry import SourceRegistry
from edge_analysis.edge_sources import EdgeSourceCatalog
from edge_analysis.attribution import EdgeAttributionEngine

__all__ = ["SourceRegistry", "EdgeSourceCatalog", "EdgeAttributionEngine"]
