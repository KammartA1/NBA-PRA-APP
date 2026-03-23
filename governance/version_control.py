"""
governance/version_control.py
==============================
Model versioning system with unique IDs, parameter tracking,
training data hashing, and deployment history.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database.connection import session_scope
from database.models import ModelVersion

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelVersionManager:
    """Manages model versions with full traceability.

    Every model deployed must be registered here with:
      - Unique version ID
      - Timestamp
      - Parameters (hyperparameters, config)
      - Training data hash
      - Feature list
      - Performance metrics on validation set
    """

    def __init__(self, sport: str = "NBA"):
        self.sport = sport

    def register_version(
        self,
        parameters: Dict[str, Any],
        features: List[str],
        hyperparameters: Dict[str, Any] | None = None,
        training_data_hash: str | None = None,
        performance_metrics: Dict[str, float] | None = None,
        version_tag: str | None = None,
    ) -> str:
        """Register a new model version.

        Args:
            parameters: Model parameters (any serializable dict).
            features: List of feature names used.
            hyperparameters: Hyperparameter settings.
            training_data_hash: Hash of training data for reproducibility.
            performance_metrics: Validation performance metrics.
            version_tag: Optional human-readable tag.

        Returns:
            Version ID string.
        """
        version_id = version_tag or f"v{_utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        full_params = {
            "parameters": parameters,
            "features": features,
            "hyperparameters": hyperparameters or {},
            "n_features": len(features),
        }

        metrics = performance_metrics or {}

        try:
            with session_scope() as session:
                mv = ModelVersion(
                    version=version_id,
                    sport=self.sport,
                    parameters_json=json.dumps(full_params),
                    training_data_hash=training_data_hash or self._hash_params(full_params),
                    performance_metrics_json=json.dumps(metrics),
                    is_active=True,
                )
                session.add(mv)

                # Deactivate previous versions
                session.query(ModelVersion).filter(
                    ModelVersion.sport == self.sport,
                    ModelVersion.version != version_id,
                    ModelVersion.is_active == True,
                ).update({"is_active": False})

            log.info("Registered model version %s with %d features", version_id, len(features))
        except Exception as e:
            log.error("Failed to register version: %s", e)

        return version_id

    def get_active_version(self) -> Optional[Dict[str, Any]]:
        """Get the currently active model version."""
        try:
            with session_scope() as session:
                mv = (
                    session.query(ModelVersion)
                    .filter(ModelVersion.sport == self.sport, ModelVersion.is_active == True)
                    .order_by(ModelVersion.created_at.desc())
                    .first()
                )
                if mv:
                    return {
                        "version": mv.version,
                        "created_at": mv.created_at.isoformat() if mv.created_at else None,
                        "parameters": mv.parameters,
                        "performance_metrics": mv.performance_metrics,
                        "training_data_hash": mv.training_data_hash,
                        "is_active": mv.is_active,
                    }
        except Exception as e:
            log.warning("Failed to get active version: %s", e)
        return None

    def get_version_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get version history, most recent first."""
        try:
            with session_scope() as session:
                versions = (
                    session.query(ModelVersion)
                    .filter(ModelVersion.sport == self.sport)
                    .order_by(ModelVersion.created_at.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "version": v.version,
                        "created_at": v.created_at.isoformat() if v.created_at else None,
                        "is_active": v.is_active,
                        "parameters": v.parameters,
                        "performance_metrics": v.performance_metrics,
                        "training_data_hash": v.training_data_hash,
                    }
                    for v in versions
                ]
        except Exception as e:
            log.warning("Failed to get version history: %s", e)
            return []

    def activate_version(self, version_id: str) -> bool:
        """Activate a specific version (and deactivate all others)."""
        try:
            with session_scope() as session:
                # Deactivate all
                session.query(ModelVersion).filter(
                    ModelVersion.sport == self.sport,
                ).update({"is_active": False})

                # Activate target
                updated = session.query(ModelVersion).filter(
                    ModelVersion.version == version_id,
                    ModelVersion.sport == self.sport,
                ).update({"is_active": True})

                if updated:
                    log.info("Activated version %s", version_id)
                    return True
                else:
                    log.warning("Version %s not found", version_id)
                    return False
        except Exception as e:
            log.error("Failed to activate version: %s", e)
            return False

    def compare_versions(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """Compare two model versions."""
        try:
            with session_scope() as session:
                va = session.query(ModelVersion).filter(ModelVersion.version == version_a).first()
                vb = session.query(ModelVersion).filter(ModelVersion.version == version_b).first()

                if not va or not vb:
                    return {"error": "One or both versions not found"}

                params_a = va.parameters
                params_b = vb.parameters
                metrics_a = va.performance_metrics
                metrics_b = vb.performance_metrics

                # Diff parameters
                features_a = set(params_a.get("features", []))
                features_b = set(params_b.get("features", []))

                # Diff metrics
                metric_diff = {}
                all_metrics = set(list(metrics_a.keys()) + list(metrics_b.keys()))
                for m in all_metrics:
                    val_a = metrics_a.get(m, 0)
                    val_b = metrics_b.get(m, 0)
                    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                        metric_diff[m] = {
                            "version_a": val_a,
                            "version_b": val_b,
                            "diff": round(val_b - val_a, 6),
                            "pct_change": round((val_b - val_a) / max(abs(val_a), 1e-10) * 100, 2),
                        }

                return {
                    "version_a": version_a,
                    "version_b": version_b,
                    "features_added": list(features_b - features_a),
                    "features_removed": list(features_a - features_b),
                    "features_unchanged": list(features_a & features_b),
                    "metric_diff": metric_diff,
                    "same_training_data": va.training_data_hash == vb.training_data_hash,
                }
        except Exception as e:
            log.error("Failed to compare versions: %s", e)
            return {"error": str(e)}

    def _hash_params(self, params: Dict) -> str:
        """Generate a hash of parameters for deduplication."""
        serialized = json.dumps(params, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:32]
