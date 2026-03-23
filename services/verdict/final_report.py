"""
services/verdict/final_report.py
=================================
FinalVerdict — pulls from ALL systems to generate a comprehensive
assessment of whether this system has real, deployable edge.

This is the single most important output of the entire quant engine.
If you only read one report, read this one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from database.connection import session_scope
from database.models import Bet, EdgeReport, ModelVersion

log = logging.getLogger(__name__)


@dataclass
class EdgeComponent:
    """A single component of the edge breakdown."""
    name: str
    category: str              # proven, fake, unproven
    contribution_pct: float    # % of total edge from this source
    confidence: float          # How confident are we in this estimate
    clv_contribution: float    # CLV attributed to this source
    brier_contribution: float  # Brier improvement from this source
    sample_size: int
    status: str                # keep, monitor, delete
    evidence: str


@dataclass
class FinalVerdictReport:
    """Complete final verdict report."""
    generated_at: datetime
    report_version: str

    # Core verdict
    has_real_edge: bool
    edge_confidence_pct: float        # 0-100%
    pct_edge_is_real: float           # What % of apparent edge is real
    pct_edge_is_illusion: float       # What % is fabricated/overfit
    million_dollar_survival_pct: float  # P(survive 12 months with $1M)

    # Edge breakdown
    total_edge_pct: float
    edge_components: List[EdgeComponent]
    proven_edge_pct: float
    fake_edge_pct: float
    unproven_edge_pct: float

    # New opportunities
    new_opportunities: List[Dict[str, Any]]

    # System health
    system_status: Dict[str, Any]
    data_quality_score: float
    simulation_accuracy: float

    # Action items
    features_to_delete: List[str]
    build_next_priorities: List[str]

    # Adversary's playbook
    adversary_playbook: List[Dict[str, str]]

    # Raw data
    summary_text: str


class FinalVerdict:
    """Generates the final verdict report by pulling from all subsystems.

    This is the top-level aggregator that answers: "Is this real?"
    """

    def __init__(self, sport: str = "NBA"):
        self.sport = sport

    def generate(self) -> FinalVerdictReport:
        """Generate the complete final verdict report."""
        # Load all data
        bets = self._load_all_bets()
        settled = [b for b in bets if b.get("status") == "settled" and b.get("profit") is not None]

        # Edge analysis
        edge_components = self._analyze_edge_components(settled)
        proven = [c for c in edge_components if c.category == "proven"]
        fake = [c for c in edge_components if c.category == "fake"]
        unproven = [c for c in edge_components if c.category == "unproven"]

        proven_pct = sum(c.contribution_pct for c in proven)
        fake_pct = sum(c.contribution_pct for c in fake)
        unproven_pct = sum(c.contribution_pct for c in unproven)

        # Total edge from CLV
        total_edge = self._compute_total_edge(settled)

        # Core metrics
        has_edge, edge_confidence = self._assess_edge_reality(settled, edge_components)

        # System health
        system_status = self._assess_system_health()
        data_quality = self._assess_data_quality(bets)
        sim_accuracy = self._assess_simulation_accuracy(settled)

        # Opportunities
        opportunities = self._identify_opportunities(edge_components)

        # Cleanup recommendations
        features_to_delete = self._recommend_deletions(edge_components)
        build_priorities = self._recommend_build_priorities(edge_components, system_status)

        # Adversary playbook
        adversary = self._generate_adversary_playbook(edge_components, settled)

        # Survival estimate
        survival_pct = self._estimate_million_survival(
            total_edge, proven_pct, has_edge, edge_confidence
        )

        # Generate summary text
        summary = self._generate_summary(
            has_edge, edge_confidence, total_edge, proven_pct, fake_pct,
            survival_pct, len(settled), features_to_delete,
        )

        report = FinalVerdictReport(
            generated_at=datetime.now(timezone.utc),
            report_version="1.0",
            has_real_edge=has_edge,
            edge_confidence_pct=round(edge_confidence, 1),
            pct_edge_is_real=round(proven_pct, 1),
            pct_edge_is_illusion=round(fake_pct, 1),
            million_dollar_survival_pct=round(survival_pct, 1),
            total_edge_pct=round(total_edge, 2),
            edge_components=edge_components,
            proven_edge_pct=round(proven_pct, 1),
            fake_edge_pct=round(fake_pct, 1),
            unproven_edge_pct=round(unproven_pct, 1),
            new_opportunities=opportunities,
            system_status=system_status,
            data_quality_score=round(data_quality, 1),
            simulation_accuracy=round(sim_accuracy, 1),
            features_to_delete=features_to_delete,
            build_next_priorities=build_priorities,
            adversary_playbook=adversary,
            summary_text=summary,
        )

        # Persist
        self._save_report(report)

        return report

    def _load_all_bets(self) -> List[Dict[str, Any]]:
        """Load all bets from database."""
        try:
            with session_scope() as session:
                bets = (
                    session.query(Bet)
                    .filter(Bet.sport == self.sport)
                    .order_by(Bet.timestamp.asc())
                    .all()
                )
                return [b.to_dict() for b in bets]
        except Exception as e:
            log.warning("Failed to load bets: %s", e)
            return []

    def _compute_total_edge(self, settled: List[Dict]) -> float:
        """Compute total apparent edge from bet history."""
        if not settled:
            return 0.0
        total_profit = sum(b.get("profit", 0) or 0 for b in settled)
        total_stake = sum(b.get("stake", 0) or 0 for b in settled)
        if total_stake <= 0:
            return 0.0
        return (total_profit / total_stake) * 100.0

    def _analyze_edge_components(self, settled: List[Dict]) -> List[EdgeComponent]:
        """Analyze each edge source's contribution.

        Uses features_snapshot_json from bets to decompose edge by source.
        """
        if not settled:
            return self._get_default_components()

        # Compute overall CLV
        clv_values = []
        for b in settled:
            bl = b.get("bet_line")
            cl = b.get("closing_line")
            direction = b.get("direction", "over")
            if bl is not None and cl is not None:
                clv = (cl - bl) if direction.lower() == "over" else (bl - cl)
                clv_values.append(clv)
        avg_clv = float(np.mean(clv_values)) if clv_values else 0.0

        # Analyze feature snapshots to determine which signals contributed
        signal_counts: Dict[str, List[float]] = {}
        for b in settled:
            snap = b.get("features_snapshot_json", "{}")
            try:
                features = json.loads(snap) if isinstance(snap, str) else snap
            except (json.JSONDecodeError, TypeError):
                features = {}
            profit = b.get("profit", 0) or 0
            for key, val in features.items():
                if isinstance(val, (int, float)):
                    signal_counts.setdefault(key, []).append(profit)

        # Build components from signals
        components = []
        total_explained = 0.0

        for signal_name, profits in sorted(
            signal_counts.items(), key=lambda x: abs(sum(x[1])), reverse=True
        ):
            if len(profits) < 10:
                continue

            profits_arr = np.array(profits)
            avg_profit = float(np.mean(profits_arr))
            win_rate = float(np.mean(profits_arr > 0))
            n = len(profits)

            # Statistical test
            if np.std(profits_arr, ddof=1) > 0:
                t_stat, p_value = sp_stats.ttest_1samp(profits_arr, 0.0)
            else:
                t_stat, p_value = 0.0, 1.0

            # Classify
            if p_value < 0.05 and avg_profit > 0:
                category = "proven"
                status = "keep"
            elif p_value < 0.20 and avg_profit > 0:
                category = "unproven"
                status = "monitor"
            else:
                category = "fake"
                status = "delete"

            # Contribution (approximate)
            contribution = abs(avg_profit) / max(abs(avg_clv), 0.01) * 100
            contribution = min(contribution, 50.0)
            total_explained += contribution

            clv_contrib = avg_profit * 0.5  # Rough CLV attribution

            components.append(EdgeComponent(
                name=signal_name,
                category=category,
                contribution_pct=round(contribution, 1),
                confidence=round(1.0 - p_value, 3),
                clv_contribution=round(clv_contrib, 3),
                brier_contribution=0.0,
                sample_size=n,
                status=status,
                evidence=f"avg_profit=${avg_profit:.2f}, p={p_value:.4f}, n={n}, win_rate={win_rate:.1%}",
            ))

        # If no signal data, return defaults
        if not components:
            return self._get_default_components()

        # Normalize contributions
        if total_explained > 0:
            for c in components:
                c.contribution_pct = round(c.contribution_pct / total_explained * 100, 1)

        return components[:20]  # Top 20 components

    def _get_default_components(self) -> List[EdgeComponent]:
        """Return default edge components when data is insufficient."""
        default_sources = [
            ("closing_line_value", "unproven"),
            ("model_calibration", "unproven"),
            ("market_timing", "unproven"),
            ("information_speed", "unproven"),
        ]
        return [
            EdgeComponent(
                name=name, category=cat, contribution_pct=25.0, confidence=0.0,
                clv_contribution=0.0, brier_contribution=0.0, sample_size=0,
                status="monitor",
                evidence="Insufficient data to evaluate",
            )
            for name, cat in default_sources
        ]

    def _assess_edge_reality(
        self, settled: List[Dict], components: List[EdgeComponent]
    ) -> tuple[bool, float]:
        """Determine if edge is real and confidence level."""
        if len(settled) < 50:
            return False, 0.0

        # CLV positive over 200+ bets is strong evidence
        clv_vals = []
        for b in settled:
            bl, cl = b.get("bet_line"), b.get("closing_line")
            d = b.get("direction", "over")
            if bl is not None and cl is not None:
                clv_vals.append((cl - bl) if d.lower() == "over" else (bl - cl))

        if not clv_vals:
            return False, 0.0

        clv_arr = np.array(clv_vals)
        avg_clv = float(np.mean(clv_arr))

        # t-test on CLV
        if len(clv_arr) >= 30 and np.std(clv_arr, ddof=1) > 0:
            t_stat, p_value = sp_stats.ttest_1samp(clv_arr, 0.0)
            clv_significant = t_stat > 0 and p_value < 0.05
        else:
            clv_significant = False
            p_value = 1.0

        # ROI positive
        total_profit = sum(b.get("profit", 0) or 0 for b in settled)
        total_stake = sum(b.get("stake", 0) or 0 for b in settled)
        roi = (total_profit / max(total_stake, 1)) * 100

        # Proven component fraction
        proven_frac = sum(c.contribution_pct for c in components if c.category == "proven") / 100.0

        # Composite confidence
        confidence = 0.0
        if clv_significant and avg_clv > 0:
            confidence += 40.0
        if roi > 0 and len(settled) >= 100:
            confidence += 25.0
        if proven_frac > 0.3:
            confidence += 20.0
        if len(settled) >= 200:
            confidence += 15.0

        confidence = min(confidence, 95.0)

        has_edge = confidence >= 50.0 and avg_clv > 0

        return has_edge, confidence

    def _assess_system_health(self) -> Dict[str, Any]:
        """Check health of all system components."""
        status = {
            "database": "operational",
            "edge_monitor": "unknown",
            "kill_switch": "unknown",
            "execution_layer": "unknown",
            "clv_system": "unknown",
            "data_audit": "unknown",
        }

        # Check database
        try:
            from database.connection import health_check
            db_health = health_check()
            status["database"] = "operational" if db_health.get("ok") else "degraded"
        except Exception:
            status["database"] = "error"

        # Check edge monitor
        try:
            from services.edge_monitor.daily_metrics import DailyEdgeMetrics
            DailyEdgeMetrics(sport=self.sport)
            status["edge_monitor"] = "operational"
        except Exception:
            status["edge_monitor"] = "error"

        # Check kill switch
        try:
            from services.kill_switch import KillSwitch
            KillSwitch(sport=self.sport)
            status["kill_switch"] = "operational"
        except Exception:
            status["kill_switch"] = "error"

        return status

    def _assess_data_quality(self, bets: List[Dict]) -> float:
        """Score data quality 0-100."""
        if not bets:
            return 0.0

        score = 0.0
        total = len(bets)

        # CLV data availability
        has_clv = sum(1 for b in bets if b.get("closing_line") is not None)
        score += (has_clv / max(total, 1)) * 30

        # Prediction probability available
        has_prob = sum(1 for b in bets if b.get("predicted_prob") is not None)
        score += (has_prob / max(total, 1)) * 25

        # Odds data
        has_odds = sum(1 for b in bets if b.get("odds_decimal") is not None and b["odds_decimal"] > 0)
        score += (has_odds / max(total, 1)) * 20

        # Settlement completeness
        settled = sum(1 for b in bets if b.get("status") == "settled")
        score += (settled / max(total, 1)) * 15

        # Feature snapshots
        has_features = sum(
            1 for b in bets
            if b.get("features_snapshot_json") and b["features_snapshot_json"] != "{}"
        )
        score += (has_features / max(total, 1)) * 10

        return min(score, 100.0)

    def _assess_simulation_accuracy(self, settled: List[Dict]) -> float:
        """Assess how accurate the model's predictions are (0-100)."""
        if len(settled) < 20:
            return 0.0

        pred_probs = []
        outcomes = []
        for b in settled:
            pp = b.get("predicted_prob")
            profit = b.get("profit")
            if pp is not None and profit is not None:
                pred_probs.append(pp)
                outcomes.append(1.0 if profit > 0 else 0.0)

        if not pred_probs:
            return 0.0

        pred_arr = np.array(pred_probs)
        out_arr = np.array(outcomes)

        # Brier score (lower is better, 0.25 = random)
        brier = float(np.mean((pred_arr - out_arr) ** 2))

        # Score: 0.15 Brier = 100, 0.25 = 50, 0.35 = 0
        score = max(0, min(100, (0.35 - brier) / 0.20 * 100))

        return score

    def _identify_opportunities(self, components: List[EdgeComponent]) -> List[Dict[str, Any]]:
        """Identify new edge opportunities to explore."""
        opportunities = []

        # Check for unproven signals with high potential
        for c in components:
            if c.category == "unproven" and c.confidence > 0.3:
                opportunities.append({
                    "name": c.name,
                    "potential": "high" if c.contribution_pct > 10 else "medium",
                    "action": f"Collect {max(200 - c.sample_size, 50)} more samples to validate",
                    "current_evidence": c.evidence,
                })

        # Standard opportunities
        standard_opps = [
            {"name": "live_line_tracking", "potential": "high",
             "action": "Implement real-time line movement tracking for sharper CLV",
             "current_evidence": "Not yet implemented"},
            {"name": "correlation_exploitation", "potential": "medium",
             "action": "Model same-game correlations for parlay/sgp edge",
             "current_evidence": "Structural correlation data available"},
            {"name": "market_microstructure", "potential": "high",
             "action": "Model book-specific line setting patterns",
             "current_evidence": "Line movement data available"},
        ]
        opportunities.extend(standard_opps)

        return opportunities[:10]

    def _recommend_deletions(self, components: List[EdgeComponent]) -> List[str]:
        """Recommend features/signals to delete immediately."""
        return [
            f"{c.name} ({c.evidence})"
            for c in components
            if c.status == "delete"
        ]

    def _recommend_build_priorities(
        self, components: List[EdgeComponent], system_status: Dict
    ) -> List[str]:
        """Recommend what to build next, prioritized."""
        priorities = []

        # Fix broken systems first
        for system, status in system_status.items():
            if status == "error":
                priorities.append(f"FIX: {system} is not operational")

        # Validate unproven edges
        unproven = [c for c in components if c.category == "unproven"]
        if unproven:
            priorities.append(f"VALIDATE: {len(unproven)} unproven edge sources need more data")

        # Standard priorities
        priorities.extend([
            "IMPROVE: Add more independent edge sources to reduce single-source risk",
            "MONITOR: Implement automated daily verdict checks",
            "HARDEN: Add adversarial testing to deployment pipeline",
        ])

        return priorities[:10]

    def _generate_adversary_playbook(
        self, components: List[EdgeComponent], settled: List[Dict]
    ) -> List[Dict[str, str]]:
        """Generate the adversary's playbook — how a book would destroy this system."""
        playbook = [
            {
                "attack": "Identify sharp bettor via CLV analysis",
                "method": "Track CLV > 1 cent over 100+ bets → flag account",
                "defense": "Rotate accounts, vary bet timing, spread across books",
            },
            {
                "attack": "Line shading against identified sharp",
                "method": "Show worse lines to flagged accounts (0.5-2 points)",
                "defense": "Monitor for systematic line discrepancies vs market consensus",
            },
            {
                "attack": "Progressive bet limits",
                "method": "Reduce max bet from $500 → $50 → $5 → banned",
                "defense": "Multi-book strategy, prioritize highest-limit books",
            },
            {
                "attack": "Copy other books' limits",
                "method": "Share sharp bettor lists across sportsbook networks",
                "defense": "Diversify across different book networks/ownership groups",
            },
            {
                "attack": "Improve own pricing model",
                "method": "Hire quant team, use sharp action to improve lines",
                "defense": "Continuously innovate edge sources, don't rely on one signal",
            },
            {
                "attack": "Delayed settlement / manual review",
                "method": "Review large bets manually, delay payouts, void suspicious bets",
                "defense": "Keep bet sizes below review thresholds, document everything",
            },
        ]

        # Add source-specific attacks
        for c in components:
            if c.category == "proven" and c.contribution_pct > 15:
                playbook.append({
                    "attack": f"Neutralize '{c.name}' edge source",
                    "method": f"Improve pricing model to incorporate {c.name} signal",
                    "defense": f"Monitor {c.name} contribution over time, have backup sources",
                })

        return playbook

    def _estimate_million_survival(
        self, total_edge: float, proven_pct: float,
        has_edge: bool, confidence: float,
    ) -> float:
        """Estimate P(survive 12 months with $1M deployment).

        Conservative estimate factoring in:
        - Edge magnitude and confidence
        - Market reaction (limits, shading)
        - Execution degradation
        - Variance/drawdown risk
        """
        if not has_edge:
            return 5.0  # Very unlikely

        base_survival = 50.0  # Start at 50%

        # Edge magnitude adjustment
        if total_edge > 5.0:
            base_survival += 15.0
        elif total_edge > 3.0:
            base_survival += 10.0
        elif total_edge > 1.0:
            base_survival += 5.0
        else:
            base_survival -= 10.0

        # Confidence adjustment
        base_survival += (confidence - 50) * 0.3

        # Proven edge percentage
        base_survival += (proven_pct - 50) * 0.2

        # Market reaction penalty (books WILL react at $1M level)
        base_survival -= 20.0  # Major penalty for large deployment

        return max(min(base_survival, 95.0), 1.0)

    def _generate_summary(
        self, has_edge: bool, confidence: float, total_edge: float,
        proven_pct: float, fake_pct: float, survival_pct: float,
        n_bets: int, features_to_delete: List[str],
    ) -> str:
        """Generate human-readable summary text."""
        verdict = "YES" if has_edge else "NO"
        lines = [
            f"FINAL VERDICT: Real edge = {verdict} (confidence: {confidence:.0f}%)",
            f"",
            f"Total apparent edge: {total_edge:.2f}%",
            f"  - Proven real: {proven_pct:.0f}%",
            f"  - Likely illusion: {fake_pct:.0f}%",
            f"  - Unproven: {100 - proven_pct - fake_pct:.0f}%",
            f"",
            f"$1M deployment 12-month survival: {survival_pct:.0f}%",
            f"Sample size: {n_bets} settled bets",
            f"",
        ]

        if features_to_delete:
            lines.append(f"IMMEDIATE ACTIONS: Delete {len(features_to_delete)} features")
            for f in features_to_delete[:5]:
                lines.append(f"  - {f}")

        if not has_edge:
            lines.extend([
                "",
                "WARNING: System does NOT have demonstrated edge.",
                "Do NOT deploy with real capital until edge is proven.",
            ])

        return "\n".join(lines)

    def _save_report(self, report: FinalVerdictReport) -> None:
        """Persist the report to database."""
        try:
            report_data = {
                "has_real_edge": report.has_real_edge,
                "edge_confidence_pct": report.edge_confidence_pct,
                "pct_edge_is_real": report.pct_edge_is_real,
                "pct_edge_is_illusion": report.pct_edge_is_illusion,
                "total_edge_pct": report.total_edge_pct,
                "million_dollar_survival_pct": report.million_dollar_survival_pct,
                "data_quality_score": report.data_quality_score,
                "n_components": len(report.edge_components),
                "n_features_to_delete": len(report.features_to_delete),
                "summary": report.summary_text,
            }
            with session_scope() as session:
                session.add(EdgeReport(
                    report_type="final_verdict",
                    sport=self.sport,
                    report_json=json.dumps(report_data),
                ))
        except Exception as e:
            log.warning("Failed to save final verdict: %s", e)
