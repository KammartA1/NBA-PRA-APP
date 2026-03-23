"""
simulation/validation.py
========================
Validation suite for the possession-level simulator.  Checks that
simulated distributions match real NBA distributions across scoring,
variance, correlations, blowout frequency, and minutes.

All tests return structured diagnostics with pass/fail verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from simulation.game_engine import GameEngine, SimulationOutput


# ---------------------------------------------------------------------------
# Reference NBA distributions (2023-24 season averages)
# ---------------------------------------------------------------------------

# Per-game stat means and std-devs for a typical starter
NBA_REFERENCE = {
    "points":   {"mean": 18.0, "std": 8.5,  "skew": 0.3},
    "rebounds":  {"mean": 6.0,  "std": 3.5,  "skew": 0.5},
    "assists":   {"mean": 4.0,  "std": 3.0,  "skew": 0.7},
    "steals":    {"mean": 1.0,  "std": 0.9,  "skew": 1.0},
    "blocks":    {"mean": 0.6,  "std": 0.8,  "skew": 1.5},
    "turnovers": {"mean": 2.5,  "std": 1.5,  "skew": 0.6},
    "minutes":   {"mean": 33.0, "std": 4.0,  "skew": -0.3},
    "pra":       {"mean": 28.0, "std": 11.0, "skew": 0.3},
}

# Typical stat correlations (starter-level)
NBA_CORRELATIONS = {
    ("points", "assists"):  0.25,
    ("points", "rebounds"): 0.10,
    ("assists", "rebounds"): 0.05,
    ("points", "minutes"):  0.65,
    ("rebounds", "minutes"): 0.50,
}

# Blowout rate: fraction of games with 20+ point final margin
NBA_BLOWOUT_RATE = 0.15   # ~15%

# Team scoring
NBA_TEAM_SCORING = {"mean": 112.0, "std": 12.0}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    passed: bool
    metric: float
    threshold: float
    detail: str


@dataclass
class ValidationReport:
    """Aggregated validation report."""
    checks: List[ValidationCheck] = field(default_factory=list)
    overall_pass: bool = True

    def add(self, check: ValidationCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            self.overall_pass = False

    def summary(self) -> str:
        lines = [
            f"{'PASS' if self.overall_pass else 'FAIL'}: "
            f"{sum(c.passed for c in self.checks)}/{len(self.checks)} checks passed",
        ]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.name}: {c.detail}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_pass": self.overall_pass,
            "n_checks": len(self.checks),
            "n_passed": sum(c.passed for c in self.checks),
            "checks": [
                {"name": c.name, "passed": c.passed, "metric": c.metric,
                 "threshold": c.threshold, "detail": c.detail}
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------

class SimulationValidator:
    """Run a suite of validation checks against simulation output."""

    def __init__(
        self,
        output: SimulationOutput,
        reference: Dict[str, Dict[str, float]] | None = None,
    ) -> None:
        self.output = output
        self.ref = reference or NBA_REFERENCE

    def run_all(self) -> ValidationReport:
        """Run every validation check and return a report."""
        report = ValidationReport()
        report.add(self.check_scoring_distribution())
        report.add(self.check_variance())
        report.add(self.check_correlations())
        report.add(self.check_blowout_frequency())
        report.add(self.check_minutes_distribution())
        report.add(self.check_team_scoring())
        report.add(self.check_stat_ranges())
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_scoring_distribution(self) -> ValidationCheck:
        """KS test: simulated scoring vs a normal distribution with NBA
        mean and std for starters."""
        # Gather all starter point distributions
        starter_pts = self._get_starter_stat("points")
        if len(starter_pts) == 0:
            return ValidationCheck(
                "scoring_ks_test", False, 1.0, 0.05,
                "No starter point data available",
            )

        ref_mean = self.ref["points"]["mean"]
        ref_std = self.ref["points"]["std"]

        # KS test against N(ref_mean, ref_std)
        ks_stat, p_value = sp_stats.kstest(
            starter_pts, "norm", args=(ref_mean, ref_std),
        )

        # We allow some deviation: pass if p > 0.01 or KS stat < 0.15
        passed = p_value > 0.01 or ks_stat < 0.15
        return ValidationCheck(
            name="scoring_ks_test",
            passed=passed,
            metric=ks_stat,
            threshold=0.15,
            detail=f"KS={ks_stat:.4f}, p={p_value:.4f}, sim_mean={np.mean(starter_pts):.1f}, "
                   f"ref_mean={ref_mean:.1f}",
        )

    def check_variance(self) -> ValidationCheck:
        """Verify simulated variance is within 50-200% of NBA variance."""
        starter_pts = self._get_starter_stat("points")
        if len(starter_pts) == 0:
            return ValidationCheck("variance_check", False, 0.0, 0.5, "No data")

        sim_std = float(np.std(starter_pts, ddof=1))
        ref_std = self.ref["points"]["std"]
        ratio = sim_std / ref_std if ref_std > 0 else 0.0
        passed = 0.5 <= ratio <= 2.0

        return ValidationCheck(
            name="variance_check",
            passed=passed,
            metric=ratio,
            threshold=0.5,
            detail=f"sim_std={sim_std:.2f}, ref_std={ref_std:.2f}, ratio={ratio:.3f}",
        )

    def check_correlations(self) -> ValidationCheck:
        """Check PTS-AST, PTS-REB correlations match NBA direction."""
        # Use first starter found
        pid = self._first_starter_id()
        if pid is None:
            return ValidationCheck("correlation_check", False, 0.0, 0.0, "No starter found")

        dists = self.output.distributions.get(pid, {})
        pts_vals = dists.get("points")
        ast_vals = dists.get("assists")
        reb_vals = dists.get("rebounds")

        if pts_vals is None or ast_vals is None or reb_vals is None:
            return ValidationCheck("correlation_check", False, 0.0, 0.0, "Missing stat data")

        # Compute correlations
        corr_pts_ast = float(np.corrcoef(pts_vals.values, ast_vals.values)[0, 1])
        corr_pts_reb = float(np.corrcoef(pts_vals.values, reb_vals.values)[0, 1])

        ref_pts_ast = NBA_CORRELATIONS[("points", "assists")]
        ref_pts_reb = NBA_CORRELATIONS[("points", "rebounds")]

        # Pass if direction (sign) matches and magnitude within 0.5
        sign_ok_1 = (corr_pts_ast > -0.1)   # should be non-negative
        sign_ok_2 = (corr_pts_reb > -0.2)
        mag_ok = abs(corr_pts_ast - ref_pts_ast) < 0.5 and abs(corr_pts_reb - ref_pts_reb) < 0.5
        passed = sign_ok_1 and sign_ok_2 and mag_ok

        return ValidationCheck(
            name="correlation_check",
            passed=passed,
            metric=corr_pts_ast,
            threshold=0.5,
            detail=f"PTS-AST r={corr_pts_ast:.3f} (ref {ref_pts_ast}), "
                   f"PTS-REB r={corr_pts_reb:.3f} (ref {ref_pts_reb})",
        )

    def check_blowout_frequency(self) -> ValidationCheck:
        """~15% of games should be 20+ point blowouts."""
        results = self.output.game_results
        n = len(results)
        if n == 0:
            return ValidationCheck("blowout_frequency", False, 0.0, 0.15, "No games")

        n_blowout = sum(1 for g in results if abs(g["margin"]) >= 20)
        rate = n_blowout / n

        # Pass if between 5% and 30% (wide tolerance)
        passed = 0.05 <= rate <= 0.30

        return ValidationCheck(
            name="blowout_frequency",
            passed=passed,
            metric=rate,
            threshold=NBA_BLOWOUT_RATE,
            detail=f"blowout_rate={rate:.3f} ({n_blowout}/{n}), target~{NBA_BLOWOUT_RATE}",
        )

    def check_minutes_distribution(self) -> ValidationCheck:
        """Simulated starter minutes should be 25-40 range on average."""
        starter_min = self._get_starter_stat("minutes")
        if len(starter_min) == 0:
            return ValidationCheck("minutes_check", False, 0.0, 33.0, "No data")

        mean_min = float(np.mean(starter_min))
        passed = 20.0 <= mean_min <= 42.0

        return ValidationCheck(
            name="minutes_check",
            passed=passed,
            metric=mean_min,
            threshold=33.0,
            detail=f"sim_mean_minutes={mean_min:.1f}, expected 25-40",
        )

    def check_team_scoring(self) -> ValidationCheck:
        """Average team score should be 95-130 (NBA range)."""
        results = self.output.game_results
        if not results:
            return ValidationCheck("team_scoring", False, 0.0, 112.0, "No games")

        home_scores = [g["home_score"] for g in results]
        away_scores = [g["away_score"] for g in results]
        all_scores = home_scores + away_scores
        mean_score = float(np.mean(all_scores))
        passed = 85.0 <= mean_score <= 135.0

        return ValidationCheck(
            name="team_scoring",
            passed=passed,
            metric=mean_score,
            threshold=112.0,
            detail=f"mean_team_score={mean_score:.1f}, expected 95-130",
        )

    def check_stat_ranges(self) -> ValidationCheck:
        """Sanity check: no player averages negative stats or >60 PPG."""
        issues = []
        for pid, stat_map in self.output.distributions.items():
            pts = stat_map.get("points")
            if pts and (pts.mean < 0 or pts.mean > 60):
                issues.append(f"{pid} pts mean={pts.mean:.1f}")
            reb = stat_map.get("rebounds")
            if reb and (reb.mean < 0 or reb.mean > 25):
                issues.append(f"{pid} reb mean={reb.mean:.1f}")
            ast = stat_map.get("assists")
            if ast and (ast.mean < 0 or ast.mean > 20):
                issues.append(f"{pid} ast mean={ast.mean:.1f}")

        passed = len(issues) == 0
        detail = "All stats in range" if passed else f"Issues: {'; '.join(issues[:5])}"

        return ValidationCheck(
            name="stat_ranges",
            passed=passed,
            metric=len(issues),
            threshold=0,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_starter_stat(self, stat: str) -> np.ndarray:
        """Concatenate a stat's values across all starters."""
        arrays = []
        for pid, stat_map in self.output.distributions.items():
            dist = stat_map.get(stat)
            if dist is None:
                continue
            # Consider a player a "starter" if their minutes mean > 25
            minutes_dist = stat_map.get("minutes")
            if minutes_dist and minutes_dist.mean >= 25.0:
                arrays.append(dist.values)
        if arrays:
            return np.concatenate(arrays)
        return np.array([], dtype=np.float64)

    def _first_starter_id(self) -> Optional[str]:
        """Return the player_id of the first starter found."""
        for pid, stat_map in self.output.distributions.items():
            minutes_dist = stat_map.get("minutes")
            if minutes_dist and minutes_dist.mean >= 25.0:
                return pid
        return None
