"""
tests/adversarial/runner.py
============================
AdversarialRunner — orchestrates all adversarial tests and produces
a final OVERFIT/PASS verdict.

If any critical test fails, the system is deemed potentially overfit
and should not be trusted for real-money deployment.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from database.connection import session_scope
from database.models import EdgeReport
from tests.adversarial.probability_perturbation import ProbabilityPerturbationTest
from tests.adversarial.best_bet_removal import BestBetRemovalTest
from tests.adversarial.noise_injection import NoiseInjectionTest
from tests.adversarial.assumption_distortion import AssumptionDistortionTest

log = logging.getLogger(__name__)


class AdversarialRunner:
    """Run all adversarial tests and produce final verdict.

    Tests:
      1. Probability perturbation (2%, 5%, 10% noise)
      2. Best bet removal (remove top 5%, 10%, 20%)
      3. Noise injection (find breakpoint)
      4. Assumption distortion (+/-20% on all assumptions)
    """

    def __init__(self, sport: str = "NBA"):
        self.sport = sport

    def run_all(self) -> Dict[str, Any]:
        """Execute all adversarial tests.

        Returns:
            Dict with per-test results and final verdict.
        """
        results: Dict[str, Any] = {}
        test_verdicts: Dict[str, bool] = {}

        # 1. Probability Perturbation
        log.info("Running probability perturbation test...")
        try:
            pp_test = ProbabilityPerturbationTest(sport=self.sport)
            pp_result = pp_test.run()
            results["probability_perturbation"] = pp_result
            test_verdicts["probability_perturbation"] = pp_result.get("overall_pass", False)
        except Exception as e:
            log.error("Probability perturbation test failed: %s", e)
            results["probability_perturbation"] = {"status": "error", "error": str(e)}
            test_verdicts["probability_perturbation"] = False

        # 2. Best Bet Removal
        log.info("Running best bet removal test...")
        try:
            bbr_test = BestBetRemovalTest(sport=self.sport)
            bbr_result = bbr_test.run()
            results["best_bet_removal"] = bbr_result
            test_verdicts["best_bet_removal"] = bbr_result.get("overall_pass", False)
        except Exception as e:
            log.error("Best bet removal test failed: %s", e)
            results["best_bet_removal"] = {"status": "error", "error": str(e)}
            test_verdicts["best_bet_removal"] = False

        # 3. Noise Injection
        log.info("Running noise injection test...")
        try:
            ni_test = NoiseInjectionTest(sport=self.sport)
            ni_result = ni_test.run()
            results["noise_injection"] = ni_result
            test_verdicts["noise_injection"] = ni_result.get("overall_pass", False)
        except Exception as e:
            log.error("Noise injection test failed: %s", e)
            results["noise_injection"] = {"status": "error", "error": str(e)}
            test_verdicts["noise_injection"] = False

        # 4. Assumption Distortion
        log.info("Running assumption distortion test...")
        try:
            ad_test = AssumptionDistortionTest()
            ad_result = ad_test.run()
            results["assumption_distortion"] = ad_result
            test_verdicts["assumption_distortion"] = ad_result.get("overall_pass", False)
        except Exception as e:
            log.error("Assumption distortion test failed: %s", e)
            results["assumption_distortion"] = {"status": "error", "error": str(e)}
            test_verdicts["assumption_distortion"] = False

        # Final verdict
        n_passed = sum(1 for v in test_verdicts.values() if v)
        n_total = len(test_verdicts)
        all_passed = all(test_verdicts.values())

        # Critical tests: probability perturbation and best bet removal
        critical_passed = (
            test_verdicts.get("probability_perturbation", False)
            and test_verdicts.get("best_bet_removal", False)
        )

        if all_passed:
            final_verdict = "PASS"
            interpretation = "System passed ALL adversarial tests. Edge appears robust."
        elif critical_passed:
            final_verdict = "CONDITIONAL_PASS"
            interpretation = (
                f"System passed critical tests but failed {n_total - n_passed} of {n_total} total. "
                "Deploy with caution."
            )
        else:
            final_verdict = "OVERFIT"
            interpretation = (
                f"System FAILED {n_total - n_passed} of {n_total} tests including critical ones. "
                "DO NOT deploy with real capital. Likely overfit."
            )

        summary = {
            "final_verdict": final_verdict,
            "tests_passed": n_passed,
            "tests_total": n_total,
            "test_verdicts": test_verdicts,
            "critical_tests_passed": critical_passed,
            "interpretation": interpretation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Detailed failure analysis
        failures = []
        for test_name, passed in test_verdicts.items():
            if not passed:
                test_result = results.get(test_name, {})
                failures.append({
                    "test": test_name,
                    "verdict": test_result.get("verdict", "FAIL"),
                    "interpretation": test_result.get("interpretation", "No details"),
                })

        full_report = {
            "summary": summary,
            "results": results,
            "failures": failures,
        }

        # Persist results
        self._save_results(full_report)

        return full_report

    def run_single(self, test_name: str) -> Dict[str, Any]:
        """Run a single adversarial test by name."""
        test_map = {
            "probability_perturbation": lambda: ProbabilityPerturbationTest(sport=self.sport).run(),
            "best_bet_removal": lambda: BestBetRemovalTest(sport=self.sport).run(),
            "noise_injection": lambda: NoiseInjectionTest(sport=self.sport).run(),
            "assumption_distortion": lambda: AssumptionDistortionTest().run(),
        }

        if test_name not in test_map:
            return {"error": f"Unknown test: {test_name}. Available: {list(test_map.keys())}"}

        try:
            return test_map[test_name]()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_last_results(self) -> Dict[str, Any] | None:
        """Retrieve the most recent adversarial test results."""
        try:
            with session_scope() as session:
                report = (
                    session.query(EdgeReport)
                    .filter(
                        EdgeReport.report_type == "adversarial_tests",
                        EdgeReport.sport == self.sport,
                    )
                    .order_by(EdgeReport.generated_at.desc())
                    .first()
                )
                if report:
                    return json.loads(report.report_json or "{}")
        except Exception as e:
            log.warning("Failed to load adversarial results: %s", e)
        return None

    def _save_results(self, report: Dict[str, Any]) -> None:
        """Persist adversarial test results to database."""
        try:
            # Slim down for storage (remove full result details)
            storage = {
                "summary": report.get("summary", {}),
                "failures": report.get("failures", []),
                "test_verdicts": report.get("summary", {}).get("test_verdicts", {}),
            }
            with session_scope() as session:
                session.add(EdgeReport(
                    report_type="adversarial_tests",
                    sport=self.sport,
                    report_json=json.dumps(storage),
                ))
        except Exception as e:
            log.warning("Failed to save adversarial results: %s", e)
