"""
workers/report_worker.py
========================
Generates comprehensive daily edge reports at 4 AM ET (after all NBA games).

Report contents:
  1. P&L summary (daily, weekly, monthly)
  2. CLV analysis (rolling 50, 100, 250, 500 bets)
  3. Calibration report (predicted vs actual by bucket)
  4. Edge decay analysis
  5. Book efficiency (which book gives best lines)
  6. Player / market breakdown
  7. System state recommendation

Run standalone:
    python -m workers.report_worker          # one-shot
    python -m workers.report_worker --loop   # daily loop
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from database.connection import session_scope, init_db
from database.models import (
    Bet, LineMovement, EdgeReport, Signal,
    CalibrationSnapshot, SystemState as SystemStateModel,
)
from workers.base import BaseWorker, standalone_main

log = logging.getLogger(__name__)

# Daily cadence: 24 hours in seconds
DAILY_INTERVAL = 24 * 60 * 60

# Calibration bucket edges (same as model_worker for consistency)
CALIBRATION_EDGES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ===================================================================
# Report generation functions
# ===================================================================

def _pnl_summary(session, now: datetime) -> Dict[str, Any]:
    """Compute P&L for daily, weekly, and monthly windows."""
    windows = {
        "daily": now - timedelta(days=1),
        "weekly": now - timedelta(days=7),
        "monthly": now - timedelta(days=30),
        "all_time": datetime(2020, 1, 1, tzinfo=timezone.utc),
    }
    result: Dict[str, Any] = {}

    for label, cutoff in windows.items():
        bets = (
            session.query(Bet)
            .filter(
                Bet.sport == "NBA",
                Bet.status.in_(["won", "lost", "push"]),
                Bet.settled_at >= cutoff,
            )
            .all()
        )
        total_staked = sum(b.stake or 0 for b in bets)
        total_pnl = sum(b.pnl or 0 for b in bets)
        wins = sum(1 for b in bets if b.status == "won")
        losses = sum(1 for b in bets if b.status == "lost")
        pushes = sum(1 for b in bets if b.status == "push")

        roi = (total_pnl / max(total_staked, 1.0)) * 100.0

        result[label] = {
            "total_bets": len(bets),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(wins / max(wins + losses, 1), 4),
            "total_staked": round(total_staked, 2),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(roi, 2),
        }

    return result


def _clv_analysis(session) -> Dict[str, Any]:
    """CLV analysis across rolling windows of 50, 100, 250, 500 bets."""
    bets = (
        session.query(Bet)
        .filter(
            Bet.sport == "NBA",
            Bet.status.in_(["won", "lost"]),
            Bet.closing_line != None,
        )
        .order_by(Bet.timestamp.desc())
        .limit(500)
        .all()
    )

    if not bets:
        return {"available": False, "reason": "no_bets_with_closing_lines"}

    result: Dict[str, Any] = {"available": True}

    for window_size in [50, 100, 250, 500]:
        window_bets = bets[:window_size]
        if not window_bets:
            result[f"last_{window_size}"] = None
            continue

        clv_values = []
        beat_close_count = 0
        for b in window_bets:
            if b.closing_line is None or b.bet_line is None:
                continue
            direction = (b.direction or "over").lower()
            if direction in ("over", "o"):
                clv = b.closing_line - b.bet_line
            else:
                clv = b.bet_line - b.closing_line

            clv_pct = (clv / max(abs(b.bet_line), 0.5)) * 100.0
            clv_values.append(clv_pct)
            if clv > 0:
                beat_close_count += 1

        if clv_values:
            arr = np.array(clv_values)
            result[f"last_{window_size}"] = {
                "n_bets": len(clv_values),
                "mean_clv_pct": round(float(arr.mean()), 3),
                "median_clv_pct": round(float(np.median(arr)), 3),
                "std_clv_pct": round(float(arr.std()), 3),
                "beat_close_rate": round(beat_close_count / len(clv_values), 4),
                "positive_clv": float(arr.mean()) > 0,
            }
        else:
            result[f"last_{window_size}"] = None

    return result


def _calibration_report(session) -> Dict[str, Any]:
    """Calibration report: predicted vs actual by probability bucket."""
    bets = (
        session.query(Bet)
        .filter(
            Bet.sport == "NBA",
            Bet.status.in_(["won", "lost"]),
            Bet.predicted_prob != None,
        )
        .order_by(Bet.timestamp.desc())
        .limit(1000)
        .all()
    )

    if len(bets) < 20:
        return {"available": False, "reason": f"insufficient_bets ({len(bets)})"}

    probs = np.array([b.predicted_prob for b in bets])
    outcomes = np.array([1.0 if b.status == "won" else 0.0 for b in bets])

    buckets = []
    total_abs_error = 0.0
    total_n = 0

    for i in range(len(CALIBRATION_EDGES) - 1):
        lo = CALIBRATION_EDGES[i]
        hi = CALIBRATION_EDGES[i + 1]
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            continue

        pred_avg = float(probs[mask].mean())
        actual_rate = float(outcomes[mask].mean())
        cal_error = abs(pred_avg - actual_rate)

        buckets.append({
            "range": f"{lo:.2f}-{hi:.2f}",
            "predicted_avg": round(pred_avg, 4),
            "actual_rate": round(actual_rate, 4),
            "n_bets": n,
            "calibration_error": round(cal_error, 4),
            "is_overconfident": pred_avg > actual_rate,
        })
        total_abs_error += cal_error * n
        total_n += n

    mean_cal_error = total_abs_error / max(total_n, 1)
    brier = float(np.mean((probs - outcomes) ** 2))

    return {
        "available": True,
        "n_bets": len(bets),
        "mean_calibration_error": round(mean_cal_error, 5),
        "brier_score": round(brier, 5),
        "buckets": buckets,
    }


def _edge_decay_analysis(session, now: datetime) -> Dict[str, Any]:
    """Analyse whether edge is shrinking over time.

    Computes rolling ROI in monthly windows to detect trend.
    """
    bets = (
        session.query(Bet)
        .filter(
            Bet.sport == "NBA",
            Bet.status.in_(["won", "lost"]),
        )
        .order_by(Bet.timestamp.asc())
        .all()
    )

    if len(bets) < 30:
        return {"available": False, "reason": f"insufficient_bets ({len(bets)})"}

    # Group by month
    monthly: Dict[str, Dict[str, float]] = defaultdict(lambda: {"staked": 0.0, "pnl": 0.0, "n": 0})
    for b in bets:
        if b.timestamp:
            month_key = b.timestamp.strftime("%Y-%m")
            monthly[month_key]["staked"] += b.stake or 0
            monthly[month_key]["pnl"] += b.pnl or 0
            monthly[month_key]["n"] += 1

    months_sorted = sorted(monthly.keys())
    monthly_roi = []
    for m in months_sorted:
        d = monthly[m]
        roi = (d["pnl"] / max(d["staked"], 1.0)) * 100.0
        monthly_roi.append({
            "month": m,
            "roi_pct": round(roi, 2),
            "n_bets": d["n"],
            "pnl": round(d["pnl"], 2),
        })

    # Detect trend: linear regression on monthly ROI
    if len(monthly_roi) >= 3:
        roi_values = np.array([m["roi_pct"] for m in monthly_roi])
        x = np.arange(len(roi_values), dtype=float)
        slope = float(np.polyfit(x, roi_values, 1)[0])
        is_decaying = slope < -0.5  # More than 0.5% ROI decay per month
    else:
        slope = 0.0
        is_decaying = False

    return {
        "available": True,
        "monthly_roi": monthly_roi,
        "slope_per_month": round(slope, 3),
        "is_decaying": is_decaying,
        "verdict": "EDGE DECAYING" if is_decaying else "EDGE STABLE",
    }


def _book_efficiency(session) -> Dict[str, Any]:
    """Which book gives the best lines? Based on CLV by book."""
    bets = (
        session.query(Bet)
        .filter(
            Bet.sport == "NBA",
            Bet.status.in_(["won", "lost"]),
            Bet.closing_line != None,
        )
        .all()
    )

    if not bets:
        return {"available": False}

    # Group by book (from features_snapshot)
    book_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"clv_sum": 0.0, "n": 0, "pnl": 0.0, "staked": 0.0}
    )

    for b in bets:
        features = {}
        try:
            features = json.loads(b.features_snapshot_json or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        book = features.get("book") or features.get("closing_book") or "unknown"
        direction = (b.direction or "over").lower()

        if b.closing_line is not None and b.bet_line is not None:
            if direction in ("over", "o"):
                clv = b.closing_line - b.bet_line
            else:
                clv = b.bet_line - b.closing_line
            clv_pct = (clv / max(abs(b.bet_line), 0.5)) * 100.0
            book_stats[book]["clv_sum"] += clv_pct
            book_stats[book]["n"] += 1
            book_stats[book]["pnl"] += b.pnl or 0
            book_stats[book]["staked"] += b.stake or 0

    rankings = []
    for book, stats in book_stats.items():
        avg_clv = stats["clv_sum"] / max(stats["n"], 1)
        roi = (stats["pnl"] / max(stats["staked"], 1.0)) * 100.0
        rankings.append({
            "book": book,
            "n_bets": stats["n"],
            "avg_clv_pct": round(avg_clv, 3),
            "roi_pct": round(roi, 2),
            "total_pnl": round(stats["pnl"], 2),
        })

    rankings.sort(key=lambda x: x["avg_clv_pct"], reverse=True)

    return {
        "available": True,
        "rankings": rankings,
        "best_book": rankings[0]["book"] if rankings else None,
    }


def _player_market_breakdown(session) -> Dict[str, Any]:
    """Where is edge concentrated by player and market type?"""
    bets = (
        session.query(Bet)
        .filter(
            Bet.sport == "NBA",
            Bet.status.in_(["won", "lost"]),
        )
        .all()
    )

    if not bets:
        return {"available": False}

    # By market
    market_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"pnl": 0.0, "staked": 0.0, "n": 0, "wins": 0}
    )
    # By player (top performers)
    player_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"pnl": 0.0, "staked": 0.0, "n": 0, "wins": 0}
    )

    for b in bets:
        market = b.market or "unknown"
        market_stats[market]["pnl"] += b.pnl or 0
        market_stats[market]["staked"] += b.stake or 0
        market_stats[market]["n"] += 1
        if b.status == "won":
            market_stats[market]["wins"] += 1

        player = b.player or "unknown"
        player_stats[player]["pnl"] += b.pnl or 0
        player_stats[player]["staked"] += b.stake or 0
        player_stats[player]["n"] += 1
        if b.status == "won":
            player_stats[player]["wins"] += 1

    # Format market breakdown
    market_breakdown = []
    for mkt, stats in sorted(market_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        roi = (stats["pnl"] / max(stats["staked"], 1.0)) * 100.0
        market_breakdown.append({
            "market": mkt,
            "n_bets": int(stats["n"]),
            "win_rate": round(stats["wins"] / max(stats["n"], 1), 4),
            "pnl": round(stats["pnl"], 2),
            "roi_pct": round(roi, 2),
        })

    # Top 20 players by absolute PnL
    player_breakdown = []
    for pl, stats in sorted(player_stats.items(), key=lambda x: abs(x[1]["pnl"]), reverse=True)[:20]:
        roi = (stats["pnl"] / max(stats["staked"], 1.0)) * 100.0
        player_breakdown.append({
            "player": pl,
            "n_bets": int(stats["n"]),
            "win_rate": round(stats["wins"] / max(stats["n"], 1), 4),
            "pnl": round(stats["pnl"], 2),
            "roi_pct": round(roi, 2),
        })

    return {
        "available": True,
        "by_market": market_breakdown,
        "top_players": player_breakdown,
    }


def _system_state_recommendation(
    pnl_data: Dict,
    clv_data: Dict,
    calibration_data: Dict,
    edge_decay: Dict,
) -> Dict[str, Any]:
    """Generate a system state recommendation based on all report sections."""
    warnings: List[str] = []
    actions: List[str] = []
    recommended_state = "ACTIVE"

    # CLV checks
    if clv_data.get("available"):
        clv_100 = clv_data.get("last_100")
        if clv_100:
            mean_clv = clv_100.get("mean_clv_pct", 0)
            if mean_clv < -2.0:
                warnings.append(f"Negative CLV over last 100 bets: {mean_clv:.2f}%")
                recommended_state = "REDUCED"
                actions.append("Reduce stake sizes by 50%")
            if mean_clv < -5.0:
                warnings.append(f"Severely negative CLV: {mean_clv:.2f}% -- market has adapted")
                recommended_state = "SUSPENDED"
                actions.append("Suspend all new bets until model retrain completes")

            beat_rate = clv_100.get("beat_close_rate", 0.5)
            if beat_rate < 0.45:
                warnings.append(f"Low beat-the-close rate: {beat_rate:.1%}")

    # Calibration checks
    if calibration_data.get("available"):
        cal_error = calibration_data.get("mean_calibration_error", 0)
        if cal_error > 0.08:
            warnings.append(f"High calibration error: {cal_error:.4f}")
            actions.append("Trigger emergency model retrain")
            if recommended_state == "ACTIVE":
                recommended_state = "REDUCED"

    # Edge decay
    if edge_decay.get("available") and edge_decay.get("is_decaying"):
        warnings.append(f"Edge decaying at {edge_decay['slope_per_month']:.2f}% ROI/month")
        actions.append("Review feature set for staleness")

    # P&L checks
    weekly = pnl_data.get("weekly", {})
    if weekly.get("total_bets", 0) > 10 and weekly.get("roi_pct", 0) < -15:
        warnings.append(f"Severe weekly loss: {weekly['roi_pct']:.1f}% ROI")
        if recommended_state != "SUSPENDED":
            recommended_state = "REDUCED"
        actions.append("Review recent bet distribution for anomalies")

    # Drawdown check
    monthly = pnl_data.get("monthly", {})
    if monthly.get("total_bets", 0) > 20 and monthly.get("roi_pct", 0) < -20:
        warnings.append(f"Monthly drawdown critical: {monthly['roi_pct']:.1f}% ROI")
        recommended_state = "SUSPENDED"
        actions.append("Full system review before resuming")

    if not warnings:
        actions.append("System healthy -- continue normal operations")

    return {
        "recommended_state": recommended_state,
        "warnings": warnings,
        "actions": actions,
    }


# ===================================================================
# Worker class
# ===================================================================

class ReportWorker(BaseWorker):
    """Generates daily edge reports at 4 AM ET."""

    def __init__(self, **kwargs):
        super().__init__(
            name="report_worker",
            interval_seconds=int(os.environ.get("REPORT_INTERVAL", str(DAILY_INTERVAL))),
            max_retries=2,
            retry_delay=30.0,
            **kwargs,
        )

    def execute(self) -> Dict[str, Any]:
        now = _utcnow()
        self.logger.info("Generating daily edge report for %s", now.date().isoformat())

        with session_scope() as session:
            # Generate all report sections
            pnl = _pnl_summary(session, now)
            clv = _clv_analysis(session)
            calibration = _calibration_report(session)
            edge_decay = _edge_decay_analysis(session, now)
            books = _book_efficiency(session)
            breakdown = _player_market_breakdown(session)
            system_rec = _system_state_recommendation(pnl, clv, calibration, edge_decay)

            # Assemble full report
            report = {
                "generated_at": now.isoformat(),
                "report_date": now.date().isoformat(),
                "pnl_summary": pnl,
                "clv_analysis": clv,
                "calibration": calibration,
                "edge_decay": edge_decay,
                "book_efficiency": books,
                "player_market_breakdown": breakdown,
                "system_recommendation": system_rec,
            }

            # Store in edge_reports table
            er = EdgeReport(
                report_type="daily_edge_report",
                generated_at=now,
                report_json=json.dumps(report, default=str),
                sport="NBA",
            )
            session.add(er)

            # If system recommendation changed, log a SystemState record
            current_state_row = (
                session.query(SystemStateModel)
                .filter(SystemStateModel.sport == "NBA")
                .order_by(SystemStateModel.changed_at.desc())
                .first()
            )
            current_state = current_state_row.state if current_state_row else "ACTIVE"
            new_state = system_rec["recommended_state"]

            if new_state != current_state:
                self.logger.warning(
                    "System state change recommended: %s -> %s",
                    current_state, new_state,
                )
                clv_at = None
                if clv.get("available"):
                    c100 = clv.get("last_100")
                    if c100:
                        clv_at = c100.get("mean_clv_pct")

                session.add(SystemStateModel(
                    sport="NBA",
                    state=new_state,
                    reason="; ".join(system_rec["warnings"]) or "Report-driven state change",
                    changed_at=now,
                    clv_at_change=clv_at,
                ))

            # Summary stats for the return dict
            n_warnings = len(system_rec["warnings"])
            total_bets_all = pnl.get("all_time", {}).get("total_bets", 0)
            daily_pnl = pnl.get("daily", {}).get("total_pnl", 0)

        self.logger.info(
            "Report generated: %d total bets, daily P&L $%.2f, %d warnings, state=%s",
            total_bets_all, daily_pnl, n_warnings, new_state,
        )

        return {
            "ok": True,
            "report_date": now.date().isoformat(),
            "total_bets_all_time": total_bets_all,
            "daily_pnl": round(daily_pnl, 2),
            "warnings": n_warnings,
            "recommended_state": new_state,
        }


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    standalone_main(ReportWorker)
