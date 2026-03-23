"""
workers/scheduler.py
====================
APScheduler-based scheduler that runs all workers on their configured
intervals with configurable overrides via environment variables.

Features:
  - Configurable intervals (ODDS_INTERVAL, SIGNAL_INTERVAL, etc.)
  - Graceful shutdown on SIGINT / SIGTERM
  - Health-check endpoint via a lightweight HTTP server (port 8099)
  - Can run as: ``python -m workers.scheduler``

Environment Variables:
  ODDS_INTERVAL     - Odds worker interval in seconds (default: 300)
  SIGNAL_INTERVAL   - Signal worker interval in seconds (default: 600)
  CLOSING_INTERVAL  - Closing worker interval in seconds (default: 300)
  MODEL_INTERVAL    - Model worker interval in seconds (default: 604800)
  REPORT_INTERVAL   - Report worker interval in seconds (default: 86400)
  HEALTH_PORT       - Health-check HTTP port (default: 8099)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from database.connection import init_db, session_scope
from database.models import WorkerStatus

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Worker registry
# ---------------------------------------------------------------------------

def _build_worker_configs() -> List[Dict[str, Any]]:
    """Return a list of worker configurations read from env or defaults."""
    return [
        {
            "name": "odds_worker",
            "interval": int(os.environ.get("ODDS_INTERVAL", "300")),
            "factory": "workers.odds_worker.OddsWorker",
        },
        {
            "name": "signal_worker",
            "interval": int(os.environ.get("SIGNAL_INTERVAL", "600")),
            "factory": "workers.signal_worker.SignalWorker",
        },
        {
            "name": "closing_worker",
            "interval": int(os.environ.get("CLOSING_INTERVAL", "300")),
            "factory": "workers.closing_worker.ClosingWorker",
        },
        {
            "name": "model_worker",
            "interval": int(os.environ.get("MODEL_INTERVAL", "604800")),
            "factory": "workers.model_worker.ModelWorker",
            # Model worker uses CronTrigger for Sunday 11 PM ET (4 AM UTC Monday)
            "cron": {"day_of_week": "mon", "hour": 4, "minute": 0},
        },
        {
            "name": "report_worker",
            "interval": int(os.environ.get("REPORT_INTERVAL", "86400")),
            "factory": "workers.report_worker.ReportWorker",
            # Report worker runs daily at 4 AM ET (9 AM UTC, or 8 AM during EDT)
            "cron": {"hour": 9, "minute": 0},
        },
        {
            "name": "stats_worker",
            "interval": int(os.environ.get("STATS_INTERVAL", "86400")),
            "factory": "workers.stats_worker.StatsWorker",
            # Stats worker runs daily at 6 AM ET (10 AM UTC) — ingests rosters, schedules
            "cron": {"hour": 10, "minute": 0},
        },
        {
            "name": "data_audit_worker",
            "interval": int(os.environ.get("AUDIT_INTERVAL", "1800")),
            "factory": "workers.data_audit_worker.DataAuditWorker",
        },
    ]


def _import_worker(factory_path: str):
    """Dynamically import a worker class from its dotted path."""
    module_path, class_name = factory_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# Health-check HTTP server
# ---------------------------------------------------------------------------

class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for /health endpoint."""

    # Reference to scheduler state (set at class level before serving)
    scheduler_state: Dict[str, Any] = {}

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self._respond(200, self.scheduler_state)
        elif self.path == "/workers":
            self._respond(200, self._worker_statuses())
        else:
            self._respond(404, {"error": "not_found"})

    def _respond(self, code: int, body: dict):
        payload = json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _worker_statuses(self) -> dict:
        try:
            with session_scope() as session:
                rows = session.query(WorkerStatus).all()
                return {"workers": [r.to_dict() for r in rows]}
        except Exception as exc:
            return {"error": str(exc)}

    def log_message(self, format, *args):
        # Suppress default access logging
        pass


def _start_health_server(port: int, state: Dict[str, Any]) -> Optional[HTTPServer]:
    """Start the health-check HTTP server in a daemon thread."""
    _HealthHandler.scheduler_state = state
    try:
        server = HTTPServer(("0.0.0.0", port), _HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        log.info("Health-check server running on port %d", port)
        return server
    except OSError as exc:
        log.warning("Could not start health server on port %d: %s", port, exc)
        return None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _run_worker_job(worker_instance):
    """Wrapper called by APScheduler for each job."""
    try:
        worker_instance.run()
    except Exception:
        log.exception("Unhandled exception in worker %s", worker_instance.name)


def start_scheduler():
    """Build and start the APScheduler with all registered workers.

    Blocks until interrupted.
    """
    # Initialise database
    init_db()
    log.info("Database initialised for scheduler")

    # Try APScheduler; fall back to simple threading if not installed
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.cron import CronTrigger
        _has_apscheduler = True
    except ImportError:
        _has_apscheduler = False
        log.warning(
            "APScheduler not installed -- falling back to threading-based scheduler. "
            "Install with: pip install apscheduler"
        )

    configs = _build_worker_configs()

    # Health-check state
    state: Dict[str, Any] = {
        "status": "running",
        "started_at": _utcnow().isoformat(),
        "workers": [c["name"] for c in configs],
        "scheduler_type": "apscheduler" if _has_apscheduler else "threading",
    }

    health_port = int(os.environ.get("HEALTH_PORT", "8099"))
    health_server = _start_health_server(health_port, state)

    # Instantiate workers
    worker_instances = []
    for cfg in configs:
        cls = _import_worker(cfg["factory"])
        instance = cls()
        worker_instances.append((cfg, instance))
        log.info(
            "Registered worker: %s (interval=%ds)",
            cfg["name"], cfg["interval"],
        )

    if _has_apscheduler:
        scheduler = BlockingScheduler(timezone="UTC")

        for cfg, instance in worker_instances:
            if "cron" in cfg and cfg["cron"]:
                trigger = CronTrigger(**cfg["cron"], timezone="UTC")
                scheduler.add_job(
                    _run_worker_job,
                    trigger=trigger,
                    args=[instance],
                    id=cfg["name"],
                    name=cfg["name"],
                    max_instances=1,
                    misfire_grace_time=300,
                )
                log.info("  %s: cron trigger %s", cfg["name"], cfg["cron"])
            else:
                trigger = IntervalTrigger(seconds=cfg["interval"])
                scheduler.add_job(
                    _run_worker_job,
                    trigger=trigger,
                    args=[instance],
                    id=cfg["name"],
                    name=cfg["name"],
                    max_instances=1,
                    misfire_grace_time=60,
                )
                log.info("  %s: interval trigger %ds", cfg["name"], cfg["interval"])

        # Run each worker once immediately on startup
        for cfg, instance in worker_instances:
            scheduler.add_job(
                _run_worker_job,
                args=[instance],
                id=f"{cfg['name']}_initial",
                name=f"{cfg['name']}_initial",
                max_instances=1,
            )

        log.info("Starting APScheduler with %d workers", len(worker_instances))
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("Scheduler shutting down...")
            scheduler.shutdown(wait=False)
    else:
        # Fallback: threading-based scheduler
        stop_event = threading.Event()

        def _sig_handler(signum, frame):
            log.info("Signal %d received, stopping all workers...", signum)
            stop_event.set()

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        threads = []
        for cfg, instance in worker_instances:
            t = threading.Thread(
                target=instance.run_forever,
                kwargs={"stop_event": stop_event},
                name=cfg["name"],
                daemon=True,
            )
            t.start()
            threads.append(t)
            log.info("Started thread for %s", cfg["name"])

        log.info("All %d workers running (threading mode)", len(threads))

        try:
            # Block main thread until stop event
            while not stop_event.is_set():
                stop_event.wait(timeout=5.0)
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt -- stopping workers")
            stop_event.set()

        # Wait for threads to finish
        for t in threads:
            t.join(timeout=10)

    state["status"] = "stopped"
    state["stopped_at"] = _utcnow().isoformat()

    if health_server:
        health_server.shutdown()

    log.info("Scheduler stopped.")


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    """CLI entry: ``python -m workers.scheduler``"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    log.info("=" * 60)
    log.info("NBA-PRA-APP Worker Scheduler")
    log.info("=" * 60)
    start_scheduler()


if __name__ == "__main__":
    main()
