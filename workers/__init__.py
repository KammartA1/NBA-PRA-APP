"""
workers — Background worker layer for NBA-PRA-APP
==================================================
Independent of Streamlit.  Each worker logs to the database, has error
handling with retries, and can be run standalone or via the scheduler.

Quick start::

    # Run all workers via APScheduler
    python -m workers.scheduler

    # Run a single worker one-shot
    python -m workers.odds_worker
    python -m workers.signal_worker
    python -m workers.closing_worker
    python -m workers.model_worker
    python -m workers.report_worker

    # Run a single worker in loop mode
    python -m workers.odds_worker --loop

Programmatic usage::

    from workers import run_all, OddsWorker, SignalWorker

    # Start all workers (blocking)
    run_all()

    # Or run a single worker
    w = OddsWorker()
    result = w.run()
"""

from workers.base import BaseWorker
from workers.odds_worker import OddsWorker
from workers.signal_worker import SignalWorker
from workers.closing_worker import ClosingWorker
from workers.model_worker import ModelWorker
from workers.report_worker import ReportWorker
from workers.stats_worker import StatsWorker

__all__ = [
    "BaseWorker",
    "OddsWorker",
    "SignalWorker",
    "ClosingWorker",
    "ModelWorker",
    "ReportWorker",
    "StatsWorker",
    "run_all",
    "main",
]


def run_all():
    """Start all workers via APScheduler (or threading fallback).

    This is a blocking call -- use ``Ctrl+C`` to stop.
    """
    from workers.scheduler import start_scheduler
    start_scheduler()


def main():
    """CLI entry point: ``python -m workers``"""
    import sys
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        sys.exit(0)

    run_all()


if __name__ == "__main__":
    main()
