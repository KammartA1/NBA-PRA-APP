#!/usr/bin/env python3
"""
NBA Scraper Monitor — check last successful pull time and health status.

Usage:
    python -m services.monitor              # quick status
    python -m services.monitor --watch      # live refresh every 10s
    python -m services.monitor --history 20 # last N fetch events
"""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.database import get_session, init_db, ScraperStatus, AuditLog, NbaPrizePicksLine


def print_status():
    session = get_session()
    try:
        status = session.query(ScraperStatus).filter_by(
            scraper_name="nba_prizepicks"
        ).first()

        print(f"\n{'='*55}")
        print(f"  NBA PRIZEPICKS SCRAPER STATUS")
        print(f"{'='*55}")

        if not status:
            print("  No scraper runs recorded yet.")
            return

        now = datetime.utcnow()
        if status.last_success:
            age = now - status.last_success
            age_min = int(age.total_seconds() / 60)
            health = "HEALTHY" if age_min < 45 else "STALE" if age_min < 120 else "DOWN"
            color = "\033[92m" if health == "HEALTHY" else "\033[93m" if health == "STALE" else "\033[91m"
            print(f"  Status:         {color}{health}\033[0m")
            print(f"  Last success:   {status.last_success.strftime('%Y-%m-%d %H:%M:%S UTC')} ({age_min} min ago)")
        else:
            print(f"  Status:         \033[91mNEVER SUCCEEDED\033[0m")

        print(f"  Last attempt:   {status.last_attempt.strftime('%Y-%m-%d %H:%M:%S UTC') if status.last_attempt else 'N/A'}")
        print(f"  Lines fetched:  {status.lines_fetched or 0}")
        print(f"  Total runs:     {status.total_runs or 0}")
        print(f"  Total errors:   {status.total_errors or 0}")
        if status.last_error:
            print(f"  Last error:     {status.last_error}")

        latest_count = session.query(NbaPrizePicksLine).filter(
            NbaPrizePicksLine.is_latest == True
        ).count()
        print(f"  Live lines:     {latest_count}")
        print(f"{'='*55}\n")

    finally:
        session.close()


def print_history(n: int = 10):
    session = get_session()
    try:
        events = session.query(AuditLog).filter(
            AuditLog.event_type.like("pp_scrape%")
        ).order_by(AuditLog.timestamp.desc()).limit(n).all()

        print(f"\nLast {n} scraper events:")
        print(f"{'─'*60}")
        for e in events:
            ts = e.timestamp.strftime("%m/%d %H:%M") if e.timestamp else "?"
            icon = "+" if "success" in (e.event_type or "") else "x"
            print(f"  [{ts}] {icon} {e.description}")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="NBA PrizePicks Scraper Monitor")
    parser.add_argument("--watch", action="store_true", help="Refresh every 10 seconds")
    parser.add_argument("--history", type=int, default=0, help="Show last N events")
    args = parser.parse_args()

    init_db()

    if args.history:
        print_history(args.history)
        return

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")
                print(f"  Monitoring — {datetime.now().strftime('%H:%M:%S')}  (Ctrl-C to stop)")
                print_status()
                time.sleep(10)
        except KeyboardInterrupt:
            return
    else:
        print_status()


if __name__ == "__main__":
    main()
