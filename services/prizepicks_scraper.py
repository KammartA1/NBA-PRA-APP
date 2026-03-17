#!/usr/bin/env python3
"""
NBA PrizePicks Scraper Service
===============================
Runs 24/7 as a systemd service. Pulls live NBA prop lines from PrizePicks API
and stores them in SQLite. Feeds the Streamlit dashboard with fresh data.

Schedule:
  - During NBA season (Oct–Jun): every 30 minutes
  - Off-season (Jul–Sep):        every 6 hours (monitoring for preseason)

Usage:
    python -m services.prizepicks_scraper          # foreground
    systemctl start nba-prizepicks-scraper         # via systemd
"""
import sys
import os
import time
import signal
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.database import (
    get_session, init_db, NbaPrizePicksLine, ScraperStatus, AuditLog, DB_DIR,
)

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR = DB_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "prizepicks_scraper.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("nba_pp_scraper")

# ── PrizePicks API ─────────────────────────────────────────────────────────
PRIZEPICKS_API = "https://api.prizepicks.com/projections"
NBA_LEAGUES = {"NBA", "NBA 1Q", "NBA 1H", "NBA 2H"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://app.prizepicks.com/",
    "Origin": "https://app.prizepicks.com",
}

# ── Schedule constants ─────────────────────────────────────────────────────
# NBA regular season: ~mid-Oct through mid-Jun (playoffs)
NBA_SEASON_MONTHS = {10, 11, 12, 1, 2, 3, 4, 5, 6}
INTERVAL_SEASON   = 30 * 60     # 30 minutes
INTERVAL_OFFSEASON = 6 * 60 * 60  # 6 hours

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.info(f"Received signal {signum}, shutting down gracefully...")
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def is_nba_season() -> bool:
    return datetime.now().month in NBA_SEASON_MONTHS


def get_interval() -> int:
    return INTERVAL_SEASON if is_nba_season() else INTERVAL_OFFSEASON


def _is_nba_league(league_str: str) -> bool:
    return league_str.upper().strip() in NBA_LEAGUES


def _request(per_page: int = 500, single_stat: str = "true", retries: int = 3):
    """Make a request to PrizePicks API with retry logic."""
    import requests

    params = {
        "per_page": str(per_page),
        "single_stat": single_stat,
        "in_play": "false",
    }

    for attempt in range(retries):
        try:
            # Try curl_cffi first (best anti-bot bypass)
            try:
                from curl_cffi import requests as cffi_req
                r = cffi_req.get(
                    PRIZEPICKS_API, params=params, headers=HEADERS,
                    impersonate="chrome120", timeout=25,
                )
                if r.status_code not in (403, 429):
                    return r.json(), None
                log.warning(f"curl_cffi got {r.status_code} on attempt {attempt+1}")
            except ImportError:
                pass
            except Exception as e:
                log.warning(f"curl_cffi error: {e}")

            # Fallback: plain requests
            r = requests.get(PRIZEPICKS_API, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json(), None

        except Exception as e:
            log.warning(f"Request error attempt {attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))

    return None, "All retries failed"


def parse_response(data: dict) -> list[dict]:
    """Parse PrizePicks JSON:API response into flat dicts, filtered to NBA."""
    included = {
        item["id"]: item
        for item in data.get("included", [])
        if isinstance(item, dict) and "id" in item
    }
    rows = []
    for proj in data.get("data", []):
        if not isinstance(proj, dict):
            continue
        attrs = proj.get("attributes", {}) or {}

        # League filter
        league = str(attrs.get("league", "") or "")
        if league and not _is_nba_league(league):
            continue

        stat_type = attrs.get("stat_type", "")
        line_score = attrs.get("line_score")
        if not stat_type or line_score is None:
            continue

        # Resolve player name
        rels = proj.get("relationships", {}) or {}
        player_id = (rels.get("new_player", {}).get("data", {}) or {}).get("id")
        if not player_id:
            player_id = (rels.get("player", {}).get("data", {}) or {}).get("id")
        player_attrs = included.get(player_id, {}).get("attributes", {}) if player_id else {}
        player_name = (
            player_attrs.get("name", "")
            or player_attrs.get("display_name", "")
            or attrs.get("name", "")
            or attrs.get("description", "")
        )

        if not player_name:
            continue

        odds_type = str(attrs.get("odds_type", "") or "").lower().strip() or "standard"

        try:
            rows.append({
                "player_name": player_name,
                "stat_type": stat_type,
                "line_score": float(line_score),
                "start_time": attrs.get("start_time", ""),
                "odds_type": odds_type,
                "league": league or "NBA",
            })
        except (TypeError, ValueError):
            pass

    return rows


def fetch_all_nba_lines() -> tuple[list[dict], str | None]:
    """Fetch standard + combo markets, deduplicate."""
    all_rows = []
    seen = set()
    last_err = None

    for single_stat in ("true", "false"):
        data, err = _request(500, single_stat)
        if err:
            last_err = err
            continue
        if data is None:
            last_err = "No response"
            continue

        rows = parse_response(data)
        for row in rows:
            key = (row["player_name"], row["stat_type"])
            if key not in seen:
                seen.add(key)
                all_rows.append(row)

        time.sleep(2)  # Rate limit between calls

    if all_rows:
        return all_rows, None
    return [], last_err or "No NBA props found"


def store_lines(rows: list[dict]) -> int:
    """Save fetched lines to the database."""
    session = get_session()
    try:
        # Mark previous batch as not-latest
        session.query(NbaPrizePicksLine).filter(
            NbaPrizePicksLine.is_latest == True
        ).update({"is_latest": False})

        now = datetime.utcnow()
        count = 0
        for row in rows:
            line = NbaPrizePicksLine(
                player_name=row["player_name"],
                stat_type=row["stat_type"],
                line_score=row["line_score"],
                start_time=row.get("start_time", ""),
                odds_type=row.get("odds_type", "standard"),
                league=row.get("league", "NBA"),
                fetched_at=now,
                is_latest=True,
            )
            session.add(line)
            count += 1

        session.commit()

        # Also write the disk cache that the existing app.py reads
        _save_disk_cache(rows)

        return count
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _save_disk_cache(rows: list[dict]):
    """Write lines to the JSON disk cache that app.py already reads."""
    cache_path = PROJECT_ROOT / ".pp_lines_cache.json"
    try:
        # Convert to the format app.py expects
        cache_rows = []
        for row in rows:
            cache_rows.append({
                "player": row["player_name"],
                "stat_type": row["stat_type"],
                "line": row["line_score"],
                "start_time": row.get("start_time", ""),
                "source": "prizepicks",
                "odds_type": row.get("odds_type", "standard"),
            })
        with open(cache_path, "w") as f:
            json.dump({"ts": time.time(), "rows": cache_rows}, f)
        log.info(f"Disk cache updated: {cache_path}")
    except Exception as e:
        log.warning(f"Failed to write disk cache: {e}")


def update_status(success: bool, lines_count: int = 0, error: str = None):
    session = get_session()
    try:
        status = session.query(ScraperStatus).filter_by(
            scraper_name="nba_prizepicks"
        ).first()
        if not status:
            status = ScraperStatus(scraper_name="nba_prizepicks")
            session.add(status)

        now = datetime.utcnow()
        status.last_attempt = now
        status.total_runs = (status.total_runs or 0) + 1
        if success:
            status.last_success = now
            status.lines_fetched = lines_count
            status.last_error = None
        else:
            status.last_error = error
            status.total_errors = (status.total_errors or 0) + 1

        session.commit()
    except Exception as e:
        session.rollback()
        log.error(f"Failed to update scraper status: {e}")
    finally:
        session.close()


def log_audit(event_type: str, description: str, data: dict = None):
    session = get_session()
    try:
        entry = AuditLog(
            event_type=event_type,
            description=description,
            data_json=json.dumps(data) if data else None,
        )
        session.add(entry)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


def run_once() -> bool:
    """Execute a single scrape cycle. Returns True on success."""
    try:
        log.info("Fetching PrizePicks NBA lines...")
        rows, err = fetch_all_nba_lines()

        if err and not rows:
            log.warning(f"Fetch returned error: {err}")
            update_status(success=False, error=err)
            log_audit("pp_scrape_error", f"NBA scrape failed: {err}")
            return False

        if not rows:
            log.info("No NBA props available (off-hours or empty slate)")
            update_status(success=True, lines_count=0)
            return True

        count = store_lines(rows)
        stat_types = list({r["stat_type"] for r in rows})
        log.info(f"Stored {count} NBA PrizePicks lines ({len(stat_types)} stat types)")
        update_status(success=True, lines_count=count)
        log_audit("pp_scrape_success", f"Stored {count} NBA lines", {
            "count": count,
            "stat_types": stat_types,
        })
        return True

    except Exception as e:
        log.error(f"Scrape failed: {e}", exc_info=True)
        update_status(success=False, error=str(e))
        log_audit("pp_scrape_error", f"NBA scrape failed: {e}")
        return False


def main():
    log.info("=" * 60)
    log.info("NBA PrizePicks Scraper Service starting")
    log.info(f"  Project root: {PROJECT_ROOT}")
    log.info(f"  Log file:     {LOG_FILE}")
    log.info("=" * 60)

    init_db()

    while not _shutdown:
        run_once()
        interval = get_interval()
        mode = "season" if is_nba_season() else "off-season"
        log.info(f"Next pull in {interval // 60} min ({mode} schedule)")

        waited = 0
        while waited < interval and not _shutdown:
            time.sleep(min(10, interval - waited))
            waited += 10

    log.info("Scraper service stopped.")


if __name__ == "__main__":
    main()
