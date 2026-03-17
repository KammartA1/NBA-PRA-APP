"""
PrizePicks Headless Scraper — Playwright-based
Runs as a companion to the Streamlit app.
Fetches fresh lines every N minutes and writes to a shared JSON file + SQLite DB.
Works from ANY IP (cloud, local, VPS) because it's a real browser.

Usage:
    python pp_scraper.py                    # continuous loop, 600s interval
    python pp_scraper.py --interval 300     # loop every 5 minutes
    python pp_scraper.py --once             # fetch once and exit
    python pp_scraper.py --once --visible   # fetch once, show browser window (debug)
"""
import asyncio
import json
import time
import os
import sys
import sqlite3
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pp_scraper")

PP_BOARD_URL = "https://app.prizepicks.com/board"
PP_API_PATTERN = "api.prizepicks.com/projections"
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(_HERE, ".pp_lines_cache.json")
OUTPUT_DB   = os.path.join(_HERE, "data", "nba_prizepicks.db")


# ──────────────────────────────────────────────
# DATABASE HELPERS
# ──────────────────────────────────────────────
def _ensure_db():
    os.makedirs(os.path.dirname(OUTPUT_DB), exist_ok=True)
    conn = sqlite3.connect(OUTPUT_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nba_prizepicks_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            line_score REAL NOT NULL,
            start_time TEXT,
            odds_type TEXT DEFAULT 'standard',
            is_latest INTEGER DEFAULT 1,
            fetched_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scraper_status (
            scraper_name TEXT PRIMARY KEY,
            last_success TEXT,
            last_error TEXT,
            rows_fetched INTEGER
        )
    """)
    conn.commit()
    return conn


# ──────────────────────────────────────────────
# JSON PARSER
# ──────────────────────────────────────────────
def _parse_pp_json(data: dict) -> list:
    """Parse PrizePicks API JSON into flat row dicts. Compatible with app.py _parse_pp_response logic."""
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

        # Accept NBA variants only
        league = str(attrs.get("league", "") or "").upper().strip()
        if league and not (league == "NBA" or league.startswith("NBA ")):
            continue

        rels = proj.get("relationships", {}) or {}
        player_id = ((rels.get("new_player") or {}).get("data") or {}).get("id")
        if not player_id:
            player_id = ((rels.get("player") or {}).get("data") or {}).get("id")
        player_attrs = included.get(player_id, {}).get("attributes", {}) if player_id else {}
        player_name = (
            player_attrs.get("name", "") or
            attrs.get("name", "") or
            attrs.get("display_name", "")
        )

        stat_type  = attrs.get("stat_type", "")
        line_score = attrs.get("line_score")
        odds_type  = str(attrs.get("odds_type", "") or "").lower().strip()
        rank_val   = attrs.get("rank", None)
        if not odds_type and rank_val is not None:
            try:
                odds_type = {1: "goblin", 2: "standard", 3: "demon"}.get(int(rank_val), "standard")
            except (TypeError, ValueError):
                odds_type = "standard"

        if player_name and stat_type and line_score is not None:
            try:
                rows.append({
                    "player":     player_name,
                    "stat_type":  stat_type,
                    "line":       float(line_score),
                    "start_time": attrs.get("start_time", ""),
                    "source":     "prizepicks",
                    "odds_type":  odds_type or "standard",
                })
            except (TypeError, ValueError):
                pass
    return rows


# ──────────────────────────────────────────────
# PLAYWRIGHT FETCH
# ──────────────────────────────────────────────
async def fetch_with_playwright(headless: bool = True, timeout_ms: int = 35000) -> list:
    """
    Launch Chromium, navigate to PrizePicks board, intercept the projections API response.
    Returns list of parsed prop dicts. Bypasses PerimeterX because it's a real browser.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        log.error("playwright not installed. Run: pip install playwright && playwright install chromium")
        return []

    captured_responses = []

    async def handle_response(response):
        url = response.url
        if PP_API_PATTERN in url and response.status == 200:
            try:
                body = await response.json()
                if isinstance(body, dict) and "data" in body:
                    captured_responses.append(body)
            except Exception:
                pass

    rows = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--window-size=1920,1080",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        # Stealth: remove webdriver fingerprint
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            try { delete navigator.__proto__.webdriver; } catch(e) {}
        """)

        page = await context.new_page()
        page.on("response", handle_response)

        try:
            log.info("Navigating to PrizePicks board...")
            await page.goto(PP_BOARD_URL, wait_until="networkidle", timeout=timeout_ms)
            await page.wait_for_timeout(5000)

            # If no API response yet, scroll to trigger lazy load
            if not captured_responses:
                log.info("No API response yet — scrolling to trigger lazy load...")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(3000)

            # If still nothing, try clicking the NBA tab
            if not captured_responses:
                log.info("Trying to click NBA tab...")
                try:
                    nba_tab = page.locator("text=NBA").first
                    if await nba_tab.is_visible(timeout=3000):
                        await nba_tab.click()
                        await page.wait_for_timeout(4000)
                except Exception:
                    pass

            # Also try fetching combo/specialty lines
            if not captured_responses:
                log.info("Trying direct API via page context...")
                try:
                    result = await page.evaluate("""
                        async () => {
                            const r = await fetch(
                                'https://api.prizepicks.com/projections?single_stat=true&per_page=500&league_id=7',
                                { credentials: 'include' }
                            );
                            return await r.json();
                        }
                    """)
                    if isinstance(result, dict) and "data" in result:
                        captured_responses.append(result)
                except Exception as e:
                    log.warning(f"Inline fetch fallback failed: {e}")

        except Exception as e:
            log.error(f"Navigation error: {e}")
        finally:
            await browser.close()

    # Parse all captured responses and deduplicate
    seen = set()
    for resp_body in captured_responses:
        parsed = _parse_pp_json(resp_body)
        for row in parsed:
            key = (row["player"], row["stat_type"])
            if key not in seen:
                seen.add(key)
                rows.append(row)

    log.info(f"Parsed {len(rows)} unique NBA props from {len(captured_responses)} API response(s)")
    return rows


# ──────────────────────────────────────────────
# SAVE TO DISK
# ──────────────────────────────────────────────
def save_to_disk(rows: list):
    """Write rows to JSON cache file and SQLite database."""
    # JSON cache (read by Streamlit via _load_pp_disk_cache)
    try:
        with open(OUTPUT_JSON, "w") as f:
            json.dump({"ts": time.time(), "rows": rows}, f)
        log.info(f"Wrote {len(rows)} rows to {OUTPUT_JSON}")
    except Exception as e:
        log.error(f"JSON write failed: {e}")

    # SQLite DB (read by Streamlit via _load_pp_from_scraper_db)
    try:
        conn = _ensure_db()
        now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        conn.execute("UPDATE nba_prizepicks_lines SET is_latest = 0")
        for row in rows:
            conn.execute(
                """INSERT INTO nba_prizepicks_lines
                   (player_name, stat_type, line_score, start_time, odds_type, is_latest, fetched_at)
                   VALUES (?, ?, ?, ?, ?, 1, ?)""",
                (
                    row["player"],
                    row["stat_type"],
                    row["line"],
                    row.get("start_time", ""),
                    row.get("odds_type", "standard"),
                    now,
                ),
            )
        conn.execute(
            "INSERT OR REPLACE INTO scraper_status (scraper_name, last_success, rows_fetched) VALUES (?, ?, ?)",
            ("nba_prizepicks", now, len(rows)),
        )
        conn.commit()
        conn.close()
        log.info(f"Wrote {len(rows)} rows to {OUTPUT_DB}")
    except Exception as e:
        log.error(f"SQLite write failed: {e}")


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
async def run_loop(interval_sec: int = 600, headless: bool = True):
    """Main loop: fetch → save → sleep → repeat."""
    log.info(f"PrizePicks scraper starting (interval={interval_sec}s, headless={headless})")
    while True:
        try:
            rows = await fetch_with_playwright(headless=headless)
            if rows:
                save_to_disk(rows)
                log.info(f"Success: {len(rows)} NBA props captured")
            else:
                log.warning("No rows captured this cycle")
                try:
                    conn = _ensure_db()
                    conn.execute(
                        "INSERT OR REPLACE INTO scraper_status (scraper_name, last_error) VALUES (?, ?)",
                        ("nba_prizepicks", "0 rows captured"),
                    )
                    conn.commit()
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            log.error(f"Cycle error: {e}")
            try:
                conn = _ensure_db()
                conn.execute(
                    "INSERT OR REPLACE INTO scraper_status (scraper_name, last_error) VALUES (?, ?)",
                    ("nba_prizepicks", str(e)[:500]),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass
        log.info(f"Sleeping {interval_sec}s until next fetch...")
        await asyncio.sleep(interval_sec)


def run_once(headless: bool = True) -> list:
    """Single fetch — callable from Streamlit's background thread."""
    return asyncio.run(fetch_with_playwright(headless=headless))


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PrizePicks headless scraper")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between fetches (default 600)")
    parser.add_argument("--once",    action="store_true",    help="Fetch once and exit")
    parser.add_argument("--visible", action="store_true",    help="Show browser window (debug mode)")
    args = parser.parse_args()

    if args.once:
        rows = run_once(headless=not args.visible)
        print(f"Fetched {len(rows)} NBA props")
        if rows:
            save_to_disk(rows)
            print(f"Sample: {rows[0]}")
    else:
        asyncio.run(run_loop(interval_sec=args.interval, headless=not args.visible))
