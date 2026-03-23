"""
workers/stats_worker.py
========================
Daily data ingestion pipeline for NBA player stats, game schedules, and rosters.

Runs once daily (default 10 AM UTC / 6 AM ET):
  1. Refresh NBA player roster from ESPN / NBA API
  2. Update player stats (PRA baselines, minutes, usage)
  3. Sync today's game schedule into events table
  4. Update player metadata (injury status, team changes)

All writes go to the players and events tables via the database ORM.

Run standalone:
    python -m workers.stats_worker
    python -m workers.stats_worker --loop
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workers.base import BaseWorker
from database.connection import init_db, session_scope, get_session_factory
from database.models import Player, Event

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StatsWorker(BaseWorker):
    name = "stats_worker"
    interval_seconds = int(os.environ.get("STATS_INTERVAL", "86400"))  # 24 hours
    max_retries = 2
    retry_delay_seconds = 60.0
    description = "Daily NBA data ingestion: rosters, stats, game schedules"

    def execute(self) -> dict:
        init_db()
        factory = get_session_factory()

        players_updated = 0
        games_synced = 0
        errors: list[str] = []

        # 1. Refresh player roster
        try:
            players_updated = self._refresh_roster(factory)
        except Exception as exc:
            msg = f"Roster refresh failed: {exc}"
            self._logger.error(msg, exc_info=True)
            errors.append(msg)

        # 2. Sync today's game schedule
        try:
            games_synced = self._sync_game_schedule(factory)
        except Exception as exc:
            msg = f"Game schedule sync failed: {exc}"
            self._logger.error(msg, exc_info=True)
            errors.append(msg)

        # 3. Update player metadata (injuries, status)
        try:
            self._update_player_metadata(factory)
        except Exception as exc:
            msg = f"Player metadata update failed: {exc}"
            self._logger.error(msg, exc_info=True)
            errors.append(msg)

        total = players_updated + games_synced
        if total == 0 and errors:
            raise RuntimeError(f"All stats sources failed: {'; '.join(errors)}")

        return {
            "items_processed": total,
            "players_updated": players_updated,
            "games_synced": games_synced,
            "errors": errors,
        }

    # ------------------------------------------------------------------ #
    # 1. Refresh NBA roster from ESPN
    # ------------------------------------------------------------------ #
    def _refresh_roster(self, factory) -> int:
        """Fetch NBA players from ESPN API and upsert into players table."""
        import requests

        teams_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
        count = 0

        try:
            resp = requests.get(teams_url, params={"limit": 50}, timeout=15)
            if resp.status_code != 200:
                self._logger.warning("ESPN teams API returned %d", resp.status_code)
                return 0

            teams_data = resp.json().get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])

            session = factory()
            try:
                now = _utcnow()

                for team_entry in teams_data:
                    team_info = team_entry.get("team", {})
                    team_abbr = team_info.get("abbreviation", "")
                    team_name = team_info.get("displayName", "")
                    team_id = team_info.get("id", "")

                    # Fetch roster for each team
                    roster_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
                    try:
                        roster_resp = requests.get(roster_url, timeout=10)
                        if roster_resp.status_code != 200:
                            continue

                        athletes = roster_resp.json().get("athletes", [])
                        for athlete in athletes:
                            name = athlete.get("displayName", athlete.get("fullName", ""))
                            if not name:
                                continue

                            position = athlete.get("position", {}).get("abbreviation", "")
                            espn_id = str(athlete.get("id", ""))

                            existing = session.query(Player).filter_by(
                                name=name, sport="NBA"
                            ).first()

                            if existing is None:
                                player = Player(
                                    name=name,
                                    team=team_abbr,
                                    sport="NBA",
                                    active=True,
                                    position=position,
                                    last_updated=now,
                                    metadata_json=json.dumps({
                                        "espn_id": espn_id,
                                        "team_name": team_name,
                                    }),
                                )
                                session.add(player)
                                count += 1
                            else:
                                # Update team/position if changed
                                if existing.team != team_abbr:
                                    existing.team = team_abbr
                                if position and existing.position != position:
                                    existing.position = position
                                existing.active = True
                                existing.last_updated = now

                    except Exception as exc:
                        self._logger.debug("Roster fetch failed for %s: %s", team_abbr, exc)
                        continue

                session.commit()
                self._logger.info("Roster refresh: %d new players added", count)
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        except Exception as exc:
            self._logger.error("ESPN roster fetch failed: %s", exc)

        return count

    # ------------------------------------------------------------------ #
    # 2. Sync game schedule
    # ------------------------------------------------------------------ #
    def _sync_game_schedule(self, factory) -> int:
        """Fetch today's NBA games from ESPN and upsert into events table."""
        import requests

        now = _utcnow()
        today = now.strftime("%Y%m%d")

        scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        count = 0

        try:
            resp = requests.get(
                scoreboard_url,
                params={"dates": today},
                timeout=15,
            )
            if resp.status_code != 200:
                self._logger.warning("ESPN scoreboard API returned %d", resp.status_code)
                return 0

            events_data = resp.json().get("events", [])

            session = factory()
            try:
                for game in events_data:
                    game_name = game.get("name", game.get("shortName", ""))
                    espn_id = str(game.get("id", ""))
                    status_type = game.get("status", {}).get("type", {}).get("name", "")

                    # Map ESPN status to our status
                    status_map = {
                        "STATUS_SCHEDULED": "scheduled",
                        "STATUS_IN_PROGRESS": "live",
                        "STATUS_HALFTIME": "live",
                        "STATUS_FINAL": "completed",
                        "STATUS_POSTPONED": "cancelled",
                    }
                    status = status_map.get(status_type, "scheduled")

                    # Parse start time
                    start_str = game.get("date", "")
                    start_time = None
                    if start_str:
                        try:
                            start_time = datetime.fromisoformat(
                                start_str.replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            pass

                    venue = ""
                    competitions = game.get("competitions", [])
                    if competitions:
                        venue_info = competitions[0].get("venue", {})
                        venue = venue_info.get("fullName", "")

                    # Upsert event
                    existing = session.query(Event).filter_by(
                        event_name=game_name, sport="NBA"
                    ).first()

                    if existing is None:
                        event = Event(
                            sport="NBA",
                            event_name=game_name,
                            start_time=start_time,
                            status=status,
                            venue=venue,
                            season=now.strftime("%Y"),
                            metadata_json=json.dumps({
                                "espn_id": espn_id,
                            }),
                        )
                        session.add(event)
                        count += 1
                    else:
                        # Update status if changed
                        if existing.status != status:
                            existing.status = status
                        if start_time and not existing.start_time:
                            existing.start_time = start_time

                session.commit()
                self._logger.info("Game schedule sync: %d new games added", count)
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        except Exception as exc:
            self._logger.error("ESPN schedule fetch failed: %s", exc)

        return count

    # ------------------------------------------------------------------ #
    # 3. Update player metadata
    # ------------------------------------------------------------------ #
    def _update_player_metadata(self, factory) -> None:
        """Mark stale players as inactive."""
        session = factory()
        try:
            cutoff = _utcnow() - timedelta(days=60)
            stale = (
                session.query(Player)
                .filter(Player.sport == "NBA")
                .filter(Player.active == True)
                .filter(Player.last_updated < cutoff)
                .all()
            )

            for p in stale:
                p.active = False

            if stale:
                self._logger.info(
                    "Marked %d players as inactive (no updates in 60 days)",
                    len(stale),
                )

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="NBA Stats Worker")
    parser.add_argument("--loop", action="store_true", help="Run in loop mode")
    args = parser.parse_args()

    worker = StatsWorker()
    if args.loop:
        worker.run_forever()
    else:
        result = worker.run()
        sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
