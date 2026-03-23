"""
Event Service — NBA game/event management.

Uses NBA API scoreboard and Odds API for game data.
Returns plain dicts.
"""
import logging
import os
import re
from datetime import datetime, date, timedelta

log = logging.getLogger(__name__)


def _get_quant_session():
    try:
        from quant_system.db.schema import get_session
        return get_session()
    except Exception:
        return None


def _safe_close(session):
    if session:
        try:
            session.close()
        except Exception:
            pass


def _team_id_to_abbr() -> dict:
    """Return {team_id: abbreviation} mapping from NBA API."""
    try:
        from nba_api.stats.static import teams as nba_teams
        return {int(t["id"]): t["abbreviation"] for t in nba_teams.get_teams()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_todays_games() -> list[dict]:
    """
    Get today's NBA games from the NBA scoreboard API.

    Returns list of dicts with: game_id, home_team, away_team,
    home_team_id, away_team_id, status, start_time
    """
    today = date.today()
    return _fetch_games_for_date(today)


def get_upcoming_games(days: int = 3) -> list[dict]:
    """
    Get upcoming NBA games for the next N days.
    Combines NBA scoreboard + Odds API events for broader coverage.
    """
    all_games = []
    today = date.today()

    # NBA scoreboard for each day
    for offset in range(days):
        game_date = today + timedelta(days=offset)
        day_games = _fetch_games_for_date(game_date)
        for g in day_games:
            g["game_date"] = game_date.isoformat()
        all_games.extend(day_games)

    # Supplement with Odds API events (has more coverage for future dates)
    try:
        odds_games = _fetch_odds_api_events(
            today.isoformat(),
            (today + timedelta(days=days)).isoformat(),
        )
        # Merge: add events from Odds API that aren't already in NBA scoreboard results
        existing_matchups = set()
        for g in all_games:
            key = (g.get("home_team", ""), g.get("away_team", ""), g.get("game_date", ""))
            existing_matchups.add(key)

        for og in odds_games:
            key = (og.get("home_team", ""), og.get("away_team", ""), og.get("game_date", ""))
            if key not in existing_matchups:
                all_games.append(og)
    except Exception as exc:
        log.debug("Odds API supplement failed: %s", exc)

    return all_games


def get_or_create_event(
    event_name: str,
    start_time: datetime | str,
    venue: str | None = None,
) -> dict:
    """
    Get or create an event record. Events are keyed by name + date.
    Returns event dict with id, name, start_time, venue, status.
    """
    session = _get_quant_session()
    if session is None:
        # No DB — return a transient event dict
        return {
            "id": None,
            "name": event_name,
            "start_time": str(start_time),
            "venue": venue,
            "status": "scheduled",
        }
    try:
        from sqlalchemy import text

        # Normalize start_time
        if isinstance(start_time, str):
            try:
                start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except Exception:
                start_time = datetime.utcnow()

        start_date = start_time.date().isoformat()

        # Check for existing event by name + date
        result = session.execute(
            text("""
                SELECT id, reason as name, timestamp as start_time,
                       previous_state as venue, new_state as status
                FROM system_state_log
                WHERE reason LIKE :event_pattern
                AND date(timestamp) = :start_date
                LIMIT 1
            """),
            {"event_pattern": f"event:{event_name}%", "start_date": start_date},
        ).fetchone()

        if result:
            return {
                "id": result[0],
                "name": event_name,
                "start_time": str(result[2]),
                "venue": venue or result[3],
                "status": result[4] or "scheduled",
            }

        # Create new event (store in system_state_log with event prefix)
        from quant_system.db.schema import SystemStateLog
        event_row = SystemStateLog(
            sport="nba",
            timestamp=start_time,
            previous_state=venue or "",
            new_state="scheduled",
            reason=f"event:{event_name}",
            clv_at_change=None,
            bankroll_at_change=None,
            drawdown_at_change=None,
        )
        session.add(event_row)
        session.commit()

        return {
            "id": event_row.id,
            "name": event_name,
            "start_time": str(start_time),
            "venue": venue,
            "status": "scheduled",
        }
    except Exception as exc:
        log.error("get_or_create_event failed: %s", exc)
        try:
            session.rollback()
        except Exception:
            pass
        return {
            "id": None, "name": event_name, "start_time": str(start_time),
            "venue": venue, "status": "scheduled",
        }
    finally:
        _safe_close(session)


def update_event_status(event_id: int, status: str):
    """
    Update an event's status (scheduled, live, final, postponed).
    """
    session = _get_quant_session()
    if session is None:
        return
    try:
        from sqlalchemy import text
        session.execute(
            text("UPDATE system_state_log SET new_state = :status WHERE id = :eid"),
            {"status": status, "eid": event_id},
        )
        session.commit()
        log.info("Updated event %d status to %s", event_id, status)
    except Exception as exc:
        log.error("update_event_status failed: %s", exc)
        try:
            session.rollback()
        except Exception:
            pass
    finally:
        _safe_close(session)


def get_live_games() -> list[dict]:
    """
    Get currently live NBA games from the scoreboard.
    A game is 'live' if its status code indicates it's in progress.
    """
    today = date.today()
    try:
        from nba_api.stats.endpoints import scoreboardv2
        sb = scoreboardv2.ScoreboardV2(
            game_date=today.strftime("%m/%d/%Y")
        )
        dfs = sb.get_data_frames()
        if not dfs or dfs[0].empty:
            return []

        games_df = dfs[0]
        # Line score has period-by-period data
        line_score_df = dfs[1] if len(dfs) > 1 else None

        tid_map = _team_id_to_abbr()
        live_games = []

        for _, row in games_df.iterrows():
            game_status = int(row.get("GAME_STATUS_ID", 1))
            # 1=scheduled, 2=in progress, 3=final
            if game_status != 2:
                continue

            game_id = str(row.get("GAME_ID", ""))
            home_id = int(row.get("HOME_TEAM_ID", 0))
            away_id = int(row.get("VISITOR_TEAM_ID", 0))

            game = {
                "game_id": game_id,
                "home_team": tid_map.get(home_id, "UNK"),
                "away_team": tid_map.get(away_id, "UNK"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "status": "live",
                "game_status_text": str(row.get("GAME_STATUS_TEXT", "")),
                "live_period": int(row.get("LIVE_PERIOD", 0)),
                "live_pc_time": str(row.get("LIVE_PC_TIME", "")),
            }

            # Try to get scores from line_score
            if line_score_df is not None and not line_score_df.empty:
                home_score_rows = line_score_df[
                    line_score_df["TEAM_ID"] == home_id
                ]
                away_score_rows = line_score_df[
                    line_score_df["TEAM_ID"] == away_id
                ]
                if not home_score_rows.empty:
                    game["home_score"] = int(home_score_rows.iloc[0].get("PTS", 0) or 0)
                if not away_score_rows.empty:
                    game["away_score"] = int(away_score_rows.iloc[0].get("PTS", 0) or 0)

            live_games.append(game)

        return live_games
    except Exception as exc:
        log.error("get_live_games failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_games_for_date(game_date) -> list[dict]:
    """Fetch games from NBA scoreboard for a specific date."""
    try:
        from nba_api.stats.endpoints import scoreboardv2
        sb = scoreboardv2.ScoreboardV2(
            game_date=game_date.strftime("%m/%d/%Y")
        )
        df = sb.get_data_frames()[0]
        if df.empty:
            return []

        tid_map = _team_id_to_abbr()
        games = []
        for _, row in df.iterrows():
            game_id = str(row.get("GAME_ID", ""))
            home_id = int(row.get("HOME_TEAM_ID", 0))
            away_id = int(row.get("VISITOR_TEAM_ID", 0))
            status_id = int(row.get("GAME_STATUS_ID", 1))

            status_map = {1: "scheduled", 2: "live", 3: "final"}
            games.append({
                "game_id": game_id,
                "home_team": tid_map.get(home_id, "UNK"),
                "away_team": tid_map.get(away_id, "UNK"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "status": status_map.get(status_id, "unknown"),
                "game_status_text": str(row.get("GAME_STATUS_TEXT", "")),
            })
        return games
    except Exception as exc:
        log.warning("_fetch_games_for_date(%s) failed: %s", game_date, exc)
        return []


def _fetch_odds_api_events(start_iso: str, end_iso: str) -> list[dict]:
    """Fetch events from the Odds API for a date range."""
    import requests

    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("ODDS_API_KEY", "")
        except Exception:
            pass
    if not key:
        return []

    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/events",
            params={"apiKey": key},
            timeout=15,
        )
        if not r.ok:
            return []

        data = r.json()
        if not isinstance(data, list):
            return []

        events = []
        for ev in data:
            commence = ev.get("commence_time", "")
            # Convert UTC to ET date for comparison
            try:
                dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                # Rough ET conversion (UTC-5)
                et_date = (dt - timedelta(hours=5)).date().isoformat()
            except Exception:
                et_date = commence[:10]

            if start_iso <= et_date <= end_iso:
                events.append({
                    "event_id": ev.get("id"),
                    "home_team": ev.get("home_team", ""),
                    "away_team": ev.get("away_team", ""),
                    "commence_time": commence,
                    "game_date": et_date,
                    "status": "scheduled",
                    "source": "odds_api",
                })

        return events
    except Exception as exc:
        log.debug("Odds API events fetch failed: %s", exc)
        return []
