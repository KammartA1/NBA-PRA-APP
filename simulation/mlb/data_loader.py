"""
simulation/mlb/data_loader.py — Build real BatterProfile / PitcherProfile
objects from MLB StatsAPI season data, with sample-size shrinkage.

Observed rates are regressed toward the league mean using each stat's known
stabilization point (the PA/BF at which the metric becomes ~50% signal). This
prevents small-sample players from producing extreme, unreliable projections —
a core requirement for trustworthy edges.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Optional

from .config import LEAGUE_PA_RATES
from .profiles import BatterProfile, PitcherProfile

log = logging.getLogger(__name__)

# Stabilization points (PA for batters, BF for pitchers). Sources: Russell
# Carleton / Pizza Cutter reliability research. The metric is ~50% signal at
# this sample, so it doubles as the shrinkage constant K in:
#   shrunk = (N*observed + K*league) / (N + K)
BATTER_STABILIZE = {"K": 60, "BB": 120, "HBP": 240, "HR": 170,
                    "3B": 380, "2B": 300, "1B": 300}
PITCHER_STABILIZE = {"K": 70, "BB": 170, "HBP": 300, "HR": 500,
                     "3B": 450, "2B": 400, "1B": 400}

_cache: dict = {}


def _shrink(observed: float, n: int, league: float, k: int) -> float:
    if n <= 0:
        return league
    return (n * observed + k * league) / (n + k)


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _get_people(player_id: int) -> dict:
    import statsapi
    key = ("people", player_id)
    if key in _cache:
        return _cache[key]
    data = statsapi.get("people", {"personIds": player_id})
    person = (data.get("people") or [{}])[0]
    _cache[key] = person
    return person


def _season_stats(player_id: int, group: str, season: int) -> Optional[dict]:
    import statsapi
    key = ("season", player_id, group, season)
    if key in _cache:
        return _cache[key]
    data = statsapi.get("people", {
        "personIds": player_id,
        "hydrate": f"stats(group=[{group}],type=[season],season={season})",
    })
    try:
        splits = data["people"][0]["stats"][0]["splits"]
        stat = splits[0]["stat"] if splits else None
    except (KeyError, IndexError):
        stat = None
    _cache[key] = stat
    return stat


def get_batter_profile(player_id: int, name: str, season: int | None = None,
                       lineup_slot: int = 5) -> Optional[BatterProfile]:
    season = season or date.today().year
    person = _get_people(player_id)
    bats = (person.get("batSide", {}) or {}).get("code", "R")

    stat = _season_stats(player_id, "hitting", season)
    if not stat:
        stat = _season_stats(player_id, "hitting", season - 1)
    if not stat:
        return None

    pa = _safe_float(stat.get("plateAppearances"))
    if pa < 1:
        return None
    n = int(pa)
    hits = _safe_float(stat.get("hits"))
    doubles = _safe_float(stat.get("doubles"))
    triples = _safe_float(stat.get("triples"))
    hr = _safe_float(stat.get("homeRuns"))
    singles = max(0.0, hits - doubles - triples - hr)
    k = _safe_float(stat.get("strikeOuts"))
    bb = _safe_float(stat.get("baseOnBalls"))
    hbp = _safe_float(stat.get("hitByPitch"))
    sb = _safe_float(stat.get("stolenBases"))
    cs = _safe_float(stat.get("caughtStealing"))

    def rate(x, code):
        return _shrink(x / pa, n, LEAGUE_PA_RATES[code], BATTER_STABILIZE[code])

    sb_opps = max(singles + bb + hbp, 1.0)  # rough times-on-1B
    sb_att = (sb + cs) / sb_opps
    sb_succ = sb / max(sb + cs, 1.0) if (sb + cs) > 0 else 0.72

    return BatterProfile(
        player_id=str(player_id), name=name, bats=bats, pa=n,
        k=rate(k, "K"), bb=rate(bb, "BB"), hbp=rate(hbp, "HBP"),
        hr=rate(hr, "HR"), triple=rate(triples, "3B"),
        double=rate(doubles, "2B"), single=rate(singles, "1B"),
        sb_attempt=min(max(sb_att, 0.0), 0.5),
        sb_success=min(max(sb_succ, 0.4), 0.95),
        lineup_slot=lineup_slot,
    )


def get_pitcher_profile(player_id: int, name: str, season: int | None = None,
                        is_starter: bool = True) -> Optional[PitcherProfile]:
    season = season or date.today().year
    person = _get_people(player_id)
    throws = (person.get("pitchHand", {}) or {}).get("code", "R")

    stat = _season_stats(player_id, "pitching", season)
    if not stat:
        stat = _season_stats(player_id, "pitching", season - 1)
    if not stat:
        return None

    bf = _safe_float(stat.get("battersFaced"))
    if bf < 1:
        return None
    n = int(bf)
    hits = _safe_float(stat.get("hits"))
    doubles = _safe_float(stat.get("doubles"))
    triples = _safe_float(stat.get("triples"))
    hr = _safe_float(stat.get("homeRuns"))
    singles = max(0.0, hits - doubles - triples - hr)
    k = _safe_float(stat.get("strikeOuts"))
    bb = _safe_float(stat.get("baseOnBalls"))
    hbp = _safe_float(stat.get("hitByPitch"))
    pitches = _safe_float(stat.get("numberOfPitches") or stat.get("pitchesThrown"))
    gs = _safe_float(stat.get("gamesStarted")) or _safe_float(stat.get("gamesPitched")) or 1.0
    avg_pitches = pitches / gs if (pitches > 0 and gs > 0) else (95.0 if is_starter else 20.0)
    # A real starter rarely averages <80 or >110 in this era.
    if is_starter:
        avg_pitches = min(max(avg_pitches, 78.0), 108.0)

    def rate(x, code):
        return _shrink(x / bf, n, LEAGUE_PA_RATES[code], PITCHER_STABILIZE[code])

    return PitcherProfile(
        player_id=str(player_id), name=name, throws=throws, bf=n,
        is_starter=is_starter,
        k=rate(k, "K"), bb=rate(bb, "BB"), hbp=rate(hbp, "HBP"),
        hr=rate(hr, "HR"), triple=rate(triples, "3B"),
        double=rate(doubles, "2B"), single=rate(singles, "1B"),
        avg_pitches=avg_pitches,
    )


def get_team_lineup(team_id: int, season: int | None = None,
                    top_n: int = 9) -> list[BatterProfile]:
    """Best-effort opposing lineup: a team's top batters by PA this season.

    Real posted lineups aren't available until ~hours before first pitch, so
    for projection we use the team's most-used hitters, which is a faithful
    stand-in for the run-scoring environment a pitcher faces.
    """
    import statsapi
    season = season or date.today().year
    try:
        data = statsapi.get("team_roster", {"teamId": team_id, "rosterType": "active", "season": season})
        roster = data.get("roster", [])
    except Exception as e:
        log.warning("get_team_lineup roster fetch failed: %s", e)
        return []

    candidates = []
    for r in roster:
        person = r.get("person", {})
        pid = person.get("id")
        pos = (r.get("position", {}) or {}).get("abbreviation", "")
        if not pid or pos == "P":
            continue
        candidates.append((pid, person.get("fullName", "")))

    profiles = []
    for pid, nm in candidates:
        time.sleep(0.15)
        prof = get_batter_profile(pid, nm, season)
        if prof and prof.pa >= 20:
            profiles.append(prof)
    profiles.sort(key=lambda b: b.pa, reverse=True)
    top = profiles[:top_n]
    for i, b in enumerate(top):
        b.lineup_slot = i + 1
    return top


def get_game_context(team_abbr: str, opponent_abbr: str, on_date: date | None = None):
    """Resolve probable starters, park, and (best-effort) the matchup teams.

    Returns a dict consumed by the projection layer. Park is the home team's
    venue. Probable pitchers come from the schedule when posted.
    """
    import statsapi
    on_date = on_date or date.today()
    try:
        sched = statsapi.schedule(date=on_date.strftime("%m/%d/%Y"))
    except Exception as e:
        log.warning("schedule fetch failed: %s", e)
        return {}

    for g in sched:
        home = g.get("home_name", "")
        away = g.get("away_name", "")
        if team_abbr.lower() in (home.lower(), away.lower()) or \
           opponent_abbr.lower() in (home.lower(), away.lower()):
            return {
                "game_id": g.get("game_id"),
                "home_name": home, "away_name": away,
                "venue": g.get("venue_name", ""),
                "home_probable_id": g.get("home_probable_pitcher_id"),
                "away_probable_id": g.get("away_probable_pitcher_id"),
                "home_probable": g.get("home_probable_pitcher", ""),
                "away_probable": g.get("away_probable_pitcher", ""),
            }
    return {}
