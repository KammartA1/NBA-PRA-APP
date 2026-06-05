"""
simulation/mlb/data_loader.py — Build real BatterProfile / PitcherProfile
objects from MLB StatsAPI season data with a comprehensive quant projection.

Core methodology (derived from top MLB quant syndicates):
  1. Marcel-style multi-year weighting (5/4/3 across current + 2 prior seasons)
  2. Per-stat stabilization shrinkage toward league mean
  3. Platoon splits (vs-LHP / vs-RHP) for matchup-aware projections
  4. Recent-form weighting (last 30 games) for streaky signal
  5. Team bullpen profiling for pitcher/game-environment modeling
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Optional

from .config import LEAGUE_PA_RATES
from .profiles import BatterProfile, PitcherProfile

log = logging.getLogger(__name__)

# ── Stabilization points ──────────────────────────────────────────────
# PA at which a stat is ~50% signal. Doubles as shrinkage constant K:
#   shrunk = (N × observed + K × league) / (N + K)
# Sources: Russell Carleton reliability research, Pizza Cutter, FanGraphs.
BATTER_STABILIZE = {
    "K": 60, "BB": 120, "HBP": 240, "HR": 170,
    "3B": 380, "2B": 300, "1B": 300,
}
PITCHER_STABILIZE = {
    "K": 70, "BB": 170, "HBP": 300, "HR": 500,
    "3B": 450, "2B": 400, "1B": 400,
}

# Marcel weights: current season, previous, two years ago.
MARCEL_WEIGHTS = (5, 4, 3)

# Recent-form window (games) and per-stat trust factors.
# Higher trust = more weight on recent form vs season-long.
RECENT_GAMES = 30
RECENT_TRUST = {
    "K": 0.20,   # K% stabilizes fast, recent form somewhat meaningful
    "BB": 0.15,
    "HBP": 0.05,
    "HR": 0.08,  # HR is noisy in small samples
    "3B": 0.03,
    "2B": 0.10,
    "1B": 0.10,
}

_cache: dict = {}
_RATE_DELAY = 0.03  # seconds between API calls


def _shrink(observed: float, n: int, league: float, k: int) -> float:
    if n <= 0:
        return league
    return (n * observed + k * league) / (n + k)


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _api_get(endpoint: str, params: dict) -> dict:
    import statsapi
    time.sleep(_RATE_DELAY)
    return statsapi.get(endpoint, params)


def _get_people(player_id: int) -> dict:
    key = ("people", player_id)
    if key in _cache:
        return _cache[key]
    try:
        data = _api_get("people", {"personIds": player_id})
        person = (data.get("people") or [{}])[0]
    except Exception as e:
        log.warning("people fetch failed for %s: %s", player_id, e)
        person = {}
    _cache[key] = person
    return person


def _season_stats(player_id: int, group: str, season: int) -> Optional[dict]:
    key = ("season", player_id, group, season)
    if key in _cache:
        return _cache[key]
    try:
        data = _api_get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=[{group}],type=[season],season={season})",
        })
        splits = data["people"][0]["stats"][0]["splits"]
        stat = splits[0]["stat"] if splits else None
    except (KeyError, IndexError):
        stat = None
    except Exception as e:
        log.warning("season_stats failed for %s/%s/%s: %s", player_id, group, season, e)
        stat = None
    _cache[key] = stat
    return stat


def _platoon_splits(player_id: int, group: str, season: int) -> dict:
    """Fetch vs-LHP and vs-RHP (for batters) or vs-LHB/RHB (for pitchers).

    Returns {"vl": stat_dict, "vr": stat_dict} or empty dict.
    """
    key = ("platoon", player_id, group, season)
    if key in _cache:
        return _cache[key]
    result = {}
    try:
        data = _api_get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=[{group}],type=[statSplits],sitCodes=[vl,vr],season={season})",
        })
        for sg in data["people"][0].get("stats", []):
            for sp in sg.get("splits", []):
                desc = (sp.get("split", {}).get("description", "") or "").lower()
                stat = sp.get("stat", {})
                if "left" in desc:
                    result["vl"] = stat
                elif "right" in desc:
                    result["vr"] = stat
    except Exception as e:
        log.debug("platoon splits unavailable for %s: %s", player_id, e)
    _cache[key] = result
    return result


def _recent_game_logs(player_id: int, group: str, season: int,
                      n_games: int = RECENT_GAMES) -> list[dict]:
    """Fetch the last N game logs for a player."""
    key = ("gamelog", player_id, group, season)
    if key in _cache:
        return _cache[key]
    logs = []
    try:
        data = _api_get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=[{group}],type=[gameLog],season={season})",
        })
        for sg in data["people"][0].get("stats", []):
            logs = sg.get("splits", [])
    except Exception as e:
        log.debug("game logs unavailable for %s: %s", player_id, e)
    result = logs[-n_games:] if logs else []
    _cache[key] = result
    return result


# ── Rate extraction helpers ───────────────────────────────────────────

def _extract_batter_rates(stat: dict, pa: float) -> dict:
    """Convert a stat dict into per-PA outcome rates."""
    if pa < 1:
        return {}
    hits = _safe_float(stat.get("hits"))
    doubles = _safe_float(stat.get("doubles"))
    triples = _safe_float(stat.get("triples"))
    hr = _safe_float(stat.get("homeRuns"))
    singles = max(0.0, hits - doubles - triples - hr)
    return {
        "K": _safe_float(stat.get("strikeOuts")) / pa,
        "BB": _safe_float(stat.get("baseOnBalls")) / pa,
        "HBP": _safe_float(stat.get("hitByPitch")) / pa,
        "HR": hr / pa, "3B": triples / pa,
        "2B": doubles / pa, "1B": singles / pa,
    }


def _extract_pitcher_rates(stat: dict, bf: float) -> dict:
    if bf < 1:
        return {}
    hits = _safe_float(stat.get("hits"))
    doubles = _safe_float(stat.get("doubles"))
    triples = _safe_float(stat.get("triples"))
    hr = _safe_float(stat.get("homeRuns"))
    singles = max(0.0, hits - doubles - triples - hr)
    return {
        "K": _safe_float(stat.get("strikeOuts")) / bf,
        "BB": _safe_float(stat.get("baseOnBalls")) / bf,
        "HBP": _safe_float(stat.get("hitByPitch")) / bf,
        "HR": hr / bf, "3B": triples / bf,
        "2B": doubles / bf, "1B": singles / bf,
    }


def _aggregate_game_logs(logs: list[dict], group: str) -> tuple[dict, float]:
    """Aggregate game logs into totals and PA/BF count."""
    totals = {"strikeOuts": 0, "baseOnBalls": 0, "hitByPitch": 0,
              "homeRuns": 0, "triples": 0, "doubles": 0, "hits": 0}
    n = 0.0
    pa_key = "plateAppearances" if group == "hitting" else "battersFaced"
    for entry in logs:
        stat = entry.get("stat", {})
        pa = _safe_float(stat.get(pa_key))
        if pa < 1:
            continue
        n += pa
        for k in totals:
            totals[k] += _safe_float(stat.get(k))
    return totals, n


# ── Marcel-style multi-year blending ──────────────────────────────────

def _marcel_blend(seasons: list[tuple[float, float, dict]]) -> dict:
    """Blend multiple seasons of rates using Marcel weights.

    Each entry is (weight, pa_or_bf, rates_dict).
    Returns blended per-PA rates (NOT shrunk yet — shrinkage applied later
    using the total weighted sample size).
    """
    rate_keys = ["K", "BB", "HBP", "HR", "3B", "2B", "1B"]
    blended = {}
    total_w = sum(w for w, _, _ in seasons if _)
    if total_w <= 0:
        return {k: LEAGUE_PA_RATES.get(k, 0.0) for k in rate_keys}
    for k in rate_keys:
        weighted_sum = sum(w * rates.get(k, LEAGUE_PA_RATES.get(k, 0.0))
                           for w, _, rates in seasons if rates)
        blended[k] = weighted_sum / total_w
    return blended


def _marcel_pa(seasons: list[tuple[float, float, dict]]) -> float:
    """Effective sample size from Marcel blend (weighted PA sum)."""
    total_w = sum(w for w, _, _ in seasons if _)
    if total_w <= 0:
        return 0.0
    return sum(w * n for w, n, _ in seasons if _) / total_w


# ── Recent-form adjustment ────────────────────────────────────────────

def _recent_form_adjust(season_rate: float, recent_rate: float,
                        recent_pa: float, stat_code: str) -> float:
    """Blend season rate with recent form, weighted by stat-specific trust.

    Only applies when we have enough recent data (>= 30 PA).
    """
    trust = RECENT_TRUST.get(stat_code, 0.05)
    if recent_pa < 30:
        trust *= recent_pa / 30.0
    return season_rate * (1.0 - trust) + recent_rate * trust


# ── Profile builders ──────────────────────────────────────────────────

def get_batter_profile(player_id: int, name: str, season: int | None = None,
                       lineup_slot: int = 5) -> Optional[BatterProfile]:
    """Build a comprehensive BatterProfile with multi-year + platoon + recent form."""
    season = season or date.today().year
    person = _get_people(player_id)
    bats = (person.get("batSide", {}) or {}).get("code", "R")

    # 1) Multi-year season stats (Marcel weighting)
    seasons_data = []
    for yr_offset, weight in enumerate(MARCEL_WEIGHTS):
        yr = season - yr_offset
        stat = _season_stats(player_id, "hitting", yr)
        if stat:
            pa = _safe_float(stat.get("plateAppearances"))
            rates = _extract_batter_rates(stat, pa)
            if pa >= 1 and rates:
                seasons_data.append((weight, pa, rates))

    if not seasons_data:
        return None

    # Marcel blend
    blended_rates = _marcel_blend(seasons_data)
    effective_pa = _marcel_pa(seasons_data)
    current_pa = seasons_data[0][1] if seasons_data else 0

    # 2) Recent form (last 30 games of current season)
    logs = _recent_game_logs(player_id, "hitting", season)
    if logs:
        log_totals, log_pa = _aggregate_game_logs(logs, "hitting")
        if log_pa >= 15:
            recent_rates = _extract_batter_rates(log_totals, log_pa)
            for k in blended_rates:
                blended_rates[k] = _recent_form_adjust(
                    blended_rates[k], recent_rates.get(k, blended_rates[k]),
                    log_pa, k)

    # 3) Apply stabilization shrinkage
    n = int(effective_pa)

    def rate(code):
        return _shrink(blended_rates.get(code, LEAGUE_PA_RATES[code]),
                       n, LEAGUE_PA_RATES[code], BATTER_STABILIZE[code])

    # 4) Stolen base rates from current season
    cur_stat = _season_stats(player_id, "hitting", season)
    sb = _safe_float(cur_stat.get("stolenBases")) if cur_stat else 0
    cs = _safe_float(cur_stat.get("caughtStealing")) if cur_stat else 0
    cur_pa = _safe_float(cur_stat.get("plateAppearances")) if cur_stat else 0
    cur_hits = _safe_float(cur_stat.get("hits")) if cur_stat else 0
    cur_bb = _safe_float(cur_stat.get("baseOnBalls")) if cur_stat else 0
    cur_hbp = _safe_float(cur_stat.get("hitByPitch")) if cur_stat else 0
    sb_opps = max(cur_hits - _safe_float(cur_stat.get("homeRuns", 0)) + cur_bb + cur_hbp, 1.0) if cur_stat else 1.0
    sb_att = (sb + cs) / sb_opps
    sb_succ = sb / max(sb + cs, 1.0) if (sb + cs) > 0 else 0.72

    # 5) Platoon splits
    platoon = _platoon_splits(player_id, "hitting", season)
    vs_l_rates = None
    vs_r_rates = None
    if platoon.get("vl"):
        vl_pa = _safe_float(platoon["vl"].get("plateAppearances"))
        if vl_pa >= 40:
            vl_raw = _extract_batter_rates(platoon["vl"], vl_pa)
            vs_l_rates = {}
            for code in BATTER_STABILIZE:
                vs_l_rates[code] = _shrink(
                    vl_raw.get(code, LEAGUE_PA_RATES[code]),
                    int(vl_pa), LEAGUE_PA_RATES[code],
                    BATTER_STABILIZE[code])
    if platoon.get("vr"):
        vr_pa = _safe_float(platoon["vr"].get("plateAppearances"))
        if vr_pa >= 40:
            vr_raw = _extract_batter_rates(platoon["vr"], vr_pa)
            vs_r_rates = {}
            for code in BATTER_STABILIZE:
                vs_r_rates[code] = _shrink(
                    vr_raw.get(code, LEAGUE_PA_RATES[code]),
                    int(vr_pa), LEAGUE_PA_RATES[code],
                    BATTER_STABILIZE[code])

    profile = BatterProfile(
        player_id=str(player_id), name=name, bats=bats, pa=int(current_pa),
        k=rate("K"), bb=rate("BB"), hbp=rate("HBP"),
        hr=rate("HR"), triple=rate("3B"),
        double=rate("2B"), single=rate("1B"),
        sb_attempt=min(max(sb_att, 0.0), 0.5),
        sb_success=min(max(sb_succ, 0.4), 0.95),
        lineup_slot=lineup_slot,
        vs_l=vs_l_rates, vs_r=vs_r_rates,
    )
    return profile


def get_pitcher_profile(player_id: int, name: str, season: int | None = None,
                        is_starter: bool = True) -> Optional[PitcherProfile]:
    """Build a comprehensive PitcherProfile with multi-year + platoon + recent form."""
    season = season or date.today().year
    person = _get_people(player_id)
    throws = (person.get("pitchHand", {}) or {}).get("code", "R")

    # 1) Multi-year season stats
    seasons_data = []
    for yr_offset, weight in enumerate(MARCEL_WEIGHTS):
        yr = season - yr_offset
        stat = _season_stats(player_id, "pitching", yr)
        if stat:
            bf = _safe_float(stat.get("battersFaced"))
            rates = _extract_pitcher_rates(stat, bf)
            if bf >= 1 and rates:
                seasons_data.append((weight, bf, rates))

    if not seasons_data:
        return None

    blended_rates = _marcel_blend(seasons_data)
    effective_bf = _marcel_pa(seasons_data)

    # 2) Recent form
    logs = _recent_game_logs(player_id, "pitching", season)
    if logs:
        log_totals, log_bf = _aggregate_game_logs(logs, "pitching")
        if log_bf >= 20:
            recent_rates = _extract_pitcher_rates(log_totals, log_bf)
            for k in blended_rates:
                blended_rates[k] = _recent_form_adjust(
                    blended_rates[k], recent_rates.get(k, blended_rates[k]),
                    log_bf, k)

    # 3) Stabilization shrinkage
    n = int(effective_bf)

    def rate(code):
        return _shrink(blended_rates.get(code, LEAGUE_PA_RATES[code]),
                       n, LEAGUE_PA_RATES[code], PITCHER_STABILIZE[code])

    # 4) Workload from current season
    cur_stat = _season_stats(player_id, "pitching", season)
    pitches = _safe_float((cur_stat or {}).get("numberOfPitches") or (cur_stat or {}).get("pitchesThrown"))
    gs = _safe_float((cur_stat or {}).get("gamesStarted")) or _safe_float((cur_stat or {}).get("gamesPitched")) or 1.0
    avg_pitches = pitches / gs if (pitches > 0 and gs > 0) else (95.0 if is_starter else 20.0)
    if is_starter:
        avg_pitches = min(max(avg_pitches, 78.0), 108.0)

    # 5) Platoon splits
    platoon = _platoon_splits(player_id, "pitching", season)
    vs_lhb_rates = None
    vs_rhb_rates = None
    if platoon.get("vl"):
        vl_bf = _safe_float(platoon["vl"].get("battersFaced") or platoon["vl"].get("plateAppearances"))
        if vl_bf >= 30:
            vl_raw = _extract_pitcher_rates(platoon["vl"], vl_bf)
            vs_lhb_rates = {}
            for code in PITCHER_STABILIZE:
                vs_lhb_rates[code] = _shrink(
                    vl_raw.get(code, LEAGUE_PA_RATES[code]),
                    int(vl_bf), LEAGUE_PA_RATES[code],
                    PITCHER_STABILIZE[code])
    if platoon.get("vr"):
        vr_bf = _safe_float(platoon["vr"].get("battersFaced") or platoon["vr"].get("plateAppearances"))
        if vr_bf >= 30:
            vr_raw = _extract_pitcher_rates(platoon["vr"], vr_bf)
            vs_rhb_rates = {}
            for code in PITCHER_STABILIZE:
                vs_rhb_rates[code] = _shrink(
                    vr_raw.get(code, LEAGUE_PA_RATES[code]),
                    int(vr_bf), LEAGUE_PA_RATES[code],
                    PITCHER_STABILIZE[code])

    cur_bf = _safe_float((cur_stat or {}).get("battersFaced")) if cur_stat else 0

    return PitcherProfile(
        player_id=str(player_id), name=name, throws=throws, bf=int(cur_bf),
        is_starter=is_starter,
        k=rate("K"), bb=rate("BB"), hbp=rate("HBP"),
        hr=rate("HR"), triple=rate("3B"),
        double=rate("2B"), single=rate("1B"),
        avg_pitches=avg_pitches,
        vs_lhb=vs_lhb_rates, vs_rhb=vs_rhb_rates,
    )


def get_team_lineup(team_id: int, season: int | None = None,
                    top_n: int = 9) -> list[BatterProfile]:
    """Best-effort lineup: team's most-used hitters by PA this season."""
    import statsapi
    season = season or date.today().year
    key = ("lineup", team_id, season)
    if key in _cache:
        return _cache[key]
    try:
        data = _api_get("team_roster", {"teamId": team_id, "rosterType": "active", "season": season})
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
        prof = get_batter_profile(pid, nm, season)
        if prof and prof.pa >= 20:
            profiles.append(prof)
    profiles.sort(key=lambda b: b.pa, reverse=True)
    top = profiles[:top_n]
    for i, b in enumerate(top):
        b.lineup_slot = i + 1
    _cache[key] = top
    return top


def get_team_bullpen_profile(team_id: int, season: int | None = None) -> Optional[PitcherProfile]:
    """Build a composite bullpen profile from team relievers.

    Uses season pitching stats for non-starter pitchers to build an
    aggregate reliever profile reflecting team-specific bullpen quality.
    """
    import statsapi
    season = season or date.today().year
    key = ("bullpen", team_id, season)
    if key in _cache:
        return _cache[key]

    try:
        data = _api_get("team_roster", {"teamId": team_id, "rosterType": "active", "season": season})
        roster = data.get("roster", [])
    except Exception as e:
        log.warning("bullpen roster fetch failed: %s", e)
        _cache[key] = None
        return None

    total_bf = 0.0
    totals = {"strikeOuts": 0, "baseOnBalls": 0, "hitByPitch": 0,
              "homeRuns": 0, "triples": 0, "doubles": 0, "hits": 0}

    for r in roster:
        person = r.get("person", {})
        pid = person.get("id")
        pos = (r.get("position", {}) or {}).get("abbreviation", "")
        if not pid or pos != "P":
            continue
        stat = _season_stats(pid, "pitching", season)
        if not stat:
            continue
        gs = _safe_float(stat.get("gamesStarted"))
        gp = _safe_float(stat.get("gamesPitched"))
        if gs > gp * 0.4:
            continue
        bf = _safe_float(stat.get("battersFaced"))
        if bf < 10:
            continue
        total_bf += bf
        for k in totals:
            totals[k] += _safe_float(stat.get(k))

    if total_bf < 50:
        _cache[key] = None
        return None

    rates = _extract_pitcher_rates(totals, total_bf)
    n = int(total_bf)
    profile = PitcherProfile(
        player_id=f"bullpen_{team_id}", name=f"Team {team_id} Bullpen",
        throws="R", bf=n, is_starter=False,
        k=_shrink(rates["K"], n, LEAGUE_PA_RATES["K"], PITCHER_STABILIZE["K"]),
        bb=_shrink(rates["BB"], n, LEAGUE_PA_RATES["BB"], PITCHER_STABILIZE["BB"]),
        hbp=_shrink(rates["HBP"], n, LEAGUE_PA_RATES["HBP"], PITCHER_STABILIZE["HBP"]),
        hr=_shrink(rates["HR"], n, LEAGUE_PA_RATES["HR"], PITCHER_STABILIZE["HR"]),
        triple=_shrink(rates["3B"], n, LEAGUE_PA_RATES["3B"], PITCHER_STABILIZE["3B"]),
        double=_shrink(rates["2B"], n, LEAGUE_PA_RATES["2B"], PITCHER_STABILIZE["2B"]),
        single=_shrink(rates["1B"], n, LEAGUE_PA_RATES["1B"], PITCHER_STABILIZE["1B"]),
        avg_pitches=20.0,
    )
    _cache[key] = profile
    return profile


def clear_cache():
    _cache.clear()
