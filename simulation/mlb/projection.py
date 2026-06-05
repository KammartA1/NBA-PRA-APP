"""
simulation/mlb/projection.py — End-to-end MLB prop projection.

Comprehensive quant pipeline:
  1. Resolves the player, determines batter vs pitcher.
  2. Builds a Marcel-weighted, platoon-aware, recent-form-adjusted profile.
  3. Resolves the opposing starter, park, and game context.
  4. Builds team-specific bullpen profile.
  5. Runs at-bat-level Monte Carlo (PA-level, full base-state).
  6. Returns projection, probability distribution, confidence tier.

Output schema matches the NBA engine so the app can price both identically.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from .config import MLBSimConfig
from .engine import MLBGameEngine
from .profiles import BatterProfile, PitcherProfile
from . import data_loader as dl

log = logging.getLogger(__name__)

# Market code -> simulated stat key.
BATTER_MARKET_TO_STAT = {
    "TB": "total_bases", "H": "hits", "R": "runs", "RBI": "rbi",
    "SB": "stolen_bases", "HR": "home_runs", "HRR": "hrr",
    "BB": "walks", "1B": "singles", "2B": "doubles", "3B": "triples",
    "MLB_FS": "fantasy",
}
PITCHER_MARKET_TO_STAT = {
    "K": "pitcher_k", "OUTS": "pitcher_outs", "ER": "earned_runs",
    "HA": "hits_allowed", "BB_A": "walks_allowed", "MLB_FS": "pitcher_fantasy",
}

# Display name -> code fallback (belt-and-suspenders).
_DISPLAY_TO_CODE = {
    "Total Bases": "TB", "Hits": "H", "Runs": "R", "RBIs": "RBI",
    "Hits+Runs+RBIs": "HRR", "Home Runs": "HR", "Stolen Bases": "SB",
    "Walks": "BB", "Singles": "1B", "Doubles": "2B", "Triples": "3B",
    "Pitcher Strikeouts": "K", "Pitching Outs": "OUTS",
    "Earned Runs": "ER", "Hits Allowed": "HA", "Walks Allowed": "BB_A",
    "Fantasy Score": "MLB_FS", "Strikeouts": "K",
}

# Confidence tiers based on data quality.
CONFIDENCE_TIERS = {
    "high": "High-confidence: 200+ PA, multi-year data, platoon splits available",
    "medium": "Medium-confidence: 100-200 PA or limited history",
    "low": "Low-confidence: <100 PA, no splits, heavy regression to league average",
}


def _normalize_market(market: str) -> str:
    """Convert a market string to its canonical code."""
    m = market.strip()
    if m in BATTER_MARKET_TO_STAT or m in PITCHER_MARKET_TO_STAT:
        return m
    return _DISPLAY_TO_CODE.get(m, m)


def _player_basic(name: str):
    """Return (player_id, fullName, is_pitcher, team_id, team_name) or None."""
    import statsapi
    res = statsapi.lookup_player(name)
    if not res:
        return None
    p = res[0]
    pid = p["id"]
    pos = (p.get("primaryPosition", {}) or {}).get("abbreviation", "")
    team = p.get("currentTeam", {}) or {}
    return pid, p.get("fullName", name), (pos == "P"), team.get("id"), team.get("name", "")


def _resolve_context(team_id, team_name, game_date):
    """Find today's game: opponent, park, probable SPs."""
    import statsapi
    if not team_id:
        return {}
    try:
        sched = statsapi.schedule(date=game_date.strftime("%m/%d/%Y"), team=team_id)
    except Exception as e:
        log.warning("schedule fetch failed: %s", e)
        return {}
    if not sched:
        return {}
    g = sched[0]
    is_home = (g.get("home_id") == team_id)
    own_sp_name = g.get("home_probable_pitcher") if is_home else g.get("away_probable_pitcher")
    opp_sp_name = g.get("away_probable_pitcher") if is_home else g.get("home_probable_pitcher")
    return {
        "game_id": g.get("game_id"),
        "is_home": is_home,
        "venue": g.get("venue_name", ""),
        "home_id": g.get("home_id"), "away_id": g.get("away_id"),
        "home_name": g.get("home_name", ""), "away_name": g.get("away_name", ""),
        "opp_id": g.get("away_id") if is_home else g.get("home_id"),
        "opp_name": g.get("away_name") if is_home else g.get("home_name"),
        "own_sp_id": _lookup_id(own_sp_name),
        "own_sp_name": own_sp_name,
        "opp_sp_id": _lookup_id(opp_sp_name),
        "opp_sp_name": opp_sp_name,
    }


def _lookup_id(player_name: str | None):
    if not player_name:
        return None
    try:
        import statsapi
        r = statsapi.lookup_player(player_name)
        return r[0]["id"] if r else None
    except Exception:
        return None


def _assess_confidence(pa_or_bf: int, has_platoon: bool,
                       n_seasons: int) -> str:
    if pa_or_bf >= 200 and n_seasons >= 2:
        return "high"
    if pa_or_bf >= 100:
        return "medium"
    return "low"


def project(name: str, market: str, line: float,
            game_date: date | None = None,
            n_sims: int = 12000,
            temp_f: float | None = None, wind_mph: float | None = None,
            wind_out: bool | None = None,
            opp_pitcher_name: str | None = None) -> dict:
    """Project an MLB prop. Returns a dict with proj, p_over, p_under, etc."""
    game_date = game_date or date.today()

    # Normalize market (handle display names)
    market = _normalize_market(market)

    basic = _player_basic(name)
    if not basic:
        return {"error": f"Could not resolve MLB player '{name}'"}
    pid, full, is_pitcher, team_id, team_name = basic

    if is_pitcher:
        stat_key = PITCHER_MARKET_TO_STAT.get(market)
        if not stat_key:
            stat_key = BATTER_MARKET_TO_STAT.get(market)
            if stat_key:
                is_pitcher = False
    else:
        stat_key = BATTER_MARKET_TO_STAT.get(market)
        if not stat_key:
            stat_key = PITCHER_MARKET_TO_STAT.get(market)
            if stat_key:
                is_pitcher = True
    if not stat_key:
        return {"error": f"Market '{market}' not recognized"}

    ctx = _resolve_context(team_id, team_name, game_date)
    park = ctx.get("venue", "")
    notes = []

    if is_pitcher:
        sp = dl.get_pitcher_profile(pid, full, season=game_date.year, is_starter=True)
        if not sp:
            return {"error": f"No season pitching data for {full}"}
        notes.append(f"Throws {sp.throws}HP, {sp.bf} BF season")
        if sp.vs_lhb or sp.vs_rhb:
            notes.append("Platoon splits loaded")

        opp_id = ctx.get("opp_id")
        opp_lineup = dl.get_team_lineup(opp_id, season=game_date.year) if opp_id else []
        if not opp_lineup:
            opp_lineup = _league_average_lineup("OPP")
            notes.append("No opposing lineup resolved — using league-average lineup")

        # Team-specific bullpen for the pitcher's own team
        own_bullpen = None
        if team_id:
            own_bullpen = dl.get_team_bullpen_profile(team_id, season=game_date.year)

        eng = MLBGameEngine(
            MLBSimConfig(n_sims=n_sims),
            home_lineup=opp_lineup, away_lineup=_league_average_lineup("SUP"),
            home_sp=None, away_sp=sp,
            home_name=ctx.get("opp_name", "OPP"), away_name=team_name,
            park=park, temp_f=temp_f, wind_mph=wind_mph, wind_out=wind_out,
            away_bullpen=own_bullpen,
        )
        sim_pid = sp.player_id
        confidence = _assess_confidence(sp.bf, sp.vs_lhb is not None, 1)
    else:
        lineup = dl.get_team_lineup(team_id, season=game_date.year) if team_id else []
        batter = dl.get_batter_profile(pid, full, season=game_date.year)
        if not batter:
            return {"error": f"No season hitting data for {full}"}
        notes.append(f"Bats {batter.bats}, {batter.pa} PA season")
        if batter.vs_l or batter.vs_r:
            notes.append("Platoon splits loaded")
        lineup = _ensure_in_lineup(lineup, batter)

        # Opposing starting pitcher
        opp_sp = None
        opp_sp_id = ctx.get("opp_sp_id")
        opp_sp_nm = ctx.get("opp_sp_name") or opp_pitcher_name or "Opp SP"
        if opp_pitcher_name:
            import statsapi
            r = statsapi.lookup_player(opp_pitcher_name)
            if r:
                opp_sp_id = r[0]["id"]; opp_sp_nm = r[0]["fullName"]
        if opp_sp_id:
            opp_sp = dl.get_pitcher_profile(opp_sp_id, opp_sp_nm,
                                            season=game_date.year, is_starter=True)
        if not opp_sp:
            opp_sp = _league_average_starter()
            notes.append("No opposing starter resolved — using league-average SP")
        else:
            notes.append(f"vs {opp_sp.name} ({opp_sp.throws}HP)")

        # Team-specific bullpens
        opp_team_id = ctx.get("opp_id")
        opp_bullpen = None
        if opp_team_id:
            opp_bullpen = dl.get_team_bullpen_profile(opp_team_id, season=game_date.year)

        eng = MLBGameEngine(
            MLBSimConfig(n_sims=n_sims),
            home_lineup=lineup, away_lineup=_league_average_lineup("SUP"),
            home_sp=None, away_sp=opp_sp,
            home_name=team_name, away_name=ctx.get("opp_name", "OPP"),
            park=park, temp_f=temp_f, wind_mph=wind_mph, wind_out=wind_out,
            home_bullpen=opp_bullpen,
        )
        sim_pid = batter.player_id
        confidence = _assess_confidence(batter.pa, batter.vs_l is not None, 1)

    if park:
        notes.append(f"Park: {park}")

    out = eng.run_simulation()
    dist = out.get_player_dist(sim_pid, stat_key)
    if dist is None:
        return {"error": f"Simulation produced no distribution for {stat_key}"}

    p_over = dist.prob_over(line)
    p_under = dist.prob_under(line)
    p_push = dist.prob_push(line) if hasattr(dist, "prob_push") else 0.0
    if p_push > 0 and (p_over + p_under) > 0:
        p_over = p_over + p_push * p_over / (p_over + p_under)
        p_under = 1.0 - p_over

    return {
        "player": full, "player_id": pid, "market": market, "line": float(line),
        "is_pitcher": is_pitcher, "stat_key": stat_key,
        "proj": round(dist.mean, 2), "median": round(dist.median, 2),
        "std": round(dist.std, 2),
        "p_over": round(float(p_over), 4), "p_under": round(float(p_under), 4),
        "p5": dist.p5, "p25": dist.p25, "p75": dist.p75, "p95": dist.p95,
        "park": park, "opponent": ctx.get("opp_name", ""),
        "opp_starter": ctx.get("opp_sp_name", "") if not is_pitcher else "",
        "n_sims": out.n_simulations, "notes": notes,
        "confidence": confidence,
    }


def _league_average_lineup(prefix: str) -> list[BatterProfile]:
    return [BatterProfile(player_id=f"{prefix}{i}", name=f"{prefix} Hitter {i+1}",
                          lineup_slot=i + 1) for i in range(9)]


def _league_average_starter() -> PitcherProfile:
    return PitcherProfile(player_id="lgsp", name="League-Avg SP", is_starter=True,
                          avg_pitches=92.0)


def _ensure_in_lineup(lineup: list[BatterProfile], batter: BatterProfile) -> list[BatterProfile]:
    if not lineup:
        lineup = _league_average_lineup("TM")
    replaced = False
    for i, b in enumerate(lineup):
        if b.player_id == batter.player_id:
            batter.lineup_slot = b.lineup_slot
            lineup[i] = batter
            replaced = True
            break
    if not replaced:
        batter.lineup_slot = batter.lineup_slot or 3
        slot = min(max(batter.lineup_slot, 1), 9)
        lineup = lineup[:slot - 1] + [batter] + lineup[slot - 1:]
        lineup = lineup[:9]
    return lineup
