"""
Scanner Service — live scanner logic extracted from app.py Live Scanner tab.

Scans all available lines (Odds API, PrizePicks, Underdog, Sleeper) for
value opportunities and returns results sorted by edge.

All computation uses projection_service; results are persisted to DB.
"""
import json
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, date, timedelta

log = logging.getLogger(__name__)

_SCANNER_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scanner_results_cache.pkl",
)


def _safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _safe_round(v, d=2):
    try:
        return round(float(v), d) if v is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_all_books(
    markets: list[str] | None = None,
    book: str = "all",
    game_date=None,
    settings: dict | None = None,
    min_prob: float = 0.53,
    min_adv: float = 0.01,
    min_ev: float = 0.01,
    max_results: int = 60,
    scan_source: str = "All sources",
    pp_lines: list[dict] | None = None,
    ud_lines: list[dict] | None = None,
    sl_lines: list[dict] | None = None,
) -> list[dict]:
    """
    Scan current lines for value opportunities.

    Parameters
    ----------
    markets : list of market names to scan (e.g. ["Points", "Rebounds", "Assists"])
    book : sportsbook filter ("all", "consensus", specific book, "prizepicks")
    game_date : date to scan (defaults to today)
    settings : model settings dict (bankroll, n_games, frac_kelly, etc.)
    min_prob : minimum calibrated probability
    min_adv : minimum advantage over implied probability
    min_ev : minimum EV (adjusted)
    max_results : cap on returned opportunities
    scan_source : "Odds API only", "PP + UD only", or "All sources"
    pp_lines : PrizePicks lines as list of dicts (player, stat_type, line, ...)
    ud_lines : Underdog lines
    sl_lines : Sleeper lines

    Returns
    -------
    list[dict] – opportunities sorted by composite sharpness + EV
    """
    # Import computation engine (separated from Streamlit frontend)
    import nba_engine as _app

    settings = settings or {}
    game_date = game_date or date.today()
    markets = markets or ["Points", "Rebounds", "Assists"]
    n_games = int(settings.get("n_games", 10))
    bankroll = float(settings.get("bankroll", 1000.0))
    frac_kelly = float(settings.get("frac_kelly", 0.25))
    max_risk_frac = float(settings.get("max_risk_per_bet", 3.0)) / 100.0
    market_prior_weight = float(settings.get("market_prior_weight", 0.65))
    exclude_chaotic = bool(settings.get("exclude_chaotic", True))
    injury_team_map = settings.get("injury_team_map", {})
    calibrator_map = settings.get("calibrator_map")

    use_odds_api = scan_source in ("Odds API only", "All sources") and book != "prizepicks"
    use_platforms = scan_source in ("PP + UD only", "All sources") or book == "prizepicks"

    candidates = []

    # ── Odds API candidates ──
    if use_odds_api:
        odds_candidates = _build_odds_api_candidates(_app, markets, book, game_date)
        candidates.extend(odds_candidates)

    # ── Platform candidates (PP / UD / Sleeper) ──
    if use_platforms:
        plat_candidates = _build_platform_candidates(
            _app, pp_lines, ud_lines, sl_lines, markets, book,
        )
        candidates.extend(plat_candidates)

    if not candidates:
        log.info("Scanner: 0 candidates to evaluate")
        return []

    log.info("Scanner: evaluating %d candidates", len(candidates))

    # Load bulk game logs for performance
    try:
        _app._fetch_bulk_gamelogs()
    except Exception:
        pass

    # Run projections in parallel
    out_rows = []
    dropped = []
    max_workers = min(32, len(candidates))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for (pname, mkt, line, meta) in candidates:
            fut = executor.submit(
                _app.compute_leg_projection,
                pname, mkt, line, meta,
                n_games=n_games,
                key_teammate_out=False,
                bankroll=bankroll,
                frac_kelly=frac_kelly,
                max_risk_frac=max_risk_frac,
                market_prior_weight=market_prior_weight,
                exclude_chaotic=exclude_chaotic,
                game_date=game_date,
                injury_team_map=injury_team_map,
            )
            futures.append((pname, mkt, line, meta, fut))

        for (pname, mkt, line, meta, fut) in futures:
            try:
                leg = fut.result(timeout=60)
            except FuturesTimeout:
                dropped.append({"player": pname, "market": mkt, "reason": "timeout"})
                continue
            except Exception as exc:
                dropped.append({"player": pname, "market": mkt, "reason": str(exc)})
                continue

            # Apply calibration
            if calibrator_map:
                try:
                    leg = _app.recompute_pricing_fields(dict(leg), calibrator_map)
                except Exception:
                    pass

            if not leg.get("gate_ok"):
                dropped.append({
                    "player": pname, "market": mkt,
                    "reason": leg.get("gate_reason", "gated"),
                })
                continue

            pc = leg.get("p_cal")
            if pc is None:
                dropped.append({"player": pname, "market": mkt, "reason": "p_cal None"})
                continue
            pc = float(pc)

            pi = leg.get("p_implied")
            ev = leg.get("ev_adj")
            if pi is None or ev is None:
                dropped.append({"player": pname, "market": mkt, "reason": "no price/EV"})
                continue

            adv = pc - float(pi)
            if pc < min_prob or adv < min_adv or float(ev) < min_ev:
                dropped.append({
                    "player": pname, "market": mkt,
                    "reason": f"below threshold (p={pc:.3f}, adv={adv:.3f}, ev={float(ev):.3f})",
                })
                continue

            mv = leg.get("line_movement") or {}
            src_badge = {"prizepicks": "PP", "underdog": "UD", "sleeper": "SL"}.get(
                str(meta.get("book", "")).lower(), meta.get("book", "") or "odds"
            )

            out_rows.append({
                "side": "Over",
                "src": src_badge,
                "player": pname,
                "market": mkt,
                "line": line,
                "p_cal": round(pc, 3),
                "p_implied": round(float(pi), 3),
                "advantage": round(adv, 3),
                "ev_adj_pct": round(float(ev) * 100, 2),
                "proj": _safe_round(leg.get("proj")),
                "edge_cat": leg.get("edge_cat", ""),
                "regime": leg.get("regime", ""),
                "hot_cold": leg.get("hot_cold", "Average"),
                "team": leg.get("team", ""),
                "opp": leg.get("opp", ""),
                "b2b": "B2B" if leg.get("b2b") else "",
                "dnp_risk": "DNP?" if leg.get("dnp_risk") else "",
                "vol_cv": _safe_round(leg.get("volatility_cv")),
                "rest_d": int(leg.get("rest_days", 2)),
                "line_mv": mv.get("direction", "--"),
                "mv_pips": float(mv.get("pips", 0.0)),
                "steam": "STEAM" if mv.get("steam") else ("FADE" if mv.get("fade") else ""),
                "stake_$": round(leg.get("stake", 0), 2),
                "n_games": int(leg.get("n_games_used", 0)),
                "sharp": _safe_round(leg.get("sharpness_score"), 0),
                "sharp_tier": leg.get("sharpness_tier", ""),
                "trend": leg.get("trend_label", ""),
                "fatigue": leg.get("fatigue_label", "Normal"),
                "game_tot": _safe_round(leg.get("game_total"), 0),
                "l3": _safe_round(leg.get("l3_avg"), 1),
                "l5": _safe_round(leg.get("l5_avg"), 1),
                # Full leg data for downstream use
                "_full_leg": leg,
            })

    # Sort by composite sharpness + EV
    out_rows.sort(
        key=lambda x: (float(x.get("sharp") or 0) * 0.6 + float(x.get("ev_adj_pct") or 0) * 0.4),
        reverse=True,
    )
    out_rows = out_rows[:max_results]

    # Strip internal _full_leg before returning (keep it lean)
    for row in out_rows:
        row.pop("_full_leg", None)

    # Persist results
    _save_scanner_results(out_rows, dropped)

    log.info("Scanner: %d opportunities found, %d dropped", len(out_rows), len(dropped))
    return out_rows


def scan_sharp_movements() -> list[dict]:
    """
    Scan for sharp line movements across all sources.
    Delegates to odds_service.detect_sharp_movements.
    """
    try:
        from services.odds_service import detect_sharp_movements
        return detect_sharp_movements(minutes=30, threshold=1.0)
    except Exception as exc:
        log.error("scan_sharp_movements failed: %s", exc)
        return []


def get_scanner_results() -> list[dict]:
    """
    Get last scan results from disk cache.
    Returns the most recent scan results or empty list.
    """
    try:
        if os.path.exists(_SCANNER_CACHE_PATH):
            with open(_SCANNER_CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            results = cache.get("scanner_results")
            if results is not None:
                # Handle both DataFrame and list formats
                if hasattr(results, "to_dict"):
                    return results.to_dict("records")
                elif isinstance(results, list):
                    return results
        return []
    except Exception as exc:
        log.warning("get_scanner_results failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_scanner_results(results: list[dict], dropped: list[dict]):
    """Persist scanner results to disk."""
    try:
        cache = {
            "scanner_results": results,
            "scanner_dropped": dropped,
            "scanner_scan_id": datetime.utcnow().isoformat(),
            "scanner_timestamp": time.time(),
        }
        with open(_SCANNER_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception as exc:
        log.warning("_save_scanner_results failed: %s", exc)


def _build_odds_api_candidates(app, markets, book, game_date) -> list[tuple]:
    """Build candidate list from Odds API."""
    candidates = []
    try:
        date_iso = game_date.isoformat() if hasattr(game_date, "isoformat") else str(game_date)
        evs, err = app.odds_get_events(date_iso)
        if err or not evs:
            log.warning("Odds API events: %s", err or "no events")
            return []

        # Unsupported markets for Odds API
        unsupported = getattr(app, "ODDS_API_UNSUPPORTED_MARKETS", set())
        supported_markets = [m for m in markets if m not in unsupported]

        market_keys = []
        for m in supported_markets:
            mk = app.ODDS_MARKETS.get(m)
            if mk:
                market_keys.append((m, mk))

        if not market_keys:
            return []

        specialty_keys = getattr(app, "SPECIALTY_MARKET_KEYS", set())

        for ev in evs:
            eid = ev.get("id")
            if not eid:
                continue

            for market_name, market_key in market_keys:
                regions = "us,us2,eu,uk" if market_key in specialty_keys else "us"
                odds, oerr = app.odds_get_event_odds(eid, (market_key,), regions=regions)
                if oerr or not odds:
                    continue

                book_filter = None if book in ("all", "prizepicks") else book
                parsed, _ = app._parse_player_prop_outcomes(odds, market_key, book_filter=book_filter)

                for r in parsed:
                    side = str(r.get("side", "")).lower()
                    if side not in ("over", "o"):
                        continue
                    pname = (r.get("player") or "").strip()
                    line_val = r.get("line")
                    if not pname or line_val is None:
                        continue

                    import pandas as pd
                    if pd.isna(line_val):
                        continue

                    meta = {
                        "event_id": r.get("event_id"),
                        "home_team": r.get("home_team"),
                        "away_team": r.get("away_team"),
                        "commence_time": r.get("commence_time"),
                        "price": r.get("price"),
                        "book": r.get("book"),
                        "market_key": market_key,
                        "side": r.get("side", "Over"),
                    }
                    candidates.append((pname, market_name, float(line_val), meta))

    except Exception as exc:
        log.error("_build_odds_api_candidates failed: %s", exc)

    return candidates


def _build_platform_candidates(
    app, pp_lines, ud_lines, sl_lines, markets, book,
) -> list[tuple]:
    """Build candidate list from platform lines (PrizePicks, Underdog, Sleeper)."""
    candidates = []
    import pandas as pd

    platform_data = []
    if pp_lines:
        platform_data.append(("prizepicks", pp_lines))
    if ud_lines:
        platform_data.append(("underdog", ud_lines))
    if sl_lines:
        platform_data.append(("sleeper", sl_lines))

    # If no platform data provided, try to load from DB
    if not platform_data:
        try:
            from services.odds_service import get_prizepicks_lines, get_underdog_lines, get_sleeper_lines
            pp = get_prizepicks_lines()
            if pp:
                platform_data.append(("prizepicks", pp))
            ud = get_underdog_lines()
            if ud:
                platform_data.append(("underdog", ud))
            sl = get_sleeper_lines()
            if sl:
                platform_data.append(("sleeper", sl))
        except Exception:
            pass

    market_options = set(markets)

    for plat_label, plat_lines in platform_data:
        if book == "prizepicks" and plat_label != "prizepicks":
            continue

        for item in plat_lines:
            pname = str(item.get("player", item.get("player_name", ""))).strip()
            stat_t = item.get("stat_type", "")
            mkt = app.map_platform_stat_to_market(stat_t) if hasattr(app, "map_platform_stat_to_market") else stat_t
            if not mkt or mkt not in market_options:
                continue

            line = item.get("line", item.get("line_score"))
            if not pname or line is None:
                continue
            try:
                if pd.isna(line):
                    continue
            except Exception:
                pass

            # Skip goblin/demon alternate lines for PrizePicks
            if plat_label == "prizepicks":
                odds_type = str(item.get("odds_type", "standard") or "standard").lower()
                if odds_type not in ("standard", ""):
                    continue

            # Platforms use true 50/50 (no vig): decimal 2.0
            plat_price = 2.0

            meta = {
                "event_id": None,
                "home_team": "",
                "away_team": "",
                "commence_time": "",
                "price": plat_price,
                "book": plat_label,
                "market_key": app.ODDS_MARKETS.get(mkt, ""),
                "side": "Over",
            }
            candidates.append((pname, mkt, float(line), meta))

    return candidates
