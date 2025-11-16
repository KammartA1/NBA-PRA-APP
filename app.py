
# ============================================================
# 🏀 UltraMAX Quant – NBA Projection Suite
# Version: v1.0.0  |  Season: 2025-26
# Full Production Streamlit Application
# ============================================================

import streamlit as st

# ------------------------------------------------------------
# Session Initialization (Prevents Streamlit Crashes)
# ------------------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.session_state["model_player"] = None
    st.session_state["joint_legs"] = []
    st.session_state["force_refresh"] = None

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="UltraMAX Quant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Header (Branding + Description)
# ------------------------------------------------------------
st.markdown("""
# 🏀 UltraMAX Quant – NBA Projection Suite  
### Advanced Monte Carlo • Correlated Modeling • Trend Engines • Full Statistical Baselines  
""")

# ------------------------------------------------------------
# Global CSS
# ------------------------------------------------------------
from app.core.layout import inject_css
inject_css()

# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
from app.ui.sidebar import render_sidebar
sidebar_state = render_sidebar()

# ------------------------------------------------------------
# Global Error Layer
# ------------------------------------------------------------
from app.ui.error_layer import error_layer

# ------------------------------------------------------------

# ============================================================
# Basketball Reference Game Log Extractor (Phase 5A-1)
# Full production module for extracting:
#   - last 5 / 10 / 20 logs
#   - season averages
#   - home/away splits
#   - opponent splits
#   - pace-adjusted logs
# ============================================================

import requests
import pandas as pd
from bs4 import BeautifulSoup

class BRefLogsExtractor:
    def __init__(self):
        self.base = "https://www.basketball-reference.com"

    def _fetch_page(self, url):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                return None
            return BeautifulSoup(r.text, "html.parser")
        except:
            return None

    def fetch_gamelog(self, player_id, season="2025"):
        url = f"{self.base}/players/{player_id[0]}/{player_id}/gamelog/{season}"
        soup = self._fetch_page(url)
        if soup is None:
            return None

        table = soup.find("table", {"id": "pgl_basic"})
        if table is None:
            return None

        try:
            df = pd.read_html(str(table))[0]
        except:
            return None

        df = df[df["G"].ne("G")]

        for col in ["PTS", "TRB", "AST"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def get_last_n(self, df, n=5):
        return df.tail(n)

    def get_splits(self, df):
        return {
            "home": df[df["Unnamed: 5"].ne("@")],
            "away": df[df["Unnamed: 5"].eq("@")]
        }

    def build_summary(self, df):
        return {
            "season_pts_mu": float(df["PTS"].mean()),
            "season_reb_mu": float(df["TRB"].mean()),
            "season_ast_mu": float(df["AST"].mean()),
            "pts_sd": float(df["PTS"].std()),
            "reb_sd": float(df["TRB"].std()),
            "ast_sd": float(df["AST"].std())
        }
# ============================================================
# Trend Engine (Phase 5A-2)
# Full production trend module:
#  - Rolling windows (5/10/20)
#  - Z-scores
#  - EMA smoothing
#  - Trend multipliers
#  - Volatility adjustment
# ============================================================

import numpy as np
import pandas as pd

class TrendEngine:
    def __init__(self, alpha=0.35):
        self.alpha = alpha  # EMA smoothing factor

    def rolling_windows(self, series):
        arr = np.array(series, dtype=float)
        return {
            "last5": arr[-5:] if len(arr) >= 5 else arr,
            "last10": arr[-10:] if len(arr) >= 10 else arr,
            "last20": arr[-20:] if len(arr) >= 20 else arr
        }

    def z_score(self, arr):
        if len(arr) < 3:
            return 0.0
        mu = float(np.mean(arr))
        sd = float(np.std(arr)) + 1e-6
        return float((arr[-1] - mu) / sd)

    def ema(self, series):
        arr = np.array(series, dtype=float)
        if len(arr) == 0:
            return 0.0
        ema = arr[0]
        for x in arr[1:]:
            ema = self.alpha * x + (1 - self.alpha) * ema
        return float(ema)

    def trend_multiplier(self, z):
        # Hot streak (z >= 1): above expectation
        # Cold streak (z <= -1): below expectation
        if z >= 1.5:
            return 1.12
        if z >= 1.0:
            return 1.07
        if z <= -1.5:
            return 0.88
        if z <= -1.0:
            return 0.93
        return 1.00

    def volatility_adjustment(self, series):
        sd = float(np.std(series)) + 1e-6
        if sd > 8:
            return 0.96   # high volatility suppresses MU slightly
        if sd < 3:
            return 1.04   # low variance increases confidence in boosts
        return 1.00

    def compute_trend_outputs(self, series):
        if len(series) == 0:
            return {
                "ema": 0.0,
                "z": 0.0,
                "trend_multiplier": 1.0,
                "vol_adjust": 1.0,
                "windows": {"last5": [], "last10": [], "last20": []}
            }

        windows = self.rolling_windows(series)
        last10 = windows["last10"]

        z = self.z_score(last10) if len(last10) >= 3 else 0.0
        ema_val = self.ema(last10)

        mult = self.trend_multiplier(z)
        vol_adj = self.volatility_adjustment(series)

        return {
            "ema": float(ema_val),
            "z": float(z),
            "trend_multiplier": float(mult),
            "vol_adjust": float(vol_adj),
            "windows": {
                "last5": windows["last5"].tolist(),
                "last10": windows["last10"].tolist(),
                "last20": windows["last20"].tolist()
            }
        }

# ============================================================
# Market MU Constructor (Phase 5A-3)
# Full Production MU Builder:
#  - Uses BRef logs
#  - Applies Trend Engine
#  - Applies Defense / Synergy / Rotation / Blowout / Context multipliers
#  - Outputs MU for Points, Rebounds, Assists, PRA
# ============================================================

import numpy as np
from .bref_logs_extractor import BRefLogsExtractor
from .trend_engine import TrendEngine

class MarketMUConstructor:
    def __init__(self):
        self.bref = BRefLogsExtractor()
        self.trend = TrendEngine()

    # -----------------------------
    # Safe chained multiplication
    # -----------------------------
    def _apply_chain(self, base, *mods):
        out = float(base)
        for m in mods:
            if m is not None:
                try:
                    out *= float(m)
                except:
                    pass
        return float(out)

    # -----------------------------
    # Compute MU for a single stat
    # -----------------------------
    def compute_stat_mu(self, series, trend_outputs,
                        def_mult, syn_mult, rot_mult, blow_mult, ctx_mult):

        if len(series) == 0:
            return 0.0

        ema = trend_outputs["ema"]
        mean_val = float(np.mean(series))

        # EMA-weighted baseline
        base = (0.55 * ema) + (0.45 * mean_val)

        # Apply trend multiplier
        mu = self._apply_chain(base, trend_outputs["trend_multiplier"])

        # Apply other multipliers
        mu = self._apply_chain(mu, def_mult, syn_mult, rot_mult, blow_mult, ctx_mult)

        return float(mu)

    # -----------------------------
    # Compute all market MU values
    # -----------------------------
    def compute_all_mu(self, player_id, season, engines):
        df = self.bref.fetch_gamelog(player_id, season)
        if df is None or len(df) == 0:
            return {"PTS": 0, "REB": 0, "AST": 0, "PRA": 0}

        pts = df["PTS"].tolist()
        reb = df["TRB"].tolist()
        ast = df["AST"].tolist()

        trend_pts = self.trend.compute_trend_outputs(pts)
        trend_reb = self.trend.compute_trend_outputs(reb)
        trend_ast = self.trend.compute_trend_outputs(ast)

        d = engines.get("defense", 1.0)
        s = engines.get("synergy", 1.0)
        r = engines.get("rotation", 1.0)
        b = engines.get("blowout", 1.0)
        c = engines.get("context", 1.0)

        mu_pts = self.compute_stat_mu(pts, trend_pts, d, s, r, b, c)
        mu_reb = self.compute_stat_mu(reb, trend_reb, d, s, r, b, c)
        mu_ast = self.compute_stat_mu(ast, trend_ast, d, s, r, b, c)

        mu_pra = mu_pts + mu_reb + mu_ast

        return {
            "PTS": float(mu_pts),
            "REB": float(mu_reb),
            "AST": float(mu_ast),
            "PRA": float(mu_pra)

# ============================================================
# Market SD Constructor (Phase 5A-4)
# Full Production Volatility Engine:
#  - Historical variance from BRef logs
#  - Rotation volatility
#  - Defensive volatility
#  - Synergy volatility
#  - Blowout volatility
#  - Min/max SD control
#  - PRA SD computed in covariance builder (5A-5)
# ============================================================

import numpy as np
from .bref_logs_extractor import BRefLogsExtractor
from .trend_engine import TrendEngine

class MarketSDConstructor:
    def __init__(self):
        self.bref = BRefLogsExtractor()
        self.trend = TrendEngine()

    def _apply_chain(self, base, *mods):
        out = float(base)
        for m in mods:
            try:
                out *= float(m)
            except:
                pass
        return float(out)

    def compute_stat_sd(self, series, role_vol, def_vol, syn_vol, blow_vol):
        if len(series) < 3:
            return 1.5  # minimal variance floor

        sd = float(np.std(series))

        # Apply multipliers
        sd = self._apply_chain(sd, role_vol, def_vol, syn_vol, blow_vol)

        # Hard stability floor
        if sd < 1.0:
            sd = 1.0

        # Hard ceiling to prevent runaway volatility
        if sd > 12.0:
            sd = 12.0

        return float(sd)

    def compute_all_sd(self, player_id, season, engines):
        df = self.bref.fetch_gamelog(player_id, season)
        if df is None or len(df) == 0:
            return {"PTS": 3.0, "REB": 2.0, "AST": 2.0}

        pts = df["PTS"].tolist()
        reb = df["TRB"].tolist()
        ast = df["AST"].tolist()

        role_vol = engines.get("rotation_vol", 1.0)
        def_vol = engines.get("defense_vol", 1.0)
        syn_vol = engines.get("synergy_vol", 1.0)
        blow_vol = engines.get("blowout_vol", 1.0)

        sd_pts = self.compute_stat_sd(pts, role_vol, def_vol, syn_vol, blow_vol)
        sd_reb = self.compute_stat_sd(reb, role_vol, def_vol, syn_vol, blow_vol)
        sd_ast = self.compute_stat_sd(ast, role_vol, def_vol, syn_vol, blow_vol)

        return {
            "PTS": float(sd_pts),
            "REB": float(sd_reb),
            "AST": float(sd_ast)
        }

# ============================================================
# Covariance Builder (Phase 5A-5)
# Computes:
#   - Cov(P, R)
#   - Cov(P, A)
#   - Cov(R, A)
#   - Full 3x3 covariance + correlation matrices
#   - PRA variance = var(P)+var(R)+var(A)+2covPR+2covPA+2covRA
# ============================================================

import numpy as np
from .bref_logs_extractor import BRefLogsExtractor

class CovarianceBuilder:
    def __init__(self):
        self.bref = BRefLogsExtractor()

    def safe_cov(self, x, y):
        if len(x) < 3 or len(y) < 3:
            return 0.0
        return float(np.cov(x, y)[0][1])

    def compute_covariance_matrix(self, player_id, season):
        df = self.bref.fetch_gamelog(player_id, season)
        if df is None or len(df) < 3:
            # Fallback minimal structure
            return {
                "cov_matrix": np.zeros((3,3)).tolist(),
                "cor_matrix": np.zeros((3,3)).tolist(),
                "pra_variance": 9.0  # fallback
            }

        pts = df["PTS"].astype(float).tolist()
        reb = df["TRB"].astype(float).tolist()
        ast = df["AST"].astype(float).tolist()

        # Base variances
        var_p = float(np.var(pts)) + 1e-6
        var_r = float(np.var(reb)) + 1e-6
        var_a = float(np.var(ast)) + 1e-6

        # Covariances
        cov_pr = self.safe_cov(pts, reb)
        cov_pa = self.safe_cov(pts, ast)
        cov_ra = self.safe_cov(reb, ast)

        # Covariance matrix (P, R, A)
        cov_matrix = np.array([
            [var_p,   cov_pr, cov_pa],
            [cov_pr,  var_r,  cov_ra],
            [cov_pa,  cov_ra, var_a]
        ], dtype=float)

# ============================================================
# Full Statistical Baseline Engine (Phase 5A-6)
# Integrates:
#  - Market MU Constructor
#  - Market SD Constructor
#  - Covariance Builder
# Produces unified MU/SD for PTS/REB/AST/PRA
# ============================================================

from .market_mu_constructor import MarketMUConstructor
from .market_sd_constructor import MarketSDConstructor
from .covariance_builder import CovarianceBuilder

class StatBaselineEngine:
    def __init__(self):
        self.mu_engine = MarketMUConstructor()
        self.sd_engine = MarketSDConstructor()
        self.cov_engine = CovarianceBuilder()

    def compute_market_baseline(self, player_id, season, market, engines):
        """
        market: 'PTS', 'REB', 'AST', or 'PRA'
        engines: dict of multipliers and volatility factors
        """
        mu_all = self.mu_engine.compute_all_mu(player_id, season, engines)
        sd_all = self.sd_engine.compute_all_sd(player_id, season, engines)
        cov = self.cov_engine.compute_covariance_matrix(player_id, season)

        if market == "PRA":
            pra_sd = cov["pra_variance"] ** 0.5
            return {
                "mu": mu_all["PTS"] + mu_all["REB"] + mu_all["AST"],
                "sd": float(pra_sd)
            }
        else:
            return {
                "mu": mu_all.get(market, 0.0),
                "sd": sd_all.get(market, 0.0)
            }

    def compute_all_baselines(self, player_id, season, engines):
        mu_all = self.mu_engine.compute_all_mu(player_id, season, engines)
        sd_all = self.sd_engine.compute_all_sd(player_id, season, engines)
        cov = self.cov_engine.compute_covariance_matrix(player_id, season)
        pra_sd = cov["pra_variance"] ** 0.5

        return {
            "PTS": {"mu": mu_all["PTS"], "sd": sd_all["PTS"]},
            "REB": {"mu": mu_all["REB"], "sd": sd_all["REB"]},
            "AST": {"mu": mu_all["AST"], "sd": sd_all["AST"]},
            "PRA": {
                "mu": mu_all["PTS"] + mu_all["REB"] + mu_all["AST"],
                "sd": float(pra_sd)
            },
            "cov_matrix": cov["cov_matrix"],
            "cor_matrix": cov["cor_matrix"]

# ============================================================
# Defense Engine (Phase 5B-1)
# Full Production Defensive Modeling:
#  - Opponent DRtg normalization
#  - Position defense
#  - Rim vs Perimeter metrics
#  - Team scheme multipliers
#  - Pace-adjusted defensive strength
#  - Defensive volatility factor
# ============================================================

import numpy as np

class DefenseEngine:
    def __init__(self):
        pass

    # --------------------------------------------------------
    # Normalize defensive rating (DRtg)
    # --------------------------------------------------------
    def normalize_drtg(self, opp_drtg):
        """
        League average DRtg ~ 113
        Lower = better defense
        """
        league_avg = 113.0
        delta = (opp_drtg - league_avg) / 50.0
        return float(1.0 - delta)

    # --------------------------------------------------------
    # Position-specific defense (placeholder scaling)
    # --------------------------------------------------------
    def position_defense_multiplier(self, pos_def_rating):
        """
        pos_def_rating: percentile 0-100 (lower = better D)
        """
        norm = (pos_def_rating - 50) / 200.0
        return float(1.0 - norm)

    # --------------------------------------------------------
    # Rim vs Perimeter defense modifiers
    # --------------------------------------------------------
    def rim_defense_mult(self, rim_rating):
        norm = (rim_rating - 50) / 200.0
        return float(1.0 - norm)

    def perimeter_defense_mult(self, per_rating):
        norm = (per_rating - 50) / 200.0
        return float(1.0 - norm)

    # --------------------------------------------------------
    # Team scheme adjustment (switch, drop, hedge)
    # --------------------------------------------------------
    def scheme_multiplier(self, scheme):
        table = {
            "switch": 0.97,
            "drop": 1.03,
            "hedge": 1.00
        }
        return float(table.get(scheme, 1.00))

    # --------------------------------------------------------
    # Defensive volatility based on inconsistency
    # --------------------------------------------------------
    def defensive_volatility(self, consistency_score):
        """
        consistency_score: 0-100 (higher = consistent)
        """
        if consistency_score < 40:
            return 1.08
        if consistency_score > 70:
            return 0.95
        return 1.00

    # --------------------------------------------------------
    # Combined Defensive Multiplier
    # --------------------------------------------------------
    def compute_defense_multiplier(self, inputs):
        """
        inputs = {
          "opp_drtg",
          "position_def",
          "rim_def",
          "per_def",
          "scheme",
          "consistency"
        }
        """
        m1 = self.normalize_drtg(inputs.get("opp_drtg", 113))
        m2 = self.position_defense_multiplier(inputs.get("position_def", 50))
        m3 = self.rim_defense_mult(inputs.get("rim_def", 50))
        m4 = self.perimeter_defense_mult(inputs.get("per_def", 50))
        m5 = self.scheme_multiplier(inputs.get("scheme", "switch"))
        m6 = self.defensive_volatility(inputs.get("consistency", 60))

        mult = m1 * m2 * m3 * m4 * m5
        return {
            "def_multiplier": float(mult),
            "def_volatility": float(m6)
        }

# ============================================================
# Team Context Engine (Phase 5B-2)
# Full Production Context Modeling:
#   - Pace normalization
#   - Home/Away adjustment
#   - Rest-day fatigue model
#   - Back-to-back penalty
#   - Altitude impact (Denver/Utah)
#   - Travel fatigue
#   - Team strength index
#   - Context volatility multiplier
# ============================================================

import numpy as np

class TeamContextEngine:
    def __init__(self):
        pass

    def pace_multiplier(self, team_pace, opp_pace):
        league_pace = 99.0
        avg = (team_pace + opp_pace) / 2
        delta = (avg - league_pace) / 20
        return float(1.0 + delta)

    def home_away_multiplier(self, is_home):
        return 1.03 if is_home else 0.97

    def rest_multiplier(self, days_rest):
        if days_rest >= 3:
            return 1.05
        if days_rest == 2:
            return 1.02
        if days_rest == 1:
            return 1.00
        return 0.92

    def b2b_multiplier(self, is_b2b):
        return 0.94 if is_b2b else 1.00

    def altitude_multiplier(self, city):
        if city in ["Denver", "Utah", "Salt Lake City"]:
            return 0.96
        return 1.00

    def travel_multiplier(self, miles):
        if miles > 1500:
            return 0.93
        if miles > 800:
            return 0.96
        return 1.00

    def strength_multiplier(self, off_rating, def_rating):
        league_off = 113.0
        league_def = 113.0
        off_delta = (off_rating - league_off) / 50
        def_delta = (def_rating - league_def) / 50
        return float(1.0 + off_delta*0.4 - def_delta*0.4)

    def context_volatility(self, inconsistency):
        if inconsistency > 70:
            return 1.00
        if inconsistency < 40:
            return 1.07
        return 1.03

    def compute_context_multiplier(self, ctx):
        m1 = self.pace_multiplier(ctx.get("team_pace", 98), ctx.get("opp_pace", 98))
        m2 = self.home_away_multiplier(ctx.get("is_home", True))
        m3 = self.rest_multiplier(ctx.get("days_rest", 1))
        m4 = self.b2b_multiplier(ctx.get("is_b2b", False))
        m5 = self.altitude_multiplier(ctx.get("city", ""))
        m6 = self.travel_multiplier(ctx.get("travel_miles", 0))
        m7 = self.strength_multiplier(ctx.get("team_off_rating", 113),
                                      ctx.get("team_def_rating", 113))
        m8 = self.context_volatility(ctx.get("inconsistency", 60))

        core = m1 * m2 * m7
        final = core * m3 * m4 * m5 * m6

        return {
            "context_multiplier": float(final),
            "context_volatility": float(m8)
        }

# ============================================================
# Rotation Volatility Engine (Phase 5B-3)
# Full production system for:
#   - Minute distribution modeling
#   - Role stability
#   - Foul volatility
#   - Coach trust index
#   - Bench depth volatility
#   - Injury-return volatility
#   - Garbage-time suppression
#   - Rotation multiplier + volatility factor
# ============================================================

import numpy as np

class RotationVolatilityEngine:
    def __init__(self):
        pass

    def minutes_mean(self, projected_minutes, recent_minutes):
        if len(recent_minutes) == 0:
            return float(projected_minutes)
        hist_mean = float(np.mean(recent_minutes))
        return float(0.6 * projected_minutes + 0.4 * hist_mean)

    def minutes_sd(self, recent_minutes):
        if len(recent_minutes) < 3:
            return 2.5
        sd = float(np.std(recent_minutes))
        return float(min(max(sd, 1.5), 7.0))

    def foul_volatility(self, foul_rate):
        if foul_rate >= 4.5: return 1.12
        if foul_rate >= 3.5: return 1.06
        return 1.00

    def coach_trust(self, trust_score):
        if trust_score >= 80: return 0.94
        if trust_score <= 40: return 1.08
        return 1.00

    def bench_depth_volatility(self, bench_depth):
        if bench_depth >= 10: return 1.07
        if bench_depth <= 8: return 0.97
        return 1.00

    def injury_return_volatility(self, games_back):
        if games_back <= 2: return 1.10
        if games_back <= 5: return 1.05
        return 1.00

    def garbage_time_multiplier(self, blowout_risk):
        if blowout_risk >= 0.25: return 0.94
        if blowout_risk >= 0.15: return 0.97
        return 1.00

    def compute_rotation_outputs(self, rot):
        projected = rot.get("projected_minutes", 30)
        recent = rot.get("recent_minutes", [])
        foul = rot.get("foul_rate", 2.5)
        trust = rot.get("coach_trust", 60)
        bench = rot.get("bench_depth", 9)
        games_back = rot.get("games_back", 10)
        blow = rot.get("blowout_risk", 0.10)

        mean_min = self.minutes_mean(projected, recent)
        sd_min = self.minutes_sd(recent)

        foul_m = self.foul_volatility(foul)
        trust_m = self.coach_trust(trust)
        bench_m = self.bench_depth_volatility(bench)
        inj_m = self.injury_return_volatility(games_back)
        garbage_m = self.garbage_time_multiplier(blow)

        rotation_multiplier = float(foul_m * trust_m * bench_m * inj_m * garbage_m)
        volatility_factor = float(sd_min / max(mean_min, 1.0))

        return {
            "minutes_mean": float(mean_min),
            "minutes_sd": float(sd_min),
            "rotation_multiplier": rotation_multiplier,
            "volatility_factor": volatility_factor
        }

# ============================================================
# Blowout Engine (Phase 5B-4)
# Full Production Module:
#   - Vegas spread → win probability model
#   - Blowout probability curve
#   - Pace decay in blowouts
#   - Minutes suppression mapping
#   - Blowout volatility factor
#   - Final MU/SD blowout multipliers
# ============================================================

import numpy as np
import math

class BlowoutEngine:
    def __init__(self):
        pass

    # --------------------------------------------------------
    # Convert Vegas spread → win probability using logistic curve
    # --------------------------------------------------------
    def spread_to_win_prob(self, spread):
        """
        Positive spread = favorite
        Negative spread = underdog

        Logistic approximation:
        win_prob = 1 / (1 + exp(-spread / 6.5))
        """
        return 1.0 / (1.0 + math.exp(-spread / 6.5))

    # --------------------------------------------------------
    # Blowout probability curve
    # --------------------------------------------------------
    def compute_blowout_prob(self, spread):
        """
        Probability game is decided by 15+ points.
        Approximated by:
        blowout_prob = logistic(spread * 0.55)
        """
        return 1.0 / (1.0 + math.exp(-spread * 0.55))

    # --------------------------------------------------------
    # Pace decay in blowouts
    # --------------------------------------------------------
    def pace_decay_multiplier(self, blowout_prob):
        """
        As blowout probability increases, pace drops.
        """
        return float(1.0 - (0.18 * blowout_prob))

    # --------------------------------------------------------
    # Minutes suppression due to blowout
    # --------------------------------------------------------
    def minutes_suppression_multiplier(self, blowout_prob, role):
        """
        Starters lose more minutes in blowouts, bench may gain.
        """
        if role == "starter":
            return float(1.0 - (0.30 * blowout_prob))
        else:
            return float(1.0 + (0.10 * blowout_prob))

    # --------------------------------------------------------
    # Blowout volatility factor
    # --------------------------------------------------------
    def blowout_volatility(self, blowout_prob):
        """
        High blowout probability increases variance.
        """
        if blowout_prob >= 0.35:
            return 1.12
        if blowout_prob >= 0.20:
            return 1.06
        return 1.00

    # --------------------------------------------------------
    # Main blowout output wrapper
    # --------------------------------------------------------
    def compute_blowout_outputs(self, inputs):
        """
        inputs = {
            "spread",
            "role",   # "starter" or "bench"
            "team_pace",
            "opp_pace"
        }
        """
        spread = inputs.get("spread", 0)
        role = inputs.get("role", "starter")

        # 1. Core probabilities
        win_prob = self.spread_to_win_prob(spread)
        blow_prob = self.compute_blowout_prob(spread)

        # 2. Pace decay
        pace_mult = self.pace_decay_multiplier(blow_prob)

        # 3. Minutes suppression
        minute_mult = self.minutes_suppression_multiplier(blow_prob, role)

        # 4. Volatility
        vol_mult = self.blowout_volatility(blow_prob)

        # 5. Combined blowout multiplier (affects MU)
        blowout_multiplier = float(pace_mult * minute_mult)

        return {
            "win_prob": float(win_prob),
            "blowout_prob": float(blow_prob),
            "blowout_multiplier": float(blowout_multiplier),
            "blowout_volatility": float(vol_mult)
        }

# ============================================================
# Synergy Playtype Engine (Phase 5B-5)
# Full Production Module:
#   - Offensive playtype weights
#   - Opponent defensive playtype matchups
#   - Usage-based multipliers
#   - Efficiency-driven multipliers
#   - Volatility modeling based on playtype risk profile
# ============================================================

import numpy as np

class SynergyEngine:
    def __init__(self):
        # Baseline weighting for each playtype
        self.playtypes = [
            "spot_up", "iso", "pnr_bh", "pnr_roll",
            "off_screen", "cut", "dho", "post_up", "transition"
        ]

    # --------------------------------------------------------
    # Normalize opponent defensive percentile (0-100)
    # --------------------------------------------------------
    def _normalize_def_pct(self, pct):
        return float(1.0 - ((pct - 50) / 200.0))

    # --------------------------------------------------------
    # Usage-based scaling
    # --------------------------------------------------------
    def usage_multiplier(self, usage_rate):
        if usage_rate >= 30:
            return 1.04
        if usage_rate >= 24:
            return 1.02
        if usage_rate <= 16:
            return 0.97
        return 1.00

    # --------------------------------------------------------
    # Efficiency-based scaling
    # --------------------------------------------------------
    def efficiency_multiplier(self, ppp):
        # points per possession
        if ppp >= 1.15:
            return 1.06
        if ppp >= 1.05:
            return 1.03
        if ppp <= 0.90:
            return 0.95
        return 1.00

    # --------------------------------------------------------
    # Volatility from playtype mix
    # --------------------------------------------------------
    def volatility_from_mix(self, usage_distribution):
        arr = np.array(list(usage_distribution.values()))
        sd = np.std(arr)
        if sd > 0.10:
            return 1.08
        if sd < 0.05:
            return 0.96
        return 1.00

    # --------------------------------------------------------
    # Master synergy calculation
    # --------------------------------------------------------
    def compute_synergy_outputs(self, syn):
        """
        syn = {
          "usage": {playtype: fraction},
          "ppp": {playtype: efficiency},
          "opp_def": {playtype: def_percentile},
          "usage_rate": int (overall usage %)
        }
        """

        usage = syn.get("usage", {})
        ppp = syn.get("ppp", {})
        opp = syn.get("opp_def", {})
        usage_rate = syn.get("usage_rate", 22)

        multipliers = []
        for pt in self.playtypes:
            u = usage.get(pt, 0.0)
            e = ppp.get(pt, 1.00)
            d = opp.get(pt, 50)

            def_mult = self._normalize_def_pct(d)
            eff_mult = self.efficiency_multiplier(e)

            m = 1.0 + (u * (eff_mult - 1)) + (u * (def_mult - 1))
            multipliers.append(m)

        synergy_mult = float(np.prod(multipliers))
        usage_mult = self.usage_multiplier(usage_rate)
        vol = self.volatility_from_mix(usage)

        final_mult = float(synergy_mult * usage_mult)

        return {
            "synergy_multiplier": final_mult,
            "synergy_volatility": vol
        }

# ============================================================
# Similarity Engine (Phase 5B-6)
# Multidimensional Player Comp Model:
#   - Usage rate similarity
#   - Pace environment similarity
#   - Playtype distribution similarity
#   - Efficiency vector similarity
#   - MU/SD vector distance
#   - Context similarity
#   - Outputs:
#       similarity_score (0-1)
#       similarity_multiplier
#       similarity_volatility
#       top-5 historical comps
# ============================================================

import numpy as np

class SimilarityEngine:
    def __init__(self):
        pass

    # --------------------------------------------------------
    # Cosine similarity helper
    # --------------------------------------------------------
    def _cosine(self, a, b):
        a=np.array(a,dtype=float)
        b=np.array(b,dtype=float)
        if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
            return 0.0
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

    # --------------------------------------------------------
    # Normalize vectors into same dimensionality
    # --------------------------------------------------------
    def _ensure_vec(self, dct, keys):
        return [float(dct.get(k,0)) for k in keys]

    # --------------------------------------------------------
    # Main similarity calculation
    # --------------------------------------------------------
    def compute_similarity_outputs(self, base_player, comp_pool):
        """
        base_player = {
          "usage_rate": float,
          "pace": float,
          "playtype": {pt: fraction},
          "eff": {pt: ppp},
          "mu_vec": [PTS, REB, AST],
          "sd_vec": [sd_P, sd_R, sd_A],
          "context": float
        }

        comp_pool = list of player dicts with same structure
        """

        # define consistent vector keys
        playtypes = ["spot_up","iso","pnr_bh","pnr_roll","off_screen","cut","dho","post_up","transition"]

        base_usage = base_player.get("usage_rate",20)
        base_pace = base_player.get("pace",99)
        base_pt = self._ensure_vec(base_player.get("playtype",{}), playtypes)
        base_eff = self._ensure_vec(base_player.get("eff",{}), playtypes)
        base_mu = base_player.get("mu_vec",[10,5,3])
        base_sd = base_player.get("sd_vec",[4,3,2])
        base_ctx = base_player.get("context",1.0)

        comp_scores=[]

        for comp in comp_pool:
            c_usage = comp.get("usage_rate",20)
            c_pace = comp.get("pace",99)
            c_pt = self._ensure_vec(comp.get("playtype",{}), playtypes)
            c_eff = self._ensure_vec(comp.get("eff",{}), playtypes)
            c_mu = comp.get("mu_vec",[10,5,3])
            c_sd = comp.get("sd_vec",[4,3,2])
            c_ctx = comp.get("context",1.0)

            # compute similarities
            s_usage = 1 - abs(base_usage-c_usage)/40
            s_pace = 1 - abs(base_pace-c_pace)/30
            s_pt = self._cosine(base_pt,c_pt)
            s_eff = self._cosine(base_eff,c_eff)
            s_mu = self._cosine(base_mu,c_mu)
            s_sd = self._cosine(base_sd,c_sd)
            s_ctx = 1 - abs(base_ctx-c_ctx)/1.5

            total = max(0.0, (s_usage + s_pace + s_pt + s_eff + s_mu + s_sd + s_ctx)/7)

            comp_scores.append({
                "player": comp.get("name","Unknown"),
                "score": float(total)
            })

        comp_scores.sort(key=lambda x: -x["score"])
        top = comp_scores[:5]

        # similarity multiplier = blend around score
        best = top[0]["score"] if len(top)>0 else 0
        similarity_multiplier = 1.0 + (best-0.5)*0.12

        # volatility: low similarity -> high vol
        if best >= 0.75: vol=0.96
        elif best >= 0.55: vol=1.00
        else: vol=1.07

        return {
            "similarity_score": float(best),
            "similarity_multiplier": float(similarity_multiplier),
            "similarity_volatility": float(vol),
            "top_comps": top
        }

# ============================================================
# Correlation Engine (Phase 5B-7)
# Full Production Correlation Modeling:
#   - Rolling correlations using last 5/10/20 games
#   - Synergy-driven correlation adjustments
#   - Context-driven covariance drift
#   - Rotation-driven correlation shifts
#   - Similarity-engine blending
#   - Outputs full correlation matrix + adjustment factor
# ============================================================

import numpy as np

class CorrelationEngine:
    def __init__(self):
        pass

    # --------------------------------------------------------
    # Safe correlation helper
    # --------------------------------------------------------
    def _safe_corr(self, x, y):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if len(x) < 3 or len(y) < 3:
            return 0.0
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0][1])

    # --------------------------------------------------------
    # Rolling window correlation
    # --------------------------------------------------------
    def rolling_correlations(self, pts, reb, ast):
        windows = [5, 10, 20]
        corr_list = []

        for w in windows:
            if len(pts) >= w and len(reb) >= w:
                corr_list.append(self._safe_corr(pts[-w:], reb[-w:]))
            if len(pts) >= w and len(ast) >= w:
                corr_list.append(self._safe_corr(pts[-w:], ast[-w:]))
            if len(reb) >= w and len(ast) >= w:
                corr_list.append(self._safe_corr(reb[-w:], ast[-w:]))

        if len(corr_list) == 0:
            return 0.0
        return float(np.mean(corr_list))

    # --------------------------------------------------------
    # Synergy-based correlation adjustment
    # --------------------------------------------------------
    def synergy_adjustment(self, synergy_volatility):
        if synergy_volatility > 1.05:
            return -0.04
        if synergy_volatility < 0.98:
            return +0.03
        return 0.0

    # --------------------------------------------------------
    # Context-based correlation drift
    # --------------------------------------------------------
    def context_adjustment(self, context_volatility):
        if context_volatility > 1.05:
            return +0.05
        if context_volatility < 0.98:
            return -0.03
        return 0.0

    # --------------------------------------------------------
    # Rotation-driven correlation change
    # --------------------------------------------------------
    def rotation_adjustment(self, rotation_volatility):
        if rotation_volatility > 0.14:
            return +0.06
        if rotation_volatility < 0.07:
            return -0.04
        return 0.0

    # --------------------------------------------------------
    # Similarity-based correlation reinforcement
    # --------------------------------------------------------
    def similarity_adjustment(self, similarity_score):
        if similarity_score >= 0.75:
            return -0.04
        if similarity_score <= 0.55:
            return +0.05
        return 0.0

    # --------------------------------------------------------
    # FULL correlation matrix builder
    # --------------------------------------------------------
    def compute_correlation_outputs(self, corr):
        """
        corr = {
          "pts": list,
          "reb": list,
          "ast": list,
          "synergy_vol": float,
          "context_vol": float,
          "rotation_vol": float,
          "similarity_score": float
        }
        """

        pts = corr.get("pts", [])
        reb = corr.get("reb", [])
        ast = corr.get("ast", [])

        # 1. Rolling base correlation
        base_corr = self.rolling_correlations(pts, reb, ast)

        # 2. Adjustments
        adj = 0
        adj += self.synergy_adjustment(corr.get("synergy_vol", 1.0))
        adj += self.context_adjustment(corr.get("context_vol", 1.0))
        adj += self.rotation_adjustment(corr.get("rotation_vol", 0.10))
        adj += self.similarity_adjustment(corr.get("similarity_score", 0.60))

        final_corr = max(min(base_corr + adj, 0.95), -0.95)

        # build full matrix (pts, reb, ast)
        C = np.array([
            [1.0,      final_corr, final_corr],
            [final_corr, 1.0,      final_corr],
            [final_corr, final_corr, 1.0]
        ])

        return {
            "corr_matrix": C.tolist(),
            "corr_adjustment": float(adj),
            "base_corr": float(base_corr),
            "final_corr": float(final_corr)
        }

# ============================================================
# Engine Input Builder (Phase 5C-1)
# Full Production Integration Layer
#
# Merges all Phase 5A and Phase 5B engine outputs into
# one unified projection input object used by Model Tab,
# Player Card, Joint EV, Overrides, Calibration, etc.
# ============================================================

from .stat_baseline_engine import StatBaselineEngine
from .defense_engine import DefenseEngine
from .team_context_engine import TeamContextEngine
from .synergy_engine import SynergyEngine
from .rotation_volatility_engine import RotationVolatilityEngine
from .blowout_engine import BlowoutEngine
from .similarity_engine import SimilarityEngine
from .correlation_engine import CorrelationEngine
from .bref_logs_extractor import BRefLogsExtractor

class EngineInputBuilder:

    def __init__(self):
        self.baseline = StatBaselineEngine()
        self.defense = DefenseEngine()
        self.context = TeamContextEngine()
        self.synergy = SynergyEngine()
        self.rotation = RotationVolatilityEngine()
        self.blowout = BlowoutEngine()
        self.similar = SimilarityEngine()
        self.corr_engine = CorrelationEngine()
        self.bref = BRefLogsExtractor()

    # --------------------------------------------------------
    # Unified Build Function
    # --------------------------------------------------------
    def build(self, player_id, season, raw_inputs):
        """
        raw_inputs = {
          "defense": {...},
          "context": {...},
          "synergy": {...},
          "rotation": {...},
          "blowout": {...},
          "similarity_base": {...},
          "similarity_pool": [...]
        }
        """

        # 1. Logs Extraction
        logs = self.bref.fetch_gamelog(player_id, season)
        pts = logs["PTS"].astype(float).tolist() if logs is not None else []
        reb = logs["TRB"].astype(float).tolist() if logs is not None else []
        ast = logs["AST"].astype(float).tolist() if logs is not None else []

        # 2. Individual Engines
        def_out   = self.defense.compute_defense_multiplier(raw_inputs.get("defense", {}))
        ctx_out   = self.context.compute_context_multiplier(raw_inputs.get("context", {}))
        syn_out   = self.synergy.compute_synergy_outputs(raw_inputs.get("synergy", {}))
        rot_out   = self.rotation.compute_rotation_outputs(raw_inputs.get("rotation", {}))
        blow_out  = self.blowout.compute_blowout_outputs(raw_inputs.get("blowout", {}))
        sim_out   = self.similar.compute_similarity_outputs(
                        raw_inputs.get("similarity_base", {}),
                        raw_inputs.get("similarity_pool", [])
                    )

        # 3. Engine multipliers for MU/SD engine
        engines = {
            "defense": def_out["def_multiplier"],
            "synergy": syn_out["synergy_multiplier"],
            "rotation": rot_out["rotation_multiplier"],
            "blowout": blow_out["blowout_multiplier"],
            "context": ctx_out["context_multiplier"],
            "defense_vol": def_out["def_volatility"],
            "synergy_vol": syn_out["synergy_volatility"],
            "rotation_vol": rot_out["volatility_factor"],
            "blowout_vol": blow_out["blowout_volatility"],
            "context_vol": ctx_out["context_volatility"]
        }

        # 4. Full baselines (MU, SD, Cov)
        baselines = self.baseline.compute_all_baselines(player_id, season, engines)

        # 5. Correlation engine
        corr_out = self.corr_engine.compute_correlation_outputs({
            "pts": pts,
            "reb": reb,
            "ast": ast,
            "synergy_vol": syn_out["synergy_volatility"],
            "context_vol": ctx_out["context_volatility"],
            "rotation_vol": rot_out["volatility_factor"],
            "similarity_score": sim_out["similarity_score"]
        })

        # 6. Unified Output
        return {
            "mu": {
                "PTS": baselines["PTS"]["mu"],
                "REB": baselines["REB"]["mu"],
                "AST": baselines["AST"]["mu"],
                "PRA": baselines["PRA"]["mu"]
            },
            "sd": {
                "PTS": baselines["PTS"]["sd"],
                "REB": baselines["REB"]["sd"],
                "AST": baselines["AST"]["sd"],
                "PRA": baselines["PRA"]["sd"]
            },
            "cov": baselines["cov_matrix"],
            "cor": corr_out["corr_matrix"],

            "multipliers": {
                "defense": def_out,
                "synergy": syn_out,
                "rotation": rot_out["rotation_multiplier"],
                "blowout": blow_out,
                "context": ctx_out,
                "similarity": sim_out
            },

            "volatility": {
                "defense": def_out["def_volatility"],
                "synergy": syn_out["synergy_volatility"],
                "rotation": rot_out["volatility_factor"],
                "blowout": blow_out["blowout_volatility"],
                "context": ctx_out["context_volatility"],
                "similarity": sim_out["similarity_volatility"]
            },

            "rotation": rot_out,
            "blowout": blow_out,
            "synergy": syn_out,
            "defense": def_out,
            "context": ctx_out,
            "similarity": sim_out,
            "correlation": corr_out
        }

# ============================================================
# Projection Engine Wrapper (Phase 5C-2)
# Converts unified engine inputs into:
#   - Final MU / SD adjustments
#   - Normal distribution objects
#   - Monte Carlo simulation inputs
#   - Probability over/under outputs
#   - PRA combined distribution
# ============================================================

import numpy as np
from math import erf, sqrt

class ProjectionEngine:

    # --------------------------------------------------------
    # Normal CDF
    # --------------------------------------------------------
    def _norm_cdf(self, x, mu, sd):
        if sd <= 0: sd = 1e-6
        z = (x - mu) / (sd * sqrt(2))
        return 0.5 * (1 + erf(z))

    # --------------------------------------------------------
    # Probability a stat goes over/under a line
    # --------------------------------------------------------
    def probability(self, mu, sd, line):
        p_under = self._norm_cdf(line, mu, sd)
        p_over = 1 - p_under
        return float(p_over), float(p_under)

    # --------------------------------------------------------
    # Build final projection object
    # --------------------------------------------------------
    def project(self, inputs, line_dict):
        """
        inputs = unified dictionary from EngineInputBuilder
        line_dict = {"PTS": X, "REB": Y, "AST": Z, "PRA": W}
        """

        mu = inputs["mu"]
        sd = inputs["sd"]
        cov = np.array(inputs["cov"], dtype=float)

        # Over/under probabilities
        probs = {}
        for stat in ["PTS", "REB", "AST", "PRA"]:
            if stat in line_dict:
                line = line_dict[stat]
                p_over, p_under = self.probability(mu[stat], sd[stat], line)
                probs[stat] = {
                    "p_over": p_over,
                    "p_under": p_under
                }

        # Package for Monte Carlo
        mc_inputs = {
            "mu_vec": np.array([mu["PTS"], mu["REB"], mu["AST"]], dtype=float).tolist(),
            "cov_matrix": cov.tolist()
        }

        return {
            "mu": mu,
            "sd": sd,
            "probabilities": probs,
            "mc_inputs": mc_inputs,
            "cov": cov.tolist(),
            "cor": inputs.get("cor", inputs.get("correlation", {}).get("corr_matrix")),
            "raw": inputs
        }

# ============================================================
# Monte Carlo Simulation Engine (Phase 5C-3)
# Correlated Multivariate Simulation for:
#   - PTS, REB, AST
#   - PRA distribution
#   - Joint outcome probabilities
# ============================================================

import numpy as np

class MonteCarloEngine:
    def __init__(self, sims=15000):
        self.sims = sims

    def run(self, mu_vec, cov_matrix):
        mu = np.array(mu_vec, dtype=float)
        cov = np.array(cov_matrix, dtype=float)

        samples = np.random.multivariate_normal(mu, cov, self.sims)

        pts = samples[:,0]
        reb = samples[:,1]
        ast = samples[:,2]
        pra = pts + reb + ast

        return {
            "PTS": pts.tolist(),
            "REB": reb.tolist(),
            "AST": ast.tolist(),
            "PRA": pra.tolist()
        }

    def probability_over(self, dist, line):
        arr = np.array(dist, dtype=float)
        return float(np.mean(arr > line))

    def summary(self, dist):
        arr = np.array(dist, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr)),
            "p25": float(np.percentile(arr,25)),
            "p50": float(np.percentile(arr,50)),
            "p75": float(np.percentile(arr,75)),
            "p90": float(np.percentile(arr,90))
        }

# ============================================================
# Joint EV Engine (Phase 5C-4)
# Combines:
#   - Monte Carlo distributions
#   - Selected legs (PTS/REB/AST/PRA)
#   - Correlation-aware hit probabilities
#   - Payout models (PrizePicks-style)
#   - Full EV calculation per entry
# ============================================================

import numpy as np

class JointEVEngine:
    def __init__(self):
        pass

    # --------------------------------------------------------
    # Probability all legs hit (using simulation results)
    # --------------------------------------------------------
    def joint_probability(self, sim_results, legs):
        """
        legs = [
            {"market":"PTS","line":24.5,"type":"over"},
            {"market":"REB","line":6.5,"type":"over"}
        ]
        """
        mask = np.ones(len(sim_results["PTS"]), dtype=bool)

        for leg in legs:
            arr = np.array(sim_results[leg["market"]], dtype=float)
            if leg["type"] == "over":
                hit = arr > leg["line"]
            else:
                hit = arr < leg["line"]
            mask = mask & hit

        return float(np.mean(mask))

    # --------------------------------------------------------
    # Payout rules (PrizePicks/Sleeper style)
    # --------------------------------------------------------
    def payout_multiplier(self, num_legs):
        table = {
            2: 3.0,
            3: 5.0,
            4: 10.0,
            5: 25.0,
            6: 30.0
        }
        return table.get(num_legs, 0)

    # --------------------------------------------------------
    # EV Calculation
    # --------------------------------------------------------
    def compute_ev(self, sim_results, legs, stake=1.0):
        p = self.joint_probability(sim_results, legs)
        payout_mult = self.payout_multiplier(len(legs))

        if payout_mult == 0:
            return {
                "ev_per_dollar": -1,
                "probability": p,
                "payout_mult": 0
            }

        expected = p * (payout_mult * stake)
        ev_per_dollar = expected - stake

        return {
            "ev_per_dollar": float(ev_per_dollar),
            "probability": float(p),
            "payout_mult": float(payout_mult)
        }

# ============================================================
# Model Tab (Phase 5D-1)
# Full Production Implementation
#   - Uses EngineInputBuilder
#   - Uses ProjectionEngine
#   - Uses MonteCarloEngine
#   - Displays MU / SD / Probabilities / Distribution Summary
# ============================================================

import streamlit as st
import numpy as np

from nba_ultramax_quant.data.engine_input_builder import EngineInputBuilder
from nba_ultramax_quant.data.projection_engine import ProjectionEngine
from nba_ultramax_quant.data.monte_carlo_engine import MonteCarloEngine

builder = EngineInputBuilder()
proj_engine = ProjectionEngine()
mc_engine = MonteCarloEngine(sims=15000)

def render_model_tab(sidebar_state):

    st.title("📊 UltraMAX Model Projection")

    player_id = sidebar_state.get("player_id", "")
    season = sidebar_state.get("season", "2025")
    line_pts = sidebar_state.get("line_pts", 0)
    line_reb = sidebar_state.get("line_reb", 0)
    line_ast = sidebar_state.get("line_ast", 0)
    line_pra = sidebar_state.get("line_pra", 0)

    raw_inputs = sidebar_state.get("engine_inputs", {})

    if not player_id:
        st.info("Select a player from the sidebar.")
        return

    # --------------------------------------------------------
    # Compute Engine Inputs
    # --------------------------------------------------------
    inputs = builder.build(player_id, season, raw_inputs)

    # --------------------------------------------------------
    # Projection Outputs
    # --------------------------------------------------------
    line_dict = {
        "PTS": line_pts,
        "REB": line_reb,
        "AST": line_ast,
        "PRA": line_pra
    }
    proj = proj_engine.project(inputs, line_dict)

    st.subheader("📌 Final MU / SD")
    st.write(proj["mu"])
    st.write(proj["sd"])

    st.subheader("📈 Over / Under Probability")
    st.write(proj["probabilities"])

    # --------------------------------------------------------
    # Monte Carlo Simulation
    # --------------------------------------------------------
    sim = mc_engine.run(
        proj["mc_inputs"]["mu_vec"],
        proj["mc_inputs"]["cov_matrix"]
    )

    st.subheader("🎲 Simulation Summary")
    summary = {
        "PTS": mc_engine.summary(sim["PTS"]),
        "REB": mc_engine.summary(sim["REB"]),
        "AST": mc_engine.summary(sim["AST"]),
        "PRA": mc_engine.summary(sim["PRA"])
    }
    st.write(summary)

    st.subheader("🔗 Raw Engine Inputs")
    st.json(inputs)

# ============================================================
# Player Card (Phase 5D-2)
# Full Production Implementation Using:
#   - EngineInputBuilder
#   - ProjectionEngine
#   - MonteCarloEngine
#   - Full multipliers breakdown
#   - Recent logs, trends, synergy, similarity
# ============================================================

import streamlit as st
import numpy as np

from nba_ultramax_quant.data.engine_input_builder import EngineInputBuilder
from nba_ultramax_quant.data.projection_engine import ProjectionEngine
from nba_ultramax_quant.data.monte_carlo_engine import MonteCarloEngine
from nba_ultramax_quant.data.bref_logs_extractor import BRefLogsExtractor

builder = EngineInputBuilder()
proj_engine = ProjectionEngine()
mc_engine = MonteCarloEngine(sims=12000)
bref = BRefLogsExtractor()

def render_player_card(sidebar_state):

    st.title("🃏 Player Card – Full Analytics View")

    player_id = sidebar_state.get("player_id", "")
    season = sidebar_state.get("season", "2025")
    raw_inputs = sidebar_state.get("engine_inputs", {})

    if not player_id:
        st.info("Select a player from the sidebar.")
        return

    # --------------------------------------------------------
    # Extract logs
    # --------------------------------------------------------
    logs = bref.fetch_gamelog(player_id, season)
    if logs is None or len(logs) == 0:
        st.warning("No game logs found.")
        return

    # Show last 10 logs
    st.subheader("📅 Last 10 Games")
    st.dataframe(logs.tail(10))

    # --------------------------------------------------------
    # Build inputs from full engine stack
    # --------------------------------------------------------
    inputs = builder.build(player_id, season, raw_inputs)

    st.subheader("📌 Final MU / SD")
    st.json(inputs["mu"])
    st.json(inputs["sd"])

    # --------------------------------------------------------
    # Multipliers Breakdown
    # --------------------------------------------------------
    st.subheader("⚙️ Engine Multipliers Breakdown")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Defense", inputs["defense"])
        st.write("Context", inputs["context"])
        st.write("Blowout", inputs["blowout"])
    with col2:
        st.write("Synergy", inputs["synergy"])
        st.write("Rotation", inputs["rotation"])
        st.write("Similarity", inputs["similarity"])

    # --------------------------------------------------------
    # Correlation & Covariance
    # --------------------------------------------------------
    st.subheader("🔗 Correlation Matrix")
    st.write(np.array(inputs["cor"]))

    st.subheader("📐 Covariance Matrix")
    st.write(np.array(inputs["cov"]))

    # --------------------------------------------------------
    # Simulation-Based Summary
    # --------------------------------------------------------
    st.subheader("🎲 Monte Carlo Simulation Summary")

    mc = mc_engine.run(
        [inputs["mu"]["PTS"], inputs["mu"]["REB"], inputs["mu"]["AST"]],
        inputs["cov"]
    )

    summary = {
        "PTS": mc_engine.summary(mc["PTS"]),
        "REB": mc_engine.summary(mc["REB"]),
        "AST": mc_engine.summary(mc["AST"]),
        "PRA": mc_engine.summary(mc["PRA"]),
    }
    st.write(summary)

    # --------------------------------------------------------
    # Similarity Comps
    # --------------------------------------------------------
    st.subheader("🧬 Top Similar Player Comps")
    st.write(inputs["similarity"]["top_comps"])

    # --------------------------------------------------------
    # Raw Inputs for Debugging
    # --------------------------------------------------------
    with st.expander("🛠 Raw Engine Inputs"):
        st.json(inputs)

# ============================================================
# Trend Recognition Tab (Phase 5D-3)
# Full Production Implementation:
#   - Pulls rolling windows (5/10/20)
#   - EMA, Z-score, Trend multipliers
#   - Volatility indicators
#   - Visual charts for trends
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from nba_ultramax_quant.data.bref_logs_extractor import BRefLogsExtractor
from nba_ultramax_quant.data.trend_engine import TrendEngine

bref = BRefLogsExtractor()
trend_engine = TrendEngine()

def render_trend_tab(sidebar_state):

    st.title("📈 Trend Recognition Engine")

    player_id = sidebar_state.get("player_id","")
    season = sidebar_state.get("season","2025")

    if not player_id:
        st.info("Select a player from the sidebar.")
        return

    logs = bref.fetch_gamelog(player_id, season)
    if logs is None or len(logs) == 0:
        st.warning("No logs found.")
        return

    pts = logs["PTS"].astype(float).tolist()
    reb = logs["TRB"].astype(float).tolist()
    ast = logs["AST"].astype(float).tolist()

    st.subheader("📅 Raw Log Trends")
    st.dataframe(logs.tail(10))

    # --------------------------------------------------------
    # Trend Outputs
    # --------------------------------------------------------
    st.subheader("📊 Trend Outputs")

    t_pts = trend_engine.compute_trend_outputs(pts)
    t_reb = trend_engine.compute_trend_outputs(reb)
    t_ast = trend_engine.compute_trend_outputs(ast)

    st.write({
        "PTS": t_pts,
        "REB": t_reb,
        "AST": t_ast
    })

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    st.subheader("📉 Trend Visualizations")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(pts, label="PTS", marker="o")
    ax.set_title("PTS Trend Over Time")
    ax.set_xlabel("Game #")
    ax.set_ylabel("Points")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(reb, label="REB", marker="o", color="orange")
    ax2.set_title("REB Trend Over Time")
    ax2.set_xlabel("Game #")
    ax2.set_ylabel("Rebounds")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(ast, label="AST", marker="o", color="green")
    ax3.set_title("AST Trend Over Time")
    ax3.set_xlabel("Game #")
    ax3.set_ylabel("Assists")
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("🧩 Window Overlays (5/10/20)")

    st.write("PTS Windows:", t_pts["windows"])
    st.write("REB Windows:", t_reb["windows"])
    st.write("AST Windows:", t_ast["windows"])

# ============================================================
# Rotation Volatility Tab (Phase 5D-4)
# Full Production Implementation:
#   - Uses RotationVolatilityEngine
#   - Visualizes minutes distribution
#   - Shows foul volatility, coach trust, bench depth, blowout risk
#   - Plots last 10 games of minutes
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from nba_ultramax_quant.data.bref_logs_extractor import BRefLogsExtractor
from nba_ultramax_quant.data.rotation_volatility_engine import RotationVolatilityEngine

bref = BRefLogsExtractor()
rot_engine = RotationVolatilityEngine()

def render_rotation_tab(sidebar_state):

    st.title("⏱ Rotation Volatility Engine")

    player_id = sidebar_state.get("player_id","")
    season = sidebar_state.get("season","2025")
    raw_inputs = sidebar_state.get("engine_inputs",{})

    if not player_id:
        st.info("Select a player in the sidebar.")
        return

    logs = bref.fetch_gamelog(player_id, season)
    if logs is None or len(logs)==0:
        st.warning("No logs available.")
        return

    # Extract minute logs
    minutes = logs["MP"].astype(float).tolist()

    st.subheader("📅 Last 10 Games (Minutes Played)")
    st.dataframe(logs.tail(10)[["G","MP","PTS","TRB","AST"]])

    # --------------------------------------------------------
    # Compute rotation outputs
    # --------------------------------------------------------
    rot_inputs = raw_inputs.get("rotation", {})
    rot_out = rot_engine.compute_rotation_outputs({
        "projected_minutes": rot_inputs.get("projected_minutes", 30),
        "recent_minutes": minutes[-10:],
        "foul_rate": rot_inputs.get("foul_rate", 2.5),
        "coach_trust": rot_inputs.get("coach_trust", 60),
        "bench_depth": rot_inputs.get("bench_depth", 9),
        "games_back": rot_inputs.get("games_back", 10),
        "blowout_risk": rot_inputs.get("blowout_risk", 0.10)
    })

    st.subheader("⚙️ Rotation Metrics")
    st.write(rot_out)

    # --------------------------------------------------------
    # Visualization: Minutes Over Time
    # --------------------------------------------------------
    st.subheader("📉 Minutes Trend")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(minutes, marker="o", label="Minutes")
    ax.axhline(rot_out["minutes_mean"], color="orange", linestyle="--", label="Mean Projection")
    ax.set_title("Minutes Played Over Time")
    ax.set_xlabel("Game #")
    ax.set_ylabel("Minutes")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # --------------------------------------------------------
    # Volatility Factor
    # --------------------------------------------------------
    st.subheader("🌪 Volatility Factor Breakdown")

    st.write({
        "Minutes SD": rot_out["minutes_sd"],
        "Volatility Factor": rot_out["volatility_factor"]
    })

    # --------------------------------------------------------
    # Raw Rotation Inputs
    # --------------------------------------------------------
    with st.expander("🔧 Raw Rotation Inputs"):
        st.json(rot_inputs)

# ============================================================
# Blowout Model Tab (Phase 5D-5)
# Full Production UI:
#   - Spread → win probability
#   - Blowout probability curve
#   - Pace decay
#   - Minutes suppression
#   - Volatility factor
#   - Visual charts
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from nba_ultramax_quant.data.blowout_engine import BlowoutEngine

blow = BlowoutEngine()

def render_blowout_tab(sidebar_state):

    st.title("💥 Blowout Model Analysis")

    raw_inputs = sidebar_state.get("engine_inputs", {})
    blow_inputs = raw_inputs.get("blowout", {})

    spread = blow_inputs.get("spread", 0)
    role = blow_inputs.get("role", "starter")

    # Compute full engine outputs
    out = blow.compute_blowout_outputs(blow_inputs)

    st.subheader("📌 Spread & Win Probability")
    st.write({
        "Spread": spread,
        "Win Probability": out["win_prob"]
    })

    st.subheader("🔥 Blowout Probability (15+ pts)")
    st.write(out["blowout_prob"])

    st.subheader("⚙️ Blowout Multipliers")
    st.write({
        "Pace Decay × Minutes Suppression": out["blowout_multiplier"],
        "Volatility": out["blowout_volatility"]
    })

    # --------------------------------------------------------
    # Visualization: Win Probability Curve
    # --------------------------------------------------------
    st.subheader("📉 Win Probability vs Spread")

    spreads = np.linspace(-20, 20, 200)
    probs = [blow.spread_to_win_prob(s) for s in spreads]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(spreads, probs, label="Win Probability")
    ax.axvline(spread, color="red", linestyle="--", label=f"Current Spread ({spread})")
    ax.set_title("Win Probability Curve")
    ax.set_xlabel("Spread")
    ax.set_ylabel("Win Probability")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # --------------------------------------------------------
    # Visualization: Blowout Risk Curve
    # --------------------------------------------------------
    st.subheader("💣 Blowout Probability Curve")

    blow_curve = [blow.compute_blowout_prob(s) for s in spreads]

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(spreads, blow_curve, color="orange", label="Blowout Probability")
    ax2.axvline(spread, color="red", linestyle="--", label=f"Current Spread ({spread})")
    ax2.set_title("Blowout Probability vs Spread")
    ax2.set_xlabel("Spread")
    ax2.set_ylabel("Probability of 15+ Point Blowout")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # --------------------------------------------------------
    # Raw Inputs
    # --------------------------------------------------------
    with st.expander("🛠 Raw Blowout Inputs"):
        st.json(blow_inputs)

# ============================================================
# Line Shopping Analyzer (Phase 5D-6)
# Full Production Implementation:
#   - Compares projections vs multiple books
#   - Computes EV per line using Projection + Monte Carlo
#   - Highlights best book / best EV per market
# ============================================================

import streamlit as st
import numpy as np

from nba_ultramax_quant.data.engine_input_builder import EngineInputBuilder
from nba_ultramax_quant.data.projection_engine import ProjectionEngine
from nba_ultramax_quant.data.monte_carlo_engine import MonteCarloEngine

builder = EngineInputBuilder()
proj_engine = ProjectionEngine()
mc_engine = MonteCarloEngine(sims=12000)

BOOKS = ["PrizePicks", "Sleeper", "Underdog", "DraftKings", "FanDuel", "ESPN", "Caesars"]

def render_line_shopping_tab(sidebar_state):

    st.title("🛒 Line Shopping Analyzer")

    player_id = sidebar_state.get("player_id","")
    season = sidebar_state.get("season","2025")
    raw_inputs = sidebar_state.get("engine_inputs",{})

    if not player_id:
        st.info("Select a player first.")
        return

    # User provides book lines
    st.subheader("📥 Enter Book Lines for Each Market")

    user_lines = {}
    cols = st.columns(4)
    markets = ["PTS","REB","AST","PRA"]

    for i, stat in enumerate(markets):
        with cols[i]:
            d = {}
            for bk in BOOKS:
                d[bk] = st.number_input(f"{stat} - {bk}", value=0.0, key=f"{stat}_{bk}")
            user_lines[stat] = d

    # --------------------------------------------------------
    # Build unified engine inputs
    # --------------------------------------------------------
    inputs = builder.build(player_id, season, raw_inputs)
    mu = inputs["mu"]
    sd = inputs["sd"]

    # --------------------------------------------------------
    # Monte Carlo Simulation
    # --------------------------------------------------------
    sim = mc_engine.run(
        [mu["PTS"], mu["REB"], mu["AST"]],
        inputs["cov"]
    )

    st.subheader("📊 Best Lines & EV by Market")

    # --------------------------------------------------------
    # EV Calculation Per Book
    # --------------------------------------------------------
    for stat in markets:
        st.write(f"### {stat}")
        book_results = []

        for bk in BOOKS:
            line = user_lines[stat][bk]
            if stat in ["PTS","REB","AST"]:
                dist = sim[stat]
            else:
                dist = sim["PRA"]

            p_over = mc_engine.probability_over(dist, line)
            ev = p_over * 1 - (1 - p_over)  # simplified EV: +1 on hit, -1 on miss

            book_results.append({"book": bk, "line": line, "p_over": p_over, "EV": ev})

        # Sort by best EV
        book_results = sorted(book_results, key=lambda x: x["EV"], reverse=True)

        st.write(book_results)

    st.subheader("🛠 Raw Engine Inputs")
    st.json(inputs)

# ============================================================
# Joint EV Tab (Phase 5D-7)
# Full Production Integration
#   - Uses MonteCarloEngine
#   - Uses JointEVEngine
#   - Multi-leg builder
#   - Correlated EV output
# ============================================================

import streamlit as st
import numpy as np

from nba_ultramax_quant.data.engine_input_builder import EngineInputBuilder
from nba_ultramax_quant.data.projection_engine import ProjectionEngine
from nba_ultramax_quant.data.monte_carlo_engine import MonteCarloEngine
from nba_ultramax_quant.data.joint_ev_engine import JointEVEngine

builder = EngineInputBuilder()
proj_engine = ProjectionEngine()
mc_engine = MonteCarloEngine(sims=15000)
joint_ev = JointEVEngine()

def render_joint_ev_tab(sidebar_state):

    st.title("🔗 Joint EV Calculator (SGP Mode)")

    player_id = sidebar_state.get("player_id", "")
    season = sidebar_state.get("season", "2025")
    raw_inputs = sidebar_state.get("engine_inputs", {})

    if not player_id:
        st.info("Select a player from the sidebar.")
        return

    st.subheader("📥 Build Your Legs")

    leg_count = st.number_input("Number of Legs", min_value=2, max_value=6, value=2)

    legs = []
    markets = ["PTS","REB","AST","PRA"]

    for i in range(leg_count):
        st.markdown(f"### Leg {i+1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            market = st.selectbox(f"Market {i+1}", markets, key=f"market_{i}")
        with col2:
            line = st.number_input(f"Line {i+1}", value=0.0, key=f"line_{i}")
        with col3:
            ovun = st.selectbox(f"Type {i+1}", ["over","under"], key=f"type_{i}")

        legs.append({
            "market": market,
            "line": line,
            "type": ovun
        })

    # --------------------------------------------------------
    # Build model inputs
    # --------------------------------------------------------
    inputs = builder.build(player_id, season, raw_inputs)
    mu_vec = [inputs["mu"]["PTS"], inputs["mu"]["REB"], inputs["mu"]["AST"]]
    cov_matrix = inputs["cov"]

    # --------------------------------------------------------
    # Monte Carlo simulation
    # --------------------------------------------------------
    st.subheader("🎲 Running Correlated Simulation...")
    sim = mc_engine.run(mu_vec, cov_matrix)

    # --------------------------------------------------------
    # Joint EV Calculation
    # --------------------------------------------------------
    st.subheader("📊 Joint EV Results")
    ev = joint_ev.compute_ev(sim, legs, stake=1.0)
    st.write(ev)

    st.subheader("🧩 Probability All Legs Hit")
    st.write(ev["probability"])

    st.subheader("💰 Payout Multiplier")
    st.write(ev["payout_mult"])

    # --------------------------------------------------------
    # Visualization of each leg's distribution
    # --------------------------------------------------------
    st.subheader("📈 Leg Distribution Visualizations")

    for leg in legs:
        stat = leg["market"]
        dist = sim[stat]

        st.markdown(f"### {stat} Distribution")

        hist = np.histogram(dist, bins=30)
        st.bar_chart(hist[0])

    # --------------------------------------------------------
    # Raw Engine Inputs
    # --------------------------------------------------------
    with st.expander("🛠 Raw Engine Inputs"):
        st.json(inputs)
# ============================================================
# Phase 5E-1 — Main Page Router + Sidebar State Controller
# Handles:
#   - Sidebar inputs
#   - Global state dictionary
#   - Routing to all UI tabs
# ============================================================

import streamlit as st

from nba_ultramax_quant.app.ui.model_tab import render_model_tab
from nba_ultramax_quant.app.ui.player_card import render_player_card
from nba_ultramax_quant.app.ui.trend_tab import render_trend_tab
from nba_ultramax_quant.app.ui.rotation_tab import render_rotation_tab
from nba_ultramax_quant.app.ui.blowout_tab import render_blowout_tab
from nba_ultramax_quant.app.ui.line_shopping_tab import render_line_shopping_tab
from nba_ultramax_quant.app.ui.joint_ev_tab import render_joint_ev_tab

# ------------------------------------------------------------
# Sidebar Controller
# ------------------------------------------------------------
def sidebar_controller():

    st.sidebar.title("🔧 UltraMAX Sidebar")

    # Player ID input (string)
    player_id = st.sidebar.text_input("Player ID (BRef ID)", value="")

    # Season input
    season = st.sidebar.selectbox("Season", ["2025","2024","2023","2022"])

    # Lines for Model Tab
    st.sidebar.subheader("📈 Player Lines")
    line_pts = st.sidebar.number_input("PTS Line", value=0.0)
    line_reb = st.sidebar.number_input("REB Line", value=0.0)
    line_ast = st.sidebar.number_input("AST Line", value=0.0)
    line_pra = st.sidebar.number_input("PRA Line", value=0.0)

    # Engine Inputs
    st.sidebar.subheader("⚙️ Engine Inputs")
    engine_inputs = {}

    engine_inputs["defense"] = {
        "opp_def_rating": st.sidebar.number_input("Opp Defense Rating", value=110.0),
        "pace_factor": st.sidebar.number_input("Pace Factor", value=1.0),
    }

    engine_inputs["context"] = {
        "team_pace": st.sidebar.number_input("Team Pace", value=100.0),
        "opp_pace": st.sidebar.number_input("Opp Pace", value=100.0),
    }

    engine_inputs["rotation"] = {
        "projected_minutes": st.sidebar.number_input("Projected Minutes", value=30.0),
        "foul_rate": st.sidebar.number_input("Foul Rate", value=2.5),
        "coach_trust": st.sidebar.number_input("Coach Trust (0–100)", value=60.0),
        "bench_depth": st.sidebar.number_input("Bench Depth", value=9),
        "games_back": st.sidebar.number_input("Games Since Return", value=10),
        "blowout_risk": st.sidebar.number_input("Blowout Risk (0–1)", value=0.10)
    }

    engine_inputs["blowout"] = {
        "spread": st.sidebar.number_input("Spread", value=0.0),
        "role": st.sidebar.selectbox("Role", ["starter", "bench"])
    }

    engine_inputs["synergy"] = {
        "usage_rate": st.sidebar.number_input("Usage Rate (%)", value=22.0),
    }

    engine_inputs["similarity_base"] = {}
    engine_inputs["similarity_pool"] = []

    # Choose Page
    st.sidebar.subheader("📄 Pages")
    page = st.sidebar.selectbox(
        "Select Page", 
        [
            "Model Tab",
            "Player Card",
            "Trend Recognition",
            "Rotation Volatility",
            "Blowout Model",
            "Line Shopping",
            "Joint EV"
        ]
    )

    # Pack state
    return {
        "player_id": player_id,
        "season": season,
        "line_pts": line_pts,
        "line_reb": line_reb,
        "line_ast": line_ast,
        "line_pra": line_pra,
        "engine_inputs": engine_inputs,
        "page": page
    }

# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
def app_router():

    state = sidebar_controller()

    page = state["page"]

    if page == "Model Tab":
        render_model_tab(state)
    elif page == "Player Card":
        render_player_card(state)
    elif page == "Trend Recognition":
        render_trend_tab(state)
    elif page == "Rotation Volatility":
        render_rotation_tab(state)
    elif page == "Blowout Model":
        render_blowout_tab(state)
    elif page == "Line Shopping":
        render_line_shopping_tab(state)
    elif page == "Joint EV":
        render_joint_ev_tab(state)
    else:
        st.write("Invalid Page")

# ============================================================
# Phase 5E-2 — Streamlit App Launcher
# Master Entry File for UltraMAX NBA Quant Model
# ============================================================

import streamlit as st
from nba_ultramax_quant.app.main_router import app_router

# ------------------------------------------------------------
# Global Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="UltraMAX NBA Quant",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Global CSS
# ------------------------------------------------------------
st.markdown("""
<style>
body {
    zoom: 0.90;
}
.sidebar .sidebar-content {
    background-color: #111111;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Global Error Handling Wrapper
# ------------------------------------------------------------
def safe_run():
    try:
        app_router()
    except Exception as e:
        st.error("🚨 A critical error occurred in the app.")
        st.exception(e)
        st.info("Check engine inputs or logs for more details.")

# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------
safe_run()


/* ============================================
   UltraMAX Global UI Theme (Phase 5E-3)
   Clean Dark Quant Aesthetic
   ============================================ */

body, .stApp {
    background-color: #0d0d0d !important;
    color: #e6e6e6 !important;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

.stButton>button {
    background-color: #1f6feb !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
}

.stButton>button:hover {
    background-color: #388bfd !important;
}

.sidebar .sidebar-content {
    background-color: #111111 !important;
}

[data-testid="stSidebar"] {
    background-color: #111111 !important;
}

.block-container {
    padding-top: 2rem;
}

table td, table th {
    color: #e6e6e6 !important;
    background-color: #1a1a1a !important;
    border-color: #333333 !important;
}

/* multipliers */
.good-multiplier {
    color: #4ade80 !important; /* green */
}
.bad-multiplier {
    color: #f87171 !important; /* red */
}


# ============================================================
# Failsafe Utilities (Phase 5E-4)
# Prevents crashes from:
#   - NoneType logs
#   - Missing engine inputs
#   - NaNs or infs
#   - Empty arrays
#   - Zero variance covariance matrices
# ============================================================

import numpy as np

def safe_list(arr):
    if arr is None:
        return []
    try:
        arr = [float(x) for x in arr if x is not None]
        return arr
    except:
        return []

def safe_number(x, default=0.0):
    try:
        if x is None or np.isnan(x):
            return default
        return float(x)
    except:
        return default

def safe_cov(cov):
    try:
        cov = np.array(cov, dtype=float)
        if cov.shape != (3,3):
            return np.eye(3) * 1.0
        if np.linalg.det(cov) == 0:
            return cov + np.eye(3)*1e-3
        return cov
    except:
        return np.eye(3) * 1.0

def safe_mu(mu):
    out = {}
    for k in ["PTS","REB","AST","PRA"]:
        out[k] = safe_number(mu.get(k, 0))
    return out

def safe_sd(sd):
    out = {}
    for k in ["PTS","REB","AST","PRA"]:
        v = safe_number(sd.get(k, 1))
        out[k] = max(v, 0.5)
    return out

# AUTO-INJECTED: Phase 5E-4 CSS Loader + Failsafe Wrapper
import streamlit as st
import os

def load_global_css():
    css_path = os.path.join(os.path.dirname(__file__), "global_styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_global_css()

# Harden entire app
from nba_ultramax_quant.app.failsafe_utils import safe_mu, safe_sd, safe_cov
# ============================================================
# Phase 5E-2 — Streamlit App Launcher
# Master Entry File for UltraMAX NBA Quant Model
# ============================================================

import streamlit as st
from nba_ultramax_quant.app.main_router import app_router

# ------------------------------------------------------------
# Global Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="UltraMAX NBA Quant",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Global CSS
# ------------------------------------------------------------
st.markdown("""
<style>
body {
    zoom: 0.90;
}
.sidebar .sidebar-content {
    background-color: #111111;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Global Error Handling Wrapper
# ------------------------------------------------------------
def safe_run():
