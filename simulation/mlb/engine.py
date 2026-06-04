"""
simulation/mlb/engine.py — At-bat-level MLB Monte Carlo game simulator.

Mirrors the NBA GameEngine interface (PlayerDistribution / SimulationOutput /
get_player_dist) so the app's MODEL tab can consume MLB and NBA identically.

Each simulated game:
  • Plays 9+ innings (extra innings capped), alternating half-innings.
  • Every plate appearance: the current pitcher (starter or bullpen, per the
    removal model) is matched against the batter via the odds-ratio (log5)
    method against league baselines, with park + weather applied to HR.
  • A full base-state is tracked so Runs and RBI are simulated, not assumed.
  • Earned runs are charged to the pitcher who allowed the runner to reach.

The engine accumulates raw counters per player, then exposes per-stat
distributions (including composites TB / Hits / HRR / Fantasy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from .config import MLBSimConfig, PITCHES_PER_PA, park_factor, weather_hr_factor
from .profiles import (
    BatterProfile, PitcherProfile, PA_OUTCOMES, LEAGUE_VECTOR,
    odds_ratio, league_average_reliever,
)

BATTER_STAT_KEYS = [
    "strikeouts", "walks", "singles", "doubles", "triples", "home_runs",
    "hits", "total_bases", "runs", "rbi", "stolen_bases", "hrr", "fantasy",
]
PITCHER_STAT_KEYS = [
    "pitcher_k", "pitcher_outs", "earned_runs", "hits_allowed",
    "walks_allowed", "pitcher_qs", "pitcher_fantasy",
]


@dataclass
class PlayerDistribution:
    """Statistical summary of a player's simulated stat distribution.

    Identical interface to simulation/game_engine.PlayerDistribution.
    """
    player_name: str
    player_id: str
    stat_name: str
    n_sims: int
    values: np.ndarray
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0

    def compute(self) -> None:
        v = self.values.astype(np.float64)
        self.mean = float(np.mean(v))
        self.median = float(np.median(v))
        self.std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        self.p5 = float(np.percentile(v, 5))
        self.p25 = float(np.percentile(v, 25))
        self.p75 = float(np.percentile(v, 75))
        self.p95 = float(np.percentile(v, 95))

    def prob_over(self, line: float) -> float:
        return float(np.mean(self.values > line))

    def prob_under(self, line: float) -> float:
        return float(np.mean(self.values < line))

    def prob_push(self, line: float) -> float:
        return float(np.mean(self.values == line))


@dataclass
class SimulationOutput:
    n_simulations: int
    home_team: str
    away_team: str
    distributions: Dict[str, Dict[str, PlayerDistribution]]

    def get_player_dist(self, player_id: str, stat: str) -> Optional[PlayerDistribution]:
        return self.distributions.get(player_id, {}).get(stat)

    def prob_over(self, player_id: str, stat: str, line: float) -> Optional[float]:
        d = self.get_player_dist(player_id, stat)
        return d.prob_over(line) if d else None


class MLBGameEngine:
    def __init__(
        self,
        config: MLBSimConfig | None = None,
        home_lineup: List[BatterProfile] | None = None,
        away_lineup: List[BatterProfile] | None = None,
        home_sp: PitcherProfile | None = None,
        away_sp: PitcherProfile | None = None,
        home_name: str = "HOME",
        away_name: str = "AWAY",
        park: str = "",
        temp_f: float | None = None,
        wind_mph: float | None = None,
        wind_out: bool | None = None,
    ) -> None:
        self.cfg = config or MLBSimConfig()
        self.home_lineup = home_lineup or []
        self.away_lineup = away_lineup or []
        self.home_sp = home_sp
        self.away_sp = away_sp
        self.home_name = home_name
        self.away_name = away_name
        runs_pf, hr_pf = park_factor(park)
        self.runs_pf = runs_pf
        self.hr_pf = hr_pf * weather_hr_factor(temp_f, wind_mph, wind_out)

        # Pre-compute league vector list aligned to PA_OUTCOMES order.
        self._league = np.array([LEAGUE_VECTOR[o] for o in PA_OUTCOMES])

    # ── matchup outcome distribution ───────────────────────────────
    def _pa_distribution(self, batter: BatterProfile, pitcher: PitcherProfile,
                         tto_pass: int) -> np.ndarray:
        """Categorical PA-outcome probabilities via odds-ratio, park+weather."""
        bvec = batter.outcome_vector()
        pvec = pitcher.outcome_vector()

        # Third-time-through-order drift for starters: K down, hits/HR up.
        if pitcher.is_starter and tto_pass >= 2:
            drift = self.cfg.tto_penalty_per_pass * (tto_pass - 1)
            pvec = dict(pvec)
            pvec["K"] = max(0.01, pvec["K"] * (1.0 - drift))
            for k in ("1B", "2B", "HR"):
                pvec[k] = pvec[k] * (1.0 + 0.5 * drift)

        probs = []
        for o in PA_OUTCOMES:
            l = LEAGUE_VECTOR[o]
            p = odds_ratio(bvec[o], pvec[o], l)
            if o == "HR":
                p *= self.hr_pf
            elif o in ("1B", "2B", "3B"):
                p *= self.runs_pf ** 0.5  # park nudges hits modestly
            probs.append(p)
        arr = np.array(probs, dtype=np.float64)
        s = arr.sum()
        return arr / s if s > 0 else self._league.copy()

    # ── single game simulation ─────────────────────────────────────
    def _play_full_game(self, rng, _bstat, _pstat, stats):
        away_score = 0
        home_score = 0
        away_idx = 0
        home_idx = 0
        away_state = _TeamPitchState(self.home_sp, self.cfg)  # away bats vs home pitcher
        home_state = _TeamPitchState(self.away_sp, self.cfg)  # home bats vs away pitcher

        for inning in range(1, self.cfg.max_innings + 1):
            # Top half: away bats
            runs, away_idx = self._half_inning(
                rng, self.away_lineup, away_idx, away_state, _bstat, _pstat)
            away_score += runs
            # Bottom half: home bats (skip if home already wins after 9 — walk-off
            # handled by breaking before playing if home leads entering 9th+)
            if inning >= 9 and home_score > away_score:
                break  # home doesn't bat in bottom of 9th+ when already ahead
            runs, home_idx = self._half_inning(
                rng, self.home_lineup, home_idx, home_state, _bstat, _pstat)
            home_score += runs
            if inning >= 9 and home_score > away_score:
                break  # walk-off
            if inning >= 9 and away_score != home_score:
                break  # game decided after a complete inning

        # Credit wins to starters (approximate): team that scored more, SP went 5+.
        self._finalize_pitcher_derived(stats, self.home_sp, home_score > away_score)
        self._finalize_pitcher_derived(stats, self.away_sp, away_score > home_score)
        return stats

    def _half_inning(self, rng, lineup, start_idx, pstate, _bstat, _pstat):
        """Simulate one half-inning. Returns (runs_scored, next_batter_idx)."""
        if not lineup:
            return 0, start_idx
        outs = 0
        # bases hold tuples (batter_pid, responsible_pitcher_pid) or None
        bases: List[Optional[tuple]] = [None, None, None]
        runs = 0
        idx = start_idx
        n = len(lineup)
        # steal bookkeeping: attempt at top of a PA if runner on 1B only
        while outs < 3:
            batter = lineup[idx % n]
            pitcher = pstate.current_pitcher()
            pstat = _pstat(pitcher.player_id, pitcher.name)
            bstat = _bstat(batter.player_id, batter.name)

            # ── stolen base attempt (runner on 1B, 2B empty) ──
            if bases[0] is not None and bases[1] is None:
                runner_pid, resp_p = bases[0]
                rb = self._lineup_batter(lineup, runner_pid)
                if rb is not None and rng.random() < rb.sb_attempt:
                    if rng.random() < rb.sb_success:
                        bases[1] = bases[0]; bases[0] = None
                        _bstat(runner_pid, rb.name)["SB"] += 1
                    else:
                        bases[0] = None
                        outs += 1
                        pstat["OUTS"] += 1
                        if outs >= 3:
                            break

            tto_pass = pstate.tto_pass()
            probs = self._pa_distribution(batter, pitcher, tto_pass)
            outcome = PA_OUTCOMES[rng.choice(len(PA_OUTCOMES), p=probs)]
            pstate.record_pa()
            pstat["BF"] += 1

            scored = 0
            if outcome == "K":
                outs += 1; pstat["OUTS"] += 1; pstat["K"] += 1; bstat["K"] += 1
            elif outcome in ("BB", "HBP"):
                bstat["BB" if outcome == "BB" else "HBP"] += 1
                pstat["BB"] += 1 if outcome == "BB" else 0
                scored += self._force_advance(bases, batter, pitcher, _bstat)
            elif outcome == "OUT":
                outs += 1; pstat["OUTS"] += 1
                scored += self._out_in_play(bases, batter, pitcher, outs, rng, _bstat, pstat)
            else:  # hit: 1B/2B/3B/HR
                pstat["H"] += 1; bstat[outcome] += 1
                scored += self._hit_advance(outcome, bases, batter, pitcher, rng, _bstat)
            runs += scored
            # RBI credited for every run driven in (sac flies included). Errors,
            # wild pitches and GIDP-runs are not modeled, so RBI == runs scored
            # on the play is exact for this model.
            bstat["RBI"] += scored
            idx += 1
            if (idx - start_idx) > 25:  # safety against pathological loops
                break
        return runs, idx

    # ── advancement helpers ────────────────────────────────────────
    def _force_advance(self, bases, batter, pitcher, _bstat) -> int:
        """Walk/HBP: force runners only. Returns runs scored."""
        runs = 0
        new_runner = (batter.player_id, pitcher.player_id)
        if bases[0] is None:
            bases[0] = new_runner
        elif bases[1] is None:
            bases[1] = bases[0]; bases[0] = new_runner
        elif bases[2] is None:
            bases[2] = bases[1]; bases[1] = bases[0]; bases[0] = new_runner
        else:
            # bases loaded → runner on 3rd forced home
            runs += self._score_runner(bases[2], _bstat, pitcher)
            bases[2] = bases[1]; bases[1] = bases[0]; bases[0] = new_runner
        return runs

    def _hit_advance(self, outcome, bases, batter, pitcher, rng, _bstat) -> int:
        runs = 0
        if outcome == "HR":
            for b in bases:
                if b is not None:
                    runs += self._score_runner(b, _bstat, pitcher)
            bases[0] = bases[1] = bases[2] = None
            runs += self._score_runner((batter.player_id, pitcher.player_id), _bstat, pitcher)
            return runs
        if outcome == "3B":
            for b in bases:
                if b is not None:
                    runs += self._score_runner(b, _bstat, pitcher)
            bases[0] = bases[1] = None
            bases[2] = (batter.player_id, pitcher.player_id)
            return runs
        if outcome == "2B":
            if bases[2] is not None:
                runs += self._score_runner(bases[2], _bstat, pitcher)
            if bases[1] is not None:
                runs += self._score_runner(bases[1], _bstat, pitcher)
            new3 = None
            if bases[0] is not None:
                if rng.random() < 0.5:
                    runs += self._score_runner(bases[0], _bstat, pitcher)
                else:
                    new3 = bases[0]
            bases[2] = new3
            bases[1] = (batter.player_id, pitcher.player_id)
            bases[0] = None
            return runs
        # 1B (single)
        if bases[2] is not None:
            runs += self._score_runner(bases[2], _bstat, pitcher)
            bases[2] = None
        if bases[1] is not None:
            if rng.random() < 0.6:
                runs += self._score_runner(bases[1], _bstat, pitcher)
                bases[1] = None
            else:
                bases[2] = bases[1]; bases[1] = None
        if bases[0] is not None:
            if rng.random() < 0.3 and bases[2] is None:
                bases[2] = bases[0]
            else:
                bases[1] = bases[0]
            bases[0] = None
        bases[0] = (batter.player_id, pitcher.player_id)
        return runs

    def _out_in_play(self, bases, batter, pitcher, outs, rng, _bstat, pstat) -> int:
        """Non-K out. Sac fly / productive out scoring; modest GIDP."""
        runs = 0
        # Sac fly: runner on 3rd, fewer than 2 outs after this out
        if bases[2] is not None and outs < 3 and rng.random() < 0.28:
            runs += self._score_runner(bases[2], _bstat, pitcher)
            bases[2] = None
            return runs  # the out already counted by caller
        return runs

    def _score_runner(self, runner, _bstat, scoring_pitcher) -> int:
        """Score a runner: credit R to runner, ER to the responsible pitcher."""
        runner_pid, resp_pid = runner
        _bstat(runner_pid, runner_pid)["R"] += 1
        # earned run charged to responsible pitcher tracked via stats dict
        self._charge_er(resp_pid)
        return 1

    def _charge_er(self, pitcher_pid):
        # ER accumulation is handled through the shared stats dict via closure;
        # set during simulate; here we stash on a side ledger.
        self._er_ledger[pitcher_pid] = self._er_ledger.get(pitcher_pid, 0) + 1

    def _lineup_batter(self, lineup, pid) -> Optional[BatterProfile]:
        for b in lineup:
            if b.player_id == pid:
                return b
        return None

    def _finalize_pitcher_derived(self, stats, sp, won):
        if sp is None or sp.player_id not in stats:
            return
        s = stats[sp.player_id]
        s["ER"] = self._er_ledger.get(sp.player_id, 0)
        s["W"] = 1.0 if (won and s.get("OUTS", 0) >= 15) else 0.0
        s["QS"] = 1.0 if (s.get("OUTS", 0) >= 18 and s["ER"] <= 3) else 0.0

    # ── multi-sim driver ────────────────────────────────────────────
    def run_simulation(self, n: int | None = None) -> SimulationOutput:
        n = n or self.cfg.n_sims
        rng = np.random.default_rng(self.cfg.random_seed)

        # Accumulate per-player per-stat arrays.
        batter_acc: Dict[str, Dict[str, np.ndarray]] = {}
        pitcher_acc: Dict[str, Dict[str, np.ndarray]] = {}
        names: Dict[str, str] = {}

        all_batters = {b.player_id: b for b in (self.home_lineup + self.away_lineup)}
        all_pitchers = {p.player_id: p for p in (self.home_sp, self.away_sp) if p}
        for pid, b in all_batters.items():
            names[pid] = b.name
            batter_acc[pid] = {k: np.zeros(n) for k in BATTER_STAT_KEYS}
        for pid, p in all_pitchers.items():
            names[pid] = p.name
            pitcher_acc[pid] = {k: np.zeros(n) for k in PITCHER_STAT_KEYS}

        for i in range(n):
            self._er_ledger = {}
            gstats = self._play_full_game(rng, *self._fresh_collectors())
            self._collect_game(i, gstats, batter_acc, pitcher_acc)

        distributions: Dict[str, Dict[str, PlayerDistribution]] = {}
        for pid, statmap in {**batter_acc, **pitcher_acc}.items():
            distributions[pid] = {}
            for stat, arr in statmap.items():
                d = PlayerDistribution(names.get(pid, pid), pid, stat, n, arr)
                d.compute()
                distributions[pid][stat] = d

        return SimulationOutput(n, self.home_name, self.away_name, distributions)

    def _fresh_collectors(self):
        stats: Dict[str, Dict[str, float]] = {}

        def _bstat(pid: str, name: str) -> Dict[str, float]:
            if pid not in stats:
                stats[pid] = {"_name": name, "_type": "batter",
                              **{k: 0.0 for k in ("1B", "2B", "3B", "HR", "K",
                                                  "BB", "HBP", "R", "RBI", "SB")}}
            return stats[pid]

        def _pstat(pid: str, name: str) -> Dict[str, float]:
            if pid not in stats:
                stats[pid] = {"_name": name, "_type": "pitcher",
                              **{k: 0.0 for k in ("K", "BB", "H", "ER", "OUTS", "BF")}}
            return stats[pid]

        self._stats_ref = stats
        return _bstat, _pstat, stats

    def _collect_game(self, i, gstats, batter_acc, pitcher_acc):
        for pid, s in gstats.items():
            if s.get("_type") == "batter" and pid in batter_acc:
                singles, doubles, triples, hr = s["1B"], s["2B"], s["3B"], s["HR"]
                hits = singles + doubles + triples + hr
                tb = singles + 2 * doubles + 3 * triples + 4 * hr
                a = batter_acc[pid]
                a["strikeouts"][i] = s["K"]; a["walks"][i] = s["BB"]
                a["singles"][i] = singles; a["doubles"][i] = doubles
                a["triples"][i] = triples; a["home_runs"][i] = hr
                a["hits"][i] = hits; a["total_bases"][i] = tb
                a["runs"][i] = s["R"]; a["rbi"][i] = s["RBI"]
                a["stolen_bases"][i] = s["SB"]; a["hrr"][i] = hits + s["R"] + s["RBI"]
                a["fantasy"][i] = (3 * singles + 5 * doubles + 8 * triples + 10 * hr
                                   + 2 * s["R"] + 2 * s["RBI"] + 2 * s["BB"]
                                   + 2 * s["HBP"] + 5 * s["SB"])
            elif s.get("_type") == "pitcher" and pid in pitcher_acc:
                a = pitcher_acc[pid]
                er = s.get("ER", 0); k = s["K"]; outs = s["OUTS"]
                a["pitcher_k"][i] = k; a["pitcher_outs"][i] = outs
                a["earned_runs"][i] = er; a["hits_allowed"][i] = s["H"]
                a["walks_allowed"][i] = s["BB"]; a["pitcher_qs"][i] = s.get("QS", 0)
                a["pitcher_fantasy"][i] = (6 * s.get("W", 0) + 4 * s.get("QS", 0)
                                           + 3 * k + 1 * outs - 3 * er)


class _TeamPitchState:
    """Tracks the active pitcher for a team and the starter→bullpen handoff."""
    def __init__(self, starter: Optional[PitcherProfile], cfg: MLBSimConfig):
        self.starter = starter
        self.cfg = cfg
        self.bullpen = league_average_reliever()
        self.bf = 0
        self.pitches = 0.0
        self._removed = starter is None

    def current_pitcher(self) -> PitcherProfile:
        if self._removed or self.starter is None:
            return self.bullpen
        # Pull at the pitcher's established pitch workload, hard-capped. Modern
        # managers rarely exceed ~100-110 pitches or the 3rd time through order.
        max_p = self.starter.avg_pitches if self.starter.avg_pitches else self.cfg.starter_max_pitches
        max_p = min(max_p, self.cfg.starter_max_pitches + 10)
        if self.pitches >= max_p or self.bf >= self.cfg.starter_max_batters_faced:
            self._removed = True
            return self.bullpen
        return self.starter

    def record_pa(self):
        if not self._removed:
            self.bf += 1
            self.pitches += PITCHES_PER_PA

    def tto_pass(self) -> int:
        return (self.bf // 9) + 1
