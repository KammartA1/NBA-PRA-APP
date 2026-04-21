"""
simulation/game_script.py
=========================
Hyper-realistic, score-differential-driven coaching AI.  Every possession the
game script model evaluates the margin, time remaining, momentum, and
individual hot/cold streaks to produce a rich ``ScriptDecision`` that shapes:

* Dynamic pace (trailing teams speed up, leading teams grind the clock)
* Shot selection (desperate 3s when trailing, paint attacks when leading)
* Star usage / clutch mode
* Two-for-one possessions and end-of-quarter heaves
* Timeout strategy driven by opponent runs and own scoring droughts
* Garbage-time gradient (smooth starters-to-bench transition)
* Intentional fouling and substitution triggers
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

from simulation.config import SimulationConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Game phase enumeration
# ---------------------------------------------------------------------------

class GamePhase(enum.Enum):
    """Expanded game phases for nuanced coaching decisions."""
    EARLY_GAME = "early_game"              # First ~5 % of possessions (feeling-out)
    CLOSE = "close"                        # Margin <= close_game_margin
    MODERATE_LEAD = "moderate_lead"        # Leading by 6-14
    MODERATE_DEFICIT = "moderate_deficit"   # Trailing by 6-14
    BLOWOUT_LEAD = "blowout_lead"          # Leading by 15+
    BLOWOUT_DEFICIT = "blowout_deficit"    # Trailing by 15+
    COMEBACK = "comeback"                  # Was down big, now cutting lead
    CLUTCH = "clutch"                      # Final 10 % of game, margin <= 5
    GARBAGE_TIME = "garbage_time"          # Game is effectively decided


# ---------------------------------------------------------------------------
# Script decision
# ---------------------------------------------------------------------------

@dataclass
class ScriptDecision:
    """Rich coaching decision produced every possession."""

    phase: GamePhase

    # --- Rotation / usage ---
    starter_minutes_modifier: float = 1.0   # multiplier on starter target minutes
    usage_boost: float = 0.0                # added to star usage rate (0 = normal)
    pull_starters: bool = False             # True -> send starters to bench
    sub_trigger: bool = False               # True -> make a substitution now

    # --- Pace ---
    pace_multiplier: float = 1.0            # >1 = faster, <1 = slower

    # --- Shot selection ---
    three_pt_rate_modifier: float = 0.0     # boost / penalty to 3PA rate
    two_pt_rate_modifier: float = 0.0       # boost to paint attacks
    ft_draw_modifier: float = 0.0           # leading teams draw more FTs
    transition_rate_modifier: float = 0.0   # trailing teams push transition

    # --- End-of-quarter tactics ---
    two_for_one: bool = False               # attempt a quick shot for 2-for-1
    end_quarter_heave: bool = False         # heave at the buzzer

    # --- Clock management ---
    call_timeout: bool = False              # coach should call TO
    intentional_foul: bool = False          # foul on defense to stop clock


# ---------------------------------------------------------------------------
# Internal momentum state
# ---------------------------------------------------------------------------

@dataclass
class _MomentumState:
    """Tracks scoring runs and droughts for a single team."""
    unanswered_run: int = 0           # consecutive points scored without opponent scoring
    opponent_unanswered_run: int = 0  # consecutive opponent points (mirrors the other team)
    possessions_without_scoring: int = 0
    last_possession_scored: bool = False

    def record_scored(self, points: int) -> None:
        """Team scored *points* this possession."""
        self.unanswered_run += points
        self.opponent_unanswered_run = 0
        self.possessions_without_scoring = 0
        self.last_possession_scored = True

    def record_opponent_scored(self, points: int) -> None:
        """Opponent scored *points* this possession."""
        self.opponent_unanswered_run += points
        self.unanswered_run = 0
        self.last_possession_scored = False

    def record_no_score(self) -> None:
        """Neither team scored on this possession (turnover, missed shot)."""
        self.possessions_without_scoring += 1
        self.last_possession_scored = False

    def reset(self) -> None:
        self.unanswered_run = 0
        self.opponent_unanswered_run = 0
        self.possessions_without_scoring = 0
        self.last_possession_scored = False


# ---------------------------------------------------------------------------
# Hot/cold hand tracker
# ---------------------------------------------------------------------------

@dataclass
class _HotColdTracker:
    """Per-player consecutive-make / consecutive-miss tracking."""
    # player_id -> consecutive makes (positive) or misses (negative)
    _streaks: Dict[str, int] = field(default_factory=dict)

    def record_make(self, player_id: str) -> None:
        cur = self._streaks.get(player_id, 0)
        self._streaks[player_id] = max(cur, 0) + 1

    def record_miss(self, player_id: str) -> None:
        cur = self._streaks.get(player_id, 0)
        self._streaks[player_id] = min(cur, 0) - 1

    def get_streak(self, player_id: str) -> int:
        """Positive = consecutive makes, negative = consecutive misses."""
        return self._streaks.get(player_id, 0)

    def hot_players(self, threshold: int) -> Dict[str, int]:
        """Return {player_id: streak} for players on a hot streak."""
        return {pid: s for pid, s in self._streaks.items() if s >= threshold}

    def cold_players(self, threshold: int) -> Dict[str, int]:
        """Return {player_id: streak} for players on a cold streak."""
        return {pid: s for pid, s in self._streaks.items() if s <= -threshold}

    def reset(self) -> None:
        self._streaks.clear()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GameScriptModel:
    """Evaluate the current score differential, time remaining, momentum,
    and hot/cold streaks to produce rich coaching decisions that shape
    player minutes, usage, pace, shot selection, and clock management.

    One instance is created per team per simulated game.
    """

    # Early-game threshold as a fraction of total possessions
    _EARLY_GAME_FRACTION: float = 0.05

    # Scoring-drought threshold (own team) that can trigger a timeout
    _SCORING_DROUGHT_POSSESSIONS: int = 4

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.cfg = config or DEFAULT_CONFIG

        # --- Persistent state (reset between simulations) ---
        self._prev_deficit: Optional[int] = None
        self._max_deficit: int = 0

        # Momentum / run tracking
        self._momentum: _MomentumState = _MomentumState()

        # Hot/cold hand tracking
        self._hot_cold: _HotColdTracker = _HotColdTracker()

        # Timeout bookkeeping: track how many TOs we have used vs. saved
        self._timeouts_called_this_half: int = 0

    # ------------------------------------------------------------------
    # Public API: shot-result feedback
    # ------------------------------------------------------------------

    def record_own_score(self, points: int) -> None:
        """Call after this team scores *points* on a possession."""
        self._momentum.record_scored(points)

    def record_opponent_score(self, points: int) -> None:
        """Call after the opponent scores *points*."""
        self._momentum.record_opponent_scored(points)

    def record_no_score(self) -> None:
        """Call when a possession ends without any scoring."""
        self._momentum.record_no_score()

    def record_player_make(self, player_id: str) -> None:
        """Call when a player makes a shot."""
        self._hot_cold.record_make(player_id)

    def record_player_miss(self, player_id: str) -> None:
        """Call when a player misses a shot."""
        self._hot_cold.record_miss(player_id)

    def get_hot_players(self) -> Dict[str, int]:
        """Return hot-streak players: {player_id: consecutive_makes}."""
        return self._hot_cold.hot_players(self.cfg.hot_hand_streak_threshold)

    def get_cold_players(self) -> Dict[str, int]:
        """Return cold-streak players: {player_id: consecutive_misses}."""
        return self._hot_cold.cold_players(self.cfg.cold_streak_threshold)

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        own_score: int,
        opp_score: int,
        possession_number: int,
        total_possessions: int,
        unanswered_opponent_points: int,
        timeouts_remaining: int,
    ) -> ScriptDecision:
        """Produce a ``ScriptDecision`` given the current game state.

        Parameters
        ----------
        own_score, opp_score : int
            Current scores for this team and the opponent.
        possession_number : int
            0-based index of the current possession in the game.
        total_possessions : int
            Expected total possessions for the entire game.
        unanswered_opponent_points : int
            Consecutive opponent points (kept for backward compat; also
            tracked internally via momentum).
        timeouts_remaining : int
            How many timeouts this team still has.

        Returns
        -------
        ScriptDecision
        """
        cfg = self.cfg
        margin = own_score - opp_score          # positive = leading
        abs_margin = abs(margin)
        game_progress = possession_number / max(total_possessions, 1)

        # Derived time indicators
        is_fourth_quarter = game_progress >= 0.75
        is_late_game = game_progress >= 0.90
        is_very_late = game_progress >= 0.95

        # Estimate minutes remaining (linear mapping from progress)
        total_minutes = cfg.quarter_length_minutes * cfg.num_quarters
        minutes_remaining = max(total_minutes * (1.0 - game_progress), 0.0)

        # Quarter-within-game (0-indexed): used for end-of-quarter logic
        quarter_progress = (game_progress * cfg.num_quarters) % 1.0
        seconds_left_in_quarter = (1.0 - quarter_progress) * cfg.quarter_length_minutes * 60.0

        # --- Comeback tracking ---
        deficit = -margin if margin < 0 else 0
        if deficit > self._max_deficit:
            self._max_deficit = deficit
        comeback = (
            self._max_deficit >= cfg.comeback_shrink_threshold
            and deficit < self._max_deficit - cfg.comeback_shrink_threshold
            and deficit > 0
        )
        self._prev_deficit = deficit

        # --- Garbage time score (gradient) ---
        if minutes_remaining > 0:
            garbage_time_score = abs_margin / (minutes_remaining * cfg.garbage_time_margin_per_minute)
        else:
            garbage_time_score = float("inf") if abs_margin > 0 else 0.0

        # ================================================================
        # Phase determination (priority order)
        # ================================================================
        phase = self._determine_phase(
            margin=margin,
            abs_margin=abs_margin,
            game_progress=game_progress,
            is_fourth_quarter=is_fourth_quarter,
            is_late_game=is_late_game,
            comeback=comeback,
            garbage_time_score=garbage_time_score,
        )

        # ================================================================
        # Build decision starting from defaults
        # ================================================================
        starter_mod = 1.0
        usage_boost = 0.0
        pull_starters = False
        sub_trigger = False
        pace_mult = 1.0
        three_pt_mod = 0.0
        two_pt_mod = 0.0
        ft_draw_mod = 0.0
        transition_mod = 0.0
        two_for_one = False
        end_quarter_heave = False
        call_timeout = False
        intentional_foul = False

        # ================================================================
        # 1. Dynamic pace adjustment (applies to ALL phases)
        # ================================================================
        if margin < 0:
            # Trailing: speed up
            pace_boost = abs_margin * cfg.trailing_pace_boost_per_point
            pace_mult += min(pace_boost, cfg.trailing_pace_boost_max)
        elif margin > 0:
            # Leading: slow down
            pace_slow = margin * cfg.leading_pace_slow_per_point
            pace_mult -= min(pace_slow, cfg.leading_pace_slow_max)

        # Close Q4 games: both teams get slightly frantic
        if is_fourth_quarter and abs_margin <= cfg.close_game_margin:
            pace_mult += cfg.q4_close_pace_boost

        # ================================================================
        # 2. Shot selection shifts (applies to ALL phases)
        # ================================================================
        if margin < 0 and abs_margin >= 10:
            # Trailing by 10+: jack 3s
            three_boost = abs_margin * cfg.trailing_three_pt_boost_per_point
            three_pt_mod += min(three_boost, cfg.trailing_three_pt_boost_max)
            # Also push transition
            transition_mod += min(abs_margin * 0.005, 0.08)

        if margin > 0:
            # Leading: attack paint, draw fouls, grind clock
            two_pt_mod += cfg.leading_paint_boost
            ft_draw_mod += cfg.leading_ft_draw_boost

        # ================================================================
        # 3. Phase-specific adjustments
        # ================================================================
        if phase == GamePhase.EARLY_GAME:
            # Feeling-out period: normal everything, slight conservative pace
            starter_mod = 1.0
            usage_boost = 0.0
            # Teams don't push pace early; let the game develop
            pace_mult = max(pace_mult, 0.97)

        elif phase == GamePhase.CLUTCH:
            # Final stretch of a close game: stars dominate
            usage_boost = cfg.clutch_star_usage_boost
            starter_mod = 1.25
            # Sub only for foul trouble, not fatigue
            sub_trigger = False
            # Pace ticks up slightly (urgency)
            pace_mult += 0.02

        elif phase == GamePhase.CLOSE:
            starter_mod = 1.10
            if is_late_game:
                usage_boost = 0.05
                starter_mod = 1.20

        elif phase == GamePhase.MODERATE_LEAD:
            starter_mod = 1.0
            if is_fourth_quarter:
                # Protect the lead: slightly tighter rotation
                starter_mod = 1.05

        elif phase == GamePhase.MODERATE_DEFICIT:
            starter_mod = 1.05
            if is_fourth_quarter:
                usage_boost = 0.04
                starter_mod = 1.12
                sub_trigger = True  # get starters back in

        elif phase == GamePhase.BLOWOUT_LEAD:
            if is_fourth_quarter:
                pull_starters = True
                starter_mod = 0.80
            else:
                starter_mod = 0.95

        elif phase == GamePhase.BLOWOUT_DEFICIT:
            if is_fourth_quarter:
                pull_starters = True
                starter_mod = 0.80
            else:
                # Still trying: push pace, jack 3s
                starter_mod = 1.0
                usage_boost = 0.02

        elif phase == GamePhase.COMEBACK:
            # Starters back in, high usage, push pace
            starter_mod = 1.15
            usage_boost = 0.06
            sub_trigger = True
            pace_mult += 0.02

        elif phase == GamePhase.GARBAGE_TIME:
            pull_starters = True
            starter_mod = 0.70
            usage_boost = 0.0
            # Bench players get run; pace normalizes
            pace_mult = 1.0
            three_pt_mod = 0.0
            two_pt_mod = 0.0
            ft_draw_mod = 0.0
            transition_mod = 0.0

        # ================================================================
        # 4. Garbage time gradient (smooth starters-to-bench transition)
        # ================================================================
        if (
            phase != GamePhase.GARBAGE_TIME
            and garbage_time_score > cfg.garbage_time_bench_scale_start
            and is_fourth_quarter
        ):
            # Approaching garbage time: gradually rest starters
            gradient = min(
                (garbage_time_score - cfg.garbage_time_bench_scale_start)
                / (cfg.garbage_time_bench_scale_full - cfg.garbage_time_bench_scale_start),
                1.0,
            )
            # Linearly interpolate starter_mod toward 0.80
            starter_mod = starter_mod * (1.0 - gradient) + 0.80 * gradient
            # At the high end of the gradient, pull starters outright
            if gradient >= 0.9:
                pull_starters = True

        # ================================================================
        # 5. Two-for-one possessions
        # ================================================================
        if (
            seconds_left_in_quarter <= cfg.two_for_one_window_seconds
            and seconds_left_in_quarter > 6.0  # not at the very end
        ):
            if random.random() < cfg.two_for_one_probability:
                two_for_one = True
                # Bias toward 3s on two-for-one
                three_pt_mod += cfg.two_for_one_three_pt_bias

        # ================================================================
        # 6. End-of-quarter heave
        # ================================================================
        if seconds_left_in_quarter <= 4.0 and seconds_left_in_quarter >= 0.0:
            if random.random() < cfg.end_quarter_heave_probability:
                end_quarter_heave = True

        # ================================================================
        # 7. Timeout logic (rich, multi-factor)
        # ================================================================
        call_timeout = self._evaluate_timeout(
            unanswered_opponent_points=max(
                unanswered_opponent_points,
                self._momentum.opponent_unanswered_run,
            ),
            own_scoring_drought=self._momentum.possessions_without_scoring,
            timeouts_remaining=timeouts_remaining,
            game_progress=game_progress,
            is_fourth_quarter=is_fourth_quarter,
            margin=margin,
            phase=phase,
        )

        # ================================================================
        # 8. Intentional fouling (trailing, very late)
        # ================================================================
        if is_very_late and margin < 0 and abs_margin <= 6:
            intentional_foul = True

        # ================================================================
        # 9. Hot-hand usage adjustments
        # ================================================================
        hot_count = len(self.get_hot_players())
        cold_count = len(self.get_cold_players())
        if hot_count > 0 and phase != GamePhase.GARBAGE_TIME:
            # Slight additional usage boost when hot shooters are on court
            usage_boost += min(hot_count * cfg.hot_hand_usage_boost, 0.10)
        if cold_count > 0 and phase != GamePhase.GARBAGE_TIME:
            usage_boost -= min(cold_count * cfg.cold_streak_usage_penalty, 0.06)

        # Clamp usage_boost to a reasonable range
        usage_boost = max(min(usage_boost, 0.25), -0.10)

        # ================================================================
        # Build and return
        # ================================================================
        return ScriptDecision(
            phase=phase,
            starter_minutes_modifier=round(starter_mod, 4),
            usage_boost=round(usage_boost, 4),
            pull_starters=pull_starters,
            sub_trigger=sub_trigger,
            pace_multiplier=round(pace_mult, 4),
            three_pt_rate_modifier=round(three_pt_mod, 4),
            two_pt_rate_modifier=round(two_pt_mod, 4),
            ft_draw_modifier=round(ft_draw_mod, 4),
            transition_rate_modifier=round(transition_mod, 4),
            two_for_one=two_for_one,
            end_quarter_heave=end_quarter_heave,
            call_timeout=call_timeout,
            intentional_foul=intentional_foul,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all mutable state between simulations."""
        self._prev_deficit = None
        self._max_deficit = 0
        self._momentum.reset()
        self._hot_cold.reset()
        self._timeouts_called_this_half = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _determine_phase(
        self,
        *,
        margin: int,
        abs_margin: int,
        game_progress: float,
        is_fourth_quarter: bool,
        is_late_game: bool,
        comeback: bool,
        garbage_time_score: float,
    ) -> GamePhase:
        """Classify the current game state into a ``GamePhase``.

        Priority order:
        1. GARBAGE_TIME (game truly decided)
        2. CLUTCH (late close game)
        3. EARLY_GAME (feeling-out period)
        4. COMEBACK (was down big, cutting into lead)
        5. Margin-based: CLOSE / MODERATE_LEAD / MODERATE_DEFICIT /
           BLOWOUT_LEAD / BLOWOUT_DEFICIT
        """
        cfg = self.cfg

        # Garbage time: game is effectively over (high garbage_time_score in Q4)
        if (
            garbage_time_score >= cfg.garbage_time_bench_scale_full
            and is_fourth_quarter
        ):
            return GamePhase.GARBAGE_TIME

        # Clutch: final 10 % of game AND margin is tight
        if (
            game_progress >= cfg.clutch_game_progress
            and abs_margin <= cfg.clutch_margin
        ):
            return GamePhase.CLUTCH

        # Early game: first ~5 % of possessions
        if game_progress < self._EARLY_GAME_FRACTION:
            return GamePhase.EARLY_GAME

        # Comeback: was down big, now cutting into the lead (Q4 only)
        if comeback and is_fourth_quarter:
            return GamePhase.COMEBACK

        # Margin-based classification
        if abs_margin <= cfg.close_game_margin:
            return GamePhase.CLOSE
        elif margin > cfg.moderate_margin:
            return GamePhase.BLOWOUT_LEAD
        elif margin < -cfg.moderate_margin:
            return GamePhase.BLOWOUT_DEFICIT
        elif margin > 0:
            return GamePhase.MODERATE_LEAD
        else:
            return GamePhase.MODERATE_DEFICIT

    def _evaluate_timeout(
        self,
        *,
        unanswered_opponent_points: int,
        own_scoring_drought: int,
        timeouts_remaining: int,
        game_progress: float,
        is_fourth_quarter: bool,
        margin: int,
        phase: GamePhase,
    ) -> bool:
        """Decide whether to call a timeout.

        Reasons to call a timeout:
        1. Opponent on a big run (momentum_timeout_trigger or momentum_run_threshold)
        2. Own scoring drought (4+ empty possessions)
        3. NOT in garbage time
        4. Save timeouts for late game (don't burn them early on marginal runs)
        """
        cfg = self.cfg

        if timeouts_remaining <= 0:
            return False

        # Never call timeout in garbage time
        if phase == GamePhase.GARBAGE_TIME:
            return False

        # --- Opponent run triggers ---

        # Mandatory timeout: large unanswered run
        if unanswered_opponent_points >= cfg.momentum_timeout_trigger:
            return True

        # Advisory timeout: moderate run (but be conservative early to save TOs)
        if unanswered_opponent_points >= cfg.momentum_run_threshold:
            # In the first half, only call if we have plenty of TOs
            if not is_fourth_quarter and timeouts_remaining <= 2:
                return False
            return True

        # Legacy threshold (backward compatibility with callers passing
        # unanswered_opponent_points directly)
        if unanswered_opponent_points >= cfg.timeout_run_threshold:
            if not is_fourth_quarter and timeouts_remaining <= 2:
                return False
            return True

        # --- Own scoring drought ---
        if (
            own_scoring_drought >= self._SCORING_DROUGHT_POSSESSIONS
            and timeouts_remaining >= 2
        ):
            # Don't waste a TO on a drought if we are blowing them out
            if margin > cfg.moderate_margin:
                return False
            return True

        return False
