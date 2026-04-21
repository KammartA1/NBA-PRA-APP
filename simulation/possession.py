"""
simulation/possession.py
========================
Single-possession resolution.  Determines the outcome of one
possession: shot type, make/miss, assists, rebounds, steals, blocks,
turnovers, and-ones, and free throws.  All stat credits are assigned
to specific players.

Enhanced with:
- Transition play modeling (fast-break efficiency boost)
- Dynamic shot selection (game-script modifiers, two-for-one, heave)
- Opponent defense modifier
- Clutch shooting variance
- Hot hand / streak integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from simulation.config import SimulationConfig, DEFAULT_CONFIG
from simulation.player_state import PlayerState


@dataclass
class PossessionResult:
    """Complete accounting of what happened on a single possession."""
    # Outcome type
    turnover: bool = False
    steal: bool = False
    shot_attempted: bool = False
    shot_type: str = ""                # "2pt", "3pt", "ft"
    shot_made: bool = False
    and_one: bool = False
    offensive_rebound: bool = False
    free_throws_attempted: int = 0
    free_throws_made: int = 0

    # Points scored on this possession
    points_scored: int = 0

    # Stat credits (player indices within team roster)
    shooter_idx: Optional[int] = None
    assister_idx: Optional[int] = None
    rebounder_idx: Optional[int] = None
    stealer_idx: Optional[int] = None     # defender who stole
    blocker_idx: Optional[int] = None     # defender who blocked
    turnover_player_idx: Optional[int] = None
    ft_shooter_idx: Optional[int] = None

    # Possession changes
    change_possession: bool = True     # True = other team gets ball next

    # Transition tracking
    is_transition: bool = False


class PossessionEngine:
    """Resolve a single possession, crediting stats to players."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.cfg = config or DEFAULT_CONFIG

    def resolve(
        self,
        offense_players: List[PlayerState],
        offense_indices: List[int],
        defense_players: List[PlayerState],
        defense_indices: List[int],
        rng: np.random.Generator,
        free_throw_possession: bool = False,
        ft_shooter_idx: Optional[int] = None,
        ft_count: int = 0,
        *,
        # --- Transition play ---
        previous_play_type: Optional[str] = None,
        # --- Dynamic shot selection ---
        three_pt_rate_modifier: float = 0.0,
        two_pt_rate_modifier: float = 0.0,
        ft_draw_modifier: float = 0.0,
        two_for_one: bool = False,
        end_quarter_heave: bool = False,
        # --- Opponent defense ---
        defense_adjustment: float = 0.0,
        # --- Clutch ---
        is_clutch: bool = False,
        # --- Hot hand ---
        player_hot_streaks: Optional[Dict[int, int]] = None,
    ) -> PossessionResult:
        """Simulate one possession and return the result.

        Parameters
        ----------
        offense_players : 5 on-court PlayerState objects (offense)
        offense_indices : roster indices for those players
        defense_players : 5 on-court PlayerState objects (defense)
        defense_indices : roster indices for those players
        rng : numpy random Generator
        free_throw_possession : if True, this possession is just FTs
        ft_shooter_idx : roster index of the FT shooter
        ft_count : number of free throws to shoot

        Keyword-only (all optional, backward compatible):
        previous_play_type : one of "turnover", "made_shot",
            "defensive_rebound", "offensive_rebound", or None.
            Used to determine if this possession is in transition.
        three_pt_rate_modifier : additive shift to 3PT attempt rate
        two_pt_rate_modifier : additive shift to 2PT attempt rate
        ft_draw_modifier : additive shift to FT draw rate
        two_for_one : if True, bias toward quick 3PT attempts
        end_quarter_heave : if True, take a desperation 3PT heave
        defense_adjustment : modifier to ALL shooting percentages
            (negative = tough defense, positive = weak defense)
        is_clutch : if True, apply clutch_rating boosts/penalties
        player_hot_streaks : dict of player roster idx -> consecutive
            makes; players above threshold get usage/efficiency boosts
        """
        result = PossessionResult()
        cfg = self.cfg

        # --- Free-throw-only possession (from foul on previous play) ---
        if free_throw_possession and ft_shooter_idx is not None:
            return self._resolve_free_throws(
                offense_players, offense_indices, defense_players,
                defense_indices, ft_shooter_idx, ft_count, rng, result,
            )

        # ==============================================================
        # Determine if this is a transition possession
        # ==============================================================
        is_transition = False
        if previous_play_type is not None:
            if previous_play_type == "turnover":
                is_transition = rng.random() < cfg.transition_rate_after_turnover
            elif previous_play_type == "defensive_rebound":
                is_transition = rng.random() < cfg.transition_rate_after_defensive_rebound
            elif previous_play_type == "made_shot":
                is_transition = rng.random() < cfg.transition_rate_after_made_shot
            # offensive_rebound -> never transition (already in halfcourt)
        result.is_transition = is_transition

        # Transition efficiency multiplier
        transition_shooting_scale = 1.0
        if is_transition:
            transition_shooting_scale = cfg.transition_efg / cfg.halfcourt_efg

        # ==============================================================
        # Build effective usage weights (hot hand + transition)
        # ==============================================================
        usage_weights = np.array(
            [p.current_usage_rate for p in offense_players], dtype=np.float64,
        )

        # Hot hand: boost usage for players on streaks
        if player_hot_streaks:
            for local_i, roster_idx in enumerate(offense_indices):
                streak = player_hot_streaks.get(roster_idx, 0)
                if streak >= cfg.hot_hand_streak_threshold:
                    usage_weights[local_i] *= (1.0 + cfg.hot_hand_usage_boost)

        # Transition: weight toward ball-handlers (PG/SG)
        if is_transition:
            for local_i, p in enumerate(offense_players):
                if p.profile.position in ("PG", "SG"):
                    usage_weights[local_i] *= 1.3  # handlers run the break

        # Normalize
        usage_total = usage_weights.sum()
        if usage_total > 0:
            usage_weights /= usage_total

        # ==============================================================
        # Step 1: Check for steal / turnover
        # ==============================================================
        effective_turnover_rate = cfg.turnover_rate
        # Defense adjustment affects steal probability
        effective_steal_share = 0.60  # base: ~60% of turnovers are steals
        if defense_adjustment != 0.0:
            # Tougher defense -> more steals
            effective_steal_share *= (1.0 - defense_adjustment * 0.5)
            effective_steal_share = max(0.0, min(1.0, effective_steal_share))

        if rng.random() < effective_turnover_rate:
            result.turnover = True
            result.change_possession = True

            # Who turned it over? (weighted by usage)
            to_player_local = self._select_by_weights(usage_weights, rng)
            to_idx = offense_indices[to_player_local]
            result.turnover_player_idx = to_idx
            offense_players[to_player_local].record_turnover()

            # Was it a steal?
            if rng.random() < effective_steal_share:
                result.steal = True
                steal_local = self._select_by_steal_rate(defense_players, rng)
                steal_idx = defense_indices[steal_local]
                result.stealer_idx = steal_idx
                defense_players[steal_local].record_steal()

            return result

        # ==============================================================
        # Step 2: End-of-quarter heave (special case)
        # ==============================================================
        if end_quarter_heave:
            return self._resolve_heave(
                offense_players, offense_indices, usage_weights, rng, result,
            )

        # ==============================================================
        # Step 3: Determine shot type (with dynamic modifiers)
        # ==============================================================
        base_2pt = cfg.two_pt_attempt_rate
        base_3pt = cfg.three_pt_attempt_rate
        base_ft = cfg.free_throw_rate

        # Apply game-script modifiers
        adj_3pt = base_3pt + three_pt_rate_modifier
        adj_2pt = base_2pt + two_pt_rate_modifier
        adj_ft = base_ft + ft_draw_modifier

        # Two-for-one: bias toward 3PT (quick shots)
        if two_for_one:
            adj_3pt += cfg.two_for_one_three_pt_bias
            adj_2pt -= cfg.two_for_one_three_pt_bias * 0.5
            adj_ft -= cfg.two_for_one_three_pt_bias * 0.5

        # Clamp to non-negative
        adj_3pt = max(adj_3pt, 0.01)
        adj_2pt = max(adj_2pt, 0.01)
        adj_ft = max(adj_ft, 0.01)

        # Renormalize so shot type rates sum to (1 - turnover_rate)
        shot_budget = 1.0 - cfg.turnover_rate
        raw_total = adj_2pt + adj_3pt + adj_ft
        if raw_total > 0:
            adj_2pt = adj_2pt / raw_total * shot_budget
            adj_3pt = adj_3pt / raw_total * shot_budget
            adj_ft = adj_ft / raw_total * shot_budget

        # Cumulative thresholds (within [0, 1] after excluding turnovers)
        cumulative_2pt = adj_2pt / shot_budget
        cumulative_3pt = cumulative_2pt + adj_3pt / shot_budget

        shot_roll = rng.random()
        if shot_roll < cumulative_2pt:
            shot_type = "2pt"
        elif shot_roll < cumulative_3pt:
            shot_type = "3pt"
        else:
            shot_type = "ft"

        # ==============================================================
        # Step 4: Select the shooter (usage-weighted)
        # ==============================================================
        shooter_local = self._select_by_weights(usage_weights, rng)
        shooter = offense_players[shooter_local]
        shooter_idx = offense_indices[shooter_local]
        result.shooter_idx = shooter_idx
        result.shot_attempted = True
        result.shot_type = shot_type

        # ==============================================================
        # Compute efficiency modifiers for this shooter
        # ==============================================================
        eff_mod = shooter.current_efficiency_modifier

        # Transition boost
        eff_mod *= transition_shooting_scale

        # Defense adjustment: adj_pct *= (1.0 + defense_adjustment)
        defense_mult = 1.0 + defense_adjustment

        # Clutch modifier
        clutch_mult = 1.0
        if is_clutch:
            clutch_rating = getattr(shooter.profile, "clutch_rating", 0.0)
            if clutch_rating != 0.0:
                # Positive clutch_rating -> boost; negative -> penalty
                # Scale: clutch_rating of +1.0 -> +5% efficiency
                clutch_mult = 1.0 + clutch_rating * cfg.clutch_efficiency_variance

        # Hot hand efficiency boost
        hot_hand_mult = 1.0
        if player_hot_streaks:
            streak = player_hot_streaks.get(shooter_idx, 0)
            if streak >= cfg.hot_hand_streak_threshold:
                hot_hand_mult = 1.0 + cfg.hot_hand_efficiency_boost

        # Combined efficiency modifier
        combined_eff = eff_mod * defense_mult * clutch_mult * hot_hand_mult

        # ==============================================================
        # Step 5: Resolve the shot
        # ==============================================================
        if shot_type == "2pt":
            base_pct = shooter.profile.two_pt_pct
            adj_pct = base_pct * combined_eff

            # Check for block (no blocks on transition fast breaks)
            if not is_transition:
                block_happened, blocker_idx_roster = self._check_block(
                    defense_players, defense_indices, rng,
                )
                if block_happened:
                    result.blocker_idx = blocker_idx_roster
                    result.shot_made = False
                    shooter.record_two_pt_attempt(False)
                    result = self._resolve_rebound(
                        result, offense_players, offense_indices,
                        defense_players, defense_indices, rng,
                    )
                    return result

            made = rng.random() < adj_pct
            shooter.record_two_pt_attempt(made)
            result.shot_made = made

            if made:
                result.points_scored = 2
                result.change_possession = True
                # And-one probability (boosted in transition)
                and_one_prob = cfg.and_one_probability
                if is_transition:
                    and_one_prob += cfg.transition_and_one_boost
                if rng.random() < and_one_prob:
                    result.and_one = True
                    ft_pct = shooter.profile.ft_pct * combined_eff
                    ft_made = 1 if rng.random() < ft_pct else 0
                    shooter.record_free_throws(1, ft_made)
                    result.free_throws_attempted = 1
                    result.free_throws_made = ft_made
                    result.points_scored += ft_made
                # Assist?
                result = self._resolve_assist(
                    result, shooter_local, offense_players, offense_indices, rng,
                )
            else:
                # Missed shot -> rebound
                result = self._resolve_rebound(
                    result, offense_players, offense_indices,
                    defense_players, defense_indices, rng,
                )

        elif shot_type == "3pt":
            base_pct = shooter.profile.three_pt_pct
            adj_pct = base_pct * combined_eff

            # Check for block (no blocks on transition)
            if not is_transition:
                block_happened, blocker_idx_roster = self._check_block(
                    defense_players, defense_indices, rng, three_pt=True,
                )
                if block_happened:
                    result.blocker_idx = blocker_idx_roster
                    result.shot_made = False
                    shooter.record_three_pt_attempt(False)
                    result = self._resolve_rebound(
                        result, offense_players, offense_indices,
                        defense_players, defense_indices, rng,
                    )
                    return result

            made = rng.random() < adj_pct
            shooter.record_three_pt_attempt(made)
            result.shot_made = made

            if made:
                result.points_scored = 3
                result.change_possession = True
                # And-one on 3PT (rare, slightly boosted in transition)
                and_one_prob_3 = cfg.and_one_probability * 0.3
                if is_transition:
                    and_one_prob_3 += cfg.transition_and_one_boost * 0.3
                if rng.random() < and_one_prob_3:
                    result.and_one = True
                    ft_pct = shooter.profile.ft_pct * combined_eff
                    ft_made = 1 if rng.random() < ft_pct else 0
                    shooter.record_free_throws(1, ft_made)
                    result.free_throws_attempted = 1
                    result.free_throws_made = ft_made
                    result.points_scored += ft_made
                result = self._resolve_assist(
                    result, shooter_local, offense_players, offense_indices, rng,
                )
            else:
                result = self._resolve_rebound(
                    result, offense_players, offense_indices,
                    defense_players, defense_indices, rng,
                )

        else:  # "ft" -- non-shooting foul, go to line
            result.shot_attempted = False
            result.shot_type = "ft"
            ft_count_local = 2
            ft_pct = shooter.profile.ft_pct * combined_eff
            makes = sum(1 for _ in range(ft_count_local) if rng.random() < ft_pct)
            shooter.record_free_throws(ft_count_local, makes)
            result.free_throws_attempted = ft_count_local
            result.free_throws_made = makes
            result.ft_shooter_idx = shooter_idx
            result.points_scored = makes
            if makes < ft_count_local:
                # Miss on last FT -> rebound
                result = self._resolve_rebound(
                    result, offense_players, offense_indices,
                    defense_players, defense_indices, rng,
                )
            else:
                result.change_possession = True

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_heave(
        self,
        offense_players: List[PlayerState],
        offense_indices: List[int],
        usage_weights: np.ndarray,
        rng: np.random.Generator,
        result: PossessionResult,
    ) -> PossessionResult:
        """Resolve an end-of-quarter desperation heave (low-% 3PT)."""
        cfg = self.cfg
        shooter_local = self._select_by_weights(usage_weights, rng)
        shooter = offense_players[shooter_local]
        shooter_idx = offense_indices[shooter_local]

        result.shooter_idx = shooter_idx
        result.shot_attempted = True
        result.shot_type = "3pt"

        made = rng.random() < cfg.heave_three_pt_pct
        shooter.record_three_pt_attempt(made)
        result.shot_made = made

        if made:
            result.points_scored = 3
            result.change_possession = True
        else:
            # Heave miss at buzzer -> no rebound, possession changes
            result.change_possession = True
        return result

    def _resolve_free_throws(
        self,
        offense_players: List[PlayerState],
        offense_indices: List[int],
        defense_players: List[PlayerState],
        defense_indices: List[int],
        ft_shooter_idx: int,
        ft_count: int,
        rng: np.random.Generator,
        result: PossessionResult,
    ) -> PossessionResult:
        """Resolve a free-throw-only possession."""
        # Find the shooter in the offense list
        shooter = None
        for i, idx in enumerate(offense_indices):
            if idx == ft_shooter_idx:
                shooter = offense_players[i]
                break
        if shooter is None:
            # Fallback: pick first player
            shooter = offense_players[0]
            ft_shooter_idx = offense_indices[0]

        result.shot_type = "ft"
        result.ft_shooter_idx = ft_shooter_idx
        ft_pct = shooter.profile.ft_pct * shooter.current_efficiency_modifier
        makes = sum(1 for _ in range(ft_count) if rng.random() < ft_pct)
        shooter.record_free_throws(ft_count, makes)
        result.free_throws_attempted = ft_count
        result.free_throws_made = makes
        result.points_scored = makes

        if makes < ft_count:
            # Missed last FT -> rebound
            result = self._resolve_rebound(
                result, offense_players, offense_indices,
                defense_players, defense_indices, rng,
            )
        else:
            result.change_possession = True
        return result

    def _select_by_usage(
        self, players: List[PlayerState], rng: np.random.Generator,
    ) -> int:
        """Pick a player index (local to the 5-man list) weighted by usage."""
        usage = np.array(
            [p.current_usage_rate for p in players], dtype=np.float64,
        )
        total = usage.sum()
        if total <= 0:
            return int(rng.integers(0, len(players)))
        usage /= total
        return int(rng.choice(len(players), p=usage))

    def _select_by_weights(
        self, weights: np.ndarray, rng: np.random.Generator,
    ) -> int:
        """Pick a player index using pre-computed, normalized weights."""
        total = weights.sum()
        if total <= 0:
            return int(rng.integers(0, len(weights)))
        normed = weights / total
        return int(rng.choice(len(weights), p=normed))

    def _select_by_steal_rate(
        self, players: List[PlayerState], rng: np.random.Generator,
    ) -> int:
        """Pick the defender who gets credit for a steal."""
        rates = np.array(
            [p.profile.steal_rate for p in players], dtype=np.float64,
        )
        total = rates.sum()
        if total <= 0:
            return int(rng.integers(0, len(players)))
        rates /= total
        return int(rng.choice(len(players), p=rates))

    def _check_block(
        self,
        defense_players: List[PlayerState],
        defense_indices: List[int],
        rng: np.random.Generator,
        three_pt: bool = False,
    ) -> tuple[bool, Optional[int]]:
        """Check if a defender blocks the shot."""
        # Blocks are rarer on 3-pointers
        modifier = 0.3 if three_pt else 1.0
        for i, dp in enumerate(defense_players):
            block_prob = dp.profile.block_rate * modifier * dp.current_efficiency_modifier
            if rng.random() < block_prob:
                dp.record_block()
                return True, defense_indices[i]
        return False, None

    def _resolve_assist(
        self,
        result: PossessionResult,
        shooter_local: int,
        offense_players: List[PlayerState],
        offense_indices: List[int],
        rng: np.random.Generator,
    ) -> PossessionResult:
        """Determine if the made basket was assisted and by whom."""
        if rng.random() < self.cfg.assist_rate_on_made_fg:
            # Pick assister (anyone except shooter, weighted by assist rate)
            ast_rates = np.array([
                p.profile.assist_rate if i != shooter_local else 0.0
                for i, p in enumerate(offense_players)
            ], dtype=np.float64)
            total = ast_rates.sum()
            if total > 0:
                ast_rates /= total
                ast_local = int(rng.choice(len(offense_players), p=ast_rates))
                ast_idx = offense_indices[ast_local]
                offense_players[ast_local].record_assist()
                result.assister_idx = ast_idx
        return result

    def _resolve_rebound(
        self,
        result: PossessionResult,
        offense_players: List[PlayerState],
        offense_indices: List[int],
        defense_players: List[PlayerState],
        defense_indices: List[int],
        rng: np.random.Generator,
    ) -> PossessionResult:
        """Resolve a rebound after a missed shot."""
        cfg = self.cfg

        if rng.random() < cfg.offensive_rebound_rate:
            # Offensive rebound
            result.offensive_rebound = True
            result.change_possession = False
            reb_local = self._select_rebounder(offense_players, rng)
            reb_idx = offense_indices[reb_local]
            offense_players[reb_local].record_rebound(offensive=True)
            result.rebounder_idx = reb_idx
        else:
            # Defensive rebound
            result.change_possession = True
            reb_local = self._select_rebounder(defense_players, rng)
            reb_idx = defense_indices[reb_local]
            defense_players[reb_local].record_rebound(offensive=False)
            result.rebounder_idx = reb_idx

        return result

    def _select_rebounder(
        self, players: List[PlayerState], rng: np.random.Generator,
    ) -> int:
        """Pick who gets the rebound, weighted by rebound rate and height."""
        weights = np.array([
            p.profile.rebound_rate * (p.profile.height_inches / 78.0)
            * (1.0 - 0.20 * p.fatigue_level)  # fatigue reduces rebounding
            for p in players
        ], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            return int(rng.integers(0, len(players)))
        weights /= total
        return int(rng.choice(len(players), p=weights))
