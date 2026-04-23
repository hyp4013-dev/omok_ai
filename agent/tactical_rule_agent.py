"""Rule-based tactical Gomoku agent."""

from __future__ import annotations

import random
from dataclasses import dataclass

from env.gomoku_env import Action, GomokuEnv
from utils.tactical_rules import find_forced_action


@dataclass(frozen=True)
class TacticalRuleEvaluation:
    action: Action
    score: float


@dataclass(frozen=True)
class _DirectionalPattern:
    score: float
    stones: int
    open_ends: int


class TacticalRuleAgent:
    """Choose moves from tactical and shape-based heuristics."""

    def __init__(
        self,
        name: str = "tactical_rule_agent",
        own_weight: float = 1.4,
        opp_weight: float = 1.2,
        own_fork_weight: float = 1.3,
        opp_fork_weight: float = 1.5,
        center_weight: float = 1.0,
        neighbor_weight: float = 1.0,
        enable_fork_bonus: bool = True,
        late_random_start_move: int | None = None,
        late_random_probability: float = 0.0,
        late_random_top_k: int = 10,
        include_open_three_block: bool = False,
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.own_weight = own_weight
        self.opp_weight = opp_weight
        self.own_fork_weight = own_fork_weight
        self.opp_fork_weight = opp_fork_weight
        self.center_weight = center_weight
        self.neighbor_weight = neighbor_weight
        self.enable_fork_bonus = enable_fork_bonus
        self.late_random_start_move = late_random_start_move
        self.late_random_probability = late_random_probability
        self.late_random_top_k = late_random_top_k
        self.include_open_three_block = include_open_three_block
        self.random = random.Random(seed)
        self.variant_label: str | None = None

    def select_action(
        self,
        env: GomokuEnv,
        training: bool = False,
    ) -> tuple[Action, TacticalRuleEvaluation]:
        del training
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("no valid actions available")

        forced_action = find_forced_action(env, include_open_three_block=self.include_open_three_block)
        if forced_action is not None:
            return forced_action, TacticalRuleEvaluation(action=forced_action, score=float("inf"))

        evaluations = [
            TacticalRuleEvaluation(action=action, score=self._score_action(env, action))
            for action in valid_actions
        ]
        if self._should_take_late_random_move(env, evaluations):
            ranked_evaluations = sorted(
                evaluations,
                key=lambda item: (item.score, -self._distance_to_center(env, item.action), item.action),
                reverse=True,
            )
            random_pool = ranked_evaluations[1 : min(len(ranked_evaluations), self.late_random_top_k)]
            if random_pool:
                chosen = self.random.choice(random_pool)
                return chosen.action, chosen
        best = max(
            evaluations,
            key=lambda item: (item.score, -self._distance_to_center(env, item.action), item.action),
        )
        return best.action, best

    def _should_take_late_random_move(
        self,
        env: GomokuEnv,
        evaluations: list[TacticalRuleEvaluation],
    ) -> bool:
        del evaluations
        if self.late_random_start_move is None:
            return False
        if env.move_count < self.late_random_start_move:
            return False
        if self.late_random_probability <= 0.0:
            return False
        return self.random.random() < self.late_random_probability

    def _score_action(self, env: GomokuEnv, action: Action) -> float:
        own_patterns = self._directional_patterns(env, action, env.current_player)
        opp_patterns = self._directional_patterns(env, action, -env.current_player)
        own_score = self._shape_score(own_patterns)
        opp_score = self._shape_score(opp_patterns)
        own_fork_bonus = self._fork_bonus(own_patterns) if self.enable_fork_bonus else 0.0
        opp_fork_bonus = self._fork_bonus(opp_patterns) if self.enable_fork_bonus else 0.0
        center_bonus = self.center_weight * self._center_bonus(env, action)
        neighbor_bonus = self.neighbor_weight * self._neighbor_bonus(env, action)
        return (
            (self.own_weight * own_score)
            + (self.opp_weight * opp_score)
            + (self.own_fork_weight * own_fork_bonus)
            + (self.opp_fork_weight * opp_fork_bonus)
            + center_bonus
            + neighbor_bonus
        )

    def _directional_patterns(
        self,
        env: GomokuEnv,
        action: Action,
        player: int,
    ) -> list[_DirectionalPattern]:
        row, col = action
        board = env.board
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        patterns: list[_DirectionalPattern] = []
        for delta_row, delta_col in directions:
            forward, forward_open = self._count_and_open(board, row, col, delta_row, delta_col, player)
            backward, backward_open = self._count_and_open(board, row, col, -delta_row, -delta_col, player)
            stones = 1 + forward + backward
            open_ends = int(forward_open) + int(backward_open)
            score = self._pattern_score(stones, open_ends)
            patterns.append(_DirectionalPattern(score=score, stones=stones, open_ends=open_ends))
        patterns.sort(key=lambda item: (item.score, item.stones, item.open_ends), reverse=True)
        return patterns

    def _shape_score(self, patterns: list[_DirectionalPattern]) -> float:
        if not patterns:
            return 0.0
        primary = patterns[0].score
        secondary = patterns[1].score if len(patterns) > 1 else 0.0
        return primary + (0.35 * secondary)

    def _fork_bonus(self, patterns: list[_DirectionalPattern]) -> float:
        open_fours = sum(1 for pattern in patterns if pattern.stones == 4 and pattern.open_ends == 2)
        fours = sum(1 for pattern in patterns if pattern.stones == 4 and pattern.open_ends >= 1)
        open_threes = sum(1 for pattern in patterns if pattern.stones == 3 and pattern.open_ends == 2)
        threes = sum(1 for pattern in patterns if pattern.stones == 3 and pattern.open_ends >= 1)

        if open_fours >= 2:
            return 200_000.0
        if open_fours >= 1 and open_threes >= 1:
            return 90_000.0
        if fours >= 2:
            return 45_000.0
        if open_threes >= 2:
            return 18_000.0
        if threes >= 2:
            return 4_000.0
        return 0.0

    def _count_and_open(
        self,
        board: list[list[int]],
        row: int,
        col: int,
        delta_row: int,
        delta_col: int,
        player: int,
    ) -> tuple[int, bool]:
        count = 0
        board_size = len(board)
        next_row = row + delta_row
        next_col = col + delta_col
        while 0 <= next_row < board_size and 0 <= next_col < board_size:
            cell = board[next_row][next_col]
            if cell != player:
                return count, cell == 0
            count += 1
            next_row += delta_row
            next_col += delta_col
        return count, False

    def _pattern_score(self, stones: int, open_ends: int) -> float:
        if stones >= 5:
            return 1_000_000.0
        if stones == 4 and open_ends == 2:
            return 50_000.0
        if stones == 4 and open_ends == 1:
            return 12_000.0
        if stones == 3 and open_ends == 2:
            return 4_000.0
        if stones == 3 and open_ends == 1:
            return 800.0
        if stones == 2 and open_ends == 2:
            return 180.0
        if stones == 2 and open_ends == 1:
            return 40.0
        if stones == 1 and open_ends == 2:
            return 8.0
        return 1.0

    def _center_bonus(self, env: GomokuEnv, action: Action) -> float:
        distance = self._distance_to_center(env, action)
        return max(0.0, env.board_size - distance)

    def _distance_to_center(self, env: GomokuEnv, action: Action) -> float:
        center = (env.board_size - 1) / 2.0
        row, col = action
        return abs(row - center) + abs(col - center)

    def _neighbor_bonus(self, env: GomokuEnv, action: Action) -> float:
        row, col = action
        bonus = 0.0
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                next_row = row + delta_row
                next_col = col + delta_col
                if 0 <= next_row < env.board_size and 0 <= next_col < env.board_size:
                    if env.board[next_row][next_col] != 0:
                        bonus += 2.5
        return bonus


class EasyTacticalRuleAgent(TacticalRuleAgent):
    """A weaker tactical agent that keeps immediate tactics but softens shaping pressure."""

    def __init__(self, name: str = "easy_tactical_rule_agent") -> None:
        super().__init__(
            name=name,
            own_weight=1.15,
            opp_weight=1.0,
            own_fork_weight=0.0,
            opp_fork_weight=0.0,
            center_weight=0.45,
            neighbor_weight=0.6,
            enable_fork_bonus=False,
            include_open_three_block=False,
        )


class NormalTacticalRuleAgent(TacticalRuleAgent):
    """A medium tactical agent between easy and hard."""

    def __init__(self, name: str = "normal_tactical_rule_agent") -> None:
        super().__init__(
            name=name,
            own_weight=1.25,
            opp_weight=1.1,
            own_fork_weight=0.55,
            opp_fork_weight=0.75,
            center_weight=0.75,
            neighbor_weight=0.8,
            enable_fork_bonus=True,
            include_open_three_block=False,
        )


class SuperEasyTacticalRuleAgent(TacticalRuleAgent):
    """A very weak tactical agent for curriculum warm-up."""

    def __init__(self, name: str = "super_easy_tactical_rule_agent") -> None:
        super().__init__(
            name=name,
            own_weight=0.95,
            opp_weight=0.7,
            own_fork_weight=0.0,
            opp_fork_weight=0.0,
            center_weight=0.15,
            neighbor_weight=0.2,
            enable_fork_bonus=False,
            late_random_start_move=30,
            late_random_probability=0.05,
            late_random_top_k=10,
            include_open_three_block=False,
        )


class HardTacticalRuleAgent(TacticalRuleAgent):
    """The original strong tactical rule agent."""

    def __init__(self, name: str = "hard_tactical_rule_agent", seed: int | None = None) -> None:
        super().__init__(
            name=name,
            own_weight=1.4,
            opp_weight=1.2,
            own_fork_weight=1.3,
            opp_fork_weight=1.5,
            center_weight=1.0,
            neighbor_weight=1.0,
            enable_fork_bonus=True,
            include_open_three_block=True,
            seed=seed,
        )
        self.variant_label = "neutral"


class OffensiveHardTacticalRuleAgent(TacticalRuleAgent):
    """A hard variant that leans toward aggressive creation."""

    def __init__(self, name: str = "offensive_hard_tactical_rule_agent", seed: int | None = None) -> None:
        super().__init__(
            name=name,
            own_weight=1.6,
            opp_weight=1.05,
            own_fork_weight=1.75,
            opp_fork_weight=1.15,
            center_weight=1.0,
            neighbor_weight=0.95,
            enable_fork_bonus=True,
            include_open_three_block=True,
            seed=seed,
        )
        self.variant_label = "offensive"


class DefensiveHardTacticalRuleAgent(TacticalRuleAgent):
    """A hard variant that leans toward blocking and counterplay."""

    def __init__(self, name: str = "defensive_hard_tactical_rule_agent", seed: int | None = None) -> None:
        super().__init__(
            name=name,
            own_weight=1.25,
            opp_weight=1.35,
            own_fork_weight=1.05,
            opp_fork_weight=1.85,
            center_weight=0.95,
            neighbor_weight=1.05,
            enable_fork_bonus=True,
            include_open_three_block=True,
            seed=seed,
        )
        self.variant_label = "defensive"


class NeutralHardTacticalRuleAgent(TacticalRuleAgent):
    """A balanced hard variant."""

    def __init__(self, name: str = "neutral_hard_tactical_rule_agent", seed: int | None = None) -> None:
        super().__init__(
            name=name,
            own_weight=1.4,
            opp_weight=1.2,
            own_fork_weight=1.3,
            opp_fork_weight=1.5,
            center_weight=1.0,
            neighbor_weight=1.0,
            enable_fork_bonus=True,
            include_open_three_block=True,
            seed=seed,
        )
        self.variant_label = "neutral"


def build_random_hard_tactical_rule_agent(
    name: str = "hard_tactical_rule_agent",
    seed: int | None = None,
) -> TacticalRuleAgent:
    hard_variants = (
        OffensiveHardTacticalRuleAgent,
        NeutralHardTacticalRuleAgent,
        DefensiveHardTacticalRuleAgent,
    )
    chooser = random.Random(seed) if seed is not None else random
    chosen_variant = chooser.choice(hard_variants)
    return chosen_variant(name=name, seed=seed)
