"""Random baseline agent."""

from __future__ import annotations

import random
from typing import Optional, Tuple

from env.gomoku_env import GomokuEnv


class RandomAgent:
    """Selects uniformly from currently valid actions."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, env: GomokuEnv) -> Tuple[int, int]:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("no valid actions remain")
        return self._rng.choice(valid_actions)
