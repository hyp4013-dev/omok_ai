"""A lightweight value-based Gomoku agent."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from env.gomoku_env import GomokuEnv
from utils.state_encoder import action_features
from utils.tactical_rules import find_forced_action


@dataclass
class ValueStepRecord:
    features: list[float]
    prediction: float


class ValueAgent:
    def __init__(
        self,
        name: str,
        board_size: int,
        learning_rate: float = 0.08,
        gamma: float = 0.97,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        center_bias: float = 1.0,
        blocking_bias: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.random = random.Random(seed)
        self.weights = [0.0] * 11
        self.prior_weights = [0.0] * 11
        self.weights[1] = 0.3 * center_bias
        self.weights[7] = 0.4 * blocking_bias
        self.weights[8] = 0.8 * blocking_bias
        self.prior_weights[1] = 0.2 * center_bias
        self.prior_weights[2] = 0.5
        self.prior_weights[3] = 0.8
        self.prior_weights[4] = 1.4
        self.prior_weights[5] = 2.4
        self.prior_weights[6] = 0.3 * blocking_bias
        self.prior_weights[7] = 0.9 * blocking_bias
        self.prior_weights[8] = 2.0 * blocking_bias
        self.prior_weights[9] = 0.15
        self.prior_weights[10] = 0.1
        self.episodes_trained = 0
        self.weight_clip_value = 12.0
        self.error_clip_value = 2.0
        self.update_clip_value = 0.35

    def select_action(self, env: GomokuEnv, training: bool = True) -> tuple[tuple[int, int], ValueStepRecord]:
        self._sanitize_weights()
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("no valid actions available")

        if training and env.move_count == 0:
            opening_pool = self._central_opening_pool(valid_actions, env.board_size)
            action = self.random.choice(opening_pool)
            features = action_features(env.board, env.current_player, action)
            return action, ValueStepRecord(features=features, prediction=self._predict(features))

        forced_action = find_forced_action(env)
        if forced_action is not None:
            features = action_features(env.board, env.current_player, forced_action)
            return forced_action, ValueStepRecord(features=features, prediction=self._predict(features))

        action_scores = []
        for action in valid_actions:
            features = action_features(env.board, env.current_player, action)
            action_scores.append((self._predict(features), action, features))

        if training and self.random.random() < self.epsilon:
            score, action, features = self.random.choice(action_scores)
        else:
            score, action, features = max(action_scores, key=lambda item: item[0])
        return action, ValueStepRecord(features=features, prediction=score)

    def finish_game(self, episode_records: list[ValueStepRecord], outcome_reward: float) -> None:
        self._sanitize_weights()
        for move_offset, record in enumerate(reversed(episode_records)):
            target = outcome_reward * (self.gamma ** move_offset)
            prediction = record.prediction if math.isfinite(record.prediction) else 0.0
            error = max(-self.error_clip_value, min(self.error_clip_value, target - prediction))
            for index, feature in enumerate(record.features):
                update = self.learning_rate * error * feature
                update = max(-self.update_clip_value, min(self.update_clip_value, update))
                next_weight = self.weights[index] + update
                self.weights[index] = max(-self.weight_clip_value, min(self.weight_clip_value, next_weight))

        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        self._sanitize_weights()
        payload = {
            "name": self.name,
            "board_size": self.board_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "weights": self.weights,
            "prior_weights": self.prior_weights,
            "episodes_trained": self.episodes_trained,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        seed: int | None = None,
    ) -> "ValueAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            name=name or payload.get("name", "value_agent"),
            board_size=payload["board_size"],
            learning_rate=payload.get("learning_rate", 0.08),
            gamma=payload.get("gamma", 0.97),
            epsilon_start=payload.get("epsilon", 1.0),
            epsilon_end=payload.get("epsilon_end", 0.05),
            epsilon_decay=payload.get("epsilon_decay", 0.995),
            center_bias=1.0,
            blocking_bias=1.0,
            seed=seed,
        )
        agent.weights = list(payload.get("weights", agent.weights))
        agent.prior_weights = list(payload.get("prior_weights", agent.prior_weights))
        agent.episodes_trained = int(payload.get("episodes_trained", 0))
        agent._sanitize_weights()
        return agent

    def _predict(self, features: list[float]) -> float:
        score = sum(weight * feature for weight, feature in zip(self.weights, features))
        score += sum(weight * feature for weight, feature in zip(self.prior_weights, features))
        return score if math.isfinite(score) else 0.0

    def _sanitize_weights(self) -> None:
        for index, weight in enumerate(self.weights):
            if not math.isfinite(weight):
                self.weights[index] = 0.0
