"""A minimal policy-gradient Gomoku agent."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from env.gomoku_env import GomokuEnv
from utils.state_encoder import action_features, policy_state_features
from utils.tactical_rules import find_forced_action


@dataclass
class PolicyStepRecord:
    state_features: list[float]
    action_features: list[float]
    valid_action_features: list[list[float]]
    chosen_index: int


class PolicyGradientAgent:
    def __init__(
        self,
        name: str,
        board_size: int,
        learning_rate: float = 0.03,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.992,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature
        self.random = random.Random(seed)
        self.action_weights = [0.0] * 11
        self.state_weights = [0.0] * 9
        self.action_weights[1] = 0.2
        self.action_weights[3] = 0.4
        self.action_weights[4] = 0.8
        self.action_weights[5] = 1.1
        self.action_weights[7] = 0.55
        self.action_weights[8] = 1.25
        self.episodes_trained = 0

    def select_action(self, env: GomokuEnv, training: bool = True) -> tuple[tuple[int, int], PolicyStepRecord]:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("no valid actions available")

        state_features = policy_state_features(env.board, env.current_player)
        valid_action_features = [
            action_features(env.board, env.current_player, action) for action in valid_actions
        ]
        forced_action = find_forced_action(env)
        if forced_action is not None:
            chosen_index = valid_actions.index(forced_action)
            return valid_actions[chosen_index], PolicyStepRecord(
                state_features=state_features,
                action_features=valid_action_features[chosen_index],
                valid_action_features=valid_action_features,
                chosen_index=chosen_index,
            )

        if training and self.random.random() < self.epsilon:
            chosen_index = self.random.randrange(len(valid_actions))
        else:
            probabilities = self._policy(valid_action_features, state_features)
            chosen_index = self._sample_index(probabilities)

        return valid_actions[chosen_index], PolicyStepRecord(
            state_features=state_features,
            action_features=valid_action_features[chosen_index],
            valid_action_features=valid_action_features,
            chosen_index=chosen_index,
        )

    def finish_game(self, episode_records: list[PolicyStepRecord], outcome_reward: float) -> None:
        if not episode_records:
            self._decay_epsilon()
            return

        returns = [outcome_reward * (self.gamma ** steps_from_end) for steps_from_end in range(len(episode_records))]
        returns.reverse()

        mean_return = sum(returns) / len(returns)
        normalized_returns = [value - mean_return for value in returns]
        if all(abs(value) < 1e-9 for value in normalized_returns):
            normalized_returns = returns

        for step_record, advantage in zip(episode_records, normalized_returns):
            probabilities = self._policy(step_record.valid_action_features, step_record.state_features)
            expected_action = self._expected_action_features(
                step_record.valid_action_features,
                probabilities,
            )
            for index, feature in enumerate(step_record.action_features):
                gradient = feature - expected_action[index]
                self.action_weights[index] += self.learning_rate * advantage * gradient

            for index, feature in enumerate(step_record.state_features):
                self.state_weights[index] += self.learning_rate * advantage * 0.1 * feature

        self._decay_epsilon()

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "board_size": self.board_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "temperature": self.temperature,
            "action_weights": self.action_weights,
            "state_weights": self.state_weights,
            "episodes_trained": self.episodes_trained,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _policy(self, valid_action_features: list[list[float]], state_features: list[float]) -> list[float]:
        state_bonus = self._dot(self.state_weights, state_features)
        logits = []
        for features in valid_action_features:
            logits.append((self._dot(self.action_weights, features) + state_bonus) / self.temperature)

        max_logit = max(logits)
        exp_values = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exp_values)
        return [value / total for value in exp_values]

    def _sample_index(self, probabilities: list[float]) -> int:
        roll = self.random.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if roll <= cumulative:
                return index
        return len(probabilities) - 1

    def _expected_action_features(
        self,
        valid_action_features: list[list[float]],
        probabilities: list[float],
    ) -> list[float]:
        expected = [0.0] * len(valid_action_features[0])
        for probability, features in zip(probabilities, valid_action_features):
            for index, feature in enumerate(features):
                expected[index] += probability * feature
        return expected

    def _dot(self, left: list[float], right: list[float]) -> float:
        return sum(lhs * rhs for lhs, rhs in zip(left, right))

    def _decay_epsilon(self) -> None:
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
