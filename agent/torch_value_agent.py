"""Torch-based value agent trained against a frozen reference."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from env.gomoku_env import GomokuEnv
from utils.state_encoder import action_features
from utils.tactical_rules import find_forced_action


@dataclass
class TorchValueStepRecord:
    features: list[float]
    reference_features: list[float] | None = None


class ValueMLP(nn.Module):
    def __init__(self, input_dim: int = 11) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class TorchValueAgent:
    def __init__(
        self,
        name: str,
        board_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.97,
        imitation_weight: float = 0.4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.name = name
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.imitation_weight = imitation_weight
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.random = random.Random(seed)
        self.device = torch.device(device)
        self.model = ValueMLP().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.episodes_trained = 0

    def select_action(
        self,
        env: GomokuEnv,
        training: bool = True,
    ) -> tuple[tuple[int, int], TorchValueStepRecord]:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("no valid actions available")

        forced_action = find_forced_action(env)
        if forced_action is not None:
            return forced_action, TorchValueStepRecord(
                features=action_features(env.board, env.current_player, forced_action)
            )

        feature_rows = [action_features(env.board, env.current_player, action) for action in valid_actions]
        scores = self._predict_scores(feature_rows)
        if training and self.random.random() < self.epsilon:
            chosen_index = self.random.randrange(len(valid_actions))
        else:
            chosen_index = max(range(len(valid_actions)), key=lambda index: scores[index])

        return valid_actions[chosen_index], TorchValueStepRecord(features=feature_rows[chosen_index])

    def finish_game(self, episode_records: list[TorchValueStepRecord], outcome_reward: float) -> None:
        if episode_records:
            self.model.train()
            self.optimizer.zero_grad()
            losses = []
            for move_offset, record in enumerate(reversed(episode_records)):
                target_value = outcome_reward * (self.gamma ** move_offset)
                feature_tensor = torch.tensor(record.features, dtype=torch.float32, device=self.device).unsqueeze(0)
                target_tensor = torch.tensor([target_value], dtype=torch.float32, device=self.device)
                prediction = self.model(feature_tensor)
                loss = self.loss_fn(prediction, target_tensor)
                if record.reference_features is not None:
                    reference_tensor = torch.tensor(
                        record.reference_features,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)
                    reference_prediction = self.model(reference_tensor).detach()
                    loss = loss + (self.imitation_weight * self.loss_fn(prediction, reference_prediction))
                losses.append(loss)

            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "board_size": self.board_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "imitation_weight": self.imitation_weight,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "episodes_trained": self.episodes_trained,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None, device: str = "cpu") -> "TorchValueAgent":
        payload = torch.load(Path(path), map_location=device)
        agent = cls(
            name=name or payload.get("name", "torch_value_agent"),
            board_size=payload["board_size"],
            learning_rate=payload.get("learning_rate", 1e-3),
            gamma=payload.get("gamma", 0.97),
            imitation_weight=payload.get("imitation_weight", 0.4),
            epsilon_start=payload.get("epsilon", 1.0),
            epsilon_end=payload.get("epsilon_end", 0.05),
            epsilon_decay=payload.get("epsilon_decay", 0.997),
            device=device,
        )
        agent.model.load_state_dict(payload["model_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.episodes_trained = int(payload.get("episodes_trained", 0))
        return agent

    def _predict_scores(self, feature_rows: list[list[float]]) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            features = torch.tensor(feature_rows, dtype=torch.float32, device=self.device)
            return self.model(features).detach().cpu().tolist()
