"""Torch-based CNN policy agent for Gomoku."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from env.gomoku_env import Action, GomokuEnv
from utils.tactical_rules import find_forced_action


@dataclass
class TorchPolicyStepRecord:
    board_tensor: list[list[list[float]]]
    chosen_action_index: int
    log_prob: torch.Tensor


class BoardPolicyCNN(nn.Module):
    def __init__(self, board_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, board_size * board_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TorchPolicyAgent:
    def __init__(
        self,
        name: str,
        board_size: int,
        learning_rate: float = 2e-4,
        gamma: float = 0.99,
        epsilon_start: float = 0.20,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.998,
        quick_loss_move_threshold: int = 15,
        quick_loss_lr_scale: float = 2.0,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.name = name
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.quick_loss_move_threshold = quick_loss_move_threshold
        self.quick_loss_lr_scale = quick_loss_lr_scale
        self.random = random.Random(seed)
        self.device = torch.device(device)
        self.model = BoardPolicyCNN(board_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.episodes_trained = 0
        self.last_effective_learning_rate = self.learning_rate

    def select_action(
        self,
        env: GomokuEnv,
        training: bool = True,
    ) -> tuple[Action, TorchPolicyStepRecord]:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("no valid actions available")

        board_tensor = self._board_tensor(env)
        if training and env.move_count == 0:
            opening_pool = self._central_opening_pool(valid_actions, env.board_size)
            chosen_action = self.random.choice(opening_pool)
            chosen_action_index = env.action_to_index(chosen_action)
            log_prob = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            return chosen_action, TorchPolicyStepRecord(
                board_tensor=board_tensor,
                chosen_action_index=chosen_action_index,
                log_prob=log_prob,
            )

        forced_action = find_forced_action(env)
        if forced_action is not None:
            chosen_action_index = env.action_to_index(forced_action)
            log_prob = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            return forced_action, TorchPolicyStepRecord(
                board_tensor=board_tensor,
                chosen_action_index=chosen_action_index,
                log_prob=log_prob,
            )

        logits = self.model(self._tensorize(board_tensor)).squeeze(0)
        valid_action_indices = [env.action_to_index(action) for action in valid_actions]
        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[valid_action_indices] = logits[valid_action_indices]
        probabilities = torch.softmax(masked_logits, dim=0)

        if training and self.random.random() < self.epsilon:
            chosen_action = self.random.choice(valid_actions)
            chosen_action_index = env.action_to_index(chosen_action)
        elif training:
            chosen_action_index = int(torch.multinomial(probabilities, num_samples=1).item())
            chosen_action = env.index_to_action(chosen_action_index)
        else:
            chosen_action_index = int(torch.argmax(probabilities).item())
            chosen_action = env.index_to_action(chosen_action_index)

        log_prob = torch.log(probabilities[chosen_action_index] + 1e-12)
        return chosen_action, TorchPolicyStepRecord(
            board_tensor=board_tensor,
            chosen_action_index=chosen_action_index,
            log_prob=log_prob,
        )

    def _central_opening_pool(
        self,
        valid_actions: list[tuple[int, int]],
        board_size: int,
    ) -> list[tuple[int, int]]:
        opening_span = min(10, board_size)
        opening_offset = (board_size - opening_span) // 2
        opening_limit = opening_offset + opening_span - 1
        return [
            action
            for action in valid_actions
            if opening_offset <= action[0] <= opening_limit and opening_offset <= action[1] <= opening_limit
        ]

    def finish_game(
        self,
        episode_records: list[TorchPolicyStepRecord],
        outcome_reward: float,
        game_length: int,
        lr_multiplier: float = 1.0,
    ) -> None:
        if not episode_records:
            self.last_effective_learning_rate = self.learning_rate
            self._decay_epsilon()
            return

        returns = []
        discounted = outcome_reward
        for _ in reversed(episode_records):
            returns.append(discounted)
            discounted *= self.gamma
        returns.reverse()

        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-6)

        self.model.train()
        self.optimizer.zero_grad()
        losses = [-(record.log_prob * advantage) for record, advantage in zip(episode_records, returns_tensor)]
        torch.stack(losses).sum().backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

        effective_learning_rate = self._effective_learning_rate(outcome_reward, game_length) * lr_multiplier
        original_learning_rates = [group["lr"] for group in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group["lr"] = effective_learning_rate
        self.optimizer.step()
        for group, original_learning_rate in zip(self.optimizer.param_groups, original_learning_rates):
            group["lr"] = original_learning_rate
        self.last_effective_learning_rate = effective_learning_rate
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
            "quick_loss_move_threshold": self.quick_loss_move_threshold,
            "quick_loss_lr_scale": self.quick_loss_lr_scale,
            "episodes_trained": self.episodes_trained,
            "last_effective_learning_rate": self.last_effective_learning_rate,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None, device: str = "cpu") -> "TorchPolicyAgent":
        payload = torch.load(Path(path), map_location=device)
        agent = cls(
            name=name or payload.get("name", "torch_policy_agent"),
            board_size=payload["board_size"],
            learning_rate=payload.get("learning_rate", 2e-4),
            gamma=payload.get("gamma", 0.99),
            epsilon_start=payload.get("epsilon", 0.20),
            epsilon_end=payload.get("epsilon_end", 0.02),
            epsilon_decay=payload.get("epsilon_decay", 0.998),
            quick_loss_move_threshold=payload.get("quick_loss_move_threshold", 15),
            quick_loss_lr_scale=payload.get("quick_loss_lr_scale", 2.0),
            device=device,
        )
        agent.model.load_state_dict(payload["model_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.episodes_trained = int(payload.get("episodes_trained", 0))
        agent.last_effective_learning_rate = float(
            payload.get("last_effective_learning_rate", agent.learning_rate)
        )
        return agent

    def _effective_learning_rate(self, outcome_reward: float, game_length: int) -> float:
        multiplier = 1.0
        if outcome_reward < 0 and game_length <= self.quick_loss_move_threshold:
            multiplier *= self.quick_loss_lr_scale
        return self.learning_rate * multiplier

    def _board_tensor(self, env: GomokuEnv) -> list[list[list[float]]]:
        own = []
        opp = []
        bias = []
        player = env.current_player
        center = (env.board_size - 1) / 2.0
        for row_idx in range(env.board_size):
            own_row = []
            opp_row = []
            bias_row = []
            for col_idx in range(env.board_size):
                cell = env.board[row_idx][col_idx]
                own_row.append(1.0 if cell == player else 0.0)
                opp_row.append(1.0 if cell == -player else 0.0)
                distance = (abs(row_idx - center) + abs(col_idx - center)) / max(1.0, center * 2.0)
                bias_row.append(1.0 - distance)
            own.append(own_row)
            opp.append(opp_row)
            bias.append(bias_row)
        return [own, opp, bias]

    def _tensorize(self, board_tensor: list[list[list[float]]]) -> torch.Tensor:
        return torch.tensor([board_tensor], dtype=torch.float32, device=self.device)

    def _decay_epsilon(self) -> None:
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
