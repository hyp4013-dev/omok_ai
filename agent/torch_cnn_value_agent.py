"""CNN-based value agent for board-state evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from env.gomoku_env import GomokuEnv
from utils.tactical_rules import find_forced_action


@dataclass
class TorchCNNValueStepRecord:
    board_tensor: list[list[list[float]]]
    reference_board_tensor: list[list[list[float]]] | None = None
    teacher_board_tensor: list[list[list[float]]] | None = None
    selection_reason: str = "model"


class BoardValueCNN(nn.Module):
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
            nn.Linear(64 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


class TorchCNNValueAgent:
    def __init__(
        self,
        name: str,
        board_size: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.97,
        imitation_weight: float = 1.0,
        win_short_bonus: float = 0.75,
        loss_long_bonus: float = 0.75,
        quick_win_move_threshold: int = 15,
        quick_win_lr_scale: float = 0.5,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.998,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.name = name
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.imitation_weight = imitation_weight
        self.win_short_bonus = win_short_bonus
        self.loss_long_bonus = loss_long_bonus
        self.quick_win_move_threshold = quick_win_move_threshold
        self.quick_win_lr_scale = quick_win_lr_scale
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.random = random.Random(seed)
        self.device = torch.device(device)
        self.model = BoardValueCNN(board_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.episodes_trained = 0
        self.last_effective_learning_rate = self.learning_rate

    def select_action(
        self,
        env: GomokuEnv,
        training: bool = True,
    ) -> tuple[tuple[int, int], TorchCNNValueStepRecord]:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("no valid actions available")

        if training and env.move_count == 0:
            opening_pool = self._central_opening_pool(valid_actions, env.board_size)
            chosen_action = self.random.choice(opening_pool)
            chosen_index = valid_actions.index(chosen_action)
            board_tensor = self._board_tensor_after_action(env, chosen_action)
            return chosen_action, TorchCNNValueStepRecord(
                board_tensor=board_tensor,
                selection_reason="opening_random",
            )

        forced_action = find_forced_action(env)
        if forced_action is not None:
            board_tensor = self._board_tensor_after_action(env, forced_action)
            return forced_action, TorchCNNValueStepRecord(
                board_tensor=board_tensor,
                selection_reason="forced",
            )

        board_tensors = [self._board_tensor_after_action(env, action) for action in valid_actions]
        scores = self._predict_scores(board_tensors)
        allow_random_exploration = env.move_count >= self.quick_win_move_threshold
        if training and allow_random_exploration and self.random.random() < self.epsilon:
            ranked_indices = sorted(range(len(valid_actions)), key=lambda index: scores[index], reverse=True)
            exploration_pool = ranked_indices[1:10] or ranked_indices[:1]
            chosen_index = self.random.choice(exploration_pool)
            selection_reason = "random"
        else:
            chosen_index = max(range(len(valid_actions)), key=lambda index: scores[index])
            selection_reason = "model"

        return valid_actions[chosen_index], TorchCNNValueStepRecord(
            board_tensor=board_tensors[chosen_index],
            selection_reason=selection_reason,
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
        episode_records: list[TorchCNNValueStepRecord],
        outcome_reward: float,
        game_length: int,
        lr_multiplier: float = 1.0,
        teacher_weight: float = 0.0,
    ) -> None:
        if episode_records:
            self.model.train()
            self.optimizer.zero_grad()
            losses = []
            for move_offset, record in enumerate(reversed(episode_records)):
                target_value = outcome_reward * (self.gamma ** move_offset)
                board_tensor = self._tensorize(record.board_tensor)
                target_tensor = torch.tensor([target_value], dtype=torch.float32, device=self.device)
                prediction = self.model(board_tensor)
                loss = self.loss_fn(prediction, target_tensor)
                if record.reference_board_tensor is not None:
                    reference_tensor = self._tensorize(record.reference_board_tensor)
                    reference_prediction = self.model(reference_tensor).detach()
                    loss = loss + (self.imitation_weight * self.loss_fn(prediction, reference_prediction))
                if record.teacher_board_tensor is not None and teacher_weight > 0.0:
                    teacher_tensor = self._tensorize(record.teacher_board_tensor)
                    teacher_target = torch.tensor([1.0], dtype=torch.float32, device=self.device)
                    teacher_prediction = self.model(teacher_tensor)
                    loss = loss + (teacher_weight * self.loss_fn(teacher_prediction, teacher_target))
                losses.append(loss)

            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            effective_learning_rate = self._effective_learning_rate(outcome_reward, game_length) * lr_multiplier
            original_learning_rates = [group["lr"] for group in self.optimizer.param_groups]
            for group in self.optimizer.param_groups:
                group["lr"] = effective_learning_rate
            self.optimizer.step()
            for group, original_learning_rate in zip(self.optimizer.param_groups, original_learning_rates):
                group["lr"] = original_learning_rate
            self.last_effective_learning_rate = effective_learning_rate
        else:
            self.last_effective_learning_rate = self.learning_rate

        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def supervised_update(
        self,
        positive_board_tensors: list[list[list[list[float]]]],
        negative_board_tensors: list[list[list[list[float]]]],
    ) -> float:
        if not positive_board_tensors and not negative_board_tensors:
            return 0.0

        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        if positive_board_tensors:
            positive_batch = torch.tensor(positive_board_tensors, dtype=torch.float32, device=self.device)
            positive_targets = torch.ones(len(positive_board_tensors), dtype=torch.float32, device=self.device)
            losses.append(self.loss_fn(self.model(positive_batch), positive_targets))

        if negative_board_tensors:
            negative_batch = torch.tensor(negative_board_tensors, dtype=torch.float32, device=self.device)
            negative_targets = -torch.ones(len(negative_board_tensors), dtype=torch.float32, device=self.device)
            losses.append(self.loss_fn(self.model(negative_batch), negative_targets))

        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "board_size": self.board_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "imitation_weight": self.imitation_weight,
            "win_short_bonus": self.win_short_bonus,
            "loss_long_bonus": self.loss_long_bonus,
            "quick_win_move_threshold": self.quick_win_move_threshold,
            "quick_win_lr_scale": self.quick_win_lr_scale,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "episodes_trained": self.episodes_trained,
            "last_effective_learning_rate": self.last_effective_learning_rate,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None, device: str = "cpu") -> "TorchCNNValueAgent":
        payload = torch.load(Path(path), map_location=device)
        agent = cls(
            name=name or payload.get("name", "torch_cnn_value_agent"),
            board_size=payload["board_size"],
            learning_rate=payload.get("learning_rate", 1e-4),
            gamma=payload.get("gamma", 0.97),
            imitation_weight=payload.get("imitation_weight", 1.0),
            win_short_bonus=payload.get("win_short_bonus", 0.75),
            loss_long_bonus=payload.get("loss_long_bonus", 0.75),
            quick_win_move_threshold=payload.get("quick_win_move_threshold", 15),
            quick_win_lr_scale=payload.get("quick_win_lr_scale", 0.5),
            epsilon_start=payload.get("epsilon", 1.0),
            epsilon_end=payload.get("epsilon_end", 0.05),
            epsilon_decay=payload.get("epsilon_decay", 0.998),
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
        if outcome_reward > 0:
            if game_length <= self.quick_win_move_threshold:
                multiplier *= self.quick_win_lr_scale
        return self.learning_rate * multiplier

    def _board_tensor_after_action(self, env: GomokuEnv, action: tuple[int, int]) -> list[list[list[float]]]:
        row, col = action
        player = env.current_player
        next_board = [board_row[:] for board_row in env.board]
        next_board[row][col] = player
        own = []
        opp = []
        bias = []
        center = (env.board_size - 1) / 2.0
        for board_row_idx in range(env.board_size):
            own_row = []
            opp_row = []
            bias_row = []
            for board_col_idx in range(env.board_size):
                cell = next_board[board_row_idx][board_col_idx]
                own_row.append(1.0 if cell == player else 0.0)
                opp_row.append(1.0 if cell == -player else 0.0)
                distance = abs(board_row_idx - center) + abs(board_col_idx - center)
                bias_row.append(1.0 - (distance / max(1.0, center * 2.0)))
            own.append(own_row)
            opp.append(opp_row)
            bias.append(bias_row)
        return [own, opp, bias]

    def _predict_scores(self, board_tensors: list[list[list[list[float]]]]) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            batch = torch.tensor(board_tensors, dtype=torch.float32, device=self.device)
            return self.model(batch).detach().cpu().tolist()

    def _tensorize(self, board_tensor: list[list[list[float]]]) -> torch.Tensor:
        return torch.tensor([board_tensor], dtype=torch.float32, device=self.device)
