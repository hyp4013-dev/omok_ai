"""Torch-based separated value + policy Gomoku agent."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from env.gomoku_env import Action, GomokuEnv
from utils.tactical_rules import find_forced_action


@dataclass
class TorchHybridStepRecord:
    board_tensor: list[list[list[float]]]
    chosen_action_index: int
    policy_log_prob: torch.Tensor
    selection_reason: str = "value_selection"


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


class TorchHybridAgent:
    def __init__(
        self,
        name: str,
        board_size: int,
        learning_rate: float = 2e-4,
        gamma: float = 0.99,
        value_loss_weight: float = 0.5,
        policy_loss_weight: float = 0.05,
        policy_mix_weight: float = 0.0,
        freeze_value: bool = True,
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
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.policy_mix_weight = policy_mix_weight
        self.freeze_value = freeze_value
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.quick_loss_move_threshold = quick_loss_move_threshold
        self.quick_loss_lr_scale = quick_loss_lr_scale
        self.random = random.Random(seed)
        self.device = torch.device(device)

        self.value_model = BoardValueCNN(board_size).to(self.device)
        self.policy_model = BoardPolicyCNN(board_size).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.MSELoss()
        self.policy_ce_loss_fn = nn.CrossEntropyLoss()
        self.episodes_trained = 0
        self.last_effective_learning_rate = self.learning_rate

        if self.freeze_value:
            for param in self.value_model.parameters():
                param.requires_grad = False

    def select_action(
        self,
        env: GomokuEnv,
        training: bool = True,
    ) -> tuple[Action, TorchHybridStepRecord]:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("no valid actions available")

        board_tensor = self._board_tensor(env)
        if training and env.move_count == 0:
            opening_pool = self._central_opening_pool(valid_actions, env.board_size)
            chosen_action = self.random.choice(opening_pool)
            chosen_action_index = valid_actions.index(chosen_action)
            policy_log_prob = self._policy_log_prob_for_action(env, chosen_action, valid_actions)
            return chosen_action, TorchHybridStepRecord(
                board_tensor,
                chosen_action_index,
                policy_log_prob,
                selection_reason="opening_random",
            )

        forced_action = find_forced_action(env)
        if forced_action is not None:
            chosen_action_index = valid_actions.index(forced_action)
            policy_log_prob = self._policy_log_prob_for_action(env, forced_action, valid_actions)
            return forced_action, TorchHybridStepRecord(
                board_tensor,
                chosen_action_index,
                policy_log_prob,
                selection_reason="forced",
            )

        value_scores = self._action_value_scores(env, valid_actions)

        allow_random_exploration = training and (not self.freeze_value) and env.move_count >= self.quick_loss_move_threshold
        if allow_random_exploration and self.random.random() < self.epsilon:
            ranked_indices = sorted(
                range(len(valid_actions)),
                key=lambda index: value_scores[index].item(),
                reverse=True,
            )
            exploration_pool = ranked_indices[1:10] or ranked_indices[:1]
            chosen_action_index = self.random.choice(exploration_pool)
            chosen_action = valid_actions[chosen_action_index]
            selection_reason = "random"
        else:
            chosen_action_index = int(torch.argmax(value_scores).item())
            chosen_action = valid_actions[chosen_action_index]
            selection_reason = "value_selection"

        policy_log_prob = self._policy_log_prob_for_action(env, chosen_action, valid_actions)
        return chosen_action, TorchHybridStepRecord(
            board_tensor,
            chosen_action_index,
            policy_log_prob,
            selection_reason=selection_reason,
        )

    def finish_game(
        self,
        episode_records: list[TorchHybridStepRecord],
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
        normalized_returns = returns_tensor
        if len(returns_tensor) > 1:
            normalized_returns = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-6)

        if not self.freeze_value:
            self.value_model.train()
            self.value_optimizer.zero_grad()
            value_losses = []
            for record, target_value in zip(episode_records, returns_tensor):
                board_tensor = self._tensorize(record.board_tensor)
                predicted_value = self.value_model(board_tensor)
                value_target = torch.tensor([target_value], dtype=torch.float32, device=self.device)
                value_losses.append(self.value_loss_fn(predicted_value, value_target))
            torch.stack(value_losses).sum().backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=2.0)

            effective_learning_rate = self._effective_learning_rate(outcome_reward, game_length) * lr_multiplier
            original_learning_rates = [group["lr"] for group in self.value_optimizer.param_groups]
            for group in self.value_optimizer.param_groups:
                group["lr"] = effective_learning_rate
            self.value_optimizer.step()
            for group, original_learning_rate in zip(self.value_optimizer.param_groups, original_learning_rates):
                group["lr"] = original_learning_rate
            self.last_effective_learning_rate = effective_learning_rate

        self.policy_model.train()
        self.policy_optimizer.zero_grad()
        policy_losses = []
        for record, advantage in zip(episode_records, normalized_returns):
            if record.selection_reason not in {"forced", "value_selection"}:
                continue
            policy_losses.append(-(record.policy_log_prob * advantage) * self.policy_loss_weight)

        if policy_losses:
            torch.stack(policy_losses).sum().backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=2.0)
            original_learning_rates = [group["lr"] for group in self.policy_optimizer.param_groups]
            effective_learning_rate = self.learning_rate * lr_multiplier
            for group in self.policy_optimizer.param_groups:
                group["lr"] = effective_learning_rate
            self.policy_optimizer.step()
            for group, original_learning_rate in zip(self.policy_optimizer.param_groups, original_learning_rates):
                group["lr"] = original_learning_rate
            self.last_effective_learning_rate = effective_learning_rate

        self._decay_epsilon()

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "board_size": self.board_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "value_loss_weight": self.value_loss_weight,
            "policy_loss_weight": self.policy_loss_weight,
            "policy_mix_weight": self.policy_mix_weight,
            "freeze_value": self.freeze_value,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "quick_loss_move_threshold": self.quick_loss_move_threshold,
            "quick_loss_lr_scale": self.quick_loss_lr_scale,
            "episodes_trained": self.episodes_trained,
            "last_effective_learning_rate": self.last_effective_learning_rate,
            "value_model_state_dict": self.value_model.state_dict(),
            "policy_model_state_dict": self.policy_model.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None, device: str = "cpu") -> "TorchHybridAgent":
        payload = torch.load(Path(path), map_location=device)
        agent = cls(
            name=name or payload.get("name", "torch_hybrid_agent"),
            board_size=payload["board_size"],
            learning_rate=payload.get("learning_rate", 2e-4),
            gamma=payload.get("gamma", 0.99),
            value_loss_weight=payload.get("value_loss_weight", 0.5),
            policy_loss_weight=payload.get("policy_loss_weight", payload.get("policy_aux_weight", 0.05)),
            policy_mix_weight=payload.get("policy_mix_weight", 0.0),
            freeze_value=payload.get("freeze_value", True),
            epsilon_start=payload.get("epsilon", 0.20),
            epsilon_end=payload.get("epsilon_end", 0.02),
            epsilon_decay=payload.get("epsilon_decay", 0.998),
            quick_loss_move_threshold=payload.get("quick_loss_move_threshold", 15),
            quick_loss_lr_scale=payload.get("quick_loss_lr_scale", 2.0),
            device=device,
        )
        if "value_model_state_dict" in payload:
            agent.value_model.load_state_dict(payload["value_model_state_dict"])
            if "policy_model_state_dict" in payload:
                agent.policy_model.load_state_dict(payload["policy_model_state_dict"])
            if "value_optimizer_state_dict" in payload:
                try:
                    agent.value_optimizer.load_state_dict(payload["value_optimizer_state_dict"])
                except ValueError:
                    pass
            if "policy_optimizer_state_dict" in payload:
                try:
                    agent.policy_optimizer.load_state_dict(payload["policy_optimizer_state_dict"])
                except ValueError:
                    pass
        else:
            model_state = payload["model_state_dict"]
            if any(key.startswith("policy_head.") or key.startswith("value_head.") for key in model_state):
                agent._load_from_shared_hybrid_state(model_state)
                optimizer_state = payload.get("optimizer_state_dict")
                if optimizer_state:
                    try:
                        agent.policy_optimizer.load_state_dict(optimizer_state)
                    except ValueError:
                        pass
            else:
                agent._load_from_value_model_state(model_state)
                optimizer_state = payload.get("optimizer_state_dict")
                if optimizer_state:
                    try:
                        agent.value_optimizer.load_state_dict(optimizer_state)
                    except ValueError:
                        pass
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

    def _action_value_scores(self, env: GomokuEnv, valid_actions: list[Action]) -> torch.Tensor:
        action_tensors = [self._board_tensor_after_action(env, action) for action in valid_actions]
        batch = torch.tensor(action_tensors, dtype=torch.float32, device=self.device)
        self.value_model.eval()
        with torch.no_grad():
            values = self.value_model(batch)
        return values.squeeze(-1)

    def _policy_log_prob_for_action(
        self,
        env: GomokuEnv,
        action: Action,
        valid_actions: list[Action],
    ) -> torch.Tensor:
        valid_action_indices = [env.action_to_index(candidate) for candidate in valid_actions]
        board_tensor = self._board_tensor(env)
        self.policy_model.eval()
        logits = self.policy_model(self._tensorize(board_tensor)).squeeze(0)
        masked_logits = logits[valid_action_indices]
        probabilities = torch.softmax(masked_logits, dim=0)
        chosen_index = valid_actions.index(action)
        return torch.log(probabilities[chosen_index] + 1e-12)

    def _masked_probabilities(self, logits: torch.Tensor, valid_action_indices: list[int]) -> torch.Tensor:
        valid_logits = logits[valid_action_indices]
        return torch.softmax(valid_logits, dim=0)

    def _central_opening_pool(
        self,
        valid_actions: list[Action],
        board_size: int,
    ) -> list[Action]:
        opening_span = min(10, board_size)
        opening_offset = (board_size - opening_span) // 2
        opening_limit = opening_offset + opening_span - 1
        return [
            action
            for action in valid_actions
            if opening_offset <= action[0] <= opening_limit and opening_offset <= action[1] <= opening_limit
        ]

    def _board_tensor_after_action(self, env: GomokuEnv, action: Action) -> list[list[list[float]]]:
        board = [row[:] for row in env.board]
        row, col = action
        board[row][col] = env.current_player
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
                cell = board[row_idx][col_idx]
                own_row.append(1.0 if cell == player else 0.0)
                opp_row.append(1.0 if cell == -player else 0.0)
                distance = (abs(row_idx - center) + abs(col_idx - center)) / max(1.0, center * 2.0)
                bias_row.append(1.0 - distance)
            own.append(own_row)
            opp.append(opp_row)
            bias.append(bias_row)
        return [own, opp, bias]

    def _load_from_value_model_state(self, value_state: dict[str, torch.Tensor]) -> None:
        current_state = self.value_model.state_dict()
        mapped_state = dict(current_state)
        for key, tensor in value_state.items():
            if key in mapped_state:
                mapped_state[key] = tensor
        self.value_model.load_state_dict(mapped_state)

    def _load_from_shared_hybrid_state(self, shared_state: dict[str, torch.Tensor]) -> None:
        value_state = {}
        policy_state = {}
        for key, tensor in shared_state.items():
            if key.startswith("features."):
                value_state[key] = tensor
                policy_state[key] = tensor
            elif key.startswith("value_head."):
                mapped_key = key.removeprefix("value_head.")
                if mapped_key.startswith("0."):
                    value_state[f"head.1.{mapped_key.removeprefix('0.')}"] = tensor
                elif mapped_key.startswith("2."):
                    value_state[f"head.3.{mapped_key.removeprefix('2.')}"] = tensor
            elif key.startswith("policy_head."):
                mapped_key = key.removeprefix("policy_head.")
                if mapped_key.startswith("0."):
                    policy_state[f"head.1.{mapped_key.removeprefix('0.')}"] = tensor
                elif mapped_key.startswith("2."):
                    policy_state[f"head.3.{mapped_key.removeprefix('2.')}"] = tensor
        if value_state:
            self.value_model.load_state_dict({**self.value_model.state_dict(), **value_state})
        if policy_state:
            self.policy_model.load_state_dict({**self.policy_model.state_dict(), **policy_state})

    def _decay_epsilon(self) -> None:
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
