"""Torch-based policy-only CNN Gomoku agent."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from env.gomoku_env import Action, GomokuEnv
from utils.tactical_rules import find_forced_action


@dataclass
class TorchPolicyOnlyStepRecord:
    board_tensor: list[list[list[float]]]
    valid_action_indices: list[int]
    chosen_action_index: int
    log_prob: torch.Tensor | None
    teacher_board_tensor: list[list[list[float]]] | None = None
    teacher_action_index: int | None = None
    selection_reason: str = "policy"
    policy_entropy: float | None = None
    policy_top1_correct: bool | None = None
    policy_top3_correct: bool | None = None
    policy_top5_correct: bool | None = None
    policy_target_logit: float | None = None
    policy_max_logit: float | None = None
    policy_logit_gap: float | None = None
    policy_target_prob: float | None = None
    policy_max_prob: float | None = None
    policy_target_rank: int | None = None
    policy_valid_action_count: int | None = None
    policy_logits_finite: bool | None = None
    policy_probabilities_finite: bool | None = None


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
            nn.Linear(64 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, board_size * board_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TorchPolicyOnlyAgent:
    def __init__(
        self,
        name: str,
        board_size: int,
        learning_rate: float = 2e-4,
        gamma: float = 0.99,
        teacher_weight: float = 1.0,
        teacher_aux_weight: float = 1.0,
        epsilon_start: float = 0.20,
        epsilon_end: float = 0.0,
        epsilon_decay: float = 0.998,
        greedy_move_threshold: int = 30,
        quick_loss_move_threshold: int = 15,
        quick_loss_lr_scale: float = 2.0,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.name = name
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.teacher_weight = teacher_weight
        self.teacher_aux_weight = teacher_aux_weight
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.greedy_move_threshold = greedy_move_threshold
        self.quick_loss_move_threshold = quick_loss_move_threshold
        self.quick_loss_lr_scale = quick_loss_lr_scale
        self.random = random.Random(seed)
        self.device = torch.device(device)
        self.model = BoardPolicyCNN(board_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.episodes_trained = 0
        self.last_effective_learning_rate = self.learning_rate
        self.last_game_metrics: dict[str, object] = {}

    @classmethod
    def create_blank(
        cls,
        name: str,
        board_size: int,
        *,
        learning_rate: float = 2e-4,
        gamma: float = 0.99,
        teacher_weight: float = 1.0,
        teacher_aux_weight: float = 1.0,
        epsilon_start: float = 0.20,
        epsilon_end: float = 0.0,
        epsilon_decay: float = 0.998,
        greedy_move_threshold: int = 30,
        quick_loss_move_threshold: int = 15,
        quick_loss_lr_scale: float = 2.0,
        seed: int | None = None,
        device: str = "cpu",
    ) -> "TorchPolicyOnlyAgent":
        return cls(
            name=name,
            board_size=board_size,
            learning_rate=learning_rate,
            gamma=gamma,
            teacher_weight=teacher_weight,
            teacher_aux_weight=teacher_aux_weight,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            greedy_move_threshold=greedy_move_threshold,
            quick_loss_move_threshold=quick_loss_move_threshold,
            quick_loss_lr_scale=quick_loss_lr_scale,
            seed=seed,
            device=device,
        )

    def select_action(
        self,
        env: GomokuEnv,
        training: bool = True,
        teacher_action_index: int | None = None,
    ) -> tuple[Action, TorchPolicyOnlyStepRecord]:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("no valid actions available")

        board_tensor = self._board_tensor(env)
        valid_action_indices = [env.action_to_index(action) for action in valid_actions]
        if training and env.move_count == 0:
            opening_pool = self._central_opening_pool(valid_actions, env.board_size)
            chosen_action = self.random.choice(opening_pool)
            chosen_action_index = valid_actions.index(chosen_action)
            log_prob = self._policy_log_prob(board_tensor, valid_action_indices, chosen_action_index)
            return chosen_action, TorchPolicyOnlyStepRecord(
                board_tensor=board_tensor,
                teacher_board_tensor=None,
                valid_action_indices=valid_action_indices,
                chosen_action_index=chosen_action_index,
                log_prob=log_prob,
                teacher_action_index=teacher_action_index,
                selection_reason="opening_random",
            )

        forced_action = find_forced_action(env)
        if forced_action is not None:
            chosen_action_index = valid_actions.index(forced_action)
            log_prob = self._policy_log_prob(board_tensor, valid_action_indices, chosen_action_index)
            return forced_action, TorchPolicyOnlyStepRecord(
                board_tensor=board_tensor,
                teacher_board_tensor=None,
                valid_action_indices=valid_action_indices,
                chosen_action_index=chosen_action_index,
                log_prob=log_prob,
                teacher_action_index=teacher_action_index,
                selection_reason="forced",
            )

        logits = self.model(self._tensorize(board_tensor)).squeeze(0)
        masked_logits = self._masked_logits(logits, valid_action_indices)
        probabilities = torch.softmax(masked_logits, dim=0)
        selection_metrics = self._selection_metrics(probabilities, valid_action_indices, teacher_action_index)

        if training and self.random.random() < self.epsilon:
            chosen_index = self.random.randrange(len(valid_actions))
            selection_reason = "random"
        elif training and env.move_count < self.greedy_move_threshold:
            chosen_index = int(torch.multinomial(probabilities, num_samples=1).item())
            selection_reason = "policy_sample"
        else:
            chosen_index = int(torch.argmax(probabilities).item())
            selection_reason = "policy_greedy"

        chosen_action = valid_actions[chosen_index]
        log_prob = torch.log(probabilities[chosen_index] + 1e-12)
        return chosen_action, TorchPolicyOnlyStepRecord(
            board_tensor=board_tensor,
            teacher_board_tensor=None,
            valid_action_indices=valid_action_indices,
            chosen_action_index=chosen_index,
            log_prob=log_prob,
            teacher_action_index=teacher_action_index,
            selection_reason=selection_reason,
            policy_entropy=selection_metrics["policy_entropy"],
            policy_top1_correct=selection_metrics.get("policy_top1_correct"),
            policy_top3_correct=selection_metrics.get("policy_top3_correct"),
            policy_top5_correct=selection_metrics.get("policy_top5_correct"),
        )

    def build_teacher_forced_record(self, env: GomokuEnv, teacher_action: Action) -> TorchPolicyOnlyStepRecord:
        board_tensor = self._board_tensor(env)
        valid_actions = env.get_valid_actions()
        valid_action_indices = [env.action_to_index(action) for action in valid_actions]
        teacher_action_index = env.action_to_index(teacher_action)
        chosen_action_index = valid_action_indices.index(teacher_action_index)
        logits = self.model(self._tensorize(board_tensor)).squeeze(0)
        masked_logits = self._masked_logits(logits, valid_action_indices)
        probabilities = torch.softmax(masked_logits, dim=0)
        selection_metrics = self._selection_metrics(probabilities, valid_action_indices, teacher_action_index)
        return TorchPolicyOnlyStepRecord(
            board_tensor=board_tensor,
            teacher_board_tensor=None,
            valid_action_indices=valid_action_indices,
            chosen_action_index=chosen_action_index,
            log_prob=None,
            teacher_action_index=teacher_action_index,
            selection_reason="teacher_forced",
            policy_entropy=selection_metrics["policy_entropy"],
            policy_top1_correct=selection_metrics.get("policy_top1_correct"),
            policy_top3_correct=selection_metrics.get("policy_top3_correct"),
            policy_top5_correct=selection_metrics.get("policy_top5_correct"),
        )

    def finish_game(
        self,
        episode_records: list[TorchPolicyOnlyStepRecord],
        outcome_reward: float,
        game_length: int,
        teacher_weight: float | None = None,
        lr_multiplier: float = 1.0,
    ) -> None:
        if not episode_records:
            self.last_effective_learning_rate = self.learning_rate
            self.last_game_metrics = {
                "policy_loss": 0.0,
                "policy_loss_max": 0.0,
                "teacher_aux_loss": 0.0,
                "teacher_aux_loss_max": 0.0,
                "total_loss": 0.0,
                "policy_entropy": 0.0,
                "teacher_eval_steps": 0,
                "teacher_top1_correct": 0,
                "teacher_top3_correct": 0,
                "teacher_top5_correct": 0,
                "debug_eval_steps": 0,
                "debug_target_logit_avg": 0.0,
                "debug_target_logit_min": 0.0,
                "debug_max_logit_avg": 0.0,
                "debug_logit_gap_avg": 0.0,
                "debug_logit_gap_max": 0.0,
                "debug_target_prob_avg": 0.0,
                "debug_max_prob_avg": 0.0,
                "debug_target_rank_avg": 0.0,
                "debug_valid_action_count_avg": 0.0,
                "debug_finite_steps": 0,
                "selection_counts": {},
                "effective_learning_rate": self.last_effective_learning_rate,
            }
            self._decay_epsilon()
            return self.last_game_metrics

        teacher_weight = self.teacher_weight if teacher_weight is None else teacher_weight
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        policy_losses = []
        policy_loss_values = []
        aux_losses = []
        aux_loss_values = []
        entropy_values = []
        target_logit_values = []
        max_logit_values = []
        logit_gap_values = []
        target_prob_values = []
        max_prob_values = []
        target_rank_values = []
        valid_action_count_values = []
        finite_debug_steps = 0
        selection_counts: dict[str, int] = {}
        teacher_eval_steps = 0
        teacher_top1_correct = 0
        teacher_top3_correct = 0
        teacher_top5_correct = 0
        for record in episode_records:
            selection_counts[record.selection_reason] = selection_counts.get(record.selection_reason, 0) + 1
            if record.policy_entropy is not None:
                entropy_values.append(record.policy_entropy)
            if (
                record.teacher_action_index is not None
                and record.policy_top1_correct is not None
                and record.policy_top3_correct is not None
                and record.policy_top5_correct is not None
            ):
                teacher_eval_steps += 1
                teacher_top1_correct += int(record.policy_top1_correct)
                teacher_top3_correct += int(record.policy_top3_correct)
                teacher_top5_correct += int(record.policy_top5_correct)
            if record.selection_reason == "forced":
                continue
            if record.teacher_action_index is None:
                continue
            target_index = record.valid_action_indices.index(record.teacher_action_index)
            logits = self.model(self._tensorize(record.board_tensor)).squeeze(0)
            masked_logits = self._masked_logits(logits, record.valid_action_indices)
            probabilities = torch.softmax(masked_logits, dim=0)
            target_prob = probabilities[target_index]
            max_prob = torch.max(probabilities)
            target_logit = masked_logits[target_index]
            max_logit = torch.max(masked_logits)
            policy_loss = teacher_weight * F.cross_entropy(
                masked_logits.unsqueeze(0),
                torch.tensor([target_index], dtype=torch.long, device=self.device),
            )
            total_loss = policy_loss
            policy_loss_value = float(policy_loss.detach().item())
            policy_losses.append(policy_loss_value)
            policy_loss_values.append(policy_loss_value)
            target_logit_values.append(float(target_logit.detach().item()))
            max_logit_values.append(float(max_logit.detach().item()))
            logit_gap_values.append(float((max_logit - target_logit).detach().item()))
            target_prob_values.append(float(target_prob.detach().item()))
            max_prob_values.append(float(max_prob.detach().item()))
            target_rank_values.append(int((probabilities > target_prob).sum().item()) + 1)
            valid_action_count_values.append(len(record.valid_action_indices))
            if torch.isfinite(masked_logits).all().item() and torch.isfinite(probabilities).all().item():
                finite_debug_steps += 1
            if record.teacher_board_tensor is not None and self.teacher_aux_weight > 0.0:
                prediction = self.model(self._tensorize(record.board_tensor)).squeeze(0)
                teacher_prediction = self.model(self._tensorize(record.teacher_board_tensor)).detach().squeeze(0)
                teacher_loss = self.loss_fn(prediction, teacher_prediction)
                aux_loss = self.teacher_aux_weight * teacher_loss
                aux_loss_value = float(aux_loss.detach().item())
                aux_losses.append(aux_loss_value)
                aux_loss_values.append(aux_loss_value)
                total_loss = total_loss + aux_loss
            losses.append(total_loss)

        if losses:
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

        self.last_game_metrics = {
            "policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
            "policy_loss_max": max(policy_loss_values) if policy_loss_values else 0.0,
            "teacher_aux_loss": sum(aux_losses) / max(1, len(aux_losses)),
            "teacher_aux_loss_max": max(aux_loss_values) if aux_loss_values else 0.0,
            "total_loss": sum(float(loss.detach().item()) for loss in losses) / max(1, len(losses)),
            "policy_entropy": sum(entropy_values) / max(1, len(entropy_values)),
            "teacher_eval_steps": teacher_eval_steps,
            "teacher_top1_correct": teacher_top1_correct,
            "teacher_top3_correct": teacher_top3_correct,
            "teacher_top5_correct": teacher_top5_correct,
            "debug_eval_steps": len(target_logit_values),
            "debug_target_logit_avg": sum(target_logit_values) / max(1, len(target_logit_values)),
            "debug_target_logit_min": min(target_logit_values) if target_logit_values else 0.0,
            "debug_max_logit_avg": sum(max_logit_values) / max(1, len(max_logit_values)),
            "debug_logit_gap_avg": sum(logit_gap_values) / max(1, len(logit_gap_values)),
            "debug_logit_gap_max": max(logit_gap_values) if logit_gap_values else 0.0,
            "debug_target_prob_avg": sum(target_prob_values) / max(1, len(target_prob_values)),
            "debug_max_prob_avg": sum(max_prob_values) / max(1, len(max_prob_values)),
            "debug_target_rank_avg": sum(target_rank_values) / max(1, len(target_rank_values)),
            "debug_valid_action_count_avg": sum(valid_action_count_values) / max(1, len(valid_action_count_values)),
            "debug_finite_steps": finite_debug_steps,
            "selection_counts": selection_counts,
            "effective_learning_rate": self.last_effective_learning_rate,
        }
        self._decay_epsilon()
        return self.last_game_metrics

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "board_size": self.board_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "teacher_weight": self.teacher_weight,
            "teacher_aux_weight": self.teacher_aux_weight,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "greedy_move_threshold": self.greedy_move_threshold,
            "quick_loss_move_threshold": self.quick_loss_move_threshold,
            "quick_loss_lr_scale": self.quick_loss_lr_scale,
            "episodes_trained": self.episodes_trained,
            "last_effective_learning_rate": self.last_effective_learning_rate,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None, device: str = "cpu") -> "TorchPolicyOnlyAgent":
        payload = torch.load(Path(path), map_location=device)
        agent = cls(
            name=name or payload.get("name", "torch_policy_only_agent"),
            board_size=payload["board_size"],
            learning_rate=payload.get("learning_rate", 2e-4),
            gamma=payload.get("gamma", 0.99),
            teacher_weight=payload.get("teacher_weight", 1.0),
            teacher_aux_weight=payload.get("teacher_aux_weight", 1.0),
            epsilon_start=payload.get("epsilon", 0.20),
            epsilon_end=payload.get("epsilon_end", 0.0),
            epsilon_decay=payload.get("epsilon_decay", 0.998),
            greedy_move_threshold=payload.get("greedy_move_threshold", 30),
            quick_loss_move_threshold=payload.get("quick_loss_move_threshold", 15),
            quick_loss_lr_scale=payload.get("quick_loss_lr_scale", 2.0),
            device=device,
        )
        agent.model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            try:
                agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
            except ValueError:
                pass
        agent.episodes_trained = int(payload.get("episodes_trained", 0))
        agent.last_effective_learning_rate = float(
            payload.get("last_effective_learning_rate", agent.learning_rate)
        )
        return agent

    @classmethod
    def load_from_value_checkpoint(
        cls,
        value_checkpoint_path: str | Path,
        *,
        name: str,
        device: str = "cpu",
    ) -> "TorchPolicyOnlyAgent":
        payload = torch.load(Path(value_checkpoint_path), map_location=device, weights_only=False)
        if "model_state_dict" in payload:
            value_state = payload["model_state_dict"]
            board_size = payload["board_size"]
        elif "value_model_state_dict" in payload:
            value_state = payload["value_model_state_dict"]
            board_size = payload["board_size"]
        else:
            value_state = payload["state_dict"]
            board_size = payload["board_size"]

        agent = cls(
            name=name,
            board_size=board_size,
            learning_rate=2e-4,
            gamma=payload.get("gamma", 0.99),
            teacher_weight=1.0,
            teacher_aux_weight=1.0,
            device=device,
        )
        agent._load_partial_policy_state(value_state)
        return agent

    @classmethod
    def load_with_feature_checkpoint(
        cls,
        feature_checkpoint_path: str | Path,
        *,
        name: str,
        device: str = "cpu",
    ) -> "TorchPolicyOnlyAgent":
        payload = torch.load(Path(feature_checkpoint_path), map_location=device, weights_only=False)
        if "policy_model_state_dict" in payload:
            feature_state = payload["policy_model_state_dict"]
            board_size = payload["board_size"]
        elif "model_state_dict" in payload:
            feature_state = payload["model_state_dict"]
            board_size = payload["board_size"]
        else:
            feature_state = payload["state_dict"]
            board_size = payload["board_size"]

        agent = cls.create_blank(
            name=name,
            board_size=board_size,
            learning_rate=payload.get("learning_rate", 2e-4),
            gamma=payload.get("gamma", 0.99),
            teacher_weight=payload.get("teacher_weight", 1.0),
            teacher_aux_weight=payload.get("teacher_aux_weight", 1.0),
            epsilon_start=payload.get("epsilon", 0.20),
            epsilon_end=payload.get("epsilon_end", 0.0),
            epsilon_decay=payload.get("epsilon_decay", 0.998),
            greedy_move_threshold=payload.get("greedy_move_threshold", 30),
            quick_loss_move_threshold=payload.get("quick_loss_move_threshold", 15),
            quick_loss_lr_scale=payload.get("quick_loss_lr_scale", 2.0),
            device=device,
        )
        agent._load_partial_policy_state(feature_state)
        return agent

    def _load_partial_policy_state(self, value_state: dict[str, torch.Tensor]) -> None:
        current_state = self.model.state_dict()
        mapped_state = dict(current_state)
        for key, tensor in value_state.items():
            if key in mapped_state and mapped_state[key].shape == tensor.shape:
                mapped_state[key] = tensor
        self.model.load_state_dict(mapped_state)

    def _policy_log_prob(
        self,
        board_tensor: list[list[list[float]]],
        valid_action_indices: list[int],
        chosen_action_index: int,
    ) -> torch.Tensor:
        logits = self.model(self._tensorize(board_tensor)).squeeze(0)
        masked_logits = self._masked_logits(logits, valid_action_indices)
        probabilities = torch.softmax(masked_logits, dim=0)
        return torch.log(probabilities[chosen_action_index] + 1e-12)

    def _masked_logits(self, logits: torch.Tensor, valid_action_indices: list[int]) -> torch.Tensor:
        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[valid_action_indices] = logits[valid_action_indices]
        return masked_logits[valid_action_indices]

    def _selection_metrics(
        self,
        probabilities: torch.Tensor,
        valid_action_indices: list[int],
        teacher_action_index: int | None = None,
    ) -> dict[str, float | bool | int]:
        metrics: dict[str, float | bool | int] = {
            "policy_entropy": float((-(probabilities * torch.log(probabilities + 1e-12)).sum()).item()),
        }
        if teacher_action_index is None:
            return metrics

        teacher_valid_index = valid_action_indices.index(teacher_action_index)
        topk_count = min(5, probabilities.numel())
        if topk_count <= 0:
            return metrics
        topk_indices = torch.topk(probabilities, k=topk_count).indices.tolist()
        metrics["policy_top1_correct"] = teacher_valid_index == topk_indices[0]
        metrics["policy_top3_correct"] = teacher_valid_index in topk_indices[: min(3, len(topk_indices))]
        metrics["policy_top5_correct"] = teacher_valid_index in topk_indices
        return metrics

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
        row, col = action
        player = env.current_player
        next_board = [board_row[:] for board_row in env.board]
        next_board[row][col] = player
        own = []
        opp = []
        bias = []
        center = (env.board_size - 1) / 2.0
        for row_idx in range(env.board_size):
            own_row = []
            opp_row = []
            bias_row = []
            for col_idx in range(env.board_size):
                cell = next_board[row_idx][col_idx]
                own_row.append(1.0 if cell == player else 0.0)
                opp_row.append(1.0 if cell == -player else 0.0)
                distance = (abs(row_idx - center) + abs(col_idx - center)) / max(1.0, center * 2.0)
                bias_row.append(1.0 - distance)
            own.append(own_row)
            opp.append(opp_row)
            bias.append(bias_row)
        return [own, opp, bias]

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
