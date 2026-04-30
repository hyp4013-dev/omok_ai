"""Torch-based hybrid Gomoku agent with a small policy decision mix."""

from __future__ import annotations

import torch

from env.gomoku_env import Action, GomokuEnv
from utils.tactical_rules import find_forced_action

from agent.torch_hybrid_agent import TorchHybridAgent, TorchHybridStepRecord


class TorchHybridMixAgent(TorchHybridAgent):
    """Hybrid agent that keeps value frozen but lets policy slightly steer action selection."""

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
        mix_weight = max(0.0, min(1.0, float(self.policy_mix_weight)))

        if mix_weight > 0.0:
            policy_probs = self._policy_action_probabilities(env, valid_actions)
            value_probs = torch.softmax(value_scores, dim=0)
            mixed_probs = (1.0 - mix_weight) * value_probs + mix_weight * policy_probs
            chosen_action_index = int(torch.argmax(mixed_probs).item())
            selection_reason = "mixed_selection"
            ranked_scores = mixed_probs
        else:
            chosen_action_index = int(torch.argmax(value_scores).item())
            selection_reason = "value_selection"
            ranked_scores = value_scores

        allow_random_exploration = (
            training
            and (not self.freeze_value)
            and env.move_count >= self.quick_loss_move_threshold
        )
        if allow_random_exploration and self.random.random() < self.epsilon:
            ranked_indices = sorted(
                range(len(valid_actions)),
                key=lambda index: ranked_scores[index].item(),
                reverse=True,
            )
            exploration_pool = ranked_indices[1:10] or ranked_indices[:1]
            chosen_action_index = self.random.choice(exploration_pool)
            selection_reason = "random"

        chosen_action = valid_actions[chosen_action_index]
        policy_log_prob = self._policy_log_prob_for_action(env, chosen_action, valid_actions)
        return chosen_action, TorchHybridStepRecord(
            board_tensor,
            chosen_action_index,
            policy_log_prob,
            selection_reason=selection_reason,
        )

    def _policy_action_probabilities(
        self,
        env: GomokuEnv,
        valid_actions: list[Action],
    ) -> torch.Tensor:
        valid_action_indices = [env.action_to_index(candidate) for candidate in valid_actions]
        board_tensor = self._board_tensor(env)
        self.policy_model.eval()
        with torch.no_grad():
            logits = self.policy_model(self._tensorize(board_tensor)).squeeze(0)
        masked_logits = logits[valid_action_indices]
        return torch.softmax(masked_logits, dim=0)
