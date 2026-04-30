"""Tests for the Gomoku environment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from agent.policy_gradient_agent import PolicyGradientAgent
from agent.tactical_rule_agent import (
    DefensiveHardTacticalRuleAgent,
    HardTacticalRuleAgent,
    NeutralHardTacticalRuleAgent,
    OffensiveHardTacticalRuleAgent,
    TacticalRuleAgent,
    build_random_hard_tactical_rule_agent,
)
from agent.torch_cnn_value_agent import TorchCNNValueAgent
from agent.torch_hybrid_mix_agent import TorchHybridMixAgent
from agent.torch_hybrid_agent import TorchHybridAgent
from agent.torch_policy_only_agent import TorchPolicyOnlyAgent
from agent.torch_value_agent import TorchValueAgent
from agent.value_agent import ValueAgent
from env.gomoku_env import GomokuEnv
from log_parser import build_board_state, parse_log_file, parse_log_text
from log_utils import find_latest_log
from play_random import simulate_game, simulate_games
from train_competitive import train_competitive, train_until_balanced
from train_hybrid_mix_reference import train_against_reference as train_hybrid_mix_against_reference
from train_hybrid_reference import train_against_reference as train_hybrid_against_reference
from train_policy_only_reference import train_against_reference as train_policy_only_against_reference
from train_policy_only_reference import _reference_overlay_level_for_path
from train_policy_only_reference import _reward_map as _policy_reward_map
from train_value_reference import (
    _all_reference_paths,
    _default_reference_paths,
    _filter_reference_paths,
    _historical_reference_lr_multiplier,
    RuleAugmentedReferenceAgent,
    RuleOnlyReferenceAgent,
    _load_reference_agent,
    _latest_reference_names,
    _latest_reference_win_rates,
    _reference_index_for_game,
    _reference_directory,
    _write_reference_winrate_log,
    _scheduled_reference_game_counts,
    train_against_reference,
    train_with_progressive_references,
    _reward_map as _value_reward_map,
)
from utils.state_encoder import action_features
from utils.tactical_rules import find_forced_action


class GomokuEnvTest(unittest.TestCase):
    def test_reset_initializes_empty_board(self) -> None:
        env = GomokuEnv()

        board = env.reset()

        self.assertEqual(len(board), 15)
        self.assertTrue(all(len(row) == 15 for row in board))
        self.assertTrue(all(cell == 0 for row in board for cell in row))
        self.assertEqual(env.current_player, 1)
        self.assertFalse(env.done)
        self.assertEqual(env.last_action, None)

    def test_get_valid_actions_excludes_played_positions(self) -> None:
        env = GomokuEnv(board_size=5)

        env.step((1, 1))

        valid_actions = env.get_valid_actions()
        self.assertEqual(len(valid_actions), 24)
        self.assertNotIn((1, 1), valid_actions)

    def test_occupied_position_is_rejected(self) -> None:
        env = GomokuEnv(board_size=5)
        env.step((0, 0))

        with self.assertRaises(ValueError):
            env.step((0, 0))

    def test_horizontal_win(self) -> None:
        env = GomokuEnv(board_size=5)

        moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4)]
        for action in moves:
            _, _, done, info = env.step(action)

        self.assertTrue(done)
        self.assertEqual(info["winner"], 1)
        self.assertEqual(env.winner, 1)

    def test_vertical_win(self) -> None:
        env = GomokuEnv(board_size=5)

        moves = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0)]
        for action in moves:
            _, _, done, info = env.step(action)

        self.assertTrue(done)
        self.assertEqual(info["winner"], 1)

    def test_diagonal_win(self) -> None:
        env = GomokuEnv(board_size=5)

        moves = [(0, 0), (0, 1), (1, 1), (0, 2), (2, 2), (0, 3), (3, 3), (1, 3), (4, 4)]
        for action in moves:
            _, _, done, info = env.step(action)

        self.assertTrue(done)
        self.assertEqual(info["winner"], 1)

    def test_anti_diagonal_win(self) -> None:
        env = GomokuEnv(board_size=5)

        moves = [(0, 4), (0, 0), (1, 3), (0, 1), (2, 2), (1, 0), (3, 1), (1, 1), (4, 0)]
        for action in moves:
            _, _, done, info = env.step(action)

        self.assertTrue(done)
        self.assertEqual(info["winner"], 1)

    def test_draw_when_board_is_full(self) -> None:
        env = GomokuEnv(board_size=5)

        draw_sequence = [
            (0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
            (0, 4), (1, 1), (1, 2), (1, 4), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 4), (2, 3),
            (3, 0), (3, 1), (3, 2), (4, 0), (3, 3),
            (3, 4), (4, 1), (4, 2), (4, 4), (4, 3),
        ]

        for action in draw_sequence:
            _, reward, done, info = env.step(action)

        self.assertTrue(done)
        self.assertEqual(reward, 0)
        self.assertEqual(info["winner"], 0)
        self.assertEqual(env.get_valid_actions(), [])

    def test_draw_is_treated_as_loss_in_reward_map(self) -> None:
        assignments = {
            1: ("reference", object()),
            -1: ("candidate", object()),
        }

        policy_rewards = _policy_reward_map(0, assignments, "reference", "candidate")
        value_rewards = _value_reward_map(0, assignments, "reference", "candidate")

        self.assertEqual(policy_rewards["reference"], 1.0)
        self.assertEqual(policy_rewards["candidate"], -1.0)
        self.assertEqual(value_rewards["reference"], 1.0)
        self.assertEqual(value_rewards["candidate"], -1.0)

    def test_cannot_step_after_game_ends(self) -> None:
        env = GomokuEnv(board_size=5)

        moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4)]
        for action in moves:
            env.step(action)

        with self.assertRaises(ValueError):
            env.step((4, 4))

    def test_action_index_round_trip(self) -> None:
        env = GomokuEnv(board_size=15)

        index = env.action_to_index((3, 7))
        action = env.index_to_action(index)

        self.assertEqual(index, 52)
        self.assertEqual(action, (3, 7))

    def test_random_simulation_finishes(self) -> None:
        result = simulate_game(board_size=5, seed=7)

        self.assertIn(result["winner"], (-1, 0, 1))
        self.assertGreaterEqual(result["moves"], 1)
        self.assertLessEqual(result["moves"], 25)
        self.assertIsNotNone(result["last_action"])
        self.assertEqual(len(result["move_history"]), result["moves"])

    def test_simulation_batch_writes_human_readable_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_result = simulate_games(num_games=3, board_size=5, seed=11, log_dir=temp_dir)

            log_path = Path(batch_result["log_path"])
            content = log_path.read_text(encoding="utf-8")
            self.assertTrue(log_path.exists())
            self.assertEqual(log_path.parent.name, Path(temp_dir).name)
            self.assertRegex(log_path.name, r"^\d{8}_\d{6}\.log$")
            self.assertIn("Gomoku Random Simulation Log", content)
            self.assertIn("Summary", content)
            self.assertIn("Game 1:", content)
            self.assertIn("Move record", content)
            self.assertIn("Black wins:", content)
            self.assertIn("White wins:", content)

    def test_log_parser_reads_saved_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_result = simulate_games(num_games=2, board_size=5, seed=3, log_dir=temp_dir)
            parsed = parse_log_file(batch_result["log_path"])

        self.assertEqual(parsed.summary.total_games, 2)
        self.assertEqual(len(parsed.games), 2)
        self.assertEqual(parsed.games[0].board_size, 5)
        self.assertEqual(parsed.games[0].moves, len(parsed.games[0].move_history))

    def test_log_parser_ignores_policy_metrics_line(self) -> None:
        log_text = "\n".join(
            [
                "Gomoku Tactical Policy Reference Training Log",
                "Generated at: 2026-04-30 12:00:00",
                "Board size: 15x15",
                "Reference init model: models/refer/value_agent_v203_reference.pt",
                "Candidate model: models/tactical_rule_policy_agent_v1.pt",
                "Candidate init model: models/tactical_rule_policy_agent_v0.pt",
                "Candidate feature init model: none",
                "Reference rule level: super_easy",
                "Reference rule overlay exclusion prefixes: tactical_rule_value_agent",
                "Candidate black priority: False",
                "Reference rule opening moves: 20",
                "Reference rule followup probability: 0.100",
                "Teacher rule agent: hard",
                "Teacher weight: 1.000",
                "Opening teacher moves: 20",
                "Game 1: winner=test_reference, moves=1, board=15x15, reference_role=black, candidate_role=white, candidate_reward=-1.0, candidate_epsilon=0.0000",
                "Move record",
                "  1. Black (test_reference) -> (row=0, col=0)",
                "Metrics: policy_loss=0.1000, aux_loss=0.0100, total_loss=0.1100, entropy=1.2345, teacher_top1=1/1 (100.00%), teacher_top3=1/1 (100.00%), teacher_top5=1/1 (100.00%), modes=policy_greedy:1, lr=0.000200",
                "",
            ]
        )

        parsed = parse_log_text(log_text)

        self.assertEqual(parsed.summary.total_games, 1)
        self.assertEqual(len(parsed.games), 1)
        self.assertEqual(parsed.games[0].moves, 1)
        self.assertEqual(parsed.games[0].move_history[0].agent_name, "test_reference")

    def test_build_board_state_replays_moves(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_result = simulate_games(num_games=1, board_size=5, seed=5, log_dir=temp_dir)
            parsed = parse_log_file(batch_result["log_path"])

        game = parsed.games[0]
        board = build_board_state(game, move_index=2)

        first_move = game.move_history[0]
        second_move = game.move_history[1]
        self.assertEqual(board[first_move.row][first_move.col], first_move.player)
        self.assertEqual(board[second_move.row][second_move.col], second_move.player)

    def test_action_features_have_stable_length(self) -> None:
        env = GomokuEnv(board_size=5)
        features = action_features(env.board, env.current_player, (2, 2))

        self.assertEqual(len(features), 11)
        self.assertGreater(features[1], 0.0)

    def test_value_agent_selects_valid_action(self) -> None:
        env = GomokuEnv(board_size=5)
        env.step((0, 0))
        agent = ValueAgent(name="value", board_size=5, seed=1)

        action, _ = agent.select_action(env)

        self.assertIn(action, env.get_valid_actions())

    def test_policy_agent_selects_valid_action(self) -> None:
        env = GomokuEnv(board_size=5)
        agent = PolicyGradientAgent(name="policy", board_size=5, seed=1)

        action, _ = agent.select_action(env)

        self.assertIn(action, env.get_valid_actions())

    def test_torch_cnn_value_agent_randomizes_opening_within_central_ten_by_ten(self) -> None:
        env = GomokuEnv(board_size=15)
        agent = TorchCNNValueAgent(name="cnn", board_size=15, seed=7)
        valid_actions = env.get_valid_actions()
        ranked_scores = list(range(len(valid_actions), 0, -1))
        captured_pool: list[tuple[int, int]] = []
        expected_pool = {
            (row, col)
            for row in range(2, 12)
            for col in range(2, 12)
        }

        def choose(pool):
            captured_pool[:] = list(pool)
            return pool[0]

        with patch.object(agent.random, "choice", side_effect=choose), patch.object(
            agent, "_predict_scores", return_value=ranked_scores
        ):
            action, record = agent.select_action(env, training=True)

        self.assertEqual(record.selection_reason, "opening_random")
        self.assertEqual(len(captured_pool), 100)
        self.assertTrue(set(captured_pool).issubset(expected_pool))
        self.assertIn(action, expected_pool)

    def test_torch_hybrid_agent_randomizes_opening_within_central_ten_by_ten_when_frozen(self) -> None:
        env = GomokuEnv(board_size=15)
        agent = TorchHybridAgent(name="hybrid", board_size=15, seed=7, freeze_value=True)

        captured_pool: list[tuple[int, int]] = []
        expected_pool = {
            (row, col)
            for row in range(2, 12)
            for col in range(2, 12)
        }

        def choose(pool):
            captured_pool[:] = list(pool)
            return pool[0]

        with patch.object(agent.random, "choice", side_effect=choose), patch.object(
            agent.value_model, "forward", side_effect=AssertionError("value forward should not be called")
        ), patch.object(
            agent.policy_model,
            "forward",
            return_value=torch.zeros((1, 225), dtype=torch.float32),
        ):
            action, record = agent.select_action(env, training=True)

        self.assertEqual(record.selection_reason, "opening_random")
        self.assertEqual(len(captured_pool), 100)
        self.assertTrue(set(captured_pool).issubset(expected_pool))
        self.assertIn(action, expected_pool)

    def test_torch_hybrid_agent_prefers_value_over_policy_with_zero_mix_weight(self) -> None:
        env = GomokuEnv(board_size=5)
        agent = TorchHybridAgent(name="hybrid", board_size=5, seed=7, policy_mix_weight=0.0)

        def fake_value_forward(x):
            values = torch.zeros(x.shape[0], dtype=torch.float32)
            values[0] = 10.0
            return values

        def fake_policy_forward(x):
            logits = torch.zeros((x.shape[0], 25), dtype=torch.float32)
            logits[:, 24] = 10.0
            return logits

        with patch.object(agent.value_model, "forward", side_effect=fake_value_forward), patch.object(
            agent.policy_model, "forward", side_effect=fake_policy_forward
        ):
            action, _ = agent.select_action(env, training=False)

        self.assertEqual(action, (0, 0))

    def test_torch_hybrid_mix_agent_lets_policy_break_ties_with_small_mix_weight(self) -> None:
        env = GomokuEnv(board_size=5)
        agent = TorchHybridMixAgent(name="mix", board_size=5, seed=7, policy_mix_weight=0.01)

        value_scores = torch.zeros(25, dtype=torch.float32)
        value_scores[0] = 0.001

        def fake_policy_forward(x):
            logits = torch.zeros((x.shape[0], 25), dtype=torch.float32)
            logits[:, 24] = 10.0
            return logits

        with patch.object(agent, "_action_value_scores", return_value=value_scores), patch.object(
            agent.policy_model, "forward", side_effect=fake_policy_forward
        ):
            action, record = agent.select_action(env, training=False)

        self.assertEqual(record.selection_reason, "mixed_selection")
        self.assertEqual(action, (4, 4))

    def test_torch_hybrid_agent_learns_policy_head_separately(self) -> None:
        env = GomokuEnv(board_size=5)
        agent = TorchHybridAgent(
            name="hybrid",
            board_size=5,
            seed=7,
            policy_mix_weight=0.0,
            policy_loss_weight=1.0,
            freeze_value=True,
            epsilon_start=0.0,
        )

        action, record = agent.select_action(env, training=False)
        before = [param.detach().clone() for param in agent.policy_model.parameters()]
        agent.finish_game([record], outcome_reward=1.0, game_length=1)
        after = list(agent.policy_model.parameters())

        self.assertIn(action, env.get_valid_actions())
        self.assertTrue(any(not torch.allclose(before_param, after_param) for before_param, after_param in zip(before, after)))

    def test_torch_hybrid_agent_frozen_value_keeps_greedy_selection(self) -> None:
        env = GomokuEnv(board_size=5)
        agent = TorchHybridAgent(
            name="hybrid",
            board_size=5,
            seed=7,
            policy_mix_weight=0.0,
            policy_loss_weight=0.0,
            freeze_value=True,
            epsilon_start=1.0,
        )

        def fake_forward(x):
            if x.shape[0] == 1:
                value = torch.tensor([0.0], dtype=torch.float32)
                return value
            values = torch.zeros(x.shape[0], dtype=torch.float32)
            values[3] = 10.0
            return values

        with patch.object(agent.value_model, "forward", side_effect=fake_forward):
            action, record = agent.select_action(env, training=True)

        self.assertEqual(record.selection_reason, "value_selection")
        self.assertEqual(action, env.index_to_action(3))

    def test_torch_cnn_value_agent_keeps_deterministic_opening_when_not_training(self) -> None:
        env = GomokuEnv(board_size=5)
        agent = TorchCNNValueAgent(name="cnn", board_size=5, seed=7)
        valid_actions = env.get_valid_actions()
        ranked_scores = list(range(len(valid_actions), 0, -1))

        with patch.object(agent, "_predict_scores", return_value=ranked_scores):
            action, record = agent.select_action(env, training=False)

        self.assertEqual(record.selection_reason, "model")
        self.assertEqual(action, valid_actions[0])

    def test_forced_action_finds_immediate_win(self) -> None:
        env = GomokuEnv(board_size=5)
        env.board[2][0] = 1
        env.board[2][1] = 1
        env.board[2][2] = 1
        env.board[2][3] = 1
        env.current_player = 1

        self.assertEqual(find_forced_action(env), (2, 4))

    def test_forced_action_blocks_opponent_win(self) -> None:
        env = GomokuEnv(board_size=5)
        env.board[1][0] = -1
        env.board[1][1] = -1
        env.board[1][2] = -1
        env.board[1][3] = -1
        env.current_player = 1

        self.assertEqual(find_forced_action(env), (1, 4))

    def test_forced_action_does_not_block_open_three_by_default(self) -> None:
        env = GomokuEnv(board_size=5)
        env.board[2][1] = -1
        env.board[2][2] = -1
        env.board[2][3] = -1
        env.current_player = 1

        self.assertIsNone(find_forced_action(env))

    def test_hard_tactical_rule_agent_blocks_open_three(self) -> None:
        env = GomokuEnv(board_size=5)
        env.board[2][1] = -1
        env.board[2][2] = -1
        env.board[2][3] = -1
        env.current_player = 1
        agent = HardTacticalRuleAgent()

        action, _ = agent.select_action(env)

        self.assertIn(action, {(2, 0), (2, 4)})

    def test_hard_tactical_rule_agent_random_builder_uses_three_variants(self) -> None:
        with patch("agent.tactical_rule_agent.random.choice", side_effect=lambda seq: seq[0]):
            agent = build_random_hard_tactical_rule_agent()
            self.assertIsInstance(agent, OffensiveHardTacticalRuleAgent)
        with patch("agent.tactical_rule_agent.random.choice", side_effect=lambda seq: seq[1]):
            agent = build_random_hard_tactical_rule_agent()
            self.assertIsInstance(agent, NeutralHardTacticalRuleAgent)
        with patch("agent.tactical_rule_agent.random.choice", side_effect=lambda seq: seq[2]):
            agent = build_random_hard_tactical_rule_agent()
            self.assertIsInstance(agent, DefensiveHardTacticalRuleAgent)

    def test_reference_training_logs_hard_variant_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            reference_dir = _reference_directory(model_dir)

            reference_path = reference_dir / "value_agent_v20_reference.pt"
            TorchCNNValueAgent(name="value_agent_v20", board_size=5, seed=20).save(reference_path)

            with patch("agent.tactical_rule_agent.random.choice", side_effect=lambda seq: seq[0]):
                result = train_against_reference(
                    num_games=1,
                    board_size=5,
                    save_every=1,
                    seed=9,
                    log_dir=log_dir,
                    model_dir=model_dir,
                    reference_model_path=[reference_path],
                    candidate_version=21,
                    pretrain_positions=0,
                    reference_cycle_length=10,
                    reference_rule_agent_level="hard",
                    reference_rule_only_agent_level="hard",
                    teacher_weight=0.0,
                )

            log_text = Path(result["training_log_path"]).read_text(encoding="utf-8")

        self.assertIn("reference_variant=offensive", log_text)
        self.assertIn("- Hard variant usage:", log_text)
        self.assertIn("reference offensive: 1", log_text)

    def test_tactical_rule_agent_prefers_double_threat_creation(self) -> None:
        env = GomokuEnv(board_size=7)
        env.board[3][2] = 1
        env.board[3][4] = 1
        env.board[2][3] = 1
        env.board[4][3] = 1
        env.current_player = 1
        agent = TacticalRuleAgent()

        action, _ = agent.select_action(env)

        self.assertEqual(action, (3, 3))

    def test_tactical_rule_agent_prefers_double_threat_block(self) -> None:
        env = GomokuEnv(board_size=7)
        env.board[3][2] = -1
        env.board[3][4] = -1
        env.board[2][3] = -1
        env.board[4][3] = -1
        env.current_player = 1
        agent = TacticalRuleAgent()

        action, _ = agent.select_action(env)

        self.assertEqual(action, (3, 3))

    def test_training_updates_agent_weights_and_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            model_dir = Path(temp_dir) / "models"
            result = train_competitive(
                num_games=6,
                board_size=5,
                save_every=3,
                seed=9,
                log_dir=log_dir,
                model_dir=model_dir,
                imbalance_threshold=1.1,
            )

            training_log = Path(result["training_log_path"])
            value_model = Path(result["value_model_path"])
            policy_model = Path(result["policy_model_path"])

            self.assertTrue(training_log.exists())
            self.assertTrue(value_model.exists())
            self.assertTrue(policy_model.exists())
            self.assertEqual(sum(result["summary"].values()), 6)
            log_content = training_log.read_text(encoding="utf-8")
            self.assertIn("Gomoku Competitive Training Log", log_content)
            self.assertIn("Summary", log_content)
            self.assertEqual(log_content.count("Move record"), 6)
            self.assertIn("Game 1:", log_content)
            self.assertIn("Game 6:", log_content)
            self.assertIn("Checkpoint 3 Evaluation:", log_content)
            self.assertIn("Checkpoint 6 Evaluation:", log_content)

    def test_competitive_log_parser_reads_training_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_competitive(
                num_games=4,
                board_size=5,
                save_every=2,
                seed=9,
                log_dir=Path(temp_dir) / "logs",
                model_dir=Path(temp_dir) / "models",
                imbalance_threshold=1.1,
            )
            parsed = parse_log_file(result["training_log_path"])

        self.assertEqual(parsed.log_type, "competitive")
        self.assertEqual(parsed.summary.total_games, 4)
        self.assertEqual(parsed.summary.value_agent_wins + parsed.summary.policy_agent_wins + parsed.summary.draws, 4)
        self.assertEqual(len(parsed.games), 4)
        self.assertIn(parsed.games[0].winner_label, {"value_agent", "policy_agent", "draw"})
        self.assertIsNotNone(parsed.games[0].black_agent)
        self.assertEqual(len(parsed.games[0].move_history), parsed.games[0].moves)

    def test_training_keeps_value_weights_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_competitive(
                num_games=20,
                board_size=5,
                save_every=10,
                seed=5,
                log_dir=Path(temp_dir) / "logs",
                model_dir=Path(temp_dir) / "models",
                imbalance_threshold=1.1,
            )
            value_model = Path(result["value_model_path"]).read_text(encoding="utf-8")

        self.assertNotIn("NaN", value_model)

    def test_training_stops_early_when_checkpoint_is_too_imbalanced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "train_competitive._evaluate_checkpoint",
                return_value={
                    "games": 4,
                    "value_agent_wins": 4,
                    "policy_agent_wins": 0,
                    "draws": 0,
                    "average_moves": 11.0,
                },
            ):
                result = train_competitive(
                    num_games=12,
                    board_size=5,
                    save_every=3,
                    seed=9,
                    eval_games=4,
                    imbalance_threshold=0.7,
                    log_dir=Path(temp_dir) / "logs",
                    model_dir=Path(temp_dir) / "models",
                )
                log_text = Path(result["training_log_path"]).read_text(encoding="utf-8")

        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["games"], 3)
        self.assertIn("Early stop:", log_text)

    def test_auto_rebalance_records_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "train_competitive.train_competitive",
                side_effect=[
                    {
                        "games": 300,
                        "training_log_path": str(Path(temp_dir) / "a.log"),
                        "value_model_path": "value.json",
                        "policy_model_path": "policy.json",
                        "summary": {"value_agent": 10, "policy_agent": 290},
                        "stopped_early": True,
                        "stop_reason": "gap",
                        "last_evaluation": {
                            "games": 20,
                            "value_agent_wins": 0,
                            "policy_agent_wins": 20,
                            "draws": 0,
                            "average_moves": 22.0,
                        },
                        "value_agent_config": {},
                        "policy_agent_config": {},
                    },
                    {
                        "games": 2000,
                        "training_log_path": str(Path(temp_dir) / "b.log"),
                        "value_model_path": "value.json",
                        "policy_model_path": "policy.json",
                        "summary": {"value_agent": 1000, "policy_agent": 1000},
                        "stopped_early": False,
                        "stop_reason": None,
                        "last_evaluation": {
                            "games": 20,
                            "value_agent_wins": 10,
                            "policy_agent_wins": 10,
                            "draws": 0,
                            "average_moves": 41.0,
                        },
                        "value_agent_config": {},
                        "policy_agent_config": {},
                    },
                ],
            ):
                result = train_until_balanced(
                    num_games=2000,
                    board_size=5,
                    save_every=300,
                    seed=9,
                    eval_games=20,
                    imbalance_threshold=0.7,
                    target_tolerance=0.1,
                    max_attempts=2,
                    log_dir=Path(temp_dir) / "logs",
                    model_dir=Path(temp_dir) / "models",
                )

        self.assertTrue(result["converged"])
        self.assertEqual(len(result["attempts"]), 2)

    def test_value_agent_can_load_saved_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "value.json"
            agent = ValueAgent(name="value", board_size=5, seed=1)
            agent.save(path)

            loaded = ValueAgent.load(path, name="loaded", seed=2)

        self.assertEqual(loaded.name, "loaded")
        self.assertEqual(loaded.board_size, 5)
        self.assertEqual(loaded.weights, agent.weights)
        self.assertEqual(loaded.prior_weights, agent.prior_weights)

    def test_reference_training_creates_v1_snapshot_and_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            base_agent = ValueAgent(name="value_agent", board_size=5, seed=1)
            base_agent.save(model_dir / "value_agent_latest.json")

            result = train_against_reference(
                num_games=6,
                board_size=5,
                save_every=3,
                seed=9,
                log_dir=log_dir,
                model_dir=model_dir,
            )

            reference_model = Path(result["reference_model_path"])
            candidate_model = Path(result["candidate_model_path"])
            promoted_reference_model = Path(result["promoted_reference_path"])
            winrate_log = Path(result["winrate_log_path"])
            log_text = Path(result["training_log_path"]).read_text(encoding="utf-8")
            self.assertTrue(reference_model.exists())
            self.assertTrue(candidate_model.exists())
            self.assertTrue(promoted_reference_model.exists())
            self.assertTrue(winrate_log.exists())
            self.assertIn("Gomoku Value Reference Training Log", log_text)
            self.assertIn("Reference model:", log_text)
            self.assertIn("Candidate model:", log_text)
            self.assertEqual(log_text.count("Move record"), 6)

    def test_value_reference_log_parser_reads_training_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            base_agent = ValueAgent(name="value_agent", board_size=5, seed=1)
            base_agent.save(model_dir / "value_agent_latest.json")

            result = train_against_reference(
                num_games=4,
                board_size=5,
                save_every=2,
                seed=9,
                log_dir=log_dir,
                model_dir=model_dir,
                pretrain_positions=20,
            )
            parsed = parse_log_file(result["training_log_path"])

        self.assertEqual(parsed.log_type, "value_reference")
        self.assertEqual(parsed.summary.total_games, 4)
        self.assertEqual(parsed.summary.reference_wins + parsed.summary.candidate_wins + parsed.summary.draws, 4)
        self.assertEqual(len(parsed.games), 4)
        self.assertIsNotNone(parsed.games[0].black_agent)

    def test_log_parser_accepts_tactical_value_reference_header(self) -> None:
        parsed = parse_log_text(
            "\n".join(
                [
                    "Gomoku Tactical Value Reference Training Log",
                    "Generated at: 2026-04-10 14:28:54",
                    "Board size: 5x5",
                    "Reference model: models/tactical_value_agent_v3_reference.pt",
                    "Candidate model: models/tactical_value_agent_v7.pt",
                    "Reference cycle length: 10",
                    "Pretrain positions: 0",
                    "Pretrain loss: 0.000000",
                    "",
                    "Game 1: winner=tactical_value_agent_v7, moves=9, board=5x5, reference_role=black, candidate_role=white, candidate_reward=1.0, candidate_epsilon=0.0500",
                    "Move record",
                    "    1. Black (tactical_value_agent_v3_reference) -> (row=0, col=0)",
                    "    2. White (tactical_value_agent_v7) -> (row=1, col=0)",
                    "    3. Black (tactical_value_agent_v3_reference) -> (row=0, col=1)",
                    "    4. White (tactical_value_agent_v7) -> (row=1, col=1)",
                    "    5. Black (tactical_value_agent_v3_reference) -> (row=0, col=2)",
                    "    6. White (tactical_value_agent_v7) -> (row=1, col=2)",
                    "    7. Black (tactical_value_agent_v3_reference) -> (row=0, col=3)",
                    "    8. White (tactical_value_agent_v7) -> (row=1, col=3)",
                    "    9. Black (tactical_value_agent_v3_reference) -> (row=0, col=4)",
                    "",
                    "Summary",
                    "",
                    "- Total games: 1",
                    "- Reference ensemble wins: 0",
                    "- tactical_value_agent_v7 wins: 1",
                    "- Draws: 0",
                ]
            )
        )

        self.assertEqual(parsed.log_type, "value_reference")
        self.assertEqual(parsed.summary.total_games, 1)
        self.assertEqual(parsed.summary.candidate_wins, 1)
        self.assertEqual(len(parsed.games), 1)

    def test_log_parser_reads_in_progress_tactical_log_without_summary(self) -> None:
        parsed = parse_log_text(
            "\n".join(
                [
                    "Gomoku Tactical Value Reference Training Log",
                    "Generated at: 2026-04-10 14:28:54",
                    "Board size: 5x5",
                    "Reference model: models/tactical_value_agent_v3_reference.pt",
                    "Candidate model: models/tactical_value_agent_v7.pt",
                    "",
                    "Game 1: winner=tactical_value_agent_v7, moves=9, board=5x5, reference_role=black, candidate_role=white, candidate_reward=1.0, candidate_epsilon=0.0500",
                    "Move record",
                    "    1. Black (tactical_value_agent_v3_reference) -> (row=0, col=0)",
                    "    2. White (tactical_value_agent_v7) -> (row=1, col=0)",
                    "    3. Black (tactical_value_agent_v3_reference) -> (row=0, col=1)",
                    "    4. White (tactical_value_agent_v7) -> (row=1, col=1)",
                    "    5. Black (tactical_value_agent_v3_reference) -> (row=0, col=2)",
                    "    6. White (tactical_value_agent_v7) -> (row=1, col=2)",
                    "    7. Black (tactical_value_agent_v3_reference) -> (row=0, col=3)",
                    "    8. White (tactical_value_agent_v7) -> (row=1, col=3)",
                    "    9. Black (tactical_value_agent_v3_reference) -> (row=0, col=4)",
                    "",
                ]
            )
        )

        self.assertEqual(parsed.summary.total_games, 1)
        self.assertEqual(parsed.summary.candidate_label, "tactical_value_agent_v7")
        self.assertEqual(parsed.summary.candidate_wins, 1)

    def test_log_parser_accepts_tactical_policy_reference_header(self) -> None:
        parsed = parse_log_text(
            "\n".join(
                [
                    "Gomoku Tactical Policy Reference Training Log",
                    "Generated at: 2026-04-28 08:37:46",
                    "Board size: 15x15",
                    "Candidate model: models/tactical_rule_policy_agent_v73.pt",
                    "Candidate init model: models/tactical_rule_policy_agent_v72.pt",
                    "Candidate feature init model: none",
                    "Reference init model: models/refer/tactical_value_agent_v201_reference.pt",
                    "Reference rule level: none",
                    "Candidate black priority: False",
                    "Reference rule opening moves: 20",
                    "Reference rule followup probability: 0.100",
                    "Teacher rule agent: hard",
                    "Teacher weight: 1.000",
                    "Opening teacher moves: 20",
                    "",
                    "Game 1: winner=tactical_rule_policy_agent_v73, moves=2, board=15x15, reference_role=black, candidate_role=white, candidate_reward=1.0, candidate_epsilon=0.0200",
                    "Move record",
                    "    1. Black (tactical_value_agent_v201_reference) -> (row=7, col=7)",
                    "    2. White (tactical_rule_policy_agent_v73) -> (row=7, col=8)",
                    "",
                    "Summary",
                    "",
                    "- Total games: 1",
                    "- Reference ensemble wins: 0",
                    "- tactical_rule_policy_agent_v73 wins: 1",
                    "- Draws: 0",
                ]
            )
        )

        self.assertEqual(parsed.log_type, "value_reference")
        self.assertEqual(parsed.summary.total_games, 1)
        self.assertEqual(parsed.summary.candidate_label, "tactical_rule_policy_agent_v73")
        self.assertEqual(parsed.summary.candidate_wins, 1)
        self.assertEqual(parsed.summary.reference_wins, 0)
        self.assertEqual(parsed.games[0].black_agent, "tactical_value_agent_v201_reference")
        self.assertEqual(parsed.games[0].white_agent, "tactical_rule_policy_agent_v73")

    def test_torch_value_agent_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "torch_value.pt"
            agent = TorchValueAgent(name="torch_value", board_size=5, seed=1)
            agent.save(path)
            loaded = TorchValueAgent.load(path, name="loaded")

        self.assertEqual(loaded.name, "loaded")
        self.assertEqual(loaded.board_size, 5)

    def test_torch_cnn_value_agent_only_reduces_learning_rate_for_quick_wins(self) -> None:
        agent = TorchCNNValueAgent(name="cnn", board_size=5, learning_rate=1e-4, seed=1)

        short_win_lr = agent._effective_learning_rate(outcome_reward=1.0, game_length=5)
        threshold_win_lr = agent._effective_learning_rate(outcome_reward=1.0, game_length=15)
        post_threshold_win_lr = agent._effective_learning_rate(outcome_reward=1.0, game_length=16)
        short_loss_lr = agent._effective_learning_rate(outcome_reward=-1.0, game_length=5)
        long_win_lr = agent._effective_learning_rate(outcome_reward=1.0, game_length=20)
        long_loss_lr = agent._effective_learning_rate(outcome_reward=-1.0, game_length=20)

        self.assertLess(short_win_lr, agent.learning_rate)
        self.assertEqual(short_win_lr, threshold_win_lr)
        self.assertEqual(post_threshold_win_lr, agent.learning_rate)
        self.assertEqual(long_win_lr, agent.learning_rate)
        self.assertEqual(short_loss_lr, agent.learning_rate)
        self.assertEqual(long_loss_lr, agent.learning_rate)

    def test_torch_cnn_value_agent_disables_random_exploration_before_move_15(self) -> None:
        agent = TorchCNNValueAgent(name="cnn", board_size=5, seed=1)
        agent.epsilon = 1.0
        env = GomokuEnv(board_size=5)

        _, early_record = agent.select_action(env, training=True)
        for action in [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
        ]:
            env.step(action)
        _, late_record = agent.select_action(env, training=True)

        self.assertEqual(early_record.selection_reason, "model")
        self.assertEqual(late_record.selection_reason, "random")

    def test_reference_training_resume_preserves_candidate_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            base_agent = ValueAgent(name="value_agent", board_size=5, seed=1)
            base_agent.save(model_dir / "value_agent_latest.json")

            initial = TorchCNNValueAgent(name="value_agent_v9", board_size=5, seed=2)
            initial.epsilon = 0.42
            initial.episodes_trained = 37
            init_path = model_dir / "value_agent_v9.pt"
            initial.save(init_path)

            result = train_against_reference(
                num_games=1,
                board_size=5,
                save_every=1,
                seed=9,
                log_dir=log_dir,
                model_dir=model_dir,
                candidate_init_model_path=init_path,
                candidate_version=10,
                pretrain_positions=0,
            )
            resumed = TorchCNNValueAgent.load(result["candidate_model_path"], device="cpu")

        self.assertEqual(resumed.episodes_trained, 38)
        self.assertLess(resumed.epsilon, 0.42)

    def test_load_reference_agent_detects_cnn_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "value_agent_v20_reference.pt"
            agent = TorchCNNValueAgent(name="value_agent_v20", board_size=5, seed=2)
            agent.save(checkpoint_path)

            loaded = _load_reference_agent(checkpoint_path, "value_agent_v20_reference", 7, "cpu")

        self.assertIsInstance(loaded, TorchCNNValueAgent)
        self.assertEqual(loaded.name, "value_agent_v20_reference")
        self.assertEqual(loaded.board_size, 5)

    def test_load_reference_agent_wraps_all_references_with_rule_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "tactical_value_agent_v20_reference.pt"
            agent = TorchCNNValueAgent(name="tactical_value_agent_v20", board_size=5, seed=2)
            agent.save(checkpoint_path)

            loaded = _load_reference_agent(
                checkpoint_path,
                "tactical_value_agent_v20_reference",
                7,
                "cpu",
                reference_rule_agent_level="super_easy",
                reference_rule_opening_moves=20,
            )

        self.assertIsInstance(loaded, RuleAugmentedReferenceAgent)
        self.assertEqual(loaded.name, "tactical_value_agent_v20_reference")

    def test_hybrid_reference_training_saves_outputs_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            reference_path = model_dir / "value_agent_v20_reference.pt"
            TorchCNNValueAgent(name="value_agent_v20", board_size=5, seed=20).save(reference_path)

            result = train_hybrid_against_reference(
                num_games=1,
                board_size=5,
                save_every=1,
                seed=9,
                log_dir=log_dir,
                model_dir=model_dir,
                reference_model_path=[reference_path],
                candidate_version=1,
                candidate_prefix="tactical_rule_hybrid_agent",
                reference_rule_agent_level="super_easy",
                reference_rule_only_agent_level="super_easy",
            )

            training_log = Path(result["training_log_path"])
            candidate_model = Path(result["candidate_model_path"])
            promoted_reference = Path(result["promoted_reference_path"])
            log_content = training_log.read_text(encoding="utf-8")

            self.assertTrue(training_log.exists())
            self.assertTrue(candidate_model.exists())
            self.assertTrue(promoted_reference.exists())
            self.assertIn("Gomoku Hybrid Reference Training Log", log_content)

    def test_hybrid_mix_reference_training_saves_outputs_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            reference_path = model_dir / "value_agent_v20_reference.pt"
            TorchCNNValueAgent(name="value_agent_v20", board_size=15, seed=20).save(reference_path)

            result = train_hybrid_mix_against_reference(
                num_games=1,
                board_size=15,
                save_every=1,
                seed=9,
                log_dir=log_dir,
                model_dir=model_dir,
                reference_model_path=[reference_path],
                candidate_version=1,
                candidate_prefix="tactical_rule_hybrid_mix_weight_agent",
            )

            training_log = Path(result["training_log_path"])
            candidate_model = Path(result["candidate_model_path"])
            promoted_reference = Path(result["promoted_reference_path"])
            log_content = training_log.read_text(encoding="utf-8")

            self.assertTrue(training_log.exists())
            self.assertTrue(candidate_model.exists())
            self.assertTrue(promoted_reference.exists())
            self.assertIn("Gomoku Hybrid Mix Reference Training Log", log_content)
            self.assertIn("Policy decision mix weight: 0.010", log_content)

    def test_torch_hybrid_agent_bootstraps_from_value_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            value_model_path = Path(temp_dir) / "value_agent_v20.pt"
            value_agent = TorchCNNValueAgent(name="value_agent_v20", board_size=5, seed=20)
            value_agent.save(value_model_path)

            hybrid_agent = TorchHybridAgent.load(value_model_path, name="hybrid", device="cpu")

        self.assertIsInstance(hybrid_agent, TorchHybridAgent)
        self.assertEqual(hybrid_agent.name, "hybrid")

    def test_torch_policy_only_agent_bootstraps_from_value_checkpoint(self) -> None:
        value_model_path = Path("/home/yphong/omok_deeplearning/models/tactical_rule_value_agent_v201.pt")
        self.assertTrue(value_model_path.exists())

        policy_agent = TorchPolicyOnlyAgent.load_from_value_checkpoint(
            value_model_path,
            name="policy_only",
            device="cpu",
        )

        self.assertEqual(policy_agent.name, "policy_only")
        self.assertEqual(policy_agent.board_size, 15)
        payload = torch.load(value_model_path, map_location="cpu")
        value_state = payload["model_state_dict"]
        policy_state = policy_agent.model.state_dict()
        self.assertTrue(torch.equal(policy_state["features.0.weight"], value_state["features.0.weight"]))
        self.assertTrue(torch.equal(policy_state["features.2.weight"], value_state["features.2.weight"]))
        self.assertTrue(torch.equal(policy_state["head.1.weight"], value_state["head.1.weight"]))

    def test_torch_policy_only_agent_bootstraps_features_from_hybrid_checkpoint(self) -> None:
        hybrid_model_path = Path("/home/yphong/omok_deeplearning/models/tactical_rule_hybrid_agent_v37.pt")
        self.assertTrue(hybrid_model_path.exists())

        policy_agent = TorchPolicyOnlyAgent.load_with_feature_checkpoint(
            hybrid_model_path,
            name="policy_only",
            device="cpu",
        )

        payload = torch.load(hybrid_model_path, map_location="cpu")
        policy_state = policy_agent.model.state_dict()
        hybrid_policy_state = payload["policy_model_state_dict"]

        self.assertTrue(torch.equal(policy_state["features.0.weight"], hybrid_policy_state["features.0.weight"]))
        self.assertTrue(torch.equal(policy_state["features.2.weight"], hybrid_policy_state["features.2.weight"]))
        self.assertTrue(torch.equal(policy_state["features.4.weight"], hybrid_policy_state["features.4.weight"]))
        self.assertNotEqual(policy_state["head.1.weight"].shape, hybrid_policy_state["head.1.weight"].shape)

    def test_torch_policy_only_agent_builds_teacher_forced_record(self) -> None:
        env = GomokuEnv(board_size=15)
        agent = TorchPolicyOnlyAgent.create_blank(name="policy_only", board_size=15, seed=7)
        teacher_action = (7, 7)

        teacher_index = env.action_to_index(teacher_action)

        def fake_forward(x):
            logits = torch.zeros((x.shape[0], 225), dtype=torch.float32)
            logits[:, teacher_index] = 10.0
            logits[:, teacher_index - 1] = 9.0
            logits[:, teacher_index + 1] = 8.0
            logits[:, teacher_index + 2] = 7.0
            logits[:, teacher_index + 3] = 6.0
            return logits

        with patch.object(agent.model, "forward", side_effect=fake_forward):
            record = agent.build_teacher_forced_record(env, teacher_action)

        self.assertEqual(record.selection_reason, "teacher_forced")
        self.assertIsNone(record.log_prob)
        self.assertEqual(record.teacher_action_index, teacher_index)
        self.assertEqual(
            record.chosen_action_index,
            record.valid_action_indices.index(teacher_index),
        )
        self.assertIsNotNone(record.policy_entropy)
        self.assertTrue(record.policy_top1_correct)
        self.assertTrue(record.policy_top3_correct)
        self.assertTrue(record.policy_top5_correct)

    def test_torch_policy_only_agent_randomizes_opening_within_central_ten_by_ten(self) -> None:
        env = GomokuEnv(board_size=15)
        agent = TorchPolicyOnlyAgent(name="policy_only", board_size=15, seed=7)
        valid_actions = env.get_valid_actions()
        captured_pool: list[tuple[int, int]] = []
        expected_pool = {
            (row, col)
            for row in range(2, 12)
            for col in range(2, 12)
        }

        def choose(pool):
            captured_pool[:] = list(pool)
            return pool[0]

        with patch.object(agent.random, "choice", side_effect=choose), patch(
            "agent.torch_policy_only_agent.find_forced_action", return_value=(0, 0)
        ):
            action, record = agent.select_action(env, training=True)

        self.assertEqual(record.selection_reason, "opening_random")
        self.assertEqual(len(captured_pool), 100)
        self.assertTrue(set(captured_pool).issubset(expected_pool))
        self.assertIn(action, expected_pool)

    def test_torch_policy_only_agent_uses_greedy_selection_when_not_training(self) -> None:
        env = GomokuEnv(board_size=5)
        agent = TorchPolicyOnlyAgent(name="policy_only", board_size=5, seed=7, epsilon_start=0.0)

        def fake_forward(x):
            logits = torch.zeros((x.shape[0], 25), dtype=torch.float32)
            logits[:, 24] = 10.0
            return logits

        with patch.object(agent.model, "forward", side_effect=fake_forward):
            action, record = agent.select_action(env, training=False)

        self.assertEqual(record.selection_reason, "policy_greedy")
        self.assertEqual(action, (4, 4))

    def test_torch_policy_only_agent_switches_to_greedy_late_in_training(self) -> None:
        env = GomokuEnv(board_size=7)
        agent = TorchPolicyOnlyAgent(name="policy_only", board_size=7, seed=7, epsilon_start=0.0)
        agent.greedy_move_threshold = 30
        env.move_count = 30
        env.current_player = 1
        env.done = False

        def fake_forward(x):
            logits = torch.zeros((x.shape[0], 49), dtype=torch.float32)
            logits[:, 48] = 10.0
            return logits

        with patch.object(agent.model, "forward", side_effect=fake_forward):
            action, record = agent.select_action(env, training=True)

        self.assertEqual(record.selection_reason, "policy_greedy")
        self.assertEqual(action, (6, 6))

    def test_tactical_rule_agent_randomizes_opening_before_forced_action(self) -> None:
        env = GomokuEnv(board_size=15)
        agent = TacticalRuleAgent(seed=7)
        captured_pool: list[tuple[int, int]] = []
        expected_pool = {
            (row, col)
            for row in range(2, 12)
            for col in range(2, 12)
        }

        def choose(pool):
            captured_pool[:] = list(pool)
            return pool[0]

        with patch.object(agent.random, "choice", side_effect=choose), patch(
            "agent.tactical_rule_agent.find_forced_action", return_value=(0, 0)
        ):
            action, evaluation = agent.select_action(env)

        self.assertEqual(evaluation.action, action)
        self.assertEqual(len(captured_pool), 100)
        self.assertTrue(set(captured_pool).issubset(expected_pool))
        self.assertIn(action, expected_pool)

    def test_policy_only_reference_training_saves_outputs_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            result = train_policy_only_against_reference(
                num_games=1,
                board_size=15,
                save_every=1,
                seed=9,
                log_dir=log_dir,
                model_dir=model_dir,
                candidate_feature_init_model_path=Path("/home/yphong/omok_deeplearning/models/tactical_rule_hybrid_agent_v37.pt"),
                reference_model_path=Path("/home/yphong/omok_deeplearning/models/refer/value_agent_v203_reference.pt"),
                candidate_version=1,
                candidate_prefix="tactical_rule_policy_agent",
                reference_rule_agent_level="none",
                reference_rule_opening_moves=20,
                reference_rule_followup_probability=0.1,
                teacher_rule_agent_level="hard",
                teacher_weight=1.0,
                opening_teacher_moves=20,
            )

            training_log = Path(result["training_log_path"])
            candidate_model = Path(result["candidate_model_path"])
            promoted_reference = Path(result["promoted_reference_path"])
            log_content = training_log.read_text(encoding="utf-8")

            self.assertTrue(training_log.exists())
            self.assertTrue(candidate_model.exists())
            self.assertTrue(promoted_reference.exists())
            self.assertIn("Gomoku Tactical Policy Reference Training Log", log_content)
            self.assertIn("Candidate init model: random", log_content)
            self.assertIn("Candidate feature init model: /home/yphong/omok_deeplearning/models/tactical_rule_hybrid_agent_v37.pt", log_content)
            self.assertIn("Opening teacher moves: 20", log_content)
            self.assertIn("Reference init model: /home/yphong/omok_deeplearning/models/refer/value_agent_v203_reference.pt", log_content)
            self.assertIn("Reference rule level: none", log_content)
            self.assertIn("Teacher rule agent: hard", log_content)
            self.assertIn("[teacher_forced]", log_content)
            self.assertIn("Metrics:", log_content)
            self.assertIn("teacher_top1=", log_content)
            self.assertIn("entropy=", log_content)
            self.assertIn("dbg=", log_content)
            self.assertIn("target_logit_avg=", log_content)

    def test_policy_only_reference_excludes_rule_value_models_from_overlay(self) -> None:
        self.assertIsNone(
            _reference_overlay_level_for_path(
                Path("models/refer/tactical_rule_value_agent_v36_reference.pt"),
                base_overlay_level="super_easy",
                exclusion_prefixes=("tactical_rule_value_agent",),
            )
        )
        self.assertEqual(
            _reference_overlay_level_for_path(
                Path("models/refer/tactical_rule_policy_agent_v74_reference.pt"),
                base_overlay_level="super_easy",
                exclusion_prefixes=("tactical_rule_value_agent",),
            ),
            "super_easy",
        )

    def test_rule_augmented_reference_agent_can_fall_back_to_rule_after_opening(self) -> None:
        rule_agent = HardTacticalRuleAgent()
        base_agent = Mock()
        base_agent.select_action.return_value = ((0, 0), object())
        reference = RuleAugmentedReferenceAgent(
            base_agent=base_agent,
            rule_agent=rule_agent,
            name="reference",
            opening_rule_moves=0,
            late_rule_probability=1.0,
            seed=7,
        )
        env = GomokuEnv(board_size=5)

        action, _ = reference.select_action(env, training=False)

        self.assertIn(action, env.get_valid_actions())
        base_agent.select_action.assert_not_called()

    def test_rule_augmented_reference_agent_randomizes_opening(self) -> None:
        rule_agent = HardTacticalRuleAgent()
        base_agent = Mock()
        base_agent.select_action.return_value = ((0, 0), object())
        reference = RuleAugmentedReferenceAgent(
            base_agent=base_agent,
            rule_agent=rule_agent,
            name="reference",
            opening_rule_moves=20,
            late_rule_probability=0.0,
            seed=7,
        )
        env = GomokuEnv(board_size=15)

        action, record = reference.select_action(env, training=False)

        self.assertEqual(getattr(record, "selection_reason", None), "opening_random")
        self.assertIn(action, env.get_valid_actions())
        base_agent.select_action.assert_not_called()

    def test_rule_only_reference_agent_uses_rule_agent_directly(self) -> None:
        rule_agent = HardTacticalRuleAgent()
        reference = RuleOnlyReferenceAgent(rule_agent=rule_agent, name="rule_only_reference")
        env = GomokuEnv(board_size=5)

        action, _ = reference.select_action(env, training=True)

        self.assertIn(action, env.get_valid_actions())
        self.assertEqual(reference.epsilon, 0.0)

    def test_rule_only_reference_agent_randomizes_opening(self) -> None:
        rule_agent = HardTacticalRuleAgent()
        reference = RuleOnlyReferenceAgent(rule_agent=rule_agent, name="rule_only_reference")
        env = GomokuEnv(board_size=15)

        action, record = reference.select_action(env, training=True)

        self.assertEqual(getattr(record, "selection_reason", None), "opening_random")
        self.assertIn(action, env.get_valid_actions())

    def test_reference_index_for_game_rotates_in_fixed_blocks(self) -> None:
        self.assertEqual(_reference_index_for_game(1, 4, 10), 0)
        self.assertEqual(_reference_index_for_game(10, 4, 10), 0)
        self.assertEqual(_reference_index_for_game(11, 4, 10), 1)
        self.assertEqual(_reference_index_for_game(20, 4, 10), 1)
        self.assertEqual(_reference_index_for_game(21, 4, 10), 2)
        self.assertEqual(_reference_index_for_game(30, 4, 10), 2)
        self.assertEqual(_reference_index_for_game(31, 4, 10), 3)
        self.assertEqual(_reference_index_for_game(40, 4, 10), 3)
        self.assertEqual(_reference_index_for_game(41, 4, 10), 0)

    def test_default_reference_paths_use_all_available_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            log_dir = model_dir / "logs"
            log_dir.mkdir()
            reference_dir = _reference_directory(model_dir)
            for version in (3, 20, 21, 22, 23):
                (reference_dir / f"value_agent_v{version}_reference.pt").write_bytes(b"test")
            (reference_dir / "tactical_rule_policy_agent_v4_reference.pt").write_bytes(b"test")

            selected = _default_reference_paths(model_dir, log_directory=log_dir)

        self.assertEqual(
            [path.name for path in selected],
            [
                "value_agent_v3_reference.pt",
                "value_agent_v20_reference.pt",
                "value_agent_v21_reference.pt",
                "value_agent_v22_reference.pt",
                "value_agent_v23_reference.pt",
                "tactical_rule_policy_agent_v4_reference.pt",
            ],
        )

    def test_explicit_reference_paths_are_preserved_without_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            reference_dir = _reference_directory(model_dir)
            paths = []
            for version in (40, 41):
                path = reference_dir / f"value_agent_v{version}_reference.pt"
                TorchCNNValueAgent(name=f"value_agent_v{version}", board_size=5, seed=version).save(path)
                paths.append(path)
            (log_dir / "20260408_120000_value_reference_training_winrates.log").write_text(
                "\n".join(
                    [
                        "Gomoku Value Reference Winrate Log",
                        "",
                        "Per-reference win rates",
                        "- value_agent_v40_reference: 19/20 (95.00%)",
                    ]
                ),
                encoding="utf-8",
            )

            result = train_against_reference(
                num_games=1,
                board_size=5,
                save_every=1,
                seed=9,
                log_dir=log_dir,
                model_dir=model_dir,
                reference_model_path=paths,
                candidate_version=42,
                pretrain_positions=0,
            )

        self.assertEqual(result["reference_model_paths"], [str(path) for path in paths])

    def test_latest_reference_win_rates_use_most_recent_entry_per_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            older = log_dir / "20260408_100000_value_reference_training_winrates.log"
            newer = log_dir / "20260408_110000_value_reference_training_winrates.log"
            older.write_text(
                "\n".join(
                    [
                        "Gomoku Value Reference Winrate Log",
                        "",
                        "Per-reference win rates",
                        "- value_agent_v45_reference: 15/20 (75.00%)",
                        "- value_agent_v46_reference: 18/20 (90.00%)",
                    ]
                ),
                encoding="utf-8",
            )
            newer.write_text(
                "\n".join(
                    [
                        "Gomoku Value Reference Winrate Log",
                        "",
                        "Per-reference win rates",
                        "- value_agent_v46_reference: 10/20 (50.00%)",
                        "- value_agent_v47_reference: 19/20 (95.00%)",
                    ]
                ),
                encoding="utf-8",
            )

            rates = _latest_reference_win_rates(log_dir)

        self.assertEqual(rates["value_agent_v45_reference"], 0.75)
        self.assertEqual(rates["value_agent_v46_reference"], 0.50)
        self.assertEqual(rates["value_agent_v47_reference"], 0.95)

    def test_write_reference_winrate_log_sorts_names_alphabetically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "winrates.log"
            _write_reference_winrate_log(
                path=path,
                candidate_name="candidate",
                candidate_version=1,
                training_log_path=Path("logs/example.log"),
                reference_names=[
                    "tactical_rule_policy_agent_v8_reference",
                    "value_agent_v203_reference",
                    "tactical_value_agent_v114_reference",
                ],
                reference_game_counts={
                    "tactical_rule_policy_agent_v8_reference": 1,
                    "value_agent_v203_reference": 1,
                    "tactical_value_agent_v114_reference": 1,
                },
                summary_counter={
                    "candidate": 0,
                    "tactical_rule_policy_agent_v8_reference": 0,
                    "value_agent_v203_reference": 0,
                    "tactical_value_agent_v114_reference": 0,
                    "draw": 0,
                },
            )
            content = path.read_text(encoding="utf-8")

        self.assertLess(
            content.index("- tactical_rule_policy_agent_v8_reference:"),
            content.index("- tactical_value_agent_v114_reference:"),
        )
        self.assertLess(
            content.index("- tactical_value_agent_v114_reference:"),
            content.index("- value_agent_v203_reference:"),
        )

    def test_filter_reference_paths_excludes_overfit_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            reference_dir = _reference_directory(model_dir)
            reference_paths = []
            for version in (45, 46, 47):
                path = reference_dir / f"value_agent_v{version}_reference.pt"
                path.write_bytes(b"test")
                reference_paths.append(path)
            (log_dir / "20260408_120000_value_reference_training_winrates.log").write_text(
                "\n".join(
                    [
                        "Gomoku Value Reference Winrate Log",
                        "",
                        "Per-reference win rates",
                        "- value_agent_v45_reference: 19/20 (95.00%)",
                        "- value_agent_v46_reference: 18/20 (90.00%)",
                        "- value_agent_v47_reference: 10/20 (50.00%)",
                    ]
                ),
                encoding="utf-8",
            )

            filtered = _filter_reference_paths(reference_paths, log_dir)

        self.assertEqual(
            [path.name for path in filtered],
            [
                "value_agent_v45_reference.pt",
                "value_agent_v46_reference.pt",
                "value_agent_v47_reference.pt",
            ],
        )

    def test_filter_reference_paths_keeps_at_least_ten_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            reference_dir = _reference_directory(model_dir)
            reference_paths = []
            for version in range(30, 42):
                path = reference_dir / f"value_agent_v{version}_reference.pt"
                path.write_bytes(b"test")
                reference_paths.append(path)
            (log_dir / "20260408_120000_value_reference_training_winrates.log").write_text(
                "\n".join(
                    [
                        "Gomoku Value Reference Winrate Log",
                        "",
                        "Per-reference win rates",
                        *[
                            f"- value_agent_v{version}_reference: 19/20 (95.00%)"
                            for version in range(30, 42)
                        ],
                    ]
                ),
                encoding="utf-8",
            )

            filtered = _filter_reference_paths(reference_paths, log_dir, min_reference_count=10)

        self.assertEqual(len(filtered), 10)
        self.assertEqual(
            [path.name for path in filtered],
            [f"value_agent_v{version}_reference.pt" for version in range(32, 42)],
        )

    def test_historical_reference_lr_multiplier_scales_overfit_opponents(self) -> None:
        self.assertEqual(_historical_reference_lr_multiplier(None), 1.0)
        self.assertEqual(_historical_reference_lr_multiplier(0.89), 1.0)
        self.assertEqual(_historical_reference_lr_multiplier(0.90), 0.5)
        self.assertEqual(_historical_reference_lr_multiplier(0.95), 0.5)
        self.assertEqual(_historical_reference_lr_multiplier(1.0), 0.1)

    def test_latest_reference_names_selects_highest_versions(self) -> None:
        reference_paths = [
            Path(f"/tmp/value_agent_v{version}_reference.pt")
            for version in (25, 26, 27, 32, 33, 44, 45, 46, 47, 48)
        ]

        selected = _latest_reference_names(reference_paths, count=5)

        self.assertEqual(
            selected,
            [
                "value_agent_v44_reference",
                "value_agent_v45_reference",
                "value_agent_v46_reference",
                "value_agent_v47_reference",
                "value_agent_v48_reference",
            ],
        )

    def test_scheduled_reference_game_counts_follow_rotation(self) -> None:
        counts = _scheduled_reference_game_counts(
            ["value_agent_v45_reference", "value_agent_v46_reference", "value_agent_v47_reference"],
            total_games=25,
            reference_cycle_length=10,
        )

        self.assertEqual(
            counts,
            {
                "value_agent_v45_reference": 10,
                "value_agent_v46_reference": 10,
                "value_agent_v47_reference": 5,
            },
        )

    def test_reference_training_accepts_multiple_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            reference_dir = _reference_directory(model_dir)

            reference_paths = []
            for version in (20, 21, 22, 23):
                path = reference_dir / f"value_agent_v{version}_reference.pt"
                TorchCNNValueAgent(name=f"value_agent_v{version}", board_size=5, seed=version).save(path)
                reference_paths.append(path)

            result = train_against_reference(
                num_games=6,
                board_size=5,
                save_every=3,
                seed=9,
                log_dir=log_dir,
                model_dir=model_dir,
                reference_model_path=reference_paths,
                candidate_version=23,
                pretrain_positions=0,
                reference_cycle_length=10,
            )

            log_text = Path(result["training_log_path"]).read_text(encoding="utf-8")

        self.assertEqual(
            result["reference_model_paths"],
            [str(path) for path in reference_paths],
        )
        self.assertEqual(result["reference_names"], [f"value_agent_v{version}_reference" for version in (20, 21, 22, 23)])
        self.assertIn("Reference cycle length: 10", log_text)
        self.assertIn("value_agent_v20_reference", log_text)

    def test_reference_training_can_include_rule_only_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            reference_dir = _reference_directory(model_dir)

            reference_path = reference_dir / "value_agent_v20_reference.pt"
            TorchCNNValueAgent(name="value_agent_v20", board_size=5, seed=20).save(reference_path)

            result = train_against_reference(
                num_games=2,
                board_size=5,
                save_every=2,
                seed=9,
                log_dir=log_dir,
                model_dir=model_dir,
                reference_model_path=[reference_path],
                candidate_version=21,
                pretrain_positions=0,
                reference_cycle_length=10,
                reference_rule_only_agent_level="hard",
            )

            log_text = Path(result["training_log_path"]).read_text(encoding="utf-8")

        self.assertIn("Reference rule-only agent: hard", log_text)
        self.assertIn("value_agent_v20_reference", result["reference_names"])
        self.assertIn("value_agent_hard_rule_only_reference", result["reference_names"])

    def test_reference_training_boosts_learning_rate_for_latest_five_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            reference_dir = _reference_directory(model_dir)

            reference_paths = []
            for version in (20, 21, 22, 23, 24, 25):
                path = reference_dir / f"value_agent_v{version}_reference.pt"
                TorchCNNValueAgent(name=f"value_agent_v{version}", board_size=5, seed=version).save(path)
                reference_paths.append(path)
            ordered_entries = [
                {"name": f"value_agent_v{version}_reference"}
                for version in (20, 21, 22, 23, 24, 25)
            ]
            choice_sequence = ordered_entries.copy()

            def choose_reference(entries):
                next_name = choice_sequence.pop(0)["name"]
                return next(entry for entry in entries if entry["name"] == next_name)

            with patch("train_value_reference.TorchCNNValueAgent.finish_game") as finish_game, patch(
                "train_value_reference.random.Random.choice"
            ) as choice_mock:
                choice_mock.side_effect = lambda entries: choose_reference(entries)
                finish_game.return_value = None
                train_against_reference(
                    num_games=6,
                    board_size=5,
                    save_every=3,
                    seed=9,
                    log_dir=log_dir,
                    model_dir=model_dir,
                    reference_model_path=reference_paths,
                    candidate_version=26,
                    pretrain_positions=0,
                    reference_cycle_length=1,
                )

        expected = [1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        observed = [call.kwargs["lr_multiplier"] for call in finish_game.call_args_list]
        self.assertEqual(observed, expected)

    def test_progressive_reference_training_promotes_each_block(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            log_dir = Path(temp_dir) / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            reference_dir = _reference_directory(model_dir)

            reference_paths = []
            for version in (22, 23, 24, 25):
                path = reference_dir / f"value_agent_v{version}_reference.pt"
                TorchCNNValueAgent(name=f"value_agent_v{version}", board_size=5, seed=version).save(path)
                reference_paths.append(path)

            init_path = model_dir / "value_agent_v25.pt"
            TorchCNNValueAgent(name="value_agent_v25", board_size=5, seed=25).save(init_path)

            result = train_with_progressive_references(
                num_games=4,
                promotion_interval=2,
                board_size=5,
                seed=9,
                save_every=2,
                log_dir=log_dir,
                model_dir=model_dir,
                reference_model_paths=reference_paths,
                candidate_init_model_path=init_path,
                starting_candidate_version=26,
                final_candidate_version=27,
                device="cpu",
                pretrain_positions=0,
                reference_cycle_length=2,
            )
            promoted_exists = [
                Path(result["block_results"][0]["promoted_reference_path"]).exists(),
                Path(result["block_results"][1]["promoted_reference_path"]).exists(),
            ]
            summary_log_exists = Path(result["summary_log_path"]).exists()

        self.assertEqual(result["starting_candidate_version"], 26)
        self.assertEqual(result["final_candidate_version"], 27)
        self.assertEqual(len(result["block_results"]), 2)
        self.assertEqual(promoted_exists, [True, True])
        self.assertTrue(summary_log_exists)
        final_reference_names = [Path(path).name for path in result["final_reference_model_paths"]]
        self.assertGreaterEqual(len(final_reference_names), 4)
        self.assertIn("value_agent_v26_reference.pt", final_reference_names)
        self.assertIn("value_agent_v27_reference.pt", final_reference_names)

    def test_build_board_state_defaults_to_final_position(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_result = simulate_games(num_games=1, board_size=5, seed=13, log_dir=temp_dir)
            parsed = parse_log_file(batch_result["log_path"])

        game = parsed.games[0]
        board = build_board_state(game)

        self.assertEqual(sum(cell != 0 for row in board for cell in row), game.moves)

    def test_find_latest_log_returns_most_recent_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            older = Path(temp_dir) / "20260406_140000_value_reference_training.log"
            newer = Path(temp_dir) / "20260406_150000_value_reference_training.log"
            older.write_text("old", encoding="utf-8")
            newer.write_text("new", encoding="utf-8")

            latest = find_latest_log(temp_dir)

        self.assertEqual(latest, str(newer))


if __name__ == "__main__":
    unittest.main()
