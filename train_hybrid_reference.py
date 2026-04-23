"""Train a hybrid policy-value candidate against frozen references."""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

import train_value_reference as tvr
from agent.torch_hybrid_agent import TorchHybridAgent, TorchHybridStepRecord
from env.gomoku_env import GomokuEnv


def train_against_reference(
    num_games: int = 1000,
    board_size: int = 15,
    seed: int = 7,
    save_every: int = 1000,
    log_dir: str | Path = "logs",
    model_dir: str | Path = "models",
    reference_model_path: str | Path | list[str | Path] | None = None,
    candidate_init_model_path: str | Path | None = None,
    candidate_version: int | None = None,
    device: str = "cpu",
    candidate_prefix: str = "tactical_rule_hybrid_agent",
    policy_mix_weight: float = 0.0,
    policy_loss_weight: float = 0.05,
    reference_rule_agent_level: str | None = "hard",
    reference_rule_opening_moves: int = 20,
    reference_rule_followup_probability: float = 0.10,
    reference_rule_only_agent_level: str | None = "hard",
) -> dict:
    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    reference_directory = tvr._reference_directory(model_directory)

    reference_paths = tvr._resolve_reference_paths(
        reference_model_path,
        model_directory,
        log_directory,
        candidate_prefix,
    )
    if not reference_paths:
        raise ValueError("at least one reference model is required")

    version_number = candidate_version if candidate_version is not None else _next_candidate_version(
        model_directory,
        candidate_prefix,
    )
    candidate_name = f"{candidate_prefix}_v{version_number}"
    candidate_path = model_directory / f"{candidate_name}.pt"
    latest_candidate_path = model_directory / f"{candidate_prefix}_candidate_latest.pt"

    candidate_agent = _initialize_candidate_agent(
        candidate_name=candidate_name,
        board_size=board_size,
        seed=seed + 1,
        device=device,
        candidate_init_model_path=candidate_init_model_path,
        policy_mix_weight=policy_mix_weight,
        policy_loss_weight=policy_loss_weight,
    )

    reference_entries = []
    for offset, reference_path in enumerate(reference_paths):
        reference_name = tvr._reference_name_from_path(reference_path)
        reference_agent = tvr._load_reference_agent(
            reference_path,
            reference_name,
            seed + offset,
            device,
            reference_rule_agent_level=reference_rule_agent_level,
            reference_rule_opening_moves=reference_rule_opening_moves,
            reference_rule_followup_probability=reference_rule_followup_probability,
        )
        reference_agent.epsilon = 0.0
        reference_entries.append(
            {
                "path": reference_path,
                "name": reference_name,
                "agent": reference_agent,
            }
        )

    if reference_rule_only_agent_level is not None:
        rule_only_name = _rule_only_reference_name(candidate_prefix, reference_rule_only_agent_level)
        reference_entries.append(
            {
                "path": None,
                "name": rule_only_name,
                "agent": tvr.RuleOnlyReferenceAgent(
                    rule_agent=tvr._build_rule_agent(reference_rule_only_agent_level),
                    name=rule_only_name,
                ),
            }
        )

    boosted_reference_names = set(tvr._latest_reference_names(reference_paths, count=5))
    historical_reference_win_rates = tvr._latest_reference_win_rates(log_directory)
    selection_rng = random.Random(seed + 10_000)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_label = tvr._log_label_for_candidate_prefix(candidate_prefix)
    training_log_path = log_directory / f"{timestamp}_{log_label}_training.log"
    winrate_log_path = training_log_path.with_name(f"{training_log_path.stem}_winrates.log")
    rolling_window = deque(maxlen=min(50, max(10, num_games)))
    summary_counter: Counter[str] = Counter()
    candidate_role_counts: Counter[tuple[str, str]] = Counter()
    reference_role_counts: Counter[tuple[str, str, str]] = Counter()
    reference_game_counts: Counter[str] = Counter()
    hard_variant_counts: Counter[tuple[str, str]] = Counter()

    _initialize_hybrid_log(
        training_log_path,
        board_size,
        reference_paths,
        candidate_path,
        reference_rule_agent_level,
        reference_rule_opening_moves,
        reference_rule_followup_probability,
        policy_mix_weight,
        policy_loss_weight,
        reference_rule_only_agent_level,
    )

    for game_index in range(1, num_games + 1):
        env = GomokuEnv(board_size=board_size)
        reference_entry = selection_rng.choice(reference_entries)
        reference_name = reference_entry["name"]
        reference_agent = reference_entry["agent"]
        reference_variant = tvr._rule_agent_variant(reference_agent)
        reference_game_counts[reference_name] += 1
        if reference_variant is not None:
            hard_variant_counts[("reference", reference_variant)] += 1

        if reference_rule_agent_level == "super_easy":
            assignments = {1: (candidate_agent.name, candidate_agent), -1: (reference_name, reference_agent)}
        elif game_index % 2 == 1:
            assignments = {1: (reference_name, reference_agent), -1: (candidate_agent.name, candidate_agent)}
        else:
            assignments = {1: (candidate_agent.name, candidate_agent), -1: (reference_name, reference_agent)}

        candidate_records: list[TorchHybridStepRecord] = []
        move_history = []
        while not env.done:
            player = env.current_player
            agent_name, agent = assignments[player]
            training = agent_name == candidate_agent.name
            action, step_record = agent.select_action(env, training=training)
            env.step(action)
            if training:
                candidate_records.append(step_record)
            move_history.append(
                {
                    "player": player,
                    "action": action,
                    "agent": agent_name,
                    "selection_reason": getattr(step_record, "selection_reason", "unknown"),
                }
            )

        rewards = tvr._reward_map(env.winner, assignments, reference_name, candidate_agent.name)
        lr_multiplier = 2.0 if reference_name in boosted_reference_names else 1.0
        lr_multiplier *= tvr._historical_reference_lr_multiplier(
            historical_reference_win_rates.get(reference_name)
        )
        if rewards[candidate_agent.name] < 0 and env.move_count <= 15:
            lr_multiplier *= 3.0

        candidate_agent.finish_game(
            candidate_records,
            rewards[candidate_agent.name],
            env.move_count,
            lr_multiplier=lr_multiplier,
        )

        winner_name = assignments[env.winner][0] if env.winner in assignments else "draw"
        summary_counter[winner_name] += 1
        rolling_window.append(winner_name)
        candidate_role = "black" if assignments[1][0] == candidate_agent.name else "white"
        if rewards[candidate_agent.name] > 0:
            candidate_outcome = "win"
        elif rewards[candidate_agent.name] < 0:
            candidate_outcome = "loss"
        else:
            candidate_outcome = "draw"
        candidate_role_counts[(candidate_role, candidate_outcome)] += 1
        reference_role_counts[(reference_name, candidate_role, candidate_outcome)] += 1

        tvr._append_reference_game_record(
            training_log_path,
            {
                "game": game_index,
                "winner": winner_name,
                "moves": env.move_count,
                "reference_role": "black" if assignments[1][0] == reference_name else "white",
                "candidate_role": candidate_role,
                "candidate_reward": rewards[candidate_agent.name],
                "candidate_epsilon": candidate_agent.epsilon,
                "reference_name": reference_name,
                "reference_variant": reference_variant,
                "teacher_variant": None,
                "move_history": move_history,
            },
            board_size,
        )

        if game_index % save_every == 0 or game_index == num_games:
            candidate_agent.save(candidate_path)
            candidate_agent.save(latest_candidate_path)

    tvr._append_reference_summary(
        training_log_path,
        summary_counter,
        rolling_window,
        num_games,
        [entry["name"] for entry in reference_entries],
        candidate_agent.name,
        hard_variant_counts,
    )
    candidate_agent.save(candidate_path)
    candidate_agent.save(latest_candidate_path)
    promoted_reference_path = reference_directory / f"{candidate_path.stem}_reference.pt"
    shutil.copyfile(candidate_path, promoted_reference_path)
    tvr._write_reference_winrate_log(
        winrate_log_path,
        candidate_name=candidate_agent.name,
        candidate_version=version_number,
        training_log_path=training_log_path,
        reference_names=[entry["name"] for entry in reference_entries],
        reference_game_counts=reference_game_counts,
        summary_counter=summary_counter,
        candidate_role_counts=candidate_role_counts,
        reference_role_counts=reference_role_counts,
    )

    candidate_wins = summary_counter[candidate_agent.name]
    reference_wins = sum(summary_counter[entry["name"]] for entry in reference_entries)
    decisive_games = max(1, candidate_wins + reference_wins)
    candidate_win_rate = candidate_wins / decisive_games

    return {
        "games": num_games,
        "training_log_path": str(training_log_path),
        "reference_model_path": ", ".join(str(path) for path in reference_paths),
        "reference_model_paths": [str(path) for path in reference_paths],
        "candidate_model_path": str(candidate_path),
        "candidate_latest_path": str(latest_candidate_path),
        "candidate_init_model_path": str(candidate_init_model_path) if candidate_init_model_path else None,
        "promoted_reference_path": str(promoted_reference_path),
        "winrate_log_path": str(winrate_log_path),
        "candidate_version": version_number,
        "candidate_name": candidate_agent.name,
        "reference_names": [entry["name"] for entry in reference_entries],
        "summary": dict(summary_counter),
        "candidate_win_rate_vs_reference": candidate_win_rate,
        "reference_rule_only_agent_level": reference_rule_only_agent_level,
        "reference_rule_agent_level": reference_rule_agent_level,
        "reference_rule_opening_moves": reference_rule_opening_moves,
        "reference_rule_followup_probability": reference_rule_followup_probability,
        "policy_mix_weight": policy_mix_weight,
        "policy_loss_weight": policy_loss_weight,
    }


def _initialize_candidate_agent(
    candidate_name: str,
    board_size: int,
    seed: int,
    device: str,
    policy_mix_weight: float,
    policy_loss_weight: float,
    candidate_init_model_path: str | Path | None = None,
) -> TorchHybridAgent:
    if candidate_init_model_path is not None:
        agent = TorchHybridAgent.load(candidate_init_model_path, name=candidate_name, device=device)
        agent.policy_mix_weight = policy_mix_weight
        agent.policy_loss_weight = policy_loss_weight
        return agent
    return TorchHybridAgent(
        name=candidate_name,
        board_size=board_size,
        learning_rate=2e-4,
        policy_mix_weight=policy_mix_weight,
        policy_loss_weight=policy_loss_weight,
        epsilon_start=0.2,
        epsilon_end=0.02,
        epsilon_decay=0.998,
        seed=seed,
        device=device,
    )


def _initialize_hybrid_log(
    path: Path,
    board_size: int,
    reference_paths: list[Path],
    candidate_path: Path,
    reference_rule_agent_level: str | None,
    reference_rule_opening_moves: int,
    reference_rule_followup_probability: float,
    policy_mix_weight: float,
    policy_loss_weight: float,
    reference_rule_only_agent_level: str | None,
) -> None:
    lines = [
        "Gomoku Hybrid Reference Training Log",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Board size: {board_size}x{board_size}",
        f"Reference model: {', '.join(str(reference_path) for reference_path in reference_paths)}",
        f"Candidate model: {candidate_path}",
        f"Reference rule overlay: {reference_rule_agent_level or 'none'}",
        f"Reference rule opening moves: {reference_rule_opening_moves}",
        f"Reference rule follow-up probability: {reference_rule_followup_probability:.3f}",
        f"Policy decision mix weight: {policy_mix_weight:.3f}",
        f"Policy loss weight: {policy_loss_weight:.3f}",
        f"Reference rule-only agent: {reference_rule_only_agent_level or 'none'}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _rule_only_reference_name(candidate_prefix: str, reference_rule_only_agent_level: str) -> str:
    return f"{candidate_prefix}_{reference_rule_only_agent_level}_rule_only_reference"


def _next_candidate_version(model_directory: Path, candidate_prefix: str) -> int:
    version_numbers = []
    pattern = re.compile(rf"{re.escape(candidate_prefix)}_v(\d+)$")
    for path in model_directory.glob(f"{candidate_prefix}_v*.pt"):
        match = pattern.fullmatch(path.stem)
        if match:
            version_numbers.append(int(match.group(1)))
    return max(version_numbers, default=0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid candidate against frozen references")
    parser.add_argument("--games", type=int, default=1000, help="number of games")
    parser.add_argument("--board-size", type=int, default=15, help="board size")
    parser.add_argument("--save-every", type=int, default=1000, help="checkpoint interval")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    parser.add_argument("--log-dir", type=str, default="logs", help="directory for training logs")
    parser.add_argument("--model-dir", type=str, default="models", help="directory for model checkpoints")
    parser.add_argument(
        "--candidate-init-model",
        type=str,
        default=None,
        help="optional hybrid checkpoint to continue training from",
    )
    parser.add_argument(
        "--candidate-version",
        type=int,
        default=None,
        help="explicit candidate version number to write",
    )
    parser.add_argument(
        "--candidate-prefix",
        type=str,
        default="tactical_rule_hybrid_agent",
        help="candidate filename/name prefix, e.g. tactical_rule_hybrid_agent",
    )
    parser.add_argument(
        "--reference-model",
        action="append",
        default=None,
        help="path to frozen reference model; repeat to use multiple references in rotation",
    )
    parser.add_argument(
        "--reference-rule-agent-level",
        type=str,
        choices=("super_easy", "easy", "normal", "hard"),
        default="hard",
        help="apply a rule-based opening overlay to value_agent references only",
    )
    parser.add_argument(
        "--reference-rule-opening-moves",
        type=int,
        default=20,
        help="opening move count where the rule overlay controls reference models",
    )
    parser.add_argument(
        "--reference-rule-followup-probability",
        type=float,
        default=0.10,
        help="probability of falling back to the rule overlay after the opening window",
    )
    parser.add_argument(
        "--reference-rule-only-agent-level",
        type=str,
        choices=("super_easy", "easy", "normal", "hard"),
        default="hard",
        help="add a pure rule-based reference opponent at the chosen strength",
    )
    parser.add_argument(
        "--policy-mix-weight",
        type=float,
        default=0.0,
        help="how much policy head probability contributes when selecting moves",
    )
    parser.add_argument(
        "--policy-aux-weight",
        type=float,
        default=0.05,
        help="policy loss weight when policy is trained separately",
    )
    parser.add_argument(
        "--reference-cycle-length",
        type=int,
        default=10,
        help="number of consecutive games to play against one reference before rotating",
    )
    args = parser.parse_args()

    result = train_against_reference(
        num_games=args.games,
        board_size=args.board_size,
        seed=args.seed,
        save_every=args.save_every,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        reference_model_path=args.reference_model,
        candidate_init_model_path=args.candidate_init_model,
        candidate_version=args.candidate_version,
        device=args.device,
        candidate_prefix=args.candidate_prefix,
        reference_rule_agent_level=args.reference_rule_agent_level,
        reference_rule_opening_moves=args.reference_rule_opening_moves,
        reference_rule_followup_probability=args.reference_rule_followup_probability,
        reference_rule_only_agent_level=args.reference_rule_only_agent_level,
        policy_mix_weight=args.policy_mix_weight,
        policy_loss_weight=args.policy_aux_weight,
    )
    print(
        f"Hybrid reference training finished. Log: {result['training_log_path']} | "
        f"Reference: {result['reference_model_path']} | "
        f"Init: {result['candidate_init_model_path'] or 'random'} | "
        f"Candidate: {result['candidate_model_path']} | "
        f"Promoted: {result['promoted_reference_path']} | "
        f"Win rate vs reference: {result['candidate_win_rate_vs_reference']:.2%}"
    )


if __name__ == "__main__":
    main()
