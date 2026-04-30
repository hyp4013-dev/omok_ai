"""Train a tactical policy candidate against a rule-augmented reference and hard teacher."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
import random
import shutil

import train_value_reference as tvr
from agent.tactical_rule_agent import build_random_hard_tactical_rule_agent
from agent.torch_policy_only_agent import TorchPolicyOnlyAgent, TorchPolicyOnlyStepRecord
from env.gomoku_env import GomokuEnv


def train_against_reference(
    num_games: int = 1000,
    board_size: int = 15,
    seed: int = 7,
    save_every: int = 1000,
    log_dir: str | Path = "logs",
    model_dir: str | Path = "models",
    candidate_init_model_path: str | Path | None = None,
    candidate_feature_init_model_path: str | Path | None = None,
    reference_model_path: str | Path | list[str | Path] | None = None,
    candidate_version: int | None = None,
    device: str = "cpu",
    candidate_prefix: str = "tactical_rule_policy_agent",
    reference_rule_agent_level: str = "none",
    reference_rule_opening_moves: int = 20,
    reference_rule_followup_probability: float = 0.1,
    teacher_rule_agent_level: str = "hard",
    teacher_weight: float = 1.0,
    opening_teacher_moves: int = 20,
) -> dict:
    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    reference_directory = tvr._reference_directory(model_directory)

    version_number = candidate_version if candidate_version is not None else _next_candidate_version(
        model_directory,
        candidate_prefix,
    )
    candidate_name = f"{candidate_prefix}_v{version_number}"
    candidate_path = model_directory / f"{candidate_name}.pt"
    latest_candidate_path = model_directory / f"{candidate_prefix}_candidate_latest.pt"

    candidate_init_description, candidate_feature_init_description, candidate_agent = _build_candidate_agent(
        candidate_init_model_path,
        candidate_feature_init_model_path,
        candidate_name,
        board_size,
        device,
    )
    candidate_agent.teacher_weight = teacher_weight
    candidate_agent.teacher_aux_weight = 0.0
    reference_paths = tvr._resolve_reference_paths(
        reference_model_path,
        model_directory,
        log_directory,
        candidate_prefix="tactical_value_agent",
    )
    for reference_path in reference_paths:
        if reference_path.exists():
            continue
        latest_path = model_directory / "value_agent_latest.json"
        if not latest_path.exists():
            raise FileNotFoundError("value_agent_latest.json not found; cannot create reference v1")
        shutil.copyfile(latest_path, reference_path)
    reference_entries = []
    reference_rule_level = reference_rule_agent_level
    reference_overlay_level = None if reference_rule_level == "none" else (
        "super_easy" if reference_rule_level == "very_easy" else reference_rule_level
    )
    reference_overlay_exclusion_prefixes = ("tactical_rule_value_agent",)
    candidate_black_priority = reference_rule_level == "very_easy"
    for offset, reference_path in enumerate(reference_paths):
        reference_name = reference_path.stem
        per_reference_overlay_level = _reference_overlay_level_for_path(
            reference_path,
            base_overlay_level=reference_overlay_level,
            exclusion_prefixes=reference_overlay_exclusion_prefixes,
        )
        reference_agent = tvr._load_reference_agent(
            reference_path,
            reference_name=reference_name,
            seed=seed + offset,
            device=device,
            reference_rule_agent_level=per_reference_overlay_level,
            reference_rule_opening_moves=reference_rule_opening_moves,
            reference_rule_followup_probability=reference_rule_followup_probability,
        )
        reference_agent.epsilon = 0.0
        reference_entries.append(
            {
                "path": reference_path,
                "name": reference_name,
                "agent": reference_agent,
                "overlay_level": per_reference_overlay_level,
            }
        )
    teacher_agent = _build_rule_agent(teacher_rule_agent_level)
    boosted_reference_names = set(tvr._latest_reference_names(reference_paths, count=5))
    historical_reference_win_rates = tvr._latest_reference_win_rates(log_directory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log_path = log_directory / f"{timestamp}_tactical_rule_policy_reference_training.log"
    winrate_log_path = training_log_path.with_name(f"{training_log_path.stem}_winrates.log")
    _initialize_policy_log(
        training_log_path,
        board_size,
        candidate_path,
        candidate_init_description,
        candidate_feature_init_description,
        reference_paths,
        reference_rule_level,
        reference_overlay_level,
        reference_overlay_exclusion_prefixes,
        candidate_black_priority,
        reference_rule_opening_moves,
        reference_rule_followup_probability,
        teacher_rule_agent_level,
        teacher_weight,
        opening_teacher_moves,
    )

    rolling_window = deque(maxlen=min(50, max(10, num_games)))
    summary_counter: Counter[str] = Counter()
    candidate_role_counts: Counter[tuple[str, str]] = Counter()
    reference_role_counts: Counter[tuple[str, str, str]] = Counter()
    reference_game_counts: Counter[str] = Counter()
    hard_variant_counts: Counter[tuple[str, str]] = Counter()
    selection_rng = random.Random(seed + 10_000)

    for game_index in range(1, num_games + 1):
        env = GomokuEnv(board_size=board_size)
        reference_entry = selection_rng.choice(reference_entries)
        reference_name = reference_entry["name"]
        reference_agent = reference_entry["agent"]
        reference_game_counts[reference_name] += 1
        if candidate_black_priority:
            assignments = {1: (candidate_agent.name, candidate_agent), -1: (reference_name, reference_agent)}
        elif game_index % 2 == 1:
            assignments = {1: (reference_name, reference_agent), -1: (candidate_agent.name, candidate_agent)}
        else:
            assignments = {1: (candidate_agent.name, candidate_agent), -1: (reference_name, reference_agent)}

        candidate_records: list[TorchPolicyOnlyStepRecord] = []
        move_history = []
        while not env.done:
            player = env.current_player
            agent_name, agent = assignments[player]
            step_record = None
            if agent_name == candidate_agent.name:
                teacher_action, _ = teacher_agent.select_action(env, training=False)
                teacher_action_index = env.action_to_index(teacher_action)
                if env.move_count == 0:
                    action, step_record = agent.select_action(
                        env,
                        training=True,
                    )
                    step_record.teacher_action_index = teacher_action_index
                    step_record.teacher_board_tensor = candidate_agent._board_tensor_after_action(env, teacher_action)
                    candidate_records.append(step_record)
                else:
                    if env.move_count < opening_teacher_moves:
                        action = teacher_action
                        step_record = candidate_agent.build_teacher_forced_record(env, teacher_action)
                        candidate_records.append(step_record)
                    else:
                        action, step_record = agent.select_action(
                            env,
                            training=True,
                            teacher_action_index=teacher_action_index,
                        )
                        step_record.teacher_board_tensor = candidate_agent._board_tensor_after_action(env, teacher_action)
                        candidate_records.append(step_record)
            else:
                action, _ = agent.select_action(env, training=False)
            env.step(action)
            move_history.append(
                {
                    "player": player,
                    "action": action,
                    "agent": agent_name,
                    "selection_reason": (
                        step_record.selection_reason if step_record is not None else "model"
                    ),
                }
            )

        rewards = _reward_map(env.winner, assignments, reference_agent.name, candidate_agent.name)
        lr_multiplier = 2.0 if reference_name in boosted_reference_names else 1.0
        lr_multiplier *= tvr._historical_reference_lr_multiplier(
            historical_reference_win_rates.get(reference_name)
        )
        if rewards[candidate_agent.name] < 0 and env.move_count <= 15:
            lr_multiplier *= 3.0
        game_metrics = candidate_agent.finish_game(
            candidate_records,
            rewards[candidate_agent.name],
            env.move_count,
            teacher_weight=teacher_weight,
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
        reference_role_counts[(reference_agent.name, candidate_role, candidate_outcome)] += 1
        teacher_variant = getattr(teacher_agent, "variant_label", None)
        reference_variant = getattr(reference_agent, "variant_label", None)
        if reference_variant is not None:
            hard_variant_counts[("reference", reference_variant)] += 1
        if teacher_variant is not None:
            hard_variant_counts[("teacher", teacher_variant)] += 1

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
                "metrics": game_metrics,
                "reference_name": reference_name,
                "reference_variant": reference_variant,
                "teacher_variant": teacher_variant,
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
        "candidate_init_model_path": candidate_init_description,
        "candidate_feature_init_model_path": candidate_feature_init_description,
        "reference_init_model_path": ", ".join(str(path) for path in reference_paths),
        "promoted_reference_path": str(promoted_reference_path),
        "winrate_log_path": str(winrate_log_path),
        "candidate_version": version_number,
        "candidate_name": candidate_agent.name,
        "reference_names": [entry["name"] for entry in reference_entries],
        "summary": dict(summary_counter),
        "candidate_win_rate_vs_reference": candidate_win_rate,
        "reference_rule_level": reference_rule_level,
        "candidate_black_priority": candidate_black_priority,
        "boosted_reference_names": list(boosted_reference_names),
        "reference_rule_opening_moves": reference_rule_opening_moves,
        "reference_rule_followup_probability": reference_rule_followup_probability,
        "teacher_rule_agent_level": teacher_rule_agent_level,
        "teacher_weight": teacher_weight,
        "opening_teacher_moves": opening_teacher_moves,
        "historical_reference_win_rates": historical_reference_win_rates,
    }


def _build_rule_agent(level: str):
    if level == "hard":
        return build_random_hard_tactical_rule_agent()
    if level == "none":
        raise ValueError("reference rule level 'none' should not build a rule agent")
    raise ValueError(f"unknown rule level: {level}")


def _reference_overlay_level_for_path(
    reference_path: Path,
    base_overlay_level: str | None,
    exclusion_prefixes: tuple[str, ...],
) -> str | None:
    if base_overlay_level is None:
        return None
    stem = reference_path.stem
    if any(stem.startswith(prefix) for prefix in exclusion_prefixes):
        return None
    return base_overlay_level


def _reward_map(
    winner: int,
    assignments: dict[int, tuple[str, object]],
    reference_name: str,
    candidate_name: str,
) -> dict[str, float]:
    rewards = {reference_name: 0.0, candidate_name: 0.0}
    if winner == 0:
        # Treat draws as a loss for training so prolonged neutral games are penalized.
        rewards[candidate_name] = -1.0
        rewards[reference_name] = 1.0
        return rewards

    winner_name = assignments[winner][0]
    loser_name = assignments[-winner][0]
    rewards[winner_name] = 1.0
    rewards[loser_name] = -1.0
    return rewards


def _initialize_policy_log(
    path: Path,
    board_size: int,
    candidate_path: Path,
    candidate_init_model_path: str | Path,
    candidate_feature_init_model_path: str | Path | None,
    reference_model_path: str | Path | list[str | Path] | None,
    reference_rule_level: str,
    reference_overlay_level: str | None,
    reference_overlay_exclusion_prefixes: tuple[str, ...],
    candidate_black_priority: bool,
    reference_rule_opening_moves: int,
    reference_rule_followup_probability: float,
    teacher_rule_agent_level: str,
    teacher_weight: float,
    opening_teacher_moves: int,
) -> None:
    if isinstance(reference_model_path, list):
        reference_model_path_text = ", ".join(str(path) for path in reference_model_path)
    else:
        reference_model_path_text = str(reference_model_path) if reference_model_path is not None else "all refer models"
    lines = [
        "Gomoku Tactical Policy Reference Training Log",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Board size: {board_size}x{board_size}",
        f"Candidate model: {candidate_path}",
        f"Candidate init model: {candidate_init_model_path}",
        f"Candidate feature init model: {candidate_feature_init_model_path if candidate_feature_init_model_path is not None else 'none'}",
        f"Reference init model: {reference_model_path_text}",
        f"Reference rule level: {reference_rule_level}",
        f"Reference rule overlay exclusion prefixes: {', '.join(reference_overlay_exclusion_prefixes) or 'none'}",
        f"Candidate black priority: {candidate_black_priority}",
        f"Reference rule opening moves: {reference_rule_opening_moves}",
        f"Reference rule followup probability: {reference_rule_followup_probability:.3f}",
        f"Teacher rule agent: {teacher_rule_agent_level}",
        f"Teacher weight: {teacher_weight:.3f}",
        f"Opening teacher moves: {opening_teacher_moves}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _next_candidate_version(model_directory: Path, candidate_prefix: str) -> int:
    versions = []
    for path in model_directory.glob(f"{candidate_prefix}_v*.pt"):
        stem = path.stem
        if stem.startswith(f"{candidate_prefix}_v"):
            suffix = stem.removeprefix(f"{candidate_prefix}_v")
            if suffix.isdigit():
                versions.append(int(suffix))
    return max(versions, default=0) + 1


def _build_candidate_agent(
    candidate_init_model_path: str | Path | None,
    candidate_feature_init_model_path: str | Path | None,
    candidate_name: str,
    board_size: int,
    device: str,
) -> tuple[str, str | None, TorchPolicyOnlyAgent]:
    if candidate_init_model_path is None:
        candidate_agent = TorchPolicyOnlyAgent.create_blank(
            name=candidate_name,
            board_size=board_size,
            device=device,
        )
        feature_description = None
        if candidate_feature_init_model_path is not None:
            feature_path = Path(candidate_feature_init_model_path)
            candidate_agent = TorchPolicyOnlyAgent.load_with_feature_checkpoint(
                feature_path,
                name=candidate_name,
                device=device,
            )
            feature_description = str(feature_path)
        return "random", feature_description, candidate_agent
    init_path = Path(candidate_init_model_path)
    try:
        candidate_agent = TorchPolicyOnlyAgent.load(init_path, name=candidate_name, device=device)
        return str(init_path), None, candidate_agent
    except Exception:
        candidate_agent = TorchPolicyOnlyAgent.load_from_value_checkpoint(
            init_path,
            name=candidate_name,
            device=device,
        )
        return str(init_path), None, candidate_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tactical policy candidate against a tactical reference")
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
        help="policy checkpoint used to bootstrap the candidate; defaults to random initialization",
    )
    parser.add_argument(
        "--candidate-feature-init-model",
        type=str,
        default=None,
        help="policy feature checkpoint to bootstrap the candidate trunk; defaults to none",
    )
    parser.add_argument(
        "--reference-model-path",
        nargs="*",
        default=None,
        help="reference model(s) to augment with rule overlay; default uses all refer models",
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
        default="tactical_rule_policy_agent",
        help="candidate filename/name prefix",
    )
    parser.add_argument(
        "--reference-rule-level",
        "--reference-rule-agent-level",
        type=str,
        choices=("none", "very_easy", "super_easy", "easy", "normal", "hard"),
        default="none",
        help="rule level applied to reference models; use none for no rule overlay",
    )
    parser.add_argument(
        "--reference-rule-opening-moves",
        type=int,
        default=20,
        help="opening move count where the reference overlay controls the reference model",
    )
    parser.add_argument(
        "--reference-rule-followup-probability",
        type=float,
        default=0.1,
        help="probability of falling back to the rule overlay after the opening window",
    )
    parser.add_argument(
        "--teacher-rule-agent-level",
        type=str,
        choices=("super_easy", "easy", "normal", "hard"),
        default="hard",
        help="teacher strength used for imitation targets",
    )
    parser.add_argument(
        "--teacher-weight",
        type=float,
        default=1.0,
        help="weight of teacher imitation loss",
    )
    parser.add_argument(
        "--opening-teacher-moves",
        type=int,
        default=20,
        help="number of early plies forced by the rule teacher before policy takes over",
    )
    args = parser.parse_args()

    result = train_against_reference(
        num_games=args.games,
        board_size=args.board_size,
        seed=args.seed,
        save_every=args.save_every,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        candidate_init_model_path=args.candidate_init_model,
        candidate_feature_init_model_path=args.candidate_feature_init_model,
        reference_model_path=args.reference_model_path,
        candidate_version=args.candidate_version,
        device=args.device,
        candidate_prefix=args.candidate_prefix,
        reference_rule_agent_level=args.reference_rule_level,
        reference_rule_opening_moves=args.reference_rule_opening_moves,
        reference_rule_followup_probability=args.reference_rule_followup_probability,
        teacher_rule_agent_level=args.teacher_rule_agent_level,
        teacher_weight=args.teacher_weight,
        opening_teacher_moves=args.opening_teacher_moves,
    )
    print(
        f"Tactical policy training finished. Log: {result['training_log_path']} | "
        f"Init: {result['candidate_init_model_path']} | "
        f"Candidate: {result['candidate_model_path']} | "
        f"Promoted: {result['promoted_reference_path']} | "
        f"Win rate vs reference: {result['candidate_win_rate_vs_reference']:.2%}"
    )


if __name__ == "__main__":
    main()
