"""Train a CNN value candidate against the rule-based tactical agent."""

from __future__ import annotations

import argparse
import re
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from agent.tactical_rule_agent import (
    EasyTacticalRuleAgent,
    HardTacticalRuleAgent,
    build_random_hard_tactical_rule_agent,
    NormalTacticalRuleAgent,
    SuperEasyTacticalRuleAgent,
    TacticalRuleAgent,
)
from agent.torch_cnn_value_agent import TorchCNNValueAgent, TorchCNNValueStepRecord
from env.gomoku_env import GomokuEnv


def _reference_directory(model_directory: Path) -> Path:
    reference_directory = model_directory / "refer"
    reference_directory.mkdir(parents=True, exist_ok=True)
    return reference_directory


def _training_log_label(candidate_prefix: str) -> str:
    if candidate_prefix == "tactical_rule_value_agent":
        return "tactical_rule_value"
    if candidate_prefix == "tactical_value_agent":
        return "tactical_value"
    return candidate_prefix.removesuffix("_agent")


def train_against_tactical_agent(
    num_games: int = 1000,
    board_size: int = 15,
    seed: int = 7,
    save_every: int = 1000,
    log_dir: str | Path = "logs",
    model_dir: str | Path = "models",
    candidate_init_model_path: str | Path | None = None,
    candidate_version: int | None = None,
    candidate_prefix: str = "tactical_value_agent",
    rule_agent_level: str = "normal",
    teacher_rule_agent_level: str | None = None,
    teacher_weight: float = 1.0,
    device: str = "cpu",
) -> dict:
    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    resolved_init_model_path = _resolve_tactical_init_model_path(
        candidate_init_model_path,
        model_directory,
        candidate_prefix,
    )

    version_number = candidate_version if candidate_version is not None else _next_tactical_candidate_version(
        model_directory,
        candidate_prefix,
    )
    candidate_name = f"{candidate_prefix}_v{version_number}"
    candidate_path = model_directory / f"{candidate_name}.pt"
    latest_candidate_path = model_directory / f"{candidate_prefix}_candidate_latest.pt"
    tactical_agent = _build_rule_agent(rule_agent_level)
    teacher_agent = _build_rule_agent(teacher_rule_agent_level or rule_agent_level)
    candidate_agent = _initialize_candidate_agent(
        candidate_name=candidate_name,
        board_size=board_size,
        seed=seed + 1,
        device=device,
        candidate_init_model_path=resolved_init_model_path,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_label = _training_log_label(candidate_prefix)
    training_log_path = log_directory / f"{timestamp}_{log_label}_training.log"
    winrate_log_path = training_log_path.with_name(f"{training_log_path.stem}_winrates.log")
    rolling_window = deque(maxlen=min(50, max(10, num_games)))
    summary_counter: Counter[str] = Counter()
    role_outcome_counts: Counter[tuple[str, str]] = Counter()
    hard_variant_counts: Counter[tuple[str, str]] = Counter()
    _initialize_log(
        training_log_path,
        board_size,
        candidate_path,
        num_games,
        resolved_init_model_path,
        candidate_prefix,
        tactical_agent.name,
    )

    for game_index in range(1, num_games + 1):
        env = GomokuEnv(board_size=board_size)
        assignments = {1: (candidate_agent.name, candidate_agent), -1: (tactical_agent.name, tactical_agent)}
        tactical_variant = _rule_agent_variant(tactical_agent)
        teacher_variant = _rule_agent_variant(teacher_agent)
        if tactical_variant is not None:
            hard_variant_counts[("tactical", tactical_variant)] += 1
        if teacher_rule_agent_level == "hard" and teacher_variant is not None:
            hard_variant_counts[("teacher", teacher_variant)] += 1

        candidate_records: list[TorchCNNValueStepRecord] = []
        move_history = []
        while not env.done:
            player = env.current_player
            agent_name, agent = assignments[player]
            training = agent_name == candidate_agent.name
            action, step_record = agent.select_action(env, training=training)
            if training:
                teacher_action, _ = teacher_agent.select_action(env, training=False)
                step_record.teacher_board_tensor = candidate_agent._board_tensor_after_action(
                    env,
                    teacher_action,
                )
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

        rewards = _reward_map(env.winner, assignments, tactical_agent.name, candidate_agent.name)
        lr_multiplier = 3.0 if rewards[candidate_agent.name] < 0 and env.move_count <= 15 else 1.0
        candidate_agent.finish_game(
            candidate_records,
            rewards[candidate_agent.name],
            env.move_count,
            lr_multiplier=lr_multiplier,
            teacher_weight=teacher_weight,
        )

        winner_name = assignments[env.winner][0] if env.winner in assignments else "draw"
        summary_counter[winner_name] += 1
        rolling_window.append(winner_name)
        candidate_role = "black" if assignments[1][0] == candidate_agent.name else "white"
        candidate_outcome = "draw"
        if winner_name == candidate_agent.name:
            candidate_outcome = "win"
        elif winner_name == tactical_agent.name:
            candidate_outcome = "loss"
        role_outcome_counts[(candidate_role, candidate_outcome)] += 1

        _append_game_record(
            training_log_path,
            {
                "game": game_index,
                "winner": winner_name,
                "moves": env.move_count,
                "tactical_role": "black" if assignments[1][0] == tactical_agent.name else "white",
                "candidate_role": candidate_role,
                "candidate_reward": rewards[candidate_agent.name],
                "candidate_epsilon": candidate_agent.epsilon,
                "tactical_variant": tactical_variant,
                "teacher_variant": teacher_variant if teacher_rule_agent_level == "hard" else None,
                "move_history": move_history,
            },
            board_size,
        )

        if game_index % save_every == 0 or game_index == num_games:
            candidate_agent.save(candidate_path)
            candidate_agent.save(latest_candidate_path)

    _append_summary(
        training_log_path,
        summary_counter,
        rolling_window,
        total_games=num_games,
        tactical_name=tactical_agent.name,
        candidate_name=candidate_agent.name,
        hard_variant_counts=hard_variant_counts,
    )
    candidate_agent.save(candidate_path)
    candidate_agent.save(latest_candidate_path)

    candidate_wins = summary_counter[candidate_agent.name]
    tactical_wins = summary_counter[tactical_agent.name]
    decisive_games = max(1, candidate_wins + tactical_wins)
    candidate_win_rate = candidate_wins / decisive_games
    _write_tactical_winrate_log(
        winrate_log_path,
        training_log_path=training_log_path,
        candidate_name=candidate_agent.name,
        candidate_version=version_number,
        tactical_name=tactical_agent.name,
        summary_counter=summary_counter,
        role_outcome_counts=role_outcome_counts,
        total_games=num_games,
    )
    return {
        "games": num_games,
        "training_log_path": str(training_log_path),
        "winrate_log_path": str(winrate_log_path),
        "candidate_model_path": str(candidate_path),
        "candidate_latest_path": str(latest_candidate_path),
        "candidate_init_model_path": str(resolved_init_model_path) if resolved_init_model_path else None,
        "candidate_version": version_number,
        "candidate_name": candidate_agent.name,
        "opponent_name": tactical_agent.name,
        "teacher_name": teacher_agent.name,
        "summary": dict(summary_counter),
        "candidate_win_rate_vs_tactical": candidate_win_rate,
        "teacher_weight": teacher_weight,
    }


def _initialize_candidate_agent(
    candidate_name: str,
    board_size: int,
    seed: int,
    device: str,
    candidate_init_model_path: str | Path | None = None,
) -> TorchCNNValueAgent:
    if candidate_init_model_path is not None:
        agent = TorchCNNValueAgent.load(candidate_init_model_path, name=candidate_name, device=device)
        agent.name = candidate_name
        return agent
    return TorchCNNValueAgent(
        name=candidate_name,
        board_size=board_size,
        learning_rate=1e-4,
        imitation_weight=0.0,
        epsilon_decay=0.999,
        seed=seed,
        device=device,
    )


def _build_rule_agent(rule_agent_level: str) -> TacticalRuleAgent:
    if rule_agent_level == "super_easy":
        return SuperEasyTacticalRuleAgent()
    if rule_agent_level == "easy":
        return EasyTacticalRuleAgent()
    if rule_agent_level == "normal":
        return NormalTacticalRuleAgent()
    if rule_agent_level == "hard":
        return build_random_hard_tactical_rule_agent()
    raise ValueError(f"unsupported rule agent level: {rule_agent_level}")


def _rule_agent_variant(agent: Any) -> str | None:
    variant = getattr(agent, "variant_label", None)
    if variant is None:
        return None
    return str(variant)


def _resolve_tactical_init_model_path(
    candidate_init_model_path: str | Path | None,
    model_directory: Path,
    candidate_prefix: str,
) -> Path | None:
    if candidate_init_model_path is not None:
        return Path(candidate_init_model_path)
    return _latest_prefixed_model_path(model_directory, candidate_prefix) or _latest_value_model_path(model_directory)


def _latest_prefixed_model_path(model_directory: Path, candidate_prefix: str) -> Path | None:
    reference_directory = _reference_directory(model_directory)
    versioned_paths: list[tuple[int, Path]] = []
    pattern = re.compile(rf"{re.escape(candidate_prefix)}_v(\d+)")
    for path in model_directory.glob(f"{candidate_prefix}_v*.pt"):
        if path.stem.endswith("_reference"):
            continue
        if candidate_prefix == "tactical_value_agent" and not (
            reference_directory / f"{path.stem}_reference.pt"
        ).exists():
            continue
        match = pattern.fullmatch(path.stem)
        if match:
            versioned_paths.append((int(match.group(1)), path))
    if not versioned_paths:
        return None
    return max(versioned_paths, key=lambda item: item[0])[1]


def _latest_value_model_path(model_directory: Path) -> Path | None:
    versioned_paths: list[tuple[int, Path]] = []
    pattern = re.compile(r"value_agent_v(\d+)")
    for path in model_directory.glob("value_agent_v*.pt"):
        if path.stem.endswith("_reference"):
            continue
        match = pattern.fullmatch(path.stem)
        if match:
            versioned_paths.append((int(match.group(1)), path))
    if not versioned_paths:
        return None
    return max(versioned_paths, key=lambda item: item[0])[1]


def _next_tactical_candidate_version(model_directory: Path, candidate_prefix: str) -> int:
    version_numbers = []
    for path in model_directory.glob(f"{candidate_prefix}_v*.pt"):
        match = re.fullmatch(rf"{re.escape(candidate_prefix)}_v(\d+)", path.stem)
        if match:
            version_numbers.append(int(match.group(1)))
    return max(version_numbers, default=0) + 1


def _reward_map(
    winner: int,
    assignments: dict[int, tuple[str, object]],
    tactical_name: str,
    candidate_name: str,
) -> dict[str, float]:
    rewards = {tactical_name: 0.0, candidate_name: 0.0}
    if winner == 0:
        return rewards
    winner_name = assignments[winner][0]
    loser_name = assignments[-winner][0]
    rewards[winner_name] = 1.0
    rewards[loser_name] = -1.0
    return rewards


def _initialize_log(
    path: Path,
    board_size: int,
    candidate_path: Path,
    total_games: int,
    init_model_path: Path | None,
    candidate_prefix: str,
    opponent_name: str,
) -> None:
    title = "Gomoku Tactical Value Training Log"
    if candidate_prefix == "tactical_rule_value_agent":
        title = "Gomoku Tactical Rule Value Training Log"
    elif candidate_prefix != "tactical_value_agent":
        title = f"Gomoku {candidate_prefix} Training Log"
    lines = [
        title,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Board size: {board_size}x{board_size}",
        f"Opponent: {opponent_name}",
        f"Candidate model: {candidate_path}",
        f"Candidate init model: {init_model_path if init_model_path is not None else 'random'}",
        f"Total target games: {total_games}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_game_record(path: Path, record: dict, board_size: int) -> None:
    extra_fields = []
    if record.get("tactical_variant") is not None:
        extra_fields.append(f"tactical_variant={record['tactical_variant']}")
    if record.get("teacher_variant") is not None:
        extra_fields.append(f"teacher_variant={record['teacher_variant']}")
    extra_suffix = f", {', '.join(extra_fields)}" if extra_fields else ""
    lines = [
        f"Game {record['game']}: winner={record['winner']}, moves={record['moves']}, "
        f"board={board_size}x{board_size}, tactical_role={record['tactical_role']}, "
        f"candidate_role={record['candidate_role']}, candidate_reward={record['candidate_reward']:.1f}, "
        f"candidate_epsilon={record['candidate_epsilon']:.4f}{extra_suffix}",
        "Move record",
    ]
    for move_index, move in enumerate(record["move_history"], start=1):
        row, col = move["action"]
        player_name = "Black" if move["player"] == 1 else "White"
        selection_reason = move.get("selection_reason")
        reason_suffix = (
            f" [{selection_reason}]"
            if selection_reason in {"random", "forced", "opening_random"}
            else ""
        )
        lines.append(
            f"  {move_index:>3}. {player_name} ({move['agent']}) -> (row={row}, col={col}){reason_suffix}"
        )
    lines.append("")
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines) + "\n")


def _append_summary(
    path: Path,
    summary_counter: Counter[str],
    rolling_window: deque[str],
    total_games: int,
    tactical_name: str,
    candidate_name: str,
    hard_variant_counts: Counter[tuple[str, str]] | None = None,
) -> None:
    tactical_wins = summary_counter[tactical_name]
    candidate_wins = summary_counter[candidate_name]
    draws = summary_counter["draw"]
    decisive_games = max(1, tactical_wins + candidate_wins)
    lines = [
        "Summary",
        "",
        f"- Total games: {total_games}",
        f"- {tactical_name} wins: {tactical_wins}",
        f"- {candidate_name} wins: {candidate_wins}",
        f"- Draws: {draws}",
        f"- Candidate win rate vs tactical agent: {candidate_wins / decisive_games:.2%}",
        f"- Rolling window size: {len(rolling_window)}",
        f"- Rolling tactical wins: {sum(1 for winner in rolling_window if winner == tactical_name)}",
        f"- Rolling candidate wins: {sum(1 for winner in rolling_window if winner == candidate_name)}",
        f"- Rolling draws: {sum(1 for winner in rolling_window if winner == 'draw')}",
    ]
    hard_variant_counts = hard_variant_counts or Counter()
    if hard_variant_counts:
        lines.append("- Hard variant usage:")
        for role in ("tactical", "teacher"):
            for variant in ("offensive", "neutral", "defensive"):
                lines.append(f"  - {role} {variant}: {hard_variant_counts[(role, variant)]}")
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines).rstrip() + "\n")


def _write_tactical_winrate_log(
    path: Path,
    training_log_path: Path,
    candidate_name: str,
    candidate_version: int,
    tactical_name: str,
    summary_counter: Counter[str],
    role_outcome_counts: Counter[tuple[str, str]],
    total_games: int,
) -> None:
    candidate_wins = summary_counter[candidate_name]
    tactical_wins = summary_counter[tactical_name]
    draws = summary_counter["draw"]
    decisive_games = max(1, candidate_wins + tactical_wins)

    black_wins = role_outcome_counts[("black", "win")]
    black_losses = role_outcome_counts[("black", "loss")]
    black_draws = role_outcome_counts[("black", "draw")]
    white_wins = role_outcome_counts[("white", "win")]
    white_losses = role_outcome_counts[("white", "loss")]
    white_draws = role_outcome_counts[("white", "draw")]

    lines = [
        "Gomoku Tactical Value Winrate Log",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Candidate: {candidate_name}",
        f"Candidate version: v{candidate_version}",
        f"Training log: {training_log_path}",
        f"Opponent: {tactical_name}",
        f"Total games: {total_games}",
        f"Candidate wins: {candidate_wins}",
        f"Opponent wins: {tactical_wins}",
        f"Draws: {draws}",
        f"Candidate win rate vs tactical agent: {candidate_wins / decisive_games:.2%}",
        (
            f"Candidate black win rate: {black_wins}/{black_wins + black_losses} "
            f"({black_wins / max(1, black_wins + black_losses):.2%}) | "
            f"losses={black_losses} | draws={black_draws}"
        ),
        (
            f"Candidate white win rate: {white_wins}/{white_wins + white_losses} "
            f"({white_wins / max(1, white_wins + white_losses):.2%}) | "
            f"losses={white_losses} | draws={white_draws}"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tactical-value candidate against a rule-based tactical agent")
    parser.add_argument("--games", type=int, default=1000, help="number of games")
    parser.add_argument("--board-size", type=int, default=15, help="board size")
    parser.add_argument("--save-every", type=int, default=1000, help="checkpoint interval")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    parser.add_argument(
        "--candidate-init-model",
        type=str,
        default=None,
        help=(
            "checkpoint to continue training from; defaults to the latest tactical_value_agent_v*.pt, "
            "then falls back to latest value_agent_v*.pt"
        ),
    )
    parser.add_argument(
        "--candidate-version",
        type=int,
        default=None,
        help="explicit tactical candidate version number to write",
    )
    parser.add_argument(
        "--rule-agent-level",
        type=str,
        choices=("super_easy", "easy", "normal", "hard"),
        default="hard",
        help="difficulty level of the rule-based opponent",
    )
    parser.add_argument(
        "--teacher-rule-agent-level",
        type=str,
        choices=("super_easy", "easy", "normal", "hard"),
        default=None,
        help="difficulty level of the teacher rule agent; defaults to the opponent level",
    )
    parser.add_argument(
        "--teacher-weight",
        type=float,
        default=1.0,
        help="extra teacher imitation loss weight using the rule agent's recommended move",
    )
    parser.add_argument(
        "--candidate-prefix",
        type=str,
        default="tactical_value_agent",
        help="candidate file prefix, e.g. tactical_value_agent or tactical_rule_value_agent",
    )
    args = parser.parse_args()
    result = train_against_tactical_agent(
        num_games=args.games,
        board_size=args.board_size,
        save_every=args.save_every,
        seed=args.seed,
        candidate_init_model_path=args.candidate_init_model,
        candidate_version=args.candidate_version,
        candidate_prefix=args.candidate_prefix,
        rule_agent_level=args.rule_agent_level,
        teacher_rule_agent_level=args.teacher_rule_agent_level,
        teacher_weight=args.teacher_weight,
        device=args.device,
    )
    print(
        f"Tactical-value training finished. Log: {result['training_log_path']} | "
        f"Init: {result['candidate_init_model_path'] or 'random'} | "
        f"Candidate: {result['candidate_model_path']} | "
        f"Win rate vs tactical: {result['candidate_win_rate_vs_tactical']:.2%}"
    )


if __name__ == "__main__":
    main()
