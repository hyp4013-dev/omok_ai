"""Train two different Gomoku agents by competitive self-play."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

from agent.policy_gradient_agent import PolicyGradientAgent, PolicyStepRecord
from agent.value_agent import ValueAgent, ValueStepRecord
from env.gomoku_env import GomokuEnv


def train_competitive(
    num_games: int = 200,
    board_size: int = 15,
    seed: int = 7,
    save_every: int = 300,
    log_dir: str | Path = "logs",
    model_dir: str | Path = "models",
    eval_games: int = 20,
    imbalance_threshold: float = 0.7,
    value_agent_kwargs: dict | None = None,
    policy_agent_kwargs: dict | None = None,
) -> dict:
    value_config = _default_value_agent_config(board_size, seed)
    if value_agent_kwargs:
        value_config.update(value_agent_kwargs)
    value_agent = ValueAgent(
        name="value_agent",
        **value_config,
    )
    policy_config = _default_policy_agent_config(board_size, seed + 1)
    if policy_agent_kwargs:
        policy_config.update(policy_agent_kwargs)
    policy_agent = PolicyGradientAgent(
        name="policy_agent",
        **policy_config,
    )

    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log_path = log_directory / f"{timestamp}_training.log"
    rolling_window = deque(maxlen=min(50, max(10, num_games)))
    summary_counter: Counter[str] = Counter()
    completed_games = 0
    stopped_early = False
    stop_reason: str | None = None
    last_evaluation: dict | None = None
    _initialize_training_log(training_log_path, board_size)

    for game_index in range(1, num_games + 1):
        env = GomokuEnv(board_size=board_size)
        if game_index % 2 == 1:
            assignments = {1: ("value_agent", value_agent), -1: ("policy_agent", policy_agent)}
        else:
            assignments = {1: ("policy_agent", policy_agent), -1: ("value_agent", value_agent)}

        episode_records: dict[str, list[ValueStepRecord | PolicyStepRecord]] = {
            "value_agent": [],
            "policy_agent": [],
        }
        move_history = []

        while not env.done:
            player = env.current_player
            agent_name, agent = assignments[player]
            action, step_record = agent.select_action(env, training=True)
            _, _, _, info = env.step(action)
            episode_records[agent_name].append(step_record)
            move_history.append({"player": player, "action": action, "agent": agent_name})

        rewards = _reward_map(env.winner, assignments)
        value_agent.finish_game(episode_records["value_agent"], rewards["value_agent"])
        policy_agent.finish_game(episode_records["policy_agent"], rewards["policy_agent"])

        winner_name = assignments[env.winner][0] if env.winner in assignments else "draw"
        summary_counter[winner_name] += 1
        rolling_window.append(winner_name)

        game_record = {
            "game": game_index,
            "winner": winner_name,
            "winner_stone": env.winner,
            "moves": env.move_count,
            "value_role": "black" if assignments[1][0] == "value_agent" else "white",
            "policy_role": "black" if assignments[1][0] == "policy_agent" else "white",
            "value_reward": rewards["value_agent"],
            "policy_reward": rewards["policy_agent"],
            "value_epsilon": value_agent.epsilon,
            "policy_epsilon": policy_agent.epsilon,
            "move_history": move_history,
        }
        _append_game_record(training_log_path, game_record, board_size)
        completed_games = game_index

        if game_index % save_every == 0 or game_index == num_games:
            value_agent.save(model_directory / "value_agent_latest.json")
            policy_agent.save(model_directory / "policy_agent_latest.json")
            evaluation = _evaluate_checkpoint(
                value_agent=value_agent,
                policy_agent=policy_agent,
                board_size=board_size,
                eval_games=eval_games,
                seed=seed + (game_index * 17),
            )
            last_evaluation = evaluation
            _append_evaluation_record(training_log_path, game_index, evaluation)
            if _should_stop_for_imbalance(evaluation, imbalance_threshold):
                stopped_early = True
                stop_reason = (
                    f"Stopped at checkpoint {game_index}: win-rate gap exceeded "
                    f"{imbalance_threshold:.0%}"
                )
                _append_stop_record(training_log_path, stop_reason, evaluation)
                break

    _append_training_summary(training_log_path, summary_counter, rolling_window, completed_games)

    return {
        "games": completed_games,
        "training_log_path": str(training_log_path),
        "value_model_path": str(model_directory / "value_agent_latest.json"),
        "policy_model_path": str(model_directory / "policy_agent_latest.json"),
        "summary": dict(summary_counter),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "last_evaluation": last_evaluation,
        "value_agent_config": value_config,
        "policy_agent_config": policy_config,
    }


def train_until_balanced(
    num_games: int = 2000,
    board_size: int = 15,
    seed: int = 7,
    save_every: int = 300,
    eval_games: int = 20,
    imbalance_threshold: float = 0.7,
    target_tolerance: float = 0.1,
    max_attempts: int = 8,
    log_dir: str | Path = "logs",
    model_dir: str | Path = "models",
) -> dict:
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adaptive_log_path = log_directory / f"{timestamp}_adaptive_training.log"

    value_config = _default_value_agent_config(board_size, seed)
    policy_config = _default_policy_agent_config(board_size, seed + 1)
    attempts: list[dict] = []
    converged = False

    for attempt in range(1, max_attempts + 1):
        attempt_seed = seed + (attempt * 1000)
        result = train_competitive(
            num_games=num_games,
            board_size=board_size,
            seed=attempt_seed,
            save_every=save_every,
            log_dir=log_dir,
            model_dir=model_dir,
            eval_games=eval_games,
            imbalance_threshold=imbalance_threshold,
            value_agent_kwargs=value_config,
            policy_agent_kwargs=policy_config,
        )
        analysis = _analyze_attempt(result["last_evaluation"], target_tolerance)
        attempts.append(
            {
                "attempt": attempt,
                "result": result,
                "analysis": analysis,
                "value_agent_config": dict(value_config),
                "policy_agent_config": dict(policy_config),
            }
        )
        _append_adaptive_attempt(
            adaptive_log_path=adaptive_log_path,
            attempt=attempt,
            result=result,
            analysis=analysis,
            value_config=value_config,
            policy_config=policy_config,
        )

        if analysis["status"] == "balanced":
            converged = True
            break

        if analysis["status"] == "inconclusive":
            break

        value_config, policy_config = _rebalance_configs(
            value_config=value_config,
            policy_config=policy_config,
            dominant_agent=analysis["dominant_agent"],
            attempt=attempt,
        )

    return {
        "converged": converged,
        "attempts": attempts,
        "adaptive_log_path": str(adaptive_log_path),
        "final_attempt": attempts[-1] if attempts else None,
    }


def _evaluate_checkpoint(
    value_agent: ValueAgent,
    policy_agent: PolicyGradientAgent,
    board_size: int,
    eval_games: int,
    seed: int,
) -> dict:
    value_eval = copy.deepcopy(value_agent)
    policy_eval = copy.deepcopy(policy_agent)
    value_eval.epsilon = 0.0
    policy_eval.epsilon = 0.0

    winners: Counter[str] = Counter()
    total_moves = 0
    for game_index in range(1, eval_games + 1):
        env = GomokuEnv(board_size=board_size)
        if game_index % 2 == 1:
            assignments = {1: ("value_agent", value_eval), -1: ("policy_agent", policy_eval)}
        else:
            assignments = {1: ("policy_agent", policy_eval), -1: ("value_agent", value_eval)}

        while not env.done:
            _, agent = assignments[env.current_player]
            action, _ = agent.select_action(env, training=False)
            env.step(action)

        winner_name = assignments[env.winner][0] if env.winner in assignments else "draw"
        winners[winner_name] += 1
        total_moves += env.move_count

    return {
        "games": eval_games,
        "value_agent_wins": winners["value_agent"],
        "policy_agent_wins": winners["policy_agent"],
        "draws": winners["draw"],
        "average_moves": total_moves / max(1, eval_games),
    }


def _default_value_agent_config(board_size: int, seed: int) -> dict:
    return {
        "board_size": board_size,
        "learning_rate": 0.03,
        "epsilon_decay": 0.997,
        "center_bias": 1.25,
        "blocking_bias": 1.4,
        "seed": seed,
    }


def _default_policy_agent_config(board_size: int, seed: int) -> dict:
    return {
        "board_size": board_size,
        "learning_rate": 0.025,
        "epsilon_decay": 0.994,
        "temperature": 0.9,
        "seed": seed,
    }


def _analyze_attempt(last_evaluation: dict | None, target_tolerance: float) -> dict:
    if not last_evaluation:
        return {"status": "inconclusive", "reason": "no evaluation produced", "dominant_agent": None}

    decisive_games = last_evaluation["value_agent_wins"] + last_evaluation["policy_agent_wins"]
    if decisive_games == 0:
        return {
            "status": "inconclusive",
            "reason": "evaluation contained only draws",
            "dominant_agent": None,
        }

    value_rate = last_evaluation["value_agent_wins"] / decisive_games
    policy_rate = last_evaluation["policy_agent_wins"] / decisive_games
    gap = abs(value_rate - policy_rate)
    if gap <= target_tolerance:
        return {
            "status": "balanced",
            "reason": f"win-rate gap {gap:.2%} is within tolerance",
            "dominant_agent": None,
            "value_rate": value_rate,
            "policy_rate": policy_rate,
        }

    dominant_agent = "value_agent" if value_rate > policy_rate else "policy_agent"
    weaker_agent = "policy_agent" if dominant_agent == "value_agent" else "value_agent"
    return {
        "status": "rebalance",
        "reason": f"{dominant_agent} is ahead by {gap:.2%}; boosting {weaker_agent}",
        "dominant_agent": dominant_agent,
        "value_rate": value_rate,
        "policy_rate": policy_rate,
    }


def _rebalance_configs(
    value_config: dict,
    policy_config: dict,
    dominant_agent: str | None,
    attempt: int,
) -> tuple[dict, dict]:
    next_value = dict(value_config)
    next_policy = dict(policy_config)
    factor = 1.0 + (attempt * 0.75)

    if dominant_agent == "policy_agent":
        next_value["learning_rate"] = min(0.08, next_value["learning_rate"] + (0.004 * factor))
        next_value["blocking_bias"] = min(4.5, next_value["blocking_bias"] + (0.35 * factor))
        next_value["center_bias"] = min(2.4, next_value["center_bias"] + (0.08 * factor))
        next_value["epsilon_decay"] = min(0.9995, next_value["epsilon_decay"] + (0.0006 * factor))
        next_policy["learning_rate"] = max(0.006, next_policy["learning_rate"] - (0.0035 * factor))
        next_policy["temperature"] = min(2.2, next_policy["temperature"] + (0.12 * factor))
        next_policy["epsilon_decay"] = max(0.985, next_policy["epsilon_decay"] - (0.0007 * factor))
    elif dominant_agent == "value_agent":
        next_policy["learning_rate"] = min(0.08, next_policy["learning_rate"] + (0.004 * factor))
        next_policy["temperature"] = max(0.55, next_policy["temperature"] - (0.05 * factor))
        next_policy["epsilon_decay"] = min(0.9995, next_policy["epsilon_decay"] + (0.0006 * factor))
        next_value["learning_rate"] = max(0.01, next_value["learning_rate"] - (0.003 * factor))
        next_value["blocking_bias"] = max(1.0, next_value["blocking_bias"] - (0.18 * factor))

    return next_value, next_policy


def _should_stop_for_imbalance(evaluation: dict, imbalance_threshold: float) -> bool:
    decisive_games = evaluation["value_agent_wins"] + evaluation["policy_agent_wins"]
    if decisive_games == 0:
        return False

    value_rate = evaluation["value_agent_wins"] / decisive_games
    policy_rate = evaluation["policy_agent_wins"] / decisive_games
    return abs(value_rate - policy_rate) >= imbalance_threshold


def _reward_map(winner: int, assignments: dict[int, tuple[str, object]]) -> dict[str, float]:
    if winner == 0:
        return {"value_agent": 0.0, "policy_agent": 0.0}

    winner_name = assignments[winner][0]
    loser_name = assignments[-winner][0]
    return {winner_name: 1.0, loser_name: -1.0}


def _initialize_training_log(path: Path, board_size: int) -> None:
    path.write_text(
        "\n".join(
            [
                "Gomoku Competitive Training Log",
                f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Board size: {board_size}x{board_size}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _append_game_record(path: Path, record: dict, board_size: int) -> None:
    lines = [
        f"Game {record['game']}: winner={record['winner']}, moves={record['moves']}, "
        f"board={board_size}x{board_size}, value_role={record['value_role']}, "
        f"policy_role={record['policy_role']}, value_reward={record['value_reward']:.1f}, "
        f"policy_reward={record['policy_reward']:.1f}, value_epsilon={record['value_epsilon']:.4f}, "
        f"policy_epsilon={record['policy_epsilon']:.4f}",
        "Move record",
    ]
    for move_index, move in enumerate(record["move_history"], start=1):
        row, col = move["action"]
        player_name = "Black" if move["player"] == 1 else "White"
        lines.append(
            f"  {move_index:>3}. {player_name} ({move['agent']}) -> (row={row}, col={col})"
        )
    lines.append("")
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines) + "\n")


def _append_training_summary(
    path: Path,
    summary_counter: Counter[str],
    rolling_window: deque[str],
    total_games: int,
) -> None:
    value_wins = summary_counter["value_agent"]
    policy_wins = summary_counter["policy_agent"]
    draws = summary_counter["draw"]
    rolling_value = sum(1 for winner in rolling_window if winner == "value_agent")
    rolling_policy = sum(1 for winner in rolling_window if winner == "policy_agent")
    rolling_draw = sum(1 for winner in rolling_window if winner == "draw")

    lines = [
        "Summary",
        "",
        f"- Total games: {total_games}",
        f"- Value agent wins: {value_wins}",
        f"- Policy agent wins: {policy_wins}",
        f"- Draws: {draws}",
        f"- Rolling window size: {len(rolling_window)}",
        f"- Rolling value wins: {rolling_value}",
        f"- Rolling policy wins: {rolling_policy}",
        f"- Rolling draws: {rolling_draw}",
    ]
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines).rstrip() + "\n")


def _append_evaluation_record(path: Path, checkpoint_game: int, evaluation: dict) -> None:
    lines = [
        (
            f"Checkpoint {checkpoint_game} Evaluation: games={evaluation['games']}, "
            f"value_agent_wins={evaluation['value_agent_wins']}, "
            f"policy_agent_wins={evaluation['policy_agent_wins']}, "
            f"draws={evaluation['draws']}, average_moves={evaluation['average_moves']:.2f}"
        ),
        "",
    ]
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))


def _append_stop_record(path: Path, reason: str, evaluation: dict) -> None:
    lines = [
        f"Early stop: {reason}",
        (
            f"Early stop evaluation snapshot: value_agent_wins={evaluation['value_agent_wins']}, "
            f"policy_agent_wins={evaluation['policy_agent_wins']}, draws={evaluation['draws']}"
        ),
        "",
    ]
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))


def _append_adaptive_attempt(
    adaptive_log_path: Path,
    attempt: int,
    result: dict,
    analysis: dict,
    value_config: dict,
    policy_config: dict,
) -> None:
    lines = [
        f"Attempt {attempt}",
        f"- Training log: {result['training_log_path']}",
        f"- Completed games: {result['games']}",
        f"- Stopped early: {result['stopped_early']}",
        f"- Stop reason: {result['stop_reason'] or '-'}",
        f"- Value config: {value_config}",
        f"- Policy config: {policy_config}",
        f"- Last evaluation: {result['last_evaluation']}",
        f"- Analysis: {analysis['reason']}",
        "",
    ]
    with adaptive_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Competitive Gomoku training")
    parser.add_argument("--games", type=int, default=200, help="number of self-play games")
    parser.add_argument("--board-size", type=int, default=15, help="board size")
    parser.add_argument("--save-every", type=int, default=300, help="checkpoint interval")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--eval-games", type=int, default=20, help="checkpoint evaluation games")
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=0.7,
        help="stop training if checkpoint win-rate gap reaches this value",
    )
    parser.add_argument(
        "--auto-rebalance",
        action="store_true",
        help="retry training with adjusted configs until win rates are near 50%",
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=0.1,
        help="acceptable checkpoint win-rate gap for auto rebalance",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="maximum auto rebalance attempts",
    )
    args = parser.parse_args()

    if args.auto_rebalance:
        result = train_until_balanced(
            num_games=args.games,
            board_size=args.board_size,
            save_every=args.save_every,
            seed=args.seed,
            eval_games=args.eval_games,
            imbalance_threshold=args.imbalance_threshold,
            target_tolerance=args.target_tolerance,
            max_attempts=args.max_attempts,
        )
        final_attempt = result["final_attempt"]
        print(
            f"Adaptive training finished. Converged: {result['converged']} | "
            f"Adaptive log: {result['adaptive_log_path']} | "
            f"Final training log: {final_attempt['result']['training_log_path'] if final_attempt else '-'}"
        )
    else:
        result = train_competitive(
            num_games=args.games,
            board_size=args.board_size,
            save_every=args.save_every,
            seed=args.seed,
            eval_games=args.eval_games,
            imbalance_threshold=args.imbalance_threshold,
        )
        print(
            f"Training finished. Log: {result['training_log_path']} | "
            f"Value model: {result['value_model_path']} | Policy model: {result['policy_model_path']}"
        )


if __name__ == "__main__":
    main()
