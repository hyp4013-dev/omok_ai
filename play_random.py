"""Run random-vs-random Gomoku simulations and persist readable logs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from agent.random_agent import RandomAgent
from env.gomoku_env import GomokuEnv


def simulate_game(board_size: int = 15, seed: int | None = None) -> dict:
    env = GomokuEnv(board_size=board_size)
    black_agent = RandomAgent(seed=seed)
    white_agent = RandomAgent(seed=None if seed is None else seed + 1)

    move_history: list[dict] = []

    while not env.done:
        player = env.current_player
        agent = black_agent if env.current_player == 1 else white_agent
        action = agent.select_action(env)
        _, reward, done, info = env.step(action)
        move_history.append(
            {
                "move_number": env.move_count,
                "player": player,
                "action": action,
                "reward": reward,
                "done": done,
                "winner": info["winner"],
            }
        )

    return {
        "winner": env.winner,
        "moves": env.move_count,
        "last_action": env.last_action,
        "board_size": env.board_size,
        "move_history": move_history,
    }


def simulate_games(
    num_games: int = 10,
    board_size: int = 15,
    seed: int | None = None,
    log_dir: str | Path = "logs",
) -> dict:
    if num_games <= 0:
        raise ValueError("num_games must be positive")

    results = []
    winners: Counter[int] = Counter()
    for game_index in range(num_games):
        game_seed = None if seed is None else seed + (game_index * 2)
        result = simulate_game(board_size=board_size, seed=game_seed)
        results.append(result)
        winners[result["winner"]] += 1

    summary = {
        "total_games": num_games,
        "black_wins": winners[1],
        "white_wins": winners[-1],
        "draws": winners[0],
        "black_win_rate": winners[1] / num_games,
        "white_win_rate": winners[-1] / num_games,
        "draw_rate": winners[0] / num_games,
    }
    log_path = write_simulation_log(results=results, summary=summary, log_dir=log_dir)
    return {
        "results": results,
        "summary": summary,
        "log_path": str(log_path),
    }


def write_simulation_log(results: list[dict], summary: dict, log_dir: str | Path = "logs") -> Path:
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_directory / f"{timestamp}.log"
    log_path.write_text(_format_log(results, summary), encoding="utf-8")
    return log_path


def _format_log(results: list[dict], summary: dict) -> str:
    lines = [
        "Gomoku Random Simulation Log",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Summary",
        f"- Total games: {summary['total_games']}",
        f"- Black wins: {summary['black_wins']} ({summary['black_win_rate']:.2%})",
        f"- White wins: {summary['white_wins']} ({summary['white_win_rate']:.2%})",
        f"- Draws: {summary['draws']} ({summary['draw_rate']:.2%})",
        "",
    ]

    for game_index, result in enumerate(results, start=1):
        lines.append(
            f"Game {game_index}: winner={_player_label(result['winner'])}, "
            f"moves={result['moves']}, board={result['board_size']}x{result['board_size']}"
        )
        lines.append("Move record")
        for move in result["move_history"]:
            row, col = move["action"]
            lines.append(
                f"  {move['move_number']:>3}. {_player_label(move['player'])} -> "
                f"(row={row}, col={col})"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _player_label(player: int) -> str:
    if player == 1:
        return "Black"
    if player == -1:
        return "White"
    return "Draw"


if __name__ == "__main__":
    batch_result = simulate_games()
    summary = batch_result["summary"]
    print(
        f"Saved log to {batch_result['log_path']} | "
        f"Black: {summary['black_wins']} ({summary['black_win_rate']:.2%}), "
        f"White: {summary['white_wins']} ({summary['white_win_rate']:.2%}), "
        f"Draw: {summary['draws']} ({summary['draw_rate']:.2%})"
    )
