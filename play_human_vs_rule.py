"""Play a human against a rule-based Gomoku agent in the terminal."""

from __future__ import annotations

import argparse

from agent.tactical_rule_agent import (
    EasyTacticalRuleAgent,
    NormalTacticalRuleAgent,
    SuperEasyTacticalRuleAgent,
    build_random_hard_tactical_rule_agent,
)
from env.gomoku_env import GomokuEnv


PLAYER_LABELS = {1: "Black", -1: "White"}
STONE_LABELS = {0: ".", 1: "X", -1: "O"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Gomoku against a rule-based agent")
    parser.add_argument("--board-size", type=int, default=15, help="board size")
    parser.add_argument(
        "--human-color",
        choices=("black", "white"),
        default="black",
        help="human player color",
    )
    parser.add_argument(
        "--opponent-difficulty",
        choices=("super_easy", "easy", "normal", "hard"),
        default="hard",
        help="rule agent difficulty",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for deterministic opponent choice")
    args = parser.parse_args()

    env = GomokuEnv(board_size=args.board_size)
    human_player = 1 if args.human_color == "black" else -1
    opponent = build_rule_agent(args.opponent_difficulty, seed=args.seed)

    print(f"Human: {PLAYER_LABELS[human_player]}")
    print(f"Opponent: {opponent.name}")
    print("Enter moves as 'row col' using 1-based coordinates. Type 'q' to quit.\n")
    print_board(env)

    while not env.done:
        current_label = PLAYER_LABELS[env.current_player]
        if env.current_player == human_player:
            action = prompt_human_action(env, current_label)
        else:
            action, _ = opponent.select_action(env)
            row, col = action
            print(f"{current_label}: {row + 1} {col + 1} ({opponent.name})")

        _, _, _, _ = env.step(action)
        print_board(env)

    if env.winner == 0:
        print("Result: Draw")
    else:
        print(f"Result: {PLAYER_LABELS[env.winner]} wins")


def build_rule_agent(difficulty: str, seed: int | None = None):
    if difficulty == "super_easy":
        return SuperEasyTacticalRuleAgent()
    if difficulty == "easy":
        return EasyTacticalRuleAgent()
    if difficulty == "normal":
        return NormalTacticalRuleAgent()
    if difficulty == "hard":
        return build_random_hard_tactical_rule_agent(seed=seed)
    raise ValueError(f"unsupported difficulty: {difficulty}")


def prompt_human_action(env: GomokuEnv, current_label: str) -> tuple[int, int]:
    while True:
        raw = input(f"{current_label} move> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            raise SystemExit(0)
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("Enter two numbers: row col")
            continue
        try:
            row = int(parts[0]) - 1
            col = int(parts[1]) - 1
        except ValueError:
            print("Coordinates must be numbers.")
            continue
        if not (0 <= row < env.board_size and 0 <= col < env.board_size):
            print("Move is out of bounds.")
            continue
        if env.board[row][col] != 0:
            print("That position is already occupied.")
            continue
        return (row, col)


def print_board(env: GomokuEnv) -> None:
    header = "   " + " ".join(f"{col + 1:>2}" for col in range(env.board_size))
    print(header)
    for row_idx, row in enumerate(env.board, start=1):
        stones = " ".join(f"{STONE_LABELS[cell]:>2}" for cell in row)
        print(f"{row_idx:>2} {stones}")
    print()


if __name__ == "__main__":
    main()
