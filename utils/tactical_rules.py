"""Immediate tactical rules shared across Gomoku agents."""

from __future__ import annotations

from env.gomoku_env import Action, GomokuEnv


def find_forced_action(
    env: GomokuEnv,
    include_open_three_block: bool = False,
) -> Action | None:
    winning_action = _find_immediate_win(env, env.current_player)
    if winning_action is not None:
        return winning_action

    blocking_action = _find_immediate_win(env, -env.current_player)
    if blocking_action is not None:
        return blocking_action

    if not include_open_three_block:
        return None

    return _find_open_three_block(env, -env.current_player)


def _find_immediate_win(env: GomokuEnv, player: int) -> Action | None:
    current_player = env.current_player
    for action in env.get_valid_actions():
        row, col = action
        env.board[row][col] = player
        env.last_action = action
        is_win = env.check_win(action)
        env.board[row][col] = 0
        env.last_action = None
        if is_win:
            env.current_player = current_player
            return action

    env.current_player = current_player
    return None


def _find_open_three_block(env: GomokuEnv, player: int) -> Action | None:
    board = env.board
    board_size = len(board)
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))

    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] != player:
                continue
            for delta_row, delta_col in directions:
                prev_row = row - delta_row
                prev_col = col - delta_col
                if 0 <= prev_row < board_size and 0 <= prev_col < board_size:
                    if board[prev_row][prev_col] == player:
                        continue

                length = 0
                next_row = row
                next_col = col
                while 0 <= next_row < board_size and 0 <= next_col < board_size:
                    if board[next_row][next_col] != player:
                        break
                    length += 1
                    next_row += delta_row
                    next_col += delta_col

                if length != 3:
                    continue

                if not (0 <= prev_row < board_size and 0 <= prev_col < board_size):
                    continue
                if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                    continue
                if board[prev_row][prev_col] != 0 or board[next_row][next_col] != 0:
                    continue
                return (prev_row, prev_col)

    return None
