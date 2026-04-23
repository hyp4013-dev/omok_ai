"""Feature encoders shared by competing learning agents."""

from __future__ import annotations

from typing import Sequence

from env.gomoku_env import Action, Board


def perspective_board(board: Board, player: int) -> Board:
    return [[cell * player for cell in row] for row in board]


def action_features(board: Board, player: int, action: Action) -> list[float]:
    row, col = action
    if board[row][col] != 0:
        raise ValueError("action must point to an empty position")

    next_board = [board_row[:] for board_row in board]
    next_board[row][col] = player

    own_lengths = _line_lengths(next_board, row, col, player)
    opp_lengths_before = _line_lengths(board, row, col, -player, pretend_place=True)

    center = (len(board) - 1) / 2.0
    distance = abs(row - center) + abs(col - center)
    center_bonus = 1.0 - (distance / max(1.0, center * 2.0))

    own_neighbors = _adjacent_count(board, row, col, player)
    opp_neighbors = _adjacent_count(board, row, col, -player)

    features = [
        1.0,
        center_bonus,
        float(max(own_lengths)),
        _count_at_least(own_lengths, 2),
        _count_at_least(own_lengths, 3),
        _count_at_least(own_lengths, 4),
        _count_at_least(opp_lengths_before, 2),
        _count_at_least(opp_lengths_before, 3),
        _count_at_least(opp_lengths_before, 4),
        own_neighbors / 8.0,
        opp_neighbors / 8.0,
    ]
    return features


def policy_state_features(board: Board, player: int) -> list[float]:
    board_size = len(board)
    own_count = 0
    opp_count = 0
    own_neighbor_pairs = 0
    opp_neighbor_pairs = 0
    center_sum = 0.0
    center = (board_size - 1) / 2.0

    for row in range(board_size):
        for col in range(board_size):
            cell = board[row][col] * player
            if cell > 0:
                own_count += 1
                center_sum += _center_bonus(board_size, row, col)
                own_neighbor_pairs += _adjacent_count(board, row, col, player)
            elif cell < 0:
                opp_count += 1
                opp_neighbor_pairs += _adjacent_count(board, row, col, -player)

    total_cells = board_size * board_size
    turn_ratio = (own_count + opp_count) / total_cells
    return [
        1.0,
        own_count / total_cells,
        opp_count / total_cells,
        (own_count - opp_count) / total_cells,
        center_sum / max(1.0, own_count if own_count else 1.0),
        own_neighbor_pairs / max(1.0, total_cells),
        opp_neighbor_pairs / max(1.0, total_cells),
        1.0 - turn_ratio,
        1.0 if player == 1 else -1.0,
    ]


def _center_bonus(board_size: int, row: int, col: int) -> float:
    center = (board_size - 1) / 2.0
    distance = abs(row - center) + abs(col - center)
    return 1.0 - (distance / max(1.0, center * 2.0))


def _count_at_least(lengths: Sequence[int], threshold: int) -> float:
    return float(sum(1 for length in lengths if length >= threshold))


def _line_lengths(
    board: Board,
    row: int,
    col: int,
    player: int,
    pretend_place: bool = False,
) -> list[int]:
    lengths = []
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for delta_row, delta_col in directions:
        count = 1 if pretend_place or board[row][col] == player else 0
        count += _count_direction(board, row, col, delta_row, delta_col, player)
        count += _count_direction(board, row, col, -delta_row, -delta_col, player)
        lengths.append(count)
    return lengths


def _count_direction(board: Board, row: int, col: int, delta_row: int, delta_col: int, player: int) -> int:
    count = 0
    next_row = row + delta_row
    next_col = col + delta_col
    board_size = len(board)

    while 0 <= next_row < board_size and 0 <= next_col < board_size:
        if board[next_row][next_col] != player:
            break
        count += 1
        next_row += delta_row
        next_col += delta_col
    return count


def _adjacent_count(board: Board, row: int, col: int, player: int) -> int:
    count = 0
    board_size = len(board)
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue
            next_row = row + delta_row
            next_col = col + delta_col
            if 0 <= next_row < board_size and 0 <= next_col < board_size:
                if board[next_row][next_col] == player:
                    count += 1
    return count
