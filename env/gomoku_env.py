"""Core Gomoku environment."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Board = List[List[int]]
Action = Tuple[int, int]


class GomokuEnv:
    """A minimal Gomoku environment for self-play experiments."""

    def __init__(self, board_size: int = 15) -> None:
        if board_size < 5:
            raise ValueError("board_size must be at least 5")
        self.board_size = board_size
        self.board: Board = []
        self.current_player = 1
        self.done = False
        self.winner = 0
        self.last_action: Optional[Action] = None
        self.move_count = 0
        self.reset()

    def reset(self) -> Board:
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 1
        self.done = False
        self.winner = 0
        self.last_action = None
        self.move_count = 0
        return self.get_state()

    def get_state(self) -> Board:
        return [row[:] for row in self.board]

    def get_valid_actions(self) -> List[Action]:
        if self.done:
            return []
        return [
            (row_idx, col_idx)
            for row_idx in range(self.board_size)
            for col_idx in range(self.board_size)
            if self.board[row_idx][col_idx] == 0
        ]

    def action_to_index(self, action: Action) -> int:
        row, col = action
        self._validate_bounds(row, col)
        return row * self.board_size + col

    def index_to_action(self, index: int) -> Action:
        if not 0 <= index < self.board_size * self.board_size:
            raise ValueError("action index out of range")
        return divmod(index, self.board_size)

    def step(self, action: Sequence[int]) -> Tuple[Board, int, bool, dict]:
        if self.done:
            raise ValueError("game is already finished")

        if len(action) != 2:
            raise ValueError("action must contain row and col")

        row, col = int(action[0]), int(action[1])
        self._validate_bounds(row, col)

        if self.board[row][col] != 0:
            raise ValueError("action points to an occupied position")

        player = self.current_player
        self.board[row][col] = player
        self.last_action = (row, col)
        self.move_count += 1

        if self.check_win(self.last_action):
            self.done = True
            self.winner = player
            reward = 1
        elif self.move_count == self.board_size * self.board_size:
            self.done = True
            self.winner = 0
            reward = 0
        else:
            self.current_player *= -1
            reward = 0

        return self.get_state(), reward, self.done, self._build_info()

    def check_win(self, action: Optional[Action] = None) -> bool:
        if action is None:
            action = self.last_action
        if action is None:
            return False

        row, col = action
        player = self.board[row][col]
        if player == 0:
            return False

        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for delta_row, delta_col in directions:
            count = 1
            count += self._count_direction(row, col, delta_row, delta_col, player)
            count += self._count_direction(row, col, -delta_row, -delta_col, player)
            if count >= 5:
                return True
        return False

    def _build_info(self) -> dict:
        return {
            "winner": self.winner,
            "current_player": self.current_player,
            "last_action": self.last_action,
            "move_count": self.move_count,
        }

    def _count_direction(
        self, row: int, col: int, delta_row: int, delta_col: int, player: int
    ) -> int:
        count = 0
        next_row = row + delta_row
        next_col = col + delta_col
        while 0 <= next_row < self.board_size and 0 <= next_col < self.board_size:
            if self.board[next_row][next_col] != player:
                break
            count += 1
            next_row += delta_row
            next_col += delta_col
        return count

    def _validate_bounds(self, row: int, col: int) -> None:
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError("action is out of bounds")
