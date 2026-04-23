"""Tkinter UI for replaying saved Gomoku log files."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from log_parser import GameRecord, MoveRecord, SimulationLog, build_board_state, parse_log_file
from log_utils import find_latest_log


class GomokuLogViewer:
    def __init__(self, root: tk.Tk, initial_path: str | None = None) -> None:
        self.root = root
        self.root.title("Gomoku Log Viewer")
        self.root.geometry("1120x780")

        self.log_data: SimulationLog | None = None
        self.current_game_index = 0
        self.current_move_index = 0

        self.board_canvas: tk.Canvas | None = None
        self.board_margin = 40
        self.cell_size = 40

        self.game_var = tk.StringVar()
        self.move_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Open a log file to begin.")
        self.summary_var = tk.StringVar(value="")
        self.move_label_var = tk.StringVar(value="Move: 0")
        self.game_label_var = tk.StringVar(value="Game: -")

        self._build_layout()

        if initial_path:
            self.load_log(initial_path)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)

        left_panel = ttk.Frame(container)
        left_panel.pack(side="left", fill="both", expand=False)

        right_panel = ttk.Frame(container, padding=(16, 0, 0, 0))
        right_panel.pack(side="left", fill="both", expand=True)

        controls = ttk.Frame(left_panel)
        controls.pack(fill="x")

        ttk.Button(controls, text="Open Log", command=self.open_log).pack(side="left")
        ttk.Label(controls, textvariable=self.game_label_var, padding=(12, 0, 0, 0)).pack(side="left")
        ttk.Label(controls, textvariable=self.move_label_var, padding=(12, 0, 0, 0)).pack(side="left")

        self.board_canvas = tk.Canvas(
            left_panel,
            width=680,
            height=680,
            bg="#d9a55a",
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.board_canvas.pack(pady=(12, 0))

        game_select = ttk.Frame(right_panel)
        game_select.pack(fill="x")
        ttk.Label(game_select, text="Game").pack(anchor="w")
        self.game_combo = ttk.Combobox(
            game_select,
            state="readonly",
            textvariable=self.game_var,
            width=36,
        )
        self.game_combo.pack(fill="x", pady=(4, 12))
        self.game_combo.bind("<<ComboboxSelected>>", self.on_game_selected)

        navigation = ttk.LabelFrame(right_panel, text="Replay", padding=12)
        navigation.pack(fill="x")

        button_row = ttk.Frame(navigation)
        button_row.pack(fill="x")
        ttk.Button(button_row, text="|<", command=self.go_to_start, width=6).pack(side="left")
        ttk.Button(button_row, text="<", command=self.prev_move, width=6).pack(side="left", padx=(6, 0))
        ttk.Button(button_row, text=">", command=self.next_move, width=6).pack(side="left", padx=(6, 0))
        ttk.Button(button_row, text=">|", command=self.go_to_end, width=6).pack(side="left", padx=(6, 0))

        self.move_scale = ttk.Scale(
            navigation,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self.move_var,
            command=self.on_move_slider_changed,
        )
        self.move_scale.pack(fill="x", pady=(12, 0))

        info = ttk.LabelFrame(right_panel, text="Game Info", padding=12)
        info.pack(fill="x", pady=(12, 0))
        ttk.Label(info, textvariable=self.summary_var, justify="left").pack(anchor="w")
        ttk.Label(info, textvariable=self.status_var, justify="left", padding=(0, 8, 0, 0)).pack(anchor="w")

        move_list_frame = ttk.LabelFrame(right_panel, text="Moves", padding=12)
        move_list_frame.pack(fill="both", expand=True, pady=(12, 0))

        self.move_list = tk.Listbox(move_list_frame, activestyle="none", font=("Consolas", 10))
        self.move_list.pack(side="left", fill="both", expand=True)
        self.move_list.bind("<<ListboxSelect>>", self.on_move_selected)

        scrollbar = ttk.Scrollbar(move_list_frame, orient="vertical", command=self.move_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.move_list.configure(yscrollcommand=scrollbar.set)

    def open_log(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Gomoku Log File",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.load_log(path)

    def load_log(self, path: str) -> None:
        try:
            self.log_data = parse_log_file(path)
        except Exception as exc:
            messagebox.showerror("Failed to open log", str(exc))
            return

        self.root.title(f"Gomoku Log Viewer - {Path(path).name}")
        self.current_game_index = 0
        self._populate_game_combo()
        self.set_current_game(0)

    def _populate_game_combo(self) -> None:
        assert self.log_data is not None
        items = [
            f"Game {game.game_number} - winner {self._winner_label(game)} - {game.moves} moves"
            for game in self.log_data.games
        ]
        self.game_combo["values"] = items
        if items:
            self.game_var.set(items[0])

    def set_current_game(self, game_index: int) -> None:
        if self.log_data is None:
            return
        if not 0 <= game_index < len(self.log_data.games):
            return

        self.current_game_index = game_index
        self._refresh_move_controls()
        self._refresh_info_panel()
        self._populate_move_list()
        self.go_to_end()

    def _refresh_move_controls(self) -> None:
        game = self.current_game
        self.current_move_index = game.moves
        self.move_var.set(game.moves)
        self.move_scale.configure(to=game.moves)
        self.game_label_var.set(f"Game: {game.game_number}/{len(self.log_data.games)}")
        self.move_label_var.set(f"Move: {game.moves}/{game.moves}")
        self.status_var.set("Final board position")

    def _refresh_info_panel(self) -> None:
        assert self.log_data is not None
        summary = self.log_data.summary
        game = self.current_game
        lines = [
            f"Generated: {self.log_data.generated_at}",
            f"Total games: {summary.total_games}",
        ]
        if self.log_data.log_type == "competitive":
            lines.extend(
                [
                    f"Value agent wins: {summary.value_agent_wins}",
                    f"Policy agent wins: {summary.policy_agent_wins}",
                    f"Black agent: {game.black_agent or '-'}",
                    f"White agent: {game.white_agent or '-'}",
                ]
            )
        elif self.log_data.log_type == "value_reference":
            lines.extend(
                [
                    f"{summary.reference_label} wins: {summary.reference_wins}",
                    f"{summary.candidate_label} wins: {summary.candidate_wins}",
                    f"Black agent: {game.black_agent or '-'}",
                    f"White agent: {game.white_agent or '-'}",
                ]
            )
        else:
            lines.extend(
                [
                    f"Black wins: {summary.black_wins}",
                    f"White wins: {summary.white_wins}",
                ]
            )
        lines.extend(
            [
                f"Draws: {summary.draws}",
                f"Board size: {game.board_size}x{game.board_size}",
                f"Winner: {self._winner_label(game)}",
            ]
        )
        self.summary_var.set("\n".join(lines))

    def _populate_move_list(self) -> None:
        self.move_list.delete(0, tk.END)
        for move in self.current_game.move_history:
            agent_text = f" {move.agent_name}" if move.agent_name else ""
            self.move_list.insert(
                tk.END,
                f"{move.move_number:>3}. {self._player_name(move.player):<5}{agent_text:<14} ({move.row:>2}, {move.col:>2})",
            )

    @property
    def current_game(self) -> GameRecord:
        assert self.log_data is not None
        return self.log_data.games[self.current_game_index]

    def on_game_selected(self, _event: object) -> None:
        if self.log_data is None:
            return
        selected = self.game_combo.current()
        if selected >= 0:
            self.set_current_game(selected)

    def on_move_slider_changed(self, _value: str) -> None:
        if self.log_data is None:
            return
        self.current_move_index = int(round(self.move_var.get()))
        self._sync_selection()
        self.render_board()

    def on_move_selected(self, _event: object) -> None:
        if not self.move_list.curselection():
            return
        self.current_move_index = self.move_list.curselection()[0] + 1
        self.move_var.set(self.current_move_index)
        self.render_board()

    def go_to_start(self) -> None:
        self.current_move_index = 0
        self.move_var.set(0)
        self._sync_selection()
        self.render_board()

    def go_to_end(self) -> None:
        self.current_move_index = self.current_game.moves
        self.move_var.set(self.current_move_index)
        self._sync_selection()
        self.render_board()

    def prev_move(self) -> None:
        if self.current_move_index > 0:
            self.current_move_index -= 1
            self.move_var.set(self.current_move_index)
            self._sync_selection()
            self.render_board()

    def next_move(self) -> None:
        if self.current_move_index < self.current_game.moves:
            self.current_move_index += 1
            self.move_var.set(self.current_move_index)
            self._sync_selection()
            self.render_board()

    def _sync_selection(self) -> None:
        self.move_list.selection_clear(0, tk.END)
        if self.current_move_index > 0:
            list_index = self.current_move_index - 1
            self.move_list.selection_set(list_index)
            self.move_list.see(list_index)

    def render_board(self) -> None:
        if self.board_canvas is None:
            return

        game = self.current_game
        board = build_board_state(game, self.current_move_index)
        last_move = None if self.current_move_index == 0 else game.move_history[self.current_move_index - 1]

        self.cell_size = min(42, 620 / max(1, game.board_size - 1))
        board_extent = self.cell_size * (game.board_size - 1)
        width = int(self.board_margin * 2 + board_extent)
        height = int(self.board_margin * 2 + board_extent)
        self.board_canvas.configure(width=width, height=height)
        self.board_canvas.delete("all")

        for index in range(game.board_size):
            offset = self.board_margin + (index * self.cell_size)
            self.board_canvas.create_line(
                self.board_margin, offset, self.board_margin + board_extent, offset, fill="#4c3111", width=1
            )
            self.board_canvas.create_line(
                offset, self.board_margin, offset, self.board_margin + board_extent, fill="#4c3111", width=1
            )
            self.board_canvas.create_text(self.board_margin - 18, offset, text=str(index), font=("Segoe UI", 9))
            self.board_canvas.create_text(offset, self.board_margin - 18, text=str(index), font=("Segoe UI", 9))

        stone_radius = max(10, self.cell_size * 0.38)
        for row in range(game.board_size):
            for col in range(game.board_size):
                player = board[row][col]
                if player == 0:
                    continue
                x = self.board_margin + (col * self.cell_size)
                y = self.board_margin + (row * self.cell_size)
                fill = "#111111" if player == 1 else "#f2f2f2"
                text_fill = "#f2f2f2" if player == 1 else "#111111"
                self.board_canvas.create_oval(
                    x - stone_radius,
                    y - stone_radius,
                    x + stone_radius,
                    y + stone_radius,
                    fill=fill,
                    outline="#333333",
                    width=1,
                )

                move_number = self._find_move_number(game.move_history, row, col, self.current_move_index)
                if move_number is not None and stone_radius >= 12:
                    self.board_canvas.create_text(
                        x,
                        y,
                        text=str(move_number),
                        fill=text_fill,
                        font=("Segoe UI", max(8, int(stone_radius * 0.55)), "bold"),
                    )

        if last_move is not None:
            self._highlight_last_move(last_move)
            self.status_var.set(
                f"Move {last_move.move_number}: {self._player_name(last_move.player)} "
                f"to ({last_move.row}, {last_move.col})"
            )
        else:
            self.status_var.set("Initial board position")

        self.move_label_var.set(f"Move: {self.current_move_index}/{game.moves}")

    def _highlight_last_move(self, move: MoveRecord) -> None:
        x = self.board_margin + (move.col * self.cell_size)
        y = self.board_margin + (move.row * self.cell_size)
        size = max(6, self.cell_size * 0.16)
        self.board_canvas.create_rectangle(
            x - size,
            y - size,
            x + size,
            y + size,
            outline="#d22f27",
            width=2,
        )

    def _find_move_number(
        self,
        moves: list[MoveRecord],
        row: int,
        col: int,
        current_move_index: int,
    ) -> int | None:
        for move in reversed(moves[:current_move_index]):
            if move.row == row and move.col == col:
                return move.move_number
        return None

    def _player_name(self, player: int) -> str:
        if player == 1:
            return "Black"
        if player == -1:
            return "White"
        return "Draw"

    def _winner_label(self, game: GameRecord) -> str:
        if self.log_data is not None and self.log_data.log_type in {"competitive", "value_reference"}:
            if game.winner_label == "draw":
                return "Draw"
            return game.winner_label
        return self._player_name(game.winner)


def main() -> None:
    root = tk.Tk()
    initial_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_log()
    viewer = GomokuLogViewer(root, initial_path=initial_path)
    if viewer.log_data is None and initial_path:
        messagebox.showwarning("No log loaded", f"Could not load {initial_path}")
    root.mainloop()


if __name__ == "__main__":
    main()
