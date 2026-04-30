"""Parse Gomoku simulation and training logs into structured data for replay tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PLAYER_MAP = {"Black": 1, "White": -1, "Draw": 0}
RANDOM_HEADER = "Gomoku Random Simulation Log"
COMPETITIVE_HEADER = "Gomoku Competitive Training Log"
VALUE_REFERENCE_HEADER = "Gomoku Value Reference Training Log"
TACTICAL_VALUE_REFERENCE_HEADER = "Gomoku Tactical Value Reference Training Log"
TACTICAL_RULE_VALUE_REFERENCE_HEADER = "Gomoku Tactical Rule Value Reference Training Log"
TACTICAL_RULE_POLICY_REFERENCE_HEADER = "Gomoku Tactical Policy Reference Training Log"
TACTICAL_VALUE_TRAINING_HEADER = "Gomoku Tactical Value Training Log"
TACTICAL_RULE_VALUE_TRAINING_HEADER = "Gomoku Tactical Rule Value Training Log"

RANDOM_GAME_HEADER_RE = re.compile(
    r"^Game (?P<game_number>\d+): winner=(?P<winner>Black|White|Draw), "
    r"moves=(?P<moves>\d+), board=(?P<board_size>\d+)x(?P=board_size)$"
)
COMPETITIVE_GAME_HEADER_RE = re.compile(
    r"^Game (?P<game_number>\d+): winner=(?P<winner>value_agent|policy_agent|draw), "
    r"moves=(?P<moves>\d+), board=(?P<board_size>\d+)x(?P=board_size), "
    r"value_role=(?P<value_role>black|white), policy_role=(?P<policy_role>black|white), "
    r"value_reward=(?P<value_reward>-?\d+(?:\.\d+)?), policy_reward=(?P<policy_reward>-?\d+(?:\.\d+)?), "
    r"value_epsilon=(?P<value_epsilon>\d+(?:\.\d+)?), policy_epsilon=(?P<policy_epsilon>\d+(?:\.\d+)?)$"
)
VALUE_REFERENCE_GAME_HEADER_RE = re.compile(
    r"^Game (?P<game_number>\d+): winner=(?P<winner>[^,]+), "
    r"moves=(?P<moves>\d+), board=(?P<board_size>\d+)x(?P=board_size), "
    r"reference_role=(?P<reference_role>black|white), candidate_role=(?P<candidate_role>black|white), "
    r"candidate_reward=(?P<candidate_reward>-?\d+(?:\.\d+)?), "
    r"candidate_epsilon=(?P<candidate_epsilon>\d+(?:\.\d+)?)(?:, (?P<extra_fields>.*))?$"
)
TACTICAL_TRAINING_GAME_HEADER_RE = re.compile(
    r"^Game (?P<game_number>\d+): winner=(?P<winner>[^,]+), "
    r"moves=(?P<moves>\d+), board=(?P<board_size>\d+)x(?P=board_size), "
    r"tactical_role=(?P<tactical_role>black|white), candidate_role=(?P<candidate_role>black|white), "
    r"candidate_reward=(?P<candidate_reward>-?\d+(?:\.\d+)?), "
    r"candidate_epsilon=(?P<candidate_epsilon>\d+(?:\.\d+)?)(?:, (?P<extra_fields>.*))?$"
)
MOVE_RE = re.compile(
    r"^\s*(?P<move_number>\d+)\. (?P<player>Black|White)(?: \((?P<agent_name>[^)]+)\))? -> "
    r"\(row=(?P<row>\d+), col=(?P<col>\d+)\)(?: \[(?P<selection_reason>[^\]]+)\])?$"
)
METRICS_RE = re.compile(r"^Metrics: ")


@dataclass(frozen=True)
class MoveRecord:
    move_number: int
    player: int
    row: int
    col: int
    agent_name: str | None = None


@dataclass(frozen=True)
class GameRecord:
    game_number: int
    winner: int
    winner_label: str
    moves: int
    board_size: int
    move_history: list[MoveRecord]
    black_agent: str | None = None
    white_agent: str | None = None


@dataclass(frozen=True)
class SimulationSummary:
    total_games: int
    black_wins: int
    white_wins: int
    draws: int
    value_agent_wins: int = 0
    policy_agent_wins: int = 0
    reference_wins: int = 0
    candidate_wins: int = 0
    reference_label: str = "reference"
    candidate_label: str = "candidate"


@dataclass(frozen=True)
class SimulationLog:
    generated_at: str
    summary: SimulationSummary
    games: list[GameRecord]
    log_type: str


def parse_log_file(path: str | Path) -> SimulationLog:
    return parse_log_text(Path(path).read_text(encoding="utf-8"))


def parse_log_text(text: str) -> SimulationLog:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    reference_headers = {
        VALUE_REFERENCE_HEADER,
        TACTICAL_VALUE_REFERENCE_HEADER,
        TACTICAL_RULE_VALUE_REFERENCE_HEADER,
        TACTICAL_RULE_POLICY_REFERENCE_HEADER,
    }
    tactical_headers = {
        TACTICAL_VALUE_TRAINING_HEADER,
        TACTICAL_RULE_VALUE_TRAINING_HEADER,
    }
    if not lines or lines[0] not in {RANDOM_HEADER, COMPETITIVE_HEADER, *reference_headers, *tactical_headers}:
        raise ValueError("invalid log header")

    if lines[0] == RANDOM_HEADER:
        log_type = "random"
    elif lines[0] == COMPETITIVE_HEADER:
        log_type = "competitive"
    elif lines[0] in tactical_headers:
        log_type = "tactical_training"
    else:
        log_type = "value_reference"
    generated_at = _value_after_prefix(lines, "Generated at: ")
    games = _parse_games(lines, log_type)
    summary = _parse_summary(lines, log_type, games)
    return SimulationLog(generated_at=generated_at, summary=summary, games=games, log_type=log_type)


def build_board_state(game: GameRecord, move_index: int | None = None) -> list[list[int]]:
    if move_index is None:
        move_index = len(game.move_history)
    if not 0 <= move_index <= len(game.move_history):
        raise ValueError("move_index out of range")

    board = [[0 for _ in range(game.board_size)] for _ in range(game.board_size)]
    for move in game.move_history[:move_index]:
        board[move.row][move.col] = move.player
    return board


def _value_after_prefix(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise ValueError(f"missing line with prefix: {prefix}")


def _parse_summary(lines: list[str], log_type: str, games: list[GameRecord]) -> SimulationSummary:
    try:
        total_games = _parse_summary_count(lines, "- Total games: ")
        draws = _parse_summary_count(lines, "- Draws: ")
    except ValueError:
        return _summary_from_games(log_type, games)

    if log_type == "random":
        black_wins = _parse_summary_count(lines, "- Black wins: ")
        white_wins = _parse_summary_count(lines, "- White wins: ")
        return SimulationSummary(
            total_games=total_games,
            black_wins=black_wins,
            white_wins=white_wins,
            draws=draws,
        )

    if log_type == "competitive":
        value_agent_wins = _parse_summary_count(lines, "- Value agent wins: ")
        policy_agent_wins = _parse_summary_count(lines, "- Policy agent wins: ")
        return SimulationSummary(
            total_games=total_games,
            black_wins=0,
            white_wins=0,
            draws=draws,
            value_agent_wins=value_agent_wins,
            policy_agent_wins=policy_agent_wins,
        )

    if log_type == "tactical_training":
        tactical_label, tactical_wins = _parse_dynamic_summary_count(lines, "wins", occurrence=1)
        candidate_label, candidate_wins = _parse_dynamic_summary_count(lines, "wins", occurrence=2)
        return SimulationSummary(
            total_games=total_games,
            black_wins=0,
            white_wins=0,
            draws=draws,
            reference_wins=tactical_wins,
            candidate_wins=candidate_wins,
            reference_label=tactical_label,
            candidate_label=candidate_label,
        )

    reference_label, reference_wins = _parse_dynamic_summary_count(lines, "wins", occurrence=1)
    candidate_label, candidate_wins = _parse_dynamic_summary_count(lines, "wins", occurrence=2)
    return SimulationSummary(
        total_games=total_games,
        black_wins=0,
        white_wins=0,
        draws=draws,
        reference_wins=reference_wins,
        candidate_wins=candidate_wins,
        reference_label=reference_label,
        candidate_label=candidate_label,
    )


def _summary_from_games(log_type: str, games: list[GameRecord]) -> SimulationSummary:
    draws = sum(1 for game in games if game.winner == 0)
    if log_type == "random":
        black_wins = sum(1 for game in games if game.winner == 1)
        white_wins = sum(1 for game in games if game.winner == -1)
        return SimulationSummary(
            total_games=len(games),
            black_wins=black_wins,
            white_wins=white_wins,
            draws=draws,
        )
    if log_type == "competitive":
        value_agent_wins = sum(1 for game in games if game.winner_label == "value_agent")
        policy_agent_wins = sum(1 for game in games if game.winner_label == "policy_agent")
        return SimulationSummary(
            total_games=len(games),
            black_wins=0,
            white_wins=0,
            draws=draws,
            value_agent_wins=value_agent_wins,
            policy_agent_wins=policy_agent_wins,
        )

    if log_type == "tactical_training":
        candidate_labels = {
            move.agent_name
            for game in games
            for move in game.move_history
            if move.agent_name and "tactical_rule_agent" not in move.agent_name
        }
        tactical_wins = sum(
            1
            for game in games
            if game.winner_label != "draw" and "tactical_rule_agent" in game.winner_label
        )
        candidate_wins = sum(
            1
            for game in games
            if game.winner_label != "draw" and "tactical_rule_agent" not in game.winner_label
        )
        candidate_label = sorted(candidate_labels)[-1] if candidate_labels else "candidate"
        return SimulationSummary(
            total_games=len(games),
            black_wins=0,
            white_wins=0,
            draws=draws,
            reference_wins=tactical_wins,
            candidate_wins=candidate_wins,
            reference_label="tactical_agent",
            candidate_label=candidate_label,
        )

    candidate_labels = {
        move.agent_name
        for game in games
        for move in game.move_history
        if move.agent_name and "_reference" not in move.agent_name
    }
    reference_wins = sum(1 for game in games if game.winner_label.endswith("_reference"))
    candidate_wins = sum(
        1 for game in games if game.winner_label != "draw" and not game.winner_label.endswith("_reference")
    )
    candidate_label = sorted(candidate_labels)[-1] if candidate_labels else "candidate"
    return SimulationSummary(
        total_games=len(games),
        black_wins=0,
        white_wins=0,
        draws=draws,
        reference_wins=reference_wins,
        candidate_wins=candidate_wins,
        reference_label="reference",
        candidate_label=candidate_label,
    )


def _parse_summary_count(lines: list[str], prefix: str) -> int:
    raw_value = _value_after_prefix(lines, prefix)
    return int(raw_value.split(" ", 1)[0])


def _parse_dynamic_summary_count(lines: list[str], suffix: str, occurrence: int) -> tuple[str, int]:
    matched = 0
    for line in lines:
        if not line.startswith("- ") or f" {suffix}:" not in line:
            continue
        matched += 1
        if matched != occurrence:
            continue
        label, raw_value = line[2:].split(f" {suffix}: ", 1)
        return label, int(raw_value.split(" ", 1)[0])
    raise ValueError(f"missing summary occurrence {occurrence} for suffix {suffix}")


def _parse_games(lines: list[str], log_type: str) -> list[GameRecord]:
    games: list[GameRecord] = []
    index = 0
    while index < len(lines):
        if log_type == "random":
            match = RANDOM_GAME_HEADER_RE.match(lines[index])
        elif log_type == "competitive":
            match = COMPETITIVE_GAME_HEADER_RE.match(lines[index])
        elif log_type == "tactical_training":
            match = TACTICAL_TRAINING_GAME_HEADER_RE.match(lines[index])
        else:
            match = VALUE_REFERENCE_GAME_HEADER_RE.match(lines[index])
        if not match:
            index += 1
            continue

        game_number = int(match.group("game_number"))
        winner_label = match.group("winner")
        moves = int(match.group("moves"))
        board_size = int(match.group("board_size"))
        black_agent = None
        white_agent = None
        if log_type == "random":
            winner = PLAYER_MAP[winner_label]
        elif log_type == "competitive":
            value_role = match.group("value_role")
            policy_role = match.group("policy_role")
            black_agent = "value_agent" if value_role == "black" else "policy_agent"
            white_agent = "policy_agent" if black_agent == "value_agent" else "value_agent"
            if winner_label == "draw":
                winner = 0
            elif winner_label == black_agent:
                winner = 1
            else:
                winner = -1
        elif log_type == "tactical_training":
            tactical_role = match.group("tactical_role")
            first_agent_name = None
            second_agent_name = None
            for probe_index in range(index + 2, min(len(lines), index + 6)):
                move_match = MOVE_RE.match(lines[probe_index])
                if move_match:
                    if first_agent_name is None:
                        first_agent_name = move_match.group("agent_name")
                    elif second_agent_name is None and move_match.group("agent_name") != first_agent_name:
                        second_agent_name = move_match.group("agent_name")
                        break
            if tactical_role == "black":
                black_agent = first_agent_name or "tactical_agent"
                white_agent = second_agent_name or "candidate"
            else:
                black_agent = first_agent_name or "candidate"
                white_agent = second_agent_name or "tactical_agent"
            if winner_label == "draw":
                winner = 0
            elif winner_label == black_agent:
                winner = 1
            else:
                winner = -1
        else:
            reference_role = match.group("reference_role")
            candidate_role = match.group("candidate_role")
            if reference_role == "black":
                black_agent = "reference"
                white_agent = "candidate"
            else:
                black_agent = "candidate"
                white_agent = "reference"
            if winner_label == "draw":
                winner = 0
            elif reference_role == "black":
                winner = 1 if winner_label != "draw" and "reference" in winner_label else -1
            else:
                winner = 1 if winner_label != "draw" and "candidate" in winner_label else -1

            first_agent_name = None
            second_agent_name = None
            for probe_index in range(index + 2, min(len(lines), index + 6)):
                move_match = MOVE_RE.match(lines[probe_index])
                if move_match:
                    if first_agent_name is None:
                        first_agent_name = move_match.group("agent_name")
                    elif second_agent_name is None and move_match.group("agent_name") != first_agent_name:
                        second_agent_name = move_match.group("agent_name")
                        break
            if reference_role == "black":
                black_agent = first_agent_name or "reference"
                white_agent = second_agent_name or "candidate"
            else:
                black_agent = first_agent_name or "candidate"
                white_agent = second_agent_name or "reference"
            if winner_label == "draw":
                winner = 0
            elif winner_label == black_agent:
                winner = 1
            else:
                winner = -1

        index += 1
        if index >= len(lines) or lines[index] != "Move record":
            if index >= len(lines):
                break
            raise ValueError(f"missing move section for game {game_number}")
        index += 1

        move_history: list[MoveRecord] = []
        while index < len(lines):
            if not lines[index].strip():
                index += 1
                break
            if METRICS_RE.match(lines[index]):
                index += 1
                continue
            move_match = MOVE_RE.match(lines[index])
            if not move_match:
                if index == len(lines) - 1 or lines[index].startswith("Summary"):
                    break
                raise ValueError(f"invalid move line: {lines[index]}")
            move_history.append(
                MoveRecord(
                    move_number=int(move_match.group("move_number")),
                    player=PLAYER_MAP[move_match.group("player")],
                    row=int(move_match.group("row")),
                    col=int(move_match.group("col")),
                    agent_name=move_match.group("agent_name"),
                )
            )
            index += 1

        if len(move_history) != moves:
            if index >= len(lines) or not lines[index].strip():
                break
            raise ValueError(
                f"game {game_number} expected {moves} moves but parsed {len(move_history)}"
            )

        games.append(
            GameRecord(
                game_number=game_number,
                winner=winner,
                winner_label=winner_label,
                moves=moves,
                board_size=board_size,
                move_history=move_history,
                black_agent=black_agent,
                white_agent=white_agent,
            )
        )
    if not games:
        raise ValueError("no games found in log")
    return games
