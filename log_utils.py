"""Shared helpers for Gomoku log files."""

from __future__ import annotations

from pathlib import Path


def find_latest_log(log_dir: str | Path = "logs") -> str | None:
    log_directory = Path(log_dir)
    if not log_directory.exists():
        return None

    # Replay viewer only understands full game logs, not summary/eval side logs.
    candidates = sorted(log_directory.glob("*_training.log"))
    if not candidates:
        return None
    return str(candidates[-1])
