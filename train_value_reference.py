"""Train value-agent versions against the current frozen reference."""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any
from types import SimpleNamespace

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
from agent.torch_value_agent import TorchValueAgent, TorchValueStepRecord
from agent.value_agent import ValueAgent
from env.gomoku_env import GomokuEnv


REFERENCE_WINRATE_LINE_RE = re.compile(
    r"^- (?P<reference_name>[^:]+): (?P<wins>\d+)/(?P<games>\d+) \((?P<win_rate>\d+(?:\.\d+)?)%\)"
)


def _reference_directory(model_directory: Path) -> Path:
    reference_directory = model_directory / "refer"
    reference_directory.mkdir(parents=True, exist_ok=True)
    return reference_directory


class RuleAugmentedReferenceAgent:
    def __init__(
        self,
        base_agent: Any,
        rule_agent: TacticalRuleAgent,
        name: str,
        opening_rule_moves: int,
        late_rule_probability: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.base_agent = base_agent
        self.rule_agent = rule_agent
        self.name = name
        self.epsilon = 0.0
        self.opening_rule_moves = opening_rule_moves
        self.late_rule_probability = late_rule_probability
        self.random = random.Random(seed)
        self.variant_label = getattr(rule_agent, "variant_label", None)

    def select_action(self, env: GomokuEnv, training: bool = False) -> tuple[tuple[int, int], Any]:
        if env.move_count == 0:
            return self._opening_random_action(env)
        if env.move_count < self.opening_rule_moves:
            return self.rule_agent.select_action(env, training=False)
        if self.late_rule_probability > 0.0 and self.random.random() < self.late_rule_probability:
            return self.rule_agent.select_action(env, training=False)
        return self.base_agent.select_action(env, training=training)

    def _opening_random_action(self, env: GomokuEnv) -> tuple[tuple[int, int], Any]:
        valid_actions = env.get_valid_actions()
        opening_pool = self._central_opening_pool(valid_actions, env.board_size)
        chosen_action = self.random.choice(opening_pool)
        return chosen_action, SimpleNamespace(selection_reason="opening_random")

    def _central_opening_pool(
        self,
        valid_actions: list[tuple[int, int]],
        board_size: int,
    ) -> list[tuple[int, int]]:
        opening_span = min(10, board_size)
        opening_offset = (board_size - opening_span) // 2
        opening_limit = opening_offset + opening_span - 1
        return [
            action
            for action in valid_actions
            if opening_offset <= action[0] <= opening_limit and opening_offset <= action[1] <= opening_limit
        ]


class RuleOnlyReferenceAgent:
    def __init__(self, rule_agent: TacticalRuleAgent, name: str) -> None:
        self.rule_agent = rule_agent
        self.name = name
        self.epsilon = 0.0
        self.variant_label = getattr(rule_agent, "variant_label", None)

    def select_action(self, env: GomokuEnv, training: bool = False) -> tuple[tuple[int, int], Any]:
        del training
        if env.move_count == 0:
            valid_actions = env.get_valid_actions()
            opening_pool = self._central_opening_pool(valid_actions, env.board_size)
            chosen_action = self.rule_agent.random.choice(opening_pool)
            return chosen_action, SimpleNamespace(selection_reason="opening_random")
        return self.rule_agent.select_action(env, training=False)

    def _central_opening_pool(
        self,
        valid_actions: list[tuple[int, int]],
        board_size: int,
    ) -> list[tuple[int, int]]:
        opening_span = min(10, board_size)
        opening_offset = (board_size - opening_span) // 2
        opening_limit = opening_offset + opening_span - 1
        return [
            action
            for action in valid_actions
            if opening_offset <= action[0] <= opening_limit and opening_offset <= action[1] <= opening_limit
        ]


def train_against_reference(
    num_games: int = 1000,
    board_size: int = 15,
    seed: int = 7,
    save_every: int = 100,
    log_dir: str | Path = "logs",
    model_dir: str | Path = "models",
    reference_model_path: str | Path | list[str | Path] | None = None,
    candidate_init_model_path: str | Path | None = None,
    candidate_version: int | None = None,
    device: str = "cpu",
    pretrain_positions: int = 400,
    reference_cycle_length: int = 10,
    candidate_prefix: str = "value_agent",
    reference_rule_agent_level: str | None = None,
    reference_rule_opening_moves: int = 20,
    reference_rule_only_agent_level: str | None = None,
    reference_rule_followup_probability: float = 0.1,
    teacher_rule_agent_level: str = "hard",
    teacher_weight: float = 1.0,
) -> dict:
    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    reference_directory = _reference_directory(model_directory)

    reference_paths = _resolve_reference_paths(
        reference_model_path,
        model_directory,
        log_directory,
        candidate_prefix,
    )
    for reference_path in reference_paths:
        if reference_path.exists():
            continue
        latest_path = model_directory / "value_agent_latest.json"
        if not latest_path.exists():
            raise FileNotFoundError("value_agent_latest.json not found; cannot create reference v1")
        shutil.copyfile(latest_path, reference_path)

    version_number = candidate_version if candidate_version is not None else _next_candidate_version(
        model_directory,
        candidate_prefix,
    )
    candidate_path = model_directory / f"{candidate_prefix}_v{version_number}.pt"
    latest_candidate_path = model_directory / f"{candidate_prefix}_candidate_latest.pt"
    resolved_candidate_init_model_path = _resolve_candidate_init_model_path(
        candidate_init_model_path,
        candidate_prefix,
        model_directory,
    )

    reference_entries = []
    for offset, reference_path in enumerate(reference_paths):
        reference_name = _reference_name_from_path(reference_path)
        reference_agent = _load_reference_agent(
            reference_path,
            reference_name,
            seed + offset,
            device,
            reference_rule_agent_level=reference_rule_agent_level,
            reference_rule_opening_moves=reference_rule_opening_moves,
            reference_rule_followup_probability=reference_rule_followup_probability,
        )
        reference_agent.epsilon = 0.0
        reference_entries.append(
            {
                "path": reference_path,
                "name": reference_name,
                "agent": reference_agent,
            }
        )
    primary_reference_agent = reference_entries[-1]["agent"]
    if reference_rule_only_agent_level is not None:
        rule_only_name = _rule_only_reference_name(candidate_prefix, reference_rule_only_agent_level)
        reference_entries.append(
            {
                "path": None,
                "name": rule_only_name,
                "agent": RuleOnlyReferenceAgent(
                    rule_agent=_build_rule_agent(reference_rule_only_agent_level),
                    name=rule_only_name,
                ),
            }
        )
    boosted_reference_names = set(_latest_reference_names(reference_paths, count=5))
    historical_reference_win_rates = _latest_reference_win_rates(log_directory)
    candidate_agent = _initialize_candidate_agent(
        reference_path=reference_paths[-1],
        candidate_name=f"{candidate_prefix}_v{version_number}",
        board_size=board_size,
        seed=seed + 1,
        device=device,
        candidate_init_model_path=resolved_candidate_init_model_path,
    )
    teacher_agent = _build_rule_agent(teacher_rule_agent_level)
    pretrain_loss = _pretrain_from_reference(
        candidate_agent=candidate_agent,
        reference_agent=primary_reference_agent,
        board_size=board_size,
        positions=pretrain_positions,
        seed=seed + 2,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_label = _log_label_for_candidate_prefix(candidate_prefix)
    training_log_path = log_directory / f"{timestamp}_{log_label}_training.log"
    rolling_window = deque(maxlen=min(50, max(10, num_games)))
    summary_counter: Counter[str] = Counter()
    reference_game_counts: Counter[str] = Counter()
    candidate_role_counts: Counter[tuple[str, str]] = Counter()
    reference_role_counts: Counter[tuple[str, str, str]] = Counter()
    hard_variant_counts: Counter[tuple[str, str]] = Counter()
    selection_rng = random.Random(seed + 10_000)
    _initialize_reference_log(
        training_log_path,
        board_size,
        reference_paths,
        candidate_path,
        pretrain_positions,
        pretrain_loss,
        reference_cycle_length,
        log_label,
        resolved_candidate_init_model_path,
        reference_rule_agent_level,
        reference_rule_opening_moves,
        reference_rule_followup_probability,
        reference_rule_only_agent_level,
        teacher_rule_agent_level,
        teacher_weight,
    )

    for game_index in range(1, num_games + 1):
        env = GomokuEnv(board_size=board_size)
        reference_entry = selection_rng.choice(reference_entries)
        reference_name = reference_entry["name"]
        reference_agent = reference_entry["agent"]
        reference_variant = _rule_agent_variant(reference_agent)
        reference_game_counts[reference_name] += 1
        if reference_rule_agent_level == "super_easy":
            assignments = {1: (candidate_agent.name, candidate_agent), -1: (reference_name, reference_agent)}
        elif game_index % 2 == 1:
            assignments = {1: (reference_name, reference_agent), -1: (candidate_agent.name, candidate_agent)}
        else:
            assignments = {1: (candidate_agent.name, candidate_agent), -1: (reference_name, reference_agent)}

        candidate_records: list[TorchCNNValueStepRecord] = []
        move_history = []
        teacher_variant = _rule_agent_variant(teacher_agent)
        if reference_variant is not None:
            hard_variant_counts[("reference", reference_variant)] += 1
        if teacher_weight > 0.0 and teacher_variant is not None:
            hard_variant_counts[("teacher", teacher_variant)] += 1

        while not env.done:
            player = env.current_player
            agent_name, agent = assignments[player]
            training = agent_name == candidate_agent.name
            action, step_record = agent.select_action(env, training=training)
            if training:
                reference_action, _ = reference_agent.select_action(env, training=False)
                step_record.reference_board_tensor = candidate_agent._board_tensor_after_action(
                    env,
                    reference_action,
                )
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

        rewards = _reward_map(env.winner, assignments, reference_name, candidate_agent.name)
        lr_multiplier = 2.0 if reference_name in boosted_reference_names else 1.0
        lr_multiplier *= _historical_reference_lr_multiplier(
            historical_reference_win_rates.get(reference_name)
        )
        if rewards[candidate_agent.name] < 0 and env.move_count <= 15:
            lr_multiplier *= 3.0
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
        if rewards[candidate_agent.name] > 0:
            candidate_outcome = "win"
        elif rewards[candidate_agent.name] < 0:
            candidate_outcome = "loss"
        else:
            candidate_outcome = "draw"
        candidate_role_counts[(candidate_role, candidate_outcome)] += 1
        reference_role_counts[(reference_name, candidate_role, candidate_outcome)] += 1

        _append_reference_game_record(
            training_log_path,
            {
                "game": game_index,
                "winner": winner_name,
                "moves": env.move_count,
                "reference_role": "black" if assignments[1][0] == reference_name else "white",
                "candidate_role": candidate_role,
                "candidate_reward": rewards[candidate_agent.name],
                "candidate_epsilon": candidate_agent.epsilon,
                "reference_name": reference_name,
                "reference_variant": reference_variant,
                "teacher_variant": teacher_variant if teacher_weight > 0.0 else None,
                "move_history": move_history,
            },
            board_size,
        )

        if game_index % save_every == 0 or game_index == num_games:
            candidate_agent.save(candidate_path)
            candidate_agent.save(latest_candidate_path)

    _append_reference_summary(
        training_log_path,
        summary_counter,
        rolling_window,
        num_games,
        [entry["name"] for entry in reference_entries],
        candidate_agent.name,
        hard_variant_counts,
    )
    candidate_agent.save(candidate_path)
    candidate_agent.save(latest_candidate_path)
    promoted_reference_path = reference_directory / f"{candidate_path.stem}_reference.pt"
    shutil.copyfile(candidate_path, promoted_reference_path)
    winrate_log_path = training_log_path.with_name(f"{training_log_path.stem}_winrates.log")
    _write_reference_winrate_log(
        winrate_log_path,
        candidate_name=candidate_agent.name,
        candidate_version=version_number,
        training_log_path=training_log_path,
        reference_names=[entry["name"] for entry in reference_entries],
        reference_game_counts=reference_game_counts,
        summary_counter=summary_counter,
        candidate_role_counts=candidate_role_counts,
        reference_role_counts=reference_role_counts,
    )

    candidate_wins = summary_counter[candidate_agent.name]
    reference_wins = sum(summary_counter[entry["name"]] for entry in reference_entries)
    decisive_games = max(1, candidate_wins + reference_wins)
    candidate_win_rate = candidate_wins / decisive_games

    return {
        "games": num_games,
        "training_log_path": str(training_log_path),
        "reference_model_path": ", ".join(str(path) for path in reference_paths),
        "reference_model_paths": [str(path) for path in reference_paths],
        "candidate_model_path": str(candidate_path),
        "candidate_latest_path": str(latest_candidate_path),
        "candidate_init_model_path": (
            str(resolved_candidate_init_model_path) if resolved_candidate_init_model_path else None
        ),
        "promoted_reference_path": str(promoted_reference_path),
        "winrate_log_path": str(winrate_log_path),
        "candidate_version": version_number,
        "candidate_name": candidate_agent.name,
        "reference_names": [entry["name"] for entry in reference_entries],
        "summary": dict(summary_counter),
        "candidate_win_rate_vs_reference": candidate_win_rate,
        "goal_reached": candidate_win_rate >= 0.7,
        "pretrain_positions": pretrain_positions,
        "pretrain_loss": pretrain_loss,
        "reference_cycle_length": reference_cycle_length,
        "teacher_name": teacher_agent.name,
        "teacher_weight": teacher_weight,
        "reference_rule_only_agent_level": reference_rule_only_agent_level,
    }


def train_with_progressive_references(
    num_games: int,
    promotion_interval: int,
    board_size: int,
    seed: int,
    save_every: int,
    log_dir: str | Path,
    model_dir: str | Path,
    reference_model_paths: str | Path | list[str | Path] | None,
    candidate_init_model_path: str | Path,
    starting_candidate_version: int,
    final_candidate_version: int | None,
    device: str,
    pretrain_positions: int,
    reference_cycle_length: int,
    reference_eval_games: int = 20,
    exclusion_threshold: float = 0.9,
    min_reference_count: int = 10,
    evaluation_interval: int = 1000,
    reference_rule_only_agent_level: str | None = None,
) -> dict:
    if promotion_interval < 1:
        raise ValueError("promotion_interval must be at least 1")
    if evaluation_interval < 1:
        raise ValueError("evaluation_interval must be at least 1")
    resolved_final_version = final_candidate_version

    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    reference_directory = _reference_directory(model_directory)
    if reference_model_paths is None:
        current_reference_paths = _all_reference_paths(model_directory)
    else:
        current_reference_paths = _resolve_reference_paths(reference_model_paths, model_directory, log_directory)
    current_candidate_init = Path(candidate_init_model_path)
    block_results = []
    summary_log_path = log_directory / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_value_reference_progressive.log"
    _initialize_progressive_reference_log(
        summary_log_path,
        starting_candidate_version=starting_candidate_version,
        final_candidate_version=resolved_final_version,
        promotion_interval=promotion_interval,
        evaluation_interval=evaluation_interval,
        total_games=num_games,
    )

    games_completed = 0
    last_evaluation_at = 0
    candidate_version = starting_candidate_version
    while games_completed < num_games:
        if not block_results:
            active_reference_paths = current_reference_paths
        else:
            active_reference_paths = current_reference_paths
        if not active_reference_paths:
            active_reference_paths = current_reference_paths
        current_interval = promotion_interval
        block_games = min(current_interval, num_games - games_completed)
        block_result = train_against_reference(
            num_games=block_games,
            board_size=board_size,
            seed=seed + ((candidate_version - starting_candidate_version) * 1000),
            save_every=min(save_every, block_games),
            log_dir=log_dir,
            model_dir=model_dir,
            reference_model_path=active_reference_paths,
            candidate_init_model_path=current_candidate_init,
            candidate_version=candidate_version,
            device=device,
            pretrain_positions=pretrain_positions,
            reference_cycle_length=reference_cycle_length,
            reference_rule_only_agent_level=reference_rule_only_agent_level,
        )
        candidate_path = Path(block_result["candidate_model_path"])
        promoted_reference_path = reference_directory / f"{candidate_path.stem}_reference.pt"
        shutil.copyfile(candidate_path, promoted_reference_path)
        current_reference_paths = [*current_reference_paths, promoted_reference_path]
        current_candidate_init = candidate_path
        block_results.append(
            {
                "block_index": len(block_results) + 1,
                "games": block_games,
                "candidate_version": candidate_version,
                "candidate_model_path": str(candidate_path),
                "promoted_reference_path": str(promoted_reference_path),
                "training_log_path": block_result["training_log_path"],
                "winrate_log_path": block_result["winrate_log_path"],
                "candidate_win_rate_vs_reference": block_result["candidate_win_rate_vs_reference"],
                "reference_model_paths": block_result["reference_model_paths"],
                "summary": block_result["summary"],
            }
        )
        _append_progressive_reference_block(summary_log_path, block_results[-1])
        games_completed += block_games
        if (
            games_completed - last_evaluation_at >= evaluation_interval
            and current_reference_paths
        ):
            evaluation_result = _evaluate_candidate_against_references(
                candidate_model_path=current_candidate_init,
                reference_paths=current_reference_paths,
                board_size=board_size,
                seed=seed + games_completed + 50_000,
                device=device,
                games_per_reference=reference_eval_games,
                log_directory=log_directory,
                candidate_version=candidate_version,
            )
            current_reference_paths, removed_reference_paths = _prune_reference_paths_by_win_rate(
                current_reference_paths,
                evaluation_result["per_reference_win_rates"],
                exclusion_threshold=exclusion_threshold,
                min_reference_count=min_reference_count,
            )
            _append_progressive_reference_evaluation(
                summary_log_path,
                games_completed=games_completed,
                evaluation_result=evaluation_result,
                removed_reference_paths=removed_reference_paths,
                min_reference_count=min_reference_count,
                exclusion_threshold=exclusion_threshold,
            )
            last_evaluation_at = games_completed
        candidate_version += 1

    return {
        "games": num_games,
        "promotion_interval": promotion_interval,
        "evaluation_interval": evaluation_interval,
        "starting_candidate_version": starting_candidate_version,
        "final_candidate_version": candidate_version - 1,
        "block_results": block_results,
        "final_candidate_model_path": str(current_candidate_init),
        "final_reference_model_paths": [str(path) for path in current_reference_paths],
        "summary_log_path": str(summary_log_path),
    }


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


def _load_reference_agent(
    reference_path: Path,
    reference_name: str,
    seed: int,
    device: str,
    reference_rule_agent_level: str | None = None,
    reference_rule_opening_moves: int = 20,
    reference_rule_followup_probability: float = 0.1,
) -> Any:
    if reference_path.suffix == ".pt":
        payload = torch.load(reference_path, map_location=device)
        model_state = payload.get("model_state_dict", {})
        if any(key.startswith("features.") or key.startswith("head.") for key in model_state):
            base_agent = TorchCNNValueAgent.load(reference_path, name=reference_name, device=device)
        else:
            base_agent = TorchValueAgent.load(reference_path, name=reference_name, device=device)
    else:
        base_agent = ValueAgent.load(reference_path, name=reference_name, seed=seed)

    if reference_rule_agent_level is not None:
        return RuleAugmentedReferenceAgent(
            base_agent=base_agent,
            rule_agent=_build_rule_agent(reference_rule_agent_level),
            name=reference_name,
            opening_rule_moves=reference_rule_opening_moves,
            late_rule_probability=reference_rule_followup_probability,
            seed=seed,
        )
    return base_agent


def _rule_only_reference_name(candidate_prefix: str, reference_rule_only_agent_level: str) -> str:
    return f"{candidate_prefix}_{reference_rule_only_agent_level}_rule_only_reference"


def _default_reference_path(model_directory: Path) -> Path:
    return _default_reference_paths(model_directory)[-1]


def _all_reference_paths(
    model_directory: Path,
    include_tactical_references: bool = False,
) -> list[Path]:
    reference_directory = _reference_directory(model_directory)
    reference_candidates: list[tuple[str, int, Path]] = []
    for path in reference_directory.glob("*_reference.pt"):
        for prefix in ("value_agent", "tactical_value_agent", "tactical_rule_value_agent"):
            version = _reference_version_from_path(path, prefix)
            if version is not None:
                reference_candidates.append((prefix, version, path))
                break
    if reference_candidates:
        return [path for _, _, path in sorted(reference_candidates, key=lambda item: (item[0], item[1]))]
    return [model_directory / "value_agent_v1.json"]


def _default_reference_paths(
    model_directory: Path,
    ensemble_size: int | None = None,
    log_directory: Path | None = None,
    include_tactical_references: bool = False,
) -> list[Path]:
    ordered = _all_reference_paths(model_directory, include_tactical_references=include_tactical_references)
    active = ordered
    if ensemble_size is None:
        return active
    return active[-ensemble_size:]


def _resolve_reference_paths(
    reference_model_path: str | Path | list[str | Path] | None,
    model_directory: Path,
    log_directory: Path,
    candidate_prefix: str = "value_agent",
) -> list[Path]:
    if reference_model_path is None:
        return _default_reference_paths(
            model_directory,
            log_directory=log_directory,
            include_tactical_references=candidate_prefix == "tactical_value_agent",
        )
    if isinstance(reference_model_path, (str, Path)):
        return [Path(reference_model_path)]
    return [Path(path) for path in reference_model_path]


def _reference_version_from_path(path: Path, prefix: str) -> int | None:
    suffix = path.stem.removeprefix(f"{prefix}_v").removesuffix("_reference")
    if suffix.isdigit():
        return int(suffix)
    return None


def _reference_index_for_game(game_index: int, num_references: int, reference_cycle_length: int) -> int:
    if game_index < 1:
        raise ValueError("game_index must be at least 1")
    if num_references < 1:
        raise ValueError("num_references must be at least 1")
    if reference_cycle_length < 1:
        raise ValueError("reference_cycle_length must be at least 1")
    return ((game_index - 1) // reference_cycle_length) % num_references


def _filter_reference_paths(
    reference_paths: list[Path],
    log_directory: Path,
    exclusion_threshold: float = 0.9,
    min_reference_count: int = 10,
    max_removals: int | None = None,
) -> list[Path]:
    if not reference_paths or reference_paths[0].suffix != ".pt":
        return reference_paths
    if len(reference_paths) <= min_reference_count:
        return reference_paths
    latest_rates = _latest_reference_win_rates(log_directory)
    removable = [
        path
        for path in reference_paths
        if latest_rates.get(_reference_name_from_path(path), -1.0) >= exclusion_threshold
    ]
    if not removable:
        return reference_paths
    removable.sort(key=lambda path: _reference_sort_key(path))
    remaining = list(reference_paths)
    removals = 0
    removal_cap = max_removals if max_removals is not None else len(removable)
    for path in removable:
        if len(remaining) <= min_reference_count or removals >= removal_cap:
            break
        remaining.remove(path)
        removals += 1
    return remaining


def _latest_reference_win_rates(log_directory: Path) -> dict[str, float]:
    if not log_directory.exists():
        return {}
    latest_rates: dict[str, float] = {}
    for path in sorted(log_directory.glob("*_value_reference_training_winrates.log"), reverse=True):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = REFERENCE_WINRATE_LINE_RE.match(line)
            if not match:
                continue
            reference_name = match.group("reference_name")
            if reference_name in latest_rates:
                continue
            latest_rates[reference_name] = float(match.group("win_rate")) / 100.0
    return latest_rates


def _reference_sort_key(path: Path) -> tuple[int, int, str]:
    for prefix_order, prefix in enumerate(("value_agent", "tactical_value_agent")):
        version = _reference_version_from_path(path, prefix)
        if version is not None:
            return (prefix_order, version, path.name)
    return (10**9, 10**9, path.name)


def _historical_reference_lr_multiplier(reference_win_rate: float | None) -> float:
    if reference_win_rate is None:
        return 1.0
    if reference_win_rate >= 1.0:
        return 0.1
    if reference_win_rate >= 0.9:
        return 0.5
    return 1.0


def _initialize_candidate_agent(
    reference_path: Path,
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

    if reference_path.suffix == ".pt":
        agent = TorchCNNValueAgent(
            name=candidate_name,
            board_size=board_size,
            learning_rate=1e-4,
            imitation_weight=1.0,
            epsilon_decay=0.999,
            seed=seed,
            device=device,
        )
        agent.epsilon = 1.0
        agent.episodes_trained = 0
        return agent

    return TorchCNNValueAgent(
        name=candidate_name,
        board_size=board_size,
        learning_rate=1e-4,
        imitation_weight=1.0,
        epsilon_decay=0.999,
        seed=seed,
        device=device,
    )


def _resolve_candidate_init_model_path(
    candidate_init_model_path: str | Path | None,
    candidate_prefix: str,
    model_directory: Path,
) -> Path | None:
    if candidate_init_model_path is not None:
        return Path(candidate_init_model_path)
    if candidate_prefix == "value_agent":
        return None
    if candidate_prefix == "tactical_value_agent":
        return _latest_tactical_value_model_path(model_directory) or _latest_value_model_path(model_directory)
    return None


def _latest_tactical_value_model_path(model_directory: Path) -> Path | None:
    reference_directory = _reference_directory(model_directory)
    versioned_paths: list[tuple[int, Path]] = []
    pattern = re.compile(r"tactical_value_agent_v(\d+)$")
    for path in model_directory.glob("tactical_value_agent_v*.pt"):
        if path.stem.endswith("_reference"):
            continue
        if not (reference_directory / f"{path.stem}_reference.pt").exists():
            continue
        match = pattern.fullmatch(path.stem)
        if match:
            versioned_paths.append((int(match.group(1)), path))
    if not versioned_paths:
        return None
    return max(versioned_paths, key=lambda item: item[0])[1]


def _latest_value_model_path(model_directory: Path) -> Path | None:
    versioned_paths: list[tuple[int, Path]] = []
    pattern = re.compile(r"value_agent_v(\d+)$")
    for path in model_directory.glob("value_agent_v*.pt"):
        if path.stem.endswith("_reference"):
            continue
        match = pattern.fullmatch(path.stem)
        if match:
            versioned_paths.append((int(match.group(1)), path))
    if not versioned_paths:
        return None
    return max(versioned_paths, key=lambda item: item[0])[1]


def _pretrain_from_reference(
    candidate_agent: TorchCNNValueAgent,
    reference_agent: Any,
    board_size: int,
    positions: int,
    seed: int,
) -> float:
    if positions <= 0:
        return 0.0

    rng = torch.Generator().manual_seed(seed)
    losses = []
    collected = 0
    game_offset = 0
    while collected < positions:
        env = GomokuEnv(board_size=board_size)
        while not env.done and collected < positions:
            reference_action, _ = reference_agent.select_action(env, training=False)
            positive = [candidate_agent._board_tensor_after_action(env, reference_action)]
            valid_actions = [action for action in env.get_valid_actions() if action != reference_action]
            negative = []
            if valid_actions:
                permutation = torch.randperm(len(valid_actions), generator=rng).tolist()
                for action_index in permutation[: min(3, len(valid_actions))]:
                    negative.append(candidate_agent._board_tensor_after_action(env, valid_actions[action_index]))
            losses.append(candidate_agent.supervised_update(positive, negative))
            env.step(reference_action)
            collected += 1
        game_offset += 1
        if game_offset > positions * 2:
            break

    return sum(losses) / max(1, len(losses))


def _next_candidate_version(model_directory: Path, candidate_prefix: str = "value_agent") -> int:
    version_numbers = []
    pattern = re.compile(rf"{re.escape(candidate_prefix)}_v(\d+)$")
    for path in model_directory.glob(f"{candidate_prefix}_v*.pt"):
        match = pattern.fullmatch(path.stem)
        if match:
            version_numbers.append(int(match.group(1)))
    return max(version_numbers, default=1) + 1


def _reference_name_from_path(reference_path: Path) -> str:
    stem = reference_path.stem
    return stem if stem else "reference_agent"


def _latest_reference_names(reference_paths: list[Path], count: int) -> list[str]:
    ordered_candidates: list[tuple[int, str]] = []
    fallback_names: list[str] = []
    for path in reference_paths:
        name = _reference_name_from_path(path)
        fallback_names.append(name)
        for prefix_order, prefix in enumerate(("value_agent", "tactical_value_agent")):
            version = _reference_version_from_path(path, prefix)
            if version is not None:
                ordered_candidates.append((prefix_order, version, name))
                break
    if ordered_candidates:
        ordered_names = [
            name for _, _, name in sorted(ordered_candidates, key=lambda item: (item[0], item[1]))
        ]
        return ordered_names[-count:]
    return fallback_names[-count:]


def _scheduled_reference_game_counts(
    reference_names: list[str],
    total_games: int,
    reference_cycle_length: int,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not reference_names:
        return {}
    for game_index in range(1, total_games + 1):
        reference_name = reference_names[
            _reference_index_for_game(game_index, len(reference_names), reference_cycle_length)
        ]
        counts[reference_name] += 1
    return {reference_name: counts[reference_name] for reference_name in reference_names}


def _reward_map(
    winner: int,
    assignments: dict[int, tuple[str, object]],
    reference_name: str,
    candidate_name: str,
) -> dict[str, float]:
    rewards = {reference_name: 0.0, candidate_name: 0.0}
    if winner == 0:
        return rewards

    winner_name = assignments[winner][0]
    loser_name = assignments[-winner][0]
    rewards[winner_name] = 1.0
    rewards[loser_name] = -1.0
    return rewards


def _initialize_reference_log(
    path: Path,
    board_size: int,
    reference_paths: list[Path],
    candidate_path: Path,
    pretrain_positions: int,
    pretrain_loss: float,
    reference_cycle_length: int,
    log_label: str = "value_reference",
    init_model_path: Path | None = None,
    reference_rule_agent_level: str | None = None,
    reference_rule_opening_moves: int = 20,
    reference_rule_followup_probability: float = 0.1,
    reference_rule_only_agent_level: str | None = None,
    teacher_rule_agent_level: str = "hard",
    teacher_weight: float = 1.0,
) -> None:
    path.write_text(
        "\n".join(
            [
                _training_log_header(log_label),
                f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Board size: {board_size}x{board_size}",
                f"Reference model: {', '.join(str(reference_path) for reference_path in reference_paths)}",
                f"Candidate model: {candidate_path}",
                f"Candidate init model: {init_model_path if init_model_path is not None else 'random'}",
                f"Reference cycle length: {reference_cycle_length}",
                f"Reference rule overlay: {reference_rule_agent_level or 'none'}",
                f"Reference rule opening moves: {reference_rule_opening_moves}",
                f"Reference rule follow-up probability: {reference_rule_followup_probability:.3f}",
                f"Reference rule-only agent: {reference_rule_only_agent_level or 'none'}",
                f"Teacher rule agent: {teacher_rule_agent_level}",
                f"Teacher weight: {teacher_weight:.3f}",
                f"Pretrain positions: {pretrain_positions}",
                f"Pretrain loss: {pretrain_loss:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _log_label_for_candidate_prefix(candidate_prefix: str) -> str:
    if candidate_prefix == "value_agent":
        return "value_reference"
    return candidate_prefix.removesuffix("_agent") + "_reference"


def _training_log_header(log_label: str) -> str:
    if log_label == "value_reference":
        return "Gomoku Value Reference Training Log"
    words = " ".join(word.capitalize() for word in log_label.split("_"))
    return f"Gomoku {words} Training Log"


def _append_reference_game_record(path: Path, record: dict, board_size: int) -> None:
    extra_fields = []
    if record.get("reference_variant") is not None:
        extra_fields.append(f"reference_variant={record['reference_variant']}")
    if record.get("teacher_variant") is not None:
        extra_fields.append(f"teacher_variant={record['teacher_variant']}")
    extra_suffix = f", {', '.join(extra_fields)}" if extra_fields else ""
    lines = [
        f"Game {record['game']}: winner={record['winner']}, moves={record['moves']}, "
        f"board={board_size}x{board_size}, reference_role={record['reference_role']}, "
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


def _append_reference_summary(
    path: Path,
    summary_counter: Counter[str],
    rolling_window: deque[str],
    total_games: int,
    reference_names: list[str],
    candidate_name: str,
    hard_variant_counts: Counter[tuple[str, str]] | None = None,
) -> None:
    reference_wins = sum(summary_counter[reference_name] for reference_name in reference_names)
    candidate_wins = summary_counter[candidate_name]
    draws = summary_counter["draw"]
    rolling_reference = sum(1 for winner in rolling_window if winner in reference_names)
    rolling_candidate = sum(1 for winner in rolling_window if winner == candidate_name)
    rolling_draws = sum(1 for winner in rolling_window if winner == "draw")
    decisive_games = max(1, reference_wins + candidate_wins)

    lines = [
        "Summary",
        "",
        f"- Total games: {total_games}",
        f"- Reference ensemble wins: {reference_wins}",
        f"- {candidate_name} wins: {candidate_wins}",
        f"- Draws: {draws}",
        f"- Candidate win rate vs reference ensemble: {candidate_wins / decisive_games:.2%}",
        f"- Rolling window size: {len(rolling_window)}",
        f"- Rolling reference wins: {rolling_reference}",
        f"- Rolling candidate wins: {rolling_candidate}",
        f"- Rolling draws: {rolling_draws}",
    ]
    for reference_name in reference_names:
        lines.append(f"- {reference_name} wins: {summary_counter[reference_name]}")
    hard_variant_counts = hard_variant_counts or Counter()
    if hard_variant_counts:
        lines.append("- Hard variant usage:")
        for role in ("reference", "teacher"):
            for variant in ("offensive", "neutral", "defensive"):
                lines.append(f"  - {role} {variant}: {hard_variant_counts[(role, variant)]}")
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines).rstrip() + "\n")


def _initialize_progressive_reference_log(
    path: Path,
    starting_candidate_version: int,
    final_candidate_version: int | None,
    promotion_interval: int,
    evaluation_interval: int,
    total_games: int,
) -> None:
    lines = [
        "Gomoku Value Reference Progressive Log",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Starting version: v{starting_candidate_version}",
        f"Final version: {f'v{final_candidate_version}' if final_candidate_version is not None else 'auto'}",
        f"Promotion interval: {promotion_interval}",
        f"Evaluation interval: {evaluation_interval}",
        f"Total games: {total_games}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_progressive_reference_block(path: Path, block_result: dict[str, Any]) -> None:
    lines = [
        f"Block {block_result['block_index']}: v{block_result['candidate_version']}",
        f"- Games: {block_result['games']}",
        f"- Win rate vs reference ensemble: {block_result['candidate_win_rate_vs_reference']:.2%}",
        f"- Candidate model: {block_result['candidate_model_path']}",
        f"- Promoted reference: {block_result['promoted_reference_path']}",
        f"- Training log: {block_result['training_log_path']}",
        f"- Winrate log: {block_result['winrate_log_path']}",
        "",
    ]
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))


def _append_progressive_reference_evaluation(
    path: Path,
    games_completed: int,
    evaluation_result: dict[str, Any],
    removed_reference_paths: list[Path],
    min_reference_count: int,
    exclusion_threshold: float,
) -> None:
    lines = [
        f"Evaluation after {games_completed} games",
        f"- Candidate: {evaluation_result['candidate_name']}",
        f"- Evaluation log: {evaluation_result['evaluation_log_path']}",
        f"- Total references before prune: {evaluation_result['reference_count']}",
        f"- Exclusion threshold: {exclusion_threshold:.0%}",
        f"- Minimum reference count: {min_reference_count}",
    ]
    for reference_name, win_rate in evaluation_result["per_reference_win_rates"].items():
        lines.append(f"- {reference_name}: {win_rate:.2%}")
    if removed_reference_paths:
        lines.append(
            f"- Removed references: {', '.join(path.name for path in removed_reference_paths)}"
        )
    else:
        lines.append("- Removed references: none")
    lines.append("")
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))


def _evaluate_candidate_against_references(
    candidate_model_path: str | Path,
    reference_paths: list[Path],
    board_size: int,
    seed: int,
    device: str,
    games_per_reference: int,
    log_directory: Path,
    candidate_version: int,
) -> dict[str, Any]:
    candidate_name = Path(candidate_model_path).stem
    candidate_agent = TorchCNNValueAgent.load(candidate_model_path, name=candidate_name, device=device)
    candidate_agent.epsilon = 0.0
    rng = random.Random(seed)
    per_reference_records = []
    per_reference_win_rates: dict[str, float] = {}

    for reference_offset, reference_path in enumerate(reference_paths):
        reference_name = _reference_name_from_path(reference_path)
        reference_agent = _load_reference_agent(
            reference_path,
            reference_name,
            seed + reference_offset + 1,
            device,
        )
        reference_agent.epsilon = 0.0
        wins = 0
        losses = 0
        draws = 0
        for game_index in range(games_per_reference):
            env = GomokuEnv(board_size=board_size)
            candidate_is_black = ((game_index + rng.randint(0, 1)) % 2) == 0
            if candidate_is_black:
                assignments = {1: (candidate_name, candidate_agent), -1: (reference_name, reference_agent)}
            else:
                assignments = {1: (reference_name, reference_agent), -1: (candidate_name, candidate_agent)}

            while not env.done:
                _, agent = assignments[env.current_player]
                action, _ = agent.select_action(env, training=False)
                env.step(action)

            if env.winner == 0:
                draws += 1
            elif assignments[env.winner][0] == candidate_name:
                wins += 1
            else:
                losses += 1

        win_rate = wins / max(1, wins + losses)
        per_reference_records.append(
            {
                "reference_name": reference_name,
                "reference_path": str(reference_path),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "games": games_per_reference,
                "win_rate": win_rate,
            }
        )
        per_reference_win_rates[reference_name] = win_rate

    evaluation_log_path = log_directory / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_value_reference_eval_v{candidate_version}.log"
    )
    _write_reference_evaluation_log(
        evaluation_log_path,
        candidate_name=candidate_name,
        candidate_model_path=Path(candidate_model_path),
        per_reference_records=per_reference_records,
        games_per_reference=games_per_reference,
    )
    return {
        "candidate_name": candidate_name,
        "candidate_model_path": str(candidate_model_path),
        "evaluation_log_path": str(evaluation_log_path),
        "reference_count": len(reference_paths),
        "per_reference_win_rates": per_reference_win_rates,
    }


def _write_reference_evaluation_log(
    path: Path,
    candidate_name: str,
    candidate_model_path: Path,
    per_reference_records: list[dict[str, Any]],
    games_per_reference: int,
) -> None:
    lines = [
        "Gomoku Value Reference Evaluation Log",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Candidate: {candidate_name}",
        f"Candidate model: {candidate_model_path}",
        f"Games per reference: {games_per_reference}",
        "",
        "Per-reference candidate win rates",
    ]
    for record in per_reference_records:
        lines.append(
            f"- {record['reference_name']}: {record['wins']}/{record['games']} "
            f"({record['win_rate']:.2%}) | losses={record['losses']} | draws={record['draws']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prune_reference_paths_by_win_rate(
    reference_paths: list[Path],
    per_reference_win_rates: dict[str, float],
    exclusion_threshold: float,
    min_reference_count: int,
) -> tuple[list[Path], list[Path]]:
    if len(reference_paths) <= min_reference_count:
        return reference_paths, []

    removable = [
        path
        for path in reference_paths
        if per_reference_win_rates.get(_reference_name_from_path(path), -1.0) >= exclusion_threshold
    ]
    if not removable:
        return reference_paths, []

    removable.sort(key=_reference_sort_key)
    remaining = list(reference_paths)
    removed: list[Path] = []
    for path in removable:
        if len(remaining) <= min_reference_count:
            break
        remaining.remove(path)
        removed.append(path)
    return remaining, removed


def _write_reference_winrate_log(
    path: Path,
    candidate_name: str,
    candidate_version: int,
    training_log_path: Path,
    reference_names: list[str],
    reference_game_counts: Counter[str] | dict[str, int],
    summary_counter: Counter[str],
    candidate_role_counts: Counter[tuple[str, str]] | None = None,
    reference_role_counts: Counter[tuple[str, str, str]] | None = None,
) -> None:
    total_reference_games = sum(int(reference_game_counts.get(reference_name, 0)) for reference_name in reference_names)
    candidate_wins = summary_counter[candidate_name]
    reference_wins = sum(summary_counter[reference_name] for reference_name in reference_names)
    draws = summary_counter["draw"]
    candidate_role_counts = candidate_role_counts or Counter()
    reference_role_counts = reference_role_counts or Counter()
    black_wins = candidate_role_counts[("black", "win")]
    black_losses = candidate_role_counts[("black", "loss")]
    black_draws = candidate_role_counts[("black", "draw")]
    white_wins = candidate_role_counts[("white", "win")]
    white_losses = candidate_role_counts[("white", "loss")]
    white_draws = candidate_role_counts[("white", "draw")]
    lines = [
        "Gomoku Value Reference Winrate Log",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Candidate: {candidate_name}",
        f"Candidate version: v{candidate_version}",
        f"Training log: {training_log_path}",
        f"Total games: {total_reference_games}",
        f"Candidate wins: {candidate_wins}",
        f"Reference wins: {reference_wins}",
        f"Draws: {draws}",
        f"Candidate win rate vs reference ensemble: {candidate_wins / max(1, candidate_wins + reference_wins):.2%}",
        f"Candidate black win rate: {black_wins}/{black_wins + black_losses} "
        f"({black_wins / max(1, black_wins + black_losses):.2%}) | losses={black_losses} | draws={black_draws}",
        f"Candidate white win rate: {white_wins}/{white_wins + white_losses} "
        f"({white_wins / max(1, white_wins + white_losses):.2%}) | losses={white_losses} | draws={white_draws}",
        "",
        "Per-reference win rates",
    ]
    for reference_name in reference_names:
        games = int(reference_game_counts.get(reference_name, 0))
        losses = summary_counter[reference_name]
        draws_for_reference = (
            reference_role_counts[(reference_name, "black", "draw")]
            + reference_role_counts[(reference_name, "white", "draw")]
        )
        wins = max(0, games - losses - draws_for_reference)
        win_rate = wins / max(1, wins + losses)
        black_wins_for_reference = reference_role_counts[(reference_name, "black", "win")]
        black_losses_for_reference = reference_role_counts[(reference_name, "black", "loss")]
        black_draws_for_reference = reference_role_counts[(reference_name, "black", "draw")]
        white_wins_for_reference = reference_role_counts[(reference_name, "white", "win")]
        white_losses_for_reference = reference_role_counts[(reference_name, "white", "loss")]
        white_draws_for_reference = reference_role_counts[(reference_name, "white", "draw")]
        lines.append(
            f"- {reference_name}: {wins}/{games} ({win_rate:.2%}) | "
            f"black={black_wins_for_reference}/{black_wins_for_reference + black_losses_for_reference} "
            f"({black_wins_for_reference / max(1, black_wins_for_reference + black_losses_for_reference):.2%}) "
            f"losses={black_losses_for_reference} draws={black_draws_for_reference} | "
            f"white={white_wins_for_reference}/{white_wins_for_reference + white_losses_for_reference} "
            f"({white_wins_for_reference / max(1, white_wins_for_reference + white_losses_for_reference):.2%}) "
            f"losses={white_losses_for_reference} draws={white_draws_for_reference}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train value-agent version against the current frozen reference")
    parser.add_argument("--games", type=int, default=1000, help="number of games")
    parser.add_argument("--board-size", type=int, default=15, help="board size")
    parser.add_argument("--save-every", type=int, default=100, help="checkpoint interval")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    parser.add_argument("--pretrain-positions", type=int, default=400, help="reference imitation positions")
    parser.add_argument(
        "--candidate-init-model",
        type=str,
        default=None,
        help="optional candidate checkpoint to continue training from",
    )
    parser.add_argument(
        "--candidate-version",
        type=int,
        default=None,
        help="explicit candidate version number to write",
    )
    parser.add_argument(
        "--candidate-prefix",
        type=str,
        default="value_agent",
        help="candidate filename/name prefix, e.g. value_agent or tactical_value_agent",
    )
    parser.add_argument(
        "--reference-model",
        action="append",
        default=None,
        help="path to frozen reference model; repeat to use multiple references in rotation",
    )
    parser.add_argument(
        "--reference-rule-agent-level",
        type=str,
        choices=("super_easy", "easy", "normal", "hard"),
        default=None,
        help="apply a rule-based opening overlay to value_agent references only",
    )
    parser.add_argument(
        "--reference-rule-opening-moves",
        type=int,
        default=20,
        help="opening move count where the rule overlay controls value_agent references",
    )
    parser.add_argument(
        "--reference-rule-followup-probability",
        type=float,
        default=0.1,
        help="probability of falling back to the rule overlay after the opening window",
    )
    parser.add_argument(
        "--reference-rule-only-agent-level",
        type=str,
        choices=("super_easy", "easy", "normal", "hard"),
        default=None,
        help="add a pure rule-based reference opponent at the chosen strength",
    )
    parser.add_argument(
        "--teacher-rule-agent-level",
        type=str,
        choices=("super_easy", "easy", "normal", "hard"),
        default="hard",
        help="rule-based teacher level applied to candidate moves",
    )
    parser.add_argument(
        "--teacher-weight",
        type=float,
        default=1.0,
        help="teacher imitation loss weight applied on every candidate move",
    )
    parser.add_argument(
        "--reference-cycle-length",
        type=int,
        default=10,
        help="number of consecutive games to play against one reference before rotating",
    )
    parser.add_argument(
        "--promotion-interval",
        type=int,
        default=0,
        help="if set, train in blocks and promote each block result into the reference set",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=1000,
        help="when using progressive promotions, evaluate the current candidate against all references every N games",
    )
    parser.add_argument(
        "--reference-eval-games",
        type=int,
        default=20,
        help="evaluation games per reference when pruning the reference pool",
    )
    parser.add_argument(
        "--reference-exclusion-threshold",
        type=float,
        default=0.9,
        help="drop references only when candidate win rate against them reaches this threshold",
    )
    parser.add_argument(
        "--min-reference-count",
        type=int,
        default=10,
        help="never prune references below this count",
    )
    parser.add_argument(
        "--final-candidate-version",
        type=int,
        default=None,
        help="final version number when using progressive promotions",
    )
    args = parser.parse_args()

    if args.promotion_interval > 0:
        if args.candidate_init_model is None:
            raise ValueError("--candidate-init-model is required when using --promotion-interval")
        if args.candidate_version is None:
            raise ValueError("--candidate-version is required when using --promotion-interval")
        result = train_with_progressive_references(
            num_games=args.games,
            promotion_interval=args.promotion_interval,
            evaluation_interval=args.evaluation_interval,
            board_size=args.board_size,
            seed=args.seed,
            save_every=args.save_every,
            log_dir="logs",
            model_dir="models",
            reference_model_paths=args.reference_model,
            candidate_init_model_path=args.candidate_init_model,
            starting_candidate_version=args.candidate_version,
            final_candidate_version=args.final_candidate_version,
            device=args.device,
            pretrain_positions=args.pretrain_positions,
            reference_cycle_length=args.reference_cycle_length,
            reference_rule_only_agent_level=args.reference_rule_only_agent_level,
            reference_eval_games=args.reference_eval_games,
            exclusion_threshold=args.reference_exclusion_threshold,
            min_reference_count=args.min_reference_count,
        )
        print(
            f"Progressive reference training finished. Final candidate: {result['final_candidate_model_path']} | "
            f"Final version: v{result['final_candidate_version']} | "
            f"Total references: {len(result['final_reference_model_paths'])} | "
            f"Summary log: {result['summary_log_path']}"
        )
        return

    result = train_against_reference(
        num_games=args.games,
        board_size=args.board_size,
        save_every=args.save_every,
        seed=args.seed,
        reference_model_path=args.reference_model,
        candidate_init_model_path=args.candidate_init_model,
        candidate_version=args.candidate_version,
        device=args.device,
        pretrain_positions=args.pretrain_positions,
        reference_cycle_length=args.reference_cycle_length,
        candidate_prefix=args.candidate_prefix,
        reference_rule_agent_level=args.reference_rule_agent_level,
        reference_rule_opening_moves=args.reference_rule_opening_moves,
        reference_rule_followup_probability=args.reference_rule_followup_probability,
        reference_rule_only_agent_level=args.reference_rule_only_agent_level,
        teacher_rule_agent_level=args.teacher_rule_agent_level,
        teacher_weight=args.teacher_weight,
    )
    print(
        f"Reference training finished. Log: {result['training_log_path']} | "
        f"Reference: {result['reference_model_path']} | "
        f"Init: {result['candidate_init_model_path'] or 'random'} | "
        f"Candidate: {result['candidate_model_path']} | "
        f"Promoted: {result['promoted_reference_path']} | "
        f"Win rate vs reference: {result['candidate_win_rate_vs_reference']:.2%}"
    )


if __name__ == "__main__":
    main()
