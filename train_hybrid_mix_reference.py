"""Train a hybrid mix candidate against frozen references."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import train_hybrid_reference as base

from agent.torch_hybrid_mix_agent import TorchHybridMixAgent


def _initialize_mix_log(
    path: Path,
    board_size: int,
    reference_paths: list[Path],
    candidate_path: Path,
    reference_rule_agent_level: str | None,
    reference_rule_opening_moves: int,
    reference_rule_followup_probability: float,
    policy_mix_weight: float,
    policy_loss_weight: float,
    reference_rule_only_agent_level: str | None,
) -> None:
    lines = [
        "Gomoku Hybrid Mix Reference Training Log",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Board size: {board_size}x{board_size}",
        f"Reference model: {', '.join(str(reference_path) for reference_path in reference_paths)}",
        f"Candidate model: {candidate_path}",
        f"Reference rule overlay: {reference_rule_agent_level or 'none'}",
        f"Reference rule opening moves: {reference_rule_opening_moves}",
        f"Reference rule follow-up probability: {reference_rule_followup_probability:.3f}",
        f"Policy decision mix weight: {policy_mix_weight:.3f}",
        f"Policy loss weight: {policy_loss_weight:.3f}",
        f"Reference rule-only agent: {reference_rule_only_agent_level or 'none'}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_against_reference(
    num_games: int = 1000,
    board_size: int = 15,
    seed: int = 7,
    save_every: int = 1000,
    log_dir: str | Path = "logs",
    model_dir: str | Path = "models",
    reference_model_path: str | Path | list[str | Path] | None = None,
    candidate_init_model_path: str | Path | None = "models/tactical_rule_hybrid_agent_v6.pt",
    candidate_version: int | None = None,
    device: str = "cpu",
    candidate_prefix: str = "tactical_rule_hybrid_mix_weight_agent",
    policy_mix_weight: float = 0.01,
    policy_loss_weight: float = 0.05,
    reference_rule_agent_level: str | None = "hard",
    reference_rule_opening_moves: int = 20,
    reference_rule_followup_probability: float = 0.10,
    reference_rule_only_agent_level: str | None = "hard",
) -> dict:
    original_agent_cls = base.TorchHybridAgent
    original_log_init = base._initialize_hybrid_log
    try:
        base.TorchHybridAgent = TorchHybridMixAgent
        base._initialize_hybrid_log = _initialize_mix_log
        return base.train_against_reference(
            num_games=num_games,
            board_size=board_size,
            seed=seed,
            save_every=save_every,
            log_dir=log_dir,
            model_dir=model_dir,
            reference_model_path=reference_model_path,
            candidate_init_model_path=candidate_init_model_path,
            candidate_version=candidate_version,
            device=device,
            candidate_prefix=candidate_prefix,
            policy_mix_weight=policy_mix_weight,
            policy_loss_weight=policy_loss_weight,
            reference_rule_agent_level=reference_rule_agent_level,
            reference_rule_opening_moves=reference_rule_opening_moves,
            reference_rule_followup_probability=reference_rule_followup_probability,
            reference_rule_only_agent_level=reference_rule_only_agent_level,
        )
    finally:
        base.TorchHybridAgent = original_agent_cls
        base._initialize_hybrid_log = original_log_init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid mix candidate against frozen references")
    parser.add_argument("--games", type=int, default=1000, help="number of games")
    parser.add_argument("--board-size", type=int, default=15, help="board size")
    parser.add_argument("--save-every", type=int, default=1000, help="checkpoint interval")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    parser.add_argument("--log-dir", type=str, default="logs", help="directory for training logs")
    parser.add_argument("--model-dir", type=str, default="models", help="directory for model checkpoints")
    parser.add_argument(
        "--candidate-init-model",
        type=str,
        default="models/tactical_rule_hybrid_agent_v6.pt",
        help="optional hybrid checkpoint to continue training from",
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
        default="tactical_rule_hybrid_mix_weight_agent",
        help="candidate filename/name prefix, e.g. tactical_rule_hybrid_mix_weight_agent",
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
        default="hard",
        help="apply a rule-based opening overlay to value_agent references only",
    )
    parser.add_argument(
        "--reference-rule-opening-moves",
        type=int,
        default=20,
        help="opening move count where the rule overlay controls reference models",
    )
    parser.add_argument(
        "--reference-rule-followup-probability",
        type=float,
        default=0.10,
        help="probability of falling back to the rule overlay after the opening window",
    )
    parser.add_argument(
        "--reference-rule-only-agent-level",
        type=str,
        choices=("super_easy", "easy", "normal", "hard"),
        default="hard",
        help="add a pure rule-based reference opponent at the chosen strength",
    )
    parser.add_argument(
        "--policy-mix-weight",
        type=float,
        default=0.01,
        help="how much policy head probability contributes when selecting moves",
    )
    parser.add_argument(
        "--policy-aux-weight",
        type=float,
        default=0.05,
        help="policy loss weight when policy is trained separately",
    )
    parser.add_argument(
        "--reference-cycle-length",
        type=int,
        default=10,
        help="number of consecutive games to play against one reference before rotating",
    )
    args = parser.parse_args()

    result = train_against_reference(
        num_games=args.games,
        board_size=args.board_size,
        seed=args.seed,
        save_every=args.save_every,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        reference_model_path=args.reference_model,
        candidate_init_model_path=args.candidate_init_model,
        candidate_version=args.candidate_version,
        device=args.device,
        candidate_prefix=args.candidate_prefix,
        reference_rule_agent_level=args.reference_rule_agent_level,
        reference_rule_opening_moves=args.reference_rule_opening_moves,
        reference_rule_followup_probability=args.reference_rule_followup_probability,
        reference_rule_only_agent_level=args.reference_rule_only_agent_level,
        policy_mix_weight=args.policy_mix_weight,
        policy_loss_weight=args.policy_aux_weight,
    )
    print(
        f"Hybrid mix reference training finished. Log: {result['training_log_path']} | "
        f"Reference: {result['reference_model_path']} | "
        f"Init: {result['candidate_init_model_path'] or 'random'} | "
        f"Candidate: {result['candidate_model_path']} | "
        f"Promoted: {result['promoted_reference_path']} | "
        f"Win rate vs reference: {result['candidate_win_rate_vs_reference']:.2%}"
    )


if __name__ == "__main__":
    main()
