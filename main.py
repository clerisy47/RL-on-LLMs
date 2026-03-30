"""CLI entry point for RL-based prompt optimization."""

from __future__ import annotations

import argparse
import json

from config import AppSettings, load_hyperparams
from optimizer import PromptOptimizer


REWARD_MODES = ("heuristics", "human", "both")
HUMAN_CHOICE_MODES = ("best", "ranking")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="RLHF-style prompt optimization for Gemini APIs"
    )
    parser.add_argument("--topic", required=True, help="Target topic to optimize for")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of responses sampled per prompt",
    )
    parser.add_argument(
        "--feedback-mode",
        choices=["none", "cli"],
        default="cli",
        help="Human feedback mode",
    )
    parser.add_argument(
        "--hyperparams",
        default="hyperparams.json",
        help="Path to hyperparameter JSON",
    )
    parser.add_argument(
        "--reward-mode",
        choices=list(REWARD_MODES),
        default=None,
        help="Reward mode: heuristics, human, or both",
    )
    return parser


def choose_reward_mode(provided_mode: str | None) -> str:
    """Ask at startup which reward mode to use unless already provided."""
    if provided_mode in REWARD_MODES:
        return provided_mode

    print("Choose reward mode: heuristics / human / both")
    raw = input("Reward mode [both]: ").strip().lower()
    if not raw:
        return "both"
    if raw not in REWARD_MODES:
        print("Invalid mode. Falling back to 'both'.")
        return "both"
    return raw


def choose_human_choice_mode(reward_mode: str) -> str | None:
    """Ask once for human choice type when human signal is involved."""
    if reward_mode not in {"human", "both"}:
        return None

    print("Choose human choice type: best / ranking")
    raw = input("Human choice [best]: ").strip().lower()
    if not raw:
        return "best"
    if raw not in HUMAN_CHOICE_MODES:
        print("Invalid choice. Falling back to 'best'.")
        return "best"
    return raw


def main() -> None:
    """Run the optimizer from command line."""
    args = build_parser().parse_args()
    reward_mode = choose_reward_mode(args.reward_mode)
    human_choice_mode = choose_human_choice_mode(reward_mode)
    if reward_mode == "human" and args.feedback_mode == "none":
        print("Human-only mode requires feedback collection. Switching --feedback-mode to cli.")
        args.feedback_mode = "cli"

    settings = AppSettings.from_env()
    hyperparams = load_hyperparams(args.hyperparams)

    optimizer = PromptOptimizer(settings=settings, hyperparams=hyperparams)
    result = optimizer.optimize(
        topic=args.topic,
        iterations=args.iterations,
        num_samples=args.samples,
        reward_mode=reward_mode,
        feedback_mode=args.feedback_mode,
        human_choice_mode=human_choice_mode,
    )

    print("\n=== Optimization Result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
