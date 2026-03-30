"""CLI entry point for evolutionary population-based prompt optimization."""

from __future__ import annotations

import argparse
import json

from rl_prompt_opt import AppSettings, EvolutionaryPromptOptimizer, load_hyperparams


REWARD_MODES = ("heuristics", "human", "llm_judge")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Evolutionary population-based prompt optimization for Gemini APIs"
    )
    parser.add_argument("--topic", required=True, help="Target topic to optimize for")
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of evolutionary generations",
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
        help="Reward mode: heuristics, human, or llm_judge",
    )
    return parser


def choose_reward_mode(provided_mode: str | None) -> str:
    """Ask at startup which reward mode to use unless already provided."""
    if provided_mode in REWARD_MODES:
        return provided_mode

    print("Choose reward mode: heuristics / human / llm_judge")
    raw = input("Reward mode: ").strip().lower()
    if not raw:
        return "heuristics"
    if raw not in REWARD_MODES:
        print("Invalid mode. Falling back to 'heuristics'.")
        return "heuristics"
    return raw


def main() -> None:
    """Run the optimizer from command line."""
    args = build_parser().parse_args()
    reward_mode = choose_reward_mode(args.reward_mode)

    settings = AppSettings.from_env()
    hyperparams = load_hyperparams(args.hyperparams)

    optimizer = EvolutionaryPromptOptimizer(settings=settings, hyperparams=hyperparams)
    result = optimizer.optimize(
        topic=args.topic,
        reward_mode=reward_mode,
        generations=args.generations,
    )

    print("\n=== Optimization Result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
