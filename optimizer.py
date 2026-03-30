"""Hybrid RL optimizer combining bandit sampling and evolution mutation."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bandit import ParameterBandit
from config import (
    AppSettings,
    HyperParams,
    SEARCH_PARAMETER_SPACE,
    setup_logging,
)
from evolution import EvolutionStrategy
from gemini_client import GeminiClient
from human_feedback import HumanFeedback, collect_human_feedback_cli
from memory import MemoryStore, build_memory_entry
from prompt_generator import generate_prompt
from reward import best_index_from_scores, compute_final_reward, heuristic_reward


@dataclass(slots=True)
class EvaluationResult:
    """Evaluation artifacts for one parameter set."""

    params: Dict[str, Any]
    prompt: str
    responses: List[str]
    heuristic_scores: List[float]
    final_scores: List[float]
    selected_response_index: int
    reward: float
    human_feedback: Optional[HumanFeedback]


class PromptOptimizer:
    """Production-style optimizer for RLHF-like prompt search over black-box APIs."""

    def __init__(self, settings: AppSettings, hyperparams: HyperParams) -> None:
        self.settings = settings
        self.hyperparams = hyperparams
        self.logger = setup_logging(settings.log_path)

        self.client = GeminiClient(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
            api_keys=settings.gemini_api_keys,
            start_key_index=0,
            max_retries=hyperparams.max_retries,
            retry_backoff_seconds=hyperparams.retry_backoff_seconds,
            enable_cache=True,
        )
        self.bandit = ParameterBandit(parameter_space=SEARCH_PARAMETER_SPACE, epsilon=hyperparams.epsilon)
        self.evolution = EvolutionStrategy(parameter_space=SEARCH_PARAMETER_SPACE, mutation_rate=hyperparams.mutation_rate)
        self.memory = MemoryStore(settings.memory_path)

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_reward: float = float("-inf")

    @staticmethod
    def _generation_config_from_params(params: Dict[str, Any], fallback_temperature: float) -> dict[str, float | int]:
        """Extract Gemini generation config from mixed parameter set."""
        return {
            "temperature": float(params.get("temperature", fallback_temperature)),
            "top_p": float(params.get("top_p", 1.0)),
            "presence_penalty": float(params.get("presence_penalty", 0.0)),
            "frequency_penalty": float(params.get("frequency_penalty", 0.0)),
            "max_tokens": int(params.get("max_tokens", 256)),
        }

    def _configure_client_for_iteration(self, iteration: int) -> int:
        """Select iteration key using modulo rotation and refresh Gemini client."""
        key_count = len(self.settings.gemini_api_keys)
        key_index = (iteration - 1) % key_count
        selected_api_key = self.settings.gemini_api_keys[key_index]

        self.client = GeminiClient(
            api_key=selected_api_key,
            model_name=self.settings.gemini_model,
            api_keys=self.settings.gemini_api_keys,
            start_key_index=key_index,
            max_retries=self.hyperparams.max_retries,
            retry_backoff_seconds=self.hyperparams.retry_backoff_seconds,
            enable_cache=True,
        )
        return key_index

    async def _evaluate_params_async(
        self,
        topic: str,
        params: Dict[str, Any],
        num_samples: int,
        reward_mode: str,
        feedback_mode: str = "none",
        human_choice_mode: str | None = None,
    ) -> EvaluationResult:
        """Generate N responses, score them, and combine with human feedback."""
        prompt = generate_prompt(params, topic)
        generation_cfg = self._generation_config_from_params(params, self.hyperparams.temperature)
        responses = await self.client.sample_responses_async(
            prompt=prompt,
            n=num_samples,
            temperature=float(generation_cfg["temperature"]),
            top_p=float(generation_cfg["top_p"]),
            presence_penalty=float(generation_cfg["presence_penalty"]),
            frequency_penalty=float(generation_cfg["frequency_penalty"]),
            max_tokens=int(generation_cfg["max_tokens"]),
        )

        heuristic_scores = [heuristic_reward(resp) for resp in responses]

        human_feedback: Optional[HumanFeedback] = None
        if reward_mode in {"human", "both"} and feedback_mode == "cli":
            choice_mode = (human_choice_mode or "best").strip().lower()
            human_feedback = collect_human_feedback_cli(responses, choice_mode=choice_mode)

        if reward_mode == "human" and human_feedback is None:
            raise ValueError(
                "Human-only mode requires human feedback. Use --feedback-mode cli or Streamlit UI."
            )

        human_scores = (
            human_feedback.to_scores(len(responses)) if human_feedback is not None else None
        )

        final_scores: List[float] = []
        for idx in range(len(responses)):
            human_score = None if human_scores is None else human_scores[idx]
            score = compute_final_reward(
                mode=reward_mode,
                heuristic_score=heuristic_scores[idx],
                human_score=human_score,
            )
            final_scores.append(score)

        if human_feedback is not None and human_feedback.primary_choice() is not None:
            selected_index = int(human_feedback.primary_choice())
            selected_index = max(0, min(len(responses) - 1, selected_index))
        else:
            selected_index = best_index_from_scores(final_scores)

        selected_reward = float(final_scores[selected_index])

        return EvaluationResult(
            params=params,
            prompt=prompt,
            responses=responses,
            heuristic_scores=heuristic_scores,
            final_scores=final_scores,
            selected_response_index=selected_index,
            reward=selected_reward,
            human_feedback=human_feedback,
        )

    def evaluate_params(
        self,
        topic: str,
        params: Dict[str, Any],
        num_samples: Optional[int] = None,
        reward_mode: str = "both",
        feedback_mode: str = "none",
        human_choice_mode: str | None = None,
    ) -> EvaluationResult:
        """Synchronous wrapper for async evaluation."""
        return asyncio.run(
            self._evaluate_params_async(
                topic=topic,
                params=params,
                num_samples=num_samples or self.hyperparams.num_samples,
                reward_mode=reward_mode,
                feedback_mode=feedback_mode,
                human_choice_mode=human_choice_mode,
            )
        )

    def _blend_with_memory(self, topic: str, sampled: Dict[str, Any]) -> Dict[str, Any]:
        """Occasionally inject strong historical settings from memory retrieval."""
        suggestion = self.memory.suggest_parameters(topic)
        if not suggestion:
            return sampled

        blended = dict(sampled)
        for key, value in suggestion.items():
            if key in blended and value in SEARCH_PARAMETER_SPACE.get(key, []) and random.random() < 0.35:
                blended[key] = value
        return blended

    def optimize(
        self,
        topic: str,
        iterations: Optional[int] = None,
        num_samples: Optional[int] = None,
        reward_mode: str = "both",
        feedback_mode: str = "none",
        human_choice_mode: str | None = None,
    ) -> Dict[str, object]:
        """Run hybrid optimization with bandit + evolution for a topic."""
        total_iterations = iterations or self.hyperparams.iterations
        samples = num_samples or self.hyperparams.num_samples

        for iteration in range(1, total_iterations + 1):
            key_index = self._configure_client_for_iteration(iteration)
            base_params = self._blend_with_memory(topic, self.bandit.sample_parameters())
            mutated_params = self.evolution.mutate(base_params)

            base_eval = self.evaluate_params(
                topic=topic,
                params=base_params,
                num_samples=samples,
                reward_mode=reward_mode,
                feedback_mode=feedback_mode,
                human_choice_mode=human_choice_mode,
            )
            mutant_eval = self.evaluate_params(
                topic=topic,
                params=mutated_params,
                num_samples=samples,
                reward_mode=reward_mode,
                feedback_mode=feedback_mode,
                human_choice_mode=human_choice_mode,
            )

            if mutant_eval.reward > base_eval.reward:
                winner = mutant_eval
                winner_source = "mutated"
            else:
                winner = base_eval
                winner_source = "current"

            self.bandit.update(winner.params, winner.reward)

            if winner.reward > self.best_reward:
                self.best_reward = winner.reward
                self.best_params = dict(winner.params)

            entry = build_memory_entry(
                topic=topic,
                params=winner.params,
                responses=winner.responses,
                reward=winner.reward,
                selected_response_index=winner.selected_response_index,
                human_choice=(
                    winner.human_feedback.to_dict()
                    if winner.human_feedback is not None
                    else None
                ),
                heuristic_scores=winner.heuristic_scores,
                final_scores=winner.final_scores,
            )
            self.memory.add(entry)

            self.logger.info(
                "iteration=%s key_slot=%s/%s winner=%s reward=%.3f selected_response=%s params=%s",
                iteration,
                key_index + 1,
                len(self.settings.gemini_api_keys),
                winner_source,
                winner.reward,
                winner.selected_response_index + 1,
                winner.params,
            )
            print(
                f"Iteration {iteration} | key_slot={key_index + 1}/{len(self.settings.gemini_api_keys)} | winner={winner_source} | "
                f"reward={winner.reward:.3f} | selected_response={winner.selected_response_index + 1}"
            )

        result = {
            "topic": topic,
            "best_params": self.best_params,
            "best_reward": self.best_reward,
            "bandit_snapshot": self.bandit.snapshot(),
        }
        self.logger.info("optimization_complete topic=%s best_reward=%.3f", topic, self.best_reward)
        return result
