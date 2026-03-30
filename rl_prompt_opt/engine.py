from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from .candidate import Candidate, CandidateEvaluation
from .feedback import collect_generation_ranking_cli
from .gemini import GeminiClient
from .prompting import build_prompt
from .rewards import heuristic_reward
from .search_space import MODEL_PARAMETER_SPACE, PROMPT_PARAMETER_SPACE
from .settings import AppSettings, HyperParams, setup_logger


class EvolutionaryPromptOptimizer:
    def __init__(self, settings: AppSettings, hyperparams: HyperParams) -> None:
        self.settings = settings
        self.hyperparams = hyperparams
        self.logger = setup_logger(settings.log_path)
        self.client = GeminiClient(
            api_keys=settings.gemini_api_keys,
            model_name=settings.gemini_model,
            max_retries=hyperparams.max_retries,
            retry_backoff_seconds=hyperparams.retry_backoff_seconds,
            enable_cache=True,
        )
        self.rng = random.Random()

        self.population: list[Candidate] = []
        self.eval_cache: dict[str, CandidateEvaluation] = {}
        self.best_evaluation: Optional[CandidateEvaluation] = None

        self._load_memory()

    def _load_memory(self) -> None:
        path = Path(self.settings.memory_path)
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        entries = raw if isinstance(raw, list) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            prompt_params = entry.get("prompt_params")
            model_params = entry.get("model_params")
            if not isinstance(prompt_params, dict) or not isinstance(model_params, dict):
                continue
            candidate = Candidate(prompt_params=prompt_params, model_params=model_params)
            responses = entry.get("responses") or []
            if not responses:
                continue
            selected = int(entry.get("selected_response_index", 0))
            selected = max(0, min(len(responses) - 1, selected))
            reward = float(entry.get("reward", 0.0))
            heuristic_scores = entry.get("heuristic_scores") or [0.0] * len(responses)
            final_scores = entry.get("final_scores") or [reward] * len(responses)

            evaluation = CandidateEvaluation(
                candidate=candidate,
                prompt=str(entry.get("prompt", "")),
                responses=[str(x) for x in responses],
                selected_response_index=selected,
                selected_response=str(responses[selected]),
                heuristic_scores=[float(x) for x in heuristic_scores],
                llm_judge_scores=None,
                human_scores=None,
                final_scores=[float(x) for x in final_scores],
                reward=reward,
            )
            self.eval_cache[candidate.signature] = evaluation
            if self.best_evaluation is None or reward > self.best_evaluation.reward:
                self.best_evaluation = evaluation

    def _persist_memory(self) -> None:
        path = Path(self.settings.memory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, Any]] = []
        for evaluation in self.eval_cache.values():
            payload.append(
                {
                    "prompt_params": evaluation.candidate.prompt_params,
                    "model_params": evaluation.candidate.model_params,
                    "prompt": evaluation.prompt,
                    "responses": evaluation.responses,
                    "selected_response_index": evaluation.selected_response_index,
                    "reward": evaluation.reward,
                    "heuristic_scores": evaluation.heuristic_scores,
                    "final_scores": evaluation.final_scores,
                }
            )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _random_candidate(self) -> Candidate:
        prompt_params = {
            key: self.rng.choice(values)
            for key, values in PROMPT_PARAMETER_SPACE.items()
        }
        model_params = {
            key: self.rng.choice(values)
            for key, values in MODEL_PARAMETER_SPACE.items()
        }
        return Candidate(prompt_params=prompt_params, model_params=model_params)

    def _initialize_population(self) -> None:
        seen = set()
        while len(self.population) < self.hyperparams.population_size:
            candidate = self._random_candidate()
            if candidate.signature in seen:
                continue
            seen.add(candidate.signature)
            self.population.append(candidate)

    def _crossover(self, parent_a: Candidate, parent_b: Candidate) -> Candidate:
        prompt_params: dict[str, Any] = {}
        model_params: dict[str, Any] = {}

        for key in PROMPT_PARAMETER_SPACE:
            source = parent_a if self.rng.random() < 0.5 else parent_b
            prompt_params[key] = source.prompt_params[key]

        for key in MODEL_PARAMETER_SPACE:
            source = parent_a if self.rng.random() < 0.5 else parent_b
            model_params[key] = source.model_params[key]

        return Candidate(prompt_params=prompt_params, model_params=model_params)

    def _mutate(self, candidate: Candidate) -> Candidate:
        prompt_params = dict(candidate.prompt_params)
        model_params = dict(candidate.model_params)

        for key, values in PROMPT_PARAMETER_SPACE.items():
            if self.rng.random() < self.hyperparams.mutation_rate:
                alternatives = [value for value in values if value != prompt_params[key]]
                if alternatives:
                    prompt_params[key] = self.rng.choice(alternatives)

        for key, values in MODEL_PARAMETER_SPACE.items():
            if self.rng.random() < self.hyperparams.mutation_rate:
                alternatives = [value for value in values if value != model_params[key]]
                if alternatives:
                    model_params[key] = self.rng.choice(alternatives)

        return Candidate(prompt_params=prompt_params, model_params=model_params)

    def _evaluate_candidate(
        self,
        topic: str,
        candidate: Candidate,
        reward_mode: str,
        llm_judge_criteria: str | None = None,
    ) -> CandidateEvaluation:
        if candidate.signature in self.eval_cache:
            return self.eval_cache[candidate.signature]

        if "temperature" not in candidate.model_params or "top_p" not in candidate.model_params:
            raise ValueError(
                "Candidate model_params must include explicit 'temperature' and 'top_p'."
            )

        temperature = float(candidate.model_params["temperature"])
        top_p = float(candidate.model_params["top_p"])

        prompt = build_prompt(topic, candidate.prompt_params)
        response = self.client.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )

        heuristic_value = heuristic_reward(response)
        heuristic_scores = [heuristic_value]

        llm_scores: Optional[list[float]] = None
        llm_value: Optional[float] = None
        if reward_mode == "llm_judge":
            llm_value = self.client.score_with_llm_judge(topic, response, llm_judge_criteria)
            llm_scores = [llm_value]

        if reward_mode == "heuristics":
            reward = heuristic_value
        elif reward_mode == "llm_judge":
            if llm_value is None:
                raise ValueError("LLM judge score missing.")
            reward = llm_value
        else:
            reward = 0.0

        responses = [response]
        selected_index = 0
        final_scores = [reward]

        evaluation = CandidateEvaluation(
            candidate=candidate,
            prompt=prompt,
            responses=responses,
            selected_response_index=selected_index,
            selected_response=response,
            heuristic_scores=heuristic_scores,
            llm_judge_scores=llm_scores,
            human_scores=None,
            final_scores=final_scores,
            reward=float(reward),
        )

        self.eval_cache[candidate.signature] = evaluation
        if self.best_evaluation is None or evaluation.reward > self.best_evaluation.reward:
            self.best_evaluation = evaluation
        return evaluation

    def _evaluate_population(
        self,
        topic: str,
        reward_mode: str,
        llm_judge_criteria: str | None = None,
    ) -> list[CandidateEvaluation]:
        evaluations: list[CandidateEvaluation] = []
        for candidate in self.population:
            evaluations.append(
                self._evaluate_candidate(
                    topic=topic,
                    candidate=candidate,
                    reward_mode=reward_mode,
                    llm_judge_criteria=llm_judge_criteria,
                )
            )
        return evaluations

    def _apply_human_generation_ranking(self, evaluations: list[CandidateEvaluation]) -> None:
        items: list[tuple[str, str]] = []
        for idx, evaluation in enumerate(evaluations):
            params = evaluation.candidate.model_params
            label = (
                f"signature={evaluation.candidate.signature[:8]} "
                f"temp={params.get('temperature')} top_p={params.get('top_p')}"
            )
            items.append((label, evaluation.selected_response))

        scores = collect_generation_ranking_cli(items)
        self.apply_human_scores(evaluations, scores)

    def apply_human_scores(self, evaluations: list[CandidateEvaluation], scores: list[float]) -> None:
        if len(scores) != len(evaluations):
            raise ValueError("Human scores length must match number of generation candidates.")

        for idx, evaluation in enumerate(evaluations):
            score = float(scores[idx])
            evaluation.human_scores = [score]
            evaluation.final_scores = [score]
            evaluation.reward = score
            self.eval_cache[evaluation.candidate.signature] = evaluation
            if self.best_evaluation is None or evaluation.reward > self.best_evaluation.reward:
                self.best_evaluation = evaluation

    def ensure_population_initialized(self) -> None:
        if not self.population:
            self._initialize_population()

    def evaluate_generation(
        self,
        topic: str,
        reward_mode: str,
        llm_judge_criteria: str | None = None,
    ) -> list[CandidateEvaluation]:
        self.ensure_population_initialized()
        return self._evaluate_population(
            topic=topic,
            reward_mode=reward_mode,
            llm_judge_criteria=llm_judge_criteria,
        )

    def complete_generation(
        self,
        generation_index: int,
        evaluations: list[CandidateEvaluation],
    ) -> CandidateEvaluation:
        ranked = sorted(evaluations, key=lambda item: item.reward, reverse=True)
        best = ranked[0]

        self.logger.info(
            "generation=%s best_reward=%.3f best_signature=%s",
            generation_index,
            best.reward,
            best.candidate.signature[:10],
        )
        print(
            f"Generation {generation_index} | best_reward={best.reward:.3f} | "
            f"best_signature={best.candidate.signature[:10]}"
        )

        self.population = self._select_and_reproduce(evaluations)
        return best

    def build_result(self, topic: str, reward_mode: str, generations: int) -> dict[str, Any]:
        self._persist_memory()

        if self.best_evaluation is None:
            raise RuntimeError("Optimization produced no evaluations.")

        return {
            "topic": topic,
            "reward_mode": reward_mode,
            "generations": generations,
            "population_size": self.hyperparams.population_size,
            "best_reward": self.best_evaluation.reward,
            "best_prompt_params": self.best_evaluation.candidate.prompt_params,
            "best_model_params": self.best_evaluation.candidate.model_params,
            "best_response": self.best_evaluation.selected_response,
            "cache_size": len(self.eval_cache),
        }

    def _select_and_reproduce(self, evaluations: list[CandidateEvaluation]) -> list[Candidate]:
        ranked = sorted(evaluations, key=lambda item: item.reward, reverse=True)

        parent_count = max(2, int(self.hyperparams.population_size * self.hyperparams.parent_fraction))
        parent_count = min(parent_count, len(ranked))
        parents = ranked[:parent_count]

        eliminate_count = int(self.hyperparams.population_size * self.hyperparams.elimination_fraction)
        eliminate_count = max(1, min(eliminate_count, self.hyperparams.population_size - parent_count))

        survivors = ranked[:-eliminate_count]
        next_generation: list[Candidate] = [item.candidate for item in survivors]

        seen = {candidate.signature for candidate in next_generation}
        parent_candidates = [item.candidate for item in parents]
        survivor_candidates = [item.candidate for item in survivors]

        crossover_count = int(self.hyperparams.population_size * self.hyperparams.crossover_fraction)
        mutation_count = int(self.hyperparams.population_size * self.hyperparams.mutation_fraction)

        replacement_slots = self.hyperparams.population_size - len(next_generation)
        crossover_count = min(crossover_count, replacement_slots)
        mutation_count = min(mutation_count, replacement_slots - crossover_count)

        replacements: list[Candidate] = []

        while len(replacements) < crossover_count:
            parent_a = self.rng.choice(parent_candidates)
            parent_b = self.rng.choice(parent_candidates)
            child = self._crossover(parent_a, parent_b)
            if child.signature in seen:
                continue
            seen.add(child.signature)
            replacements.append(child)

        while len(replacements) < crossover_count + mutation_count:
            base = self.rng.choice(survivor_candidates)
            mutant = self._mutate(base)
            if mutant.signature in seen:
                continue
            seen.add(mutant.signature)
            replacements.append(mutant)

        while len(replacements) < replacement_slots:
            parent_a = self.rng.choice(parent_candidates)
            parent_b = self.rng.choice(parent_candidates)
            child = self._mutate(self._crossover(parent_a, parent_b))
            if child.signature in seen:
                continue
            seen.add(child.signature)
            replacements.append(child)

        next_generation.extend(replacements)

        return next_generation

    def optimize(
        self,
        topic: str,
        reward_mode: str,
        generations: Optional[int] = None,
        llm_judge_criteria: str | None = None,
    ) -> dict[str, Any]:
        total_generations = generations or self.hyperparams.generations
        self.ensure_population_initialized()

        for generation in range(1, total_generations + 1):
            evaluations = self.evaluate_generation(
                topic=topic,
                reward_mode=reward_mode,
                llm_judge_criteria=llm_judge_criteria,
            )
            if reward_mode == "human":
                self._apply_human_generation_ranking(evaluations)
            self.complete_generation(generation_index=generation, evaluations=evaluations)

        return self.build_result(topic=topic, reward_mode=reward_mode, generations=total_generations)
