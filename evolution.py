"""Evolution strategy utilities for prompt parameter mutation."""

from __future__ import annotations

import random
from typing import Any, Dict


class EvolutionStrategy:
    """Mutation-based exploration over discrete prompt parameter choices."""

    def __init__(self, parameter_space: Dict[str, list[Any]], mutation_rate: float = 0.25) -> None:
        self.parameter_space = parameter_space
        self.mutation_rate = mutation_rate

    def mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a parameter set with per-key mutation probability."""
        mutated = dict(params)
        mutated_any = False

        for param_name, values in self.parameter_space.items():
            if random.random() < self.mutation_rate:
                alternatives = [v for v in values if v != mutated[param_name]]
                if alternatives:
                    mutated[param_name] = random.choice(alternatives)
                    mutated_any = True

        if not mutated_any:
            forced_key = random.choice(list(self.parameter_space.keys()))
            alternatives = [
                value
                for value in self.parameter_space[forced_key]
                if value != mutated[forced_key]
            ]
            if alternatives:
                mutated[forced_key] = random.choice(alternatives)

        return mutated

    @staticmethod
    def select_better(
        current_params: Dict[str, Any],
        current_reward: float,
        candidate_params: Dict[str, Any],
        candidate_reward: float,
    ) -> tuple[Dict[str, Any], float, str]:
        """Select and return the better of current vs candidate configurations."""
        if candidate_reward > current_reward:
            return candidate_params, candidate_reward, "mutated"
        return current_params, current_reward, "current"
