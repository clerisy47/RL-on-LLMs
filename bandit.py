"""Epsilon-greedy bandit for prompt parameter value selection."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Hashable


@dataclass(slots=True)
class RunningStat:
    """Online running average for a parameter value."""

    count: int = 0
    avg_reward: float = 5.0

    def update(self, reward: float) -> None:
        """Update average reward using incremental mean."""
        self.count += 1
        self.avg_reward += (reward - self.avg_reward) / self.count


class ParameterBandit:
    """Track and sample prompt parameter values with epsilon-greedy policy."""

    def __init__(
        self,
        parameter_space: Dict[str, list[Any]],
        epsilon: float = 0.25,
        optimistic_prior: float = 5.0,
    ) -> None:
        self.parameter_space = parameter_space
        self.epsilon = epsilon
        self.stats: Dict[str, Dict[Hashable, RunningStat]] = {}

        for param, values in parameter_space.items():
            self.stats[param] = {
                value: RunningStat(count=0, avg_reward=optimistic_prior) for value in values
            }

    def sample_parameters(self) -> Dict[str, str]:
        """Sample one value per parameter using epsilon-greedy strategy."""
        sampled: Dict[str, Any] = {}
        for param_name, values in self.parameter_space.items():
            explore = random.random() < self.epsilon
            if explore:
                sampled[param_name] = random.choice(values)
                continue

            best_reward = max(self.stats[param_name][value].avg_reward for value in values)
            best_values = [
                value
                for value in values
                if self.stats[param_name][value].avg_reward == best_reward
            ]
            sampled[param_name] = random.choice(best_values)

        return sampled

    def update(self, params: Dict[str, str], reward: float) -> None:
        """Update value-level bandit stats for one sampled parameter set."""
        for param_name, chosen_value in params.items():
            self.stats[param_name][chosen_value].update(reward)

    def snapshot(self) -> Dict[str, Dict[str, dict[str, float | int]]]:
        """Return serializable state for diagnostics and logs."""
        return {
            param_name: {
                str(value): {
                    "count": stat.count,
                    "avg_reward": round(stat.avg_reward, 4),
                }
                for value, stat in values.items()
            }
            for param_name, values in self.stats.items()
        }
