from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


@dataclass(slots=True)
class Candidate:
    prompt_params: dict[str, Any]
    model_params: dict[str, Any]
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        self.signature = candidate_signature(self.prompt_params, self.model_params)


@dataclass(slots=True)
class CandidateEvaluation:
    candidate: Candidate
    prompt: str
    responses: list[str]
    selected_response_index: int
    selected_response: str
    heuristic_scores: list[float]
    llm_judge_scores: list[float] | None
    human_scores: list[float] | None
    final_scores: list[float]
    reward: float


def candidate_signature(prompt_params: dict[str, Any], model_params: dict[str, Any]) -> str:
    payload = {
        "prompt_params": prompt_params,
        "model_params": model_params,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()
