"""Prompt and model parameter search spaces."""

from __future__ import annotations

from typing import Any

PROMPT_PARAMETER_SPACE: dict[str, list[str]] = {
    "role": ["teacher", "engineer", "researcher", "friendly tutor"],
    "audience": ["beginner", "student", "expert"],
    "style": ["intuitive", "technical", "conversational"],
    "tone": ["formal", "friendly", "concise"],
    "depth": ["shallow", "moderate", "deep"],
    "length": ["1 sentence", "short paragraph", "detailed"],
    "structure": ["paragraph", "bullet points", "numbered steps"],
    "examples": ["none", "one", "multiple"],
    "analogies": ["none", "simple", "real-world"],
    "step_by_step": ["yes", "no"],
    "clarity": ["avoid jargon", "define terms"],
    "constraints": ["none", "avoid long sentences", "avoid repetition"],
    "format": ["plain text", "markdown"],
}

MODEL_PARAMETER_SPACE: dict[str, list[float]] = {
    "temperature": [0.2, 0.5, 0.7, 1.0],
    "top_p": [0.7, 0.9, 1.0],
}

SEARCH_PARAMETER_SPACE: dict[str, list[Any]] = {
    **PROMPT_PARAMETER_SPACE,
    **MODEL_PARAMETER_SPACE,
}
