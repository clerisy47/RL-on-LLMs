"""Prompt construction logic for parameterized prompting."""

from __future__ import annotations

from typing import Any, Mapping


def generate_prompt(params: Mapping[str, Any], topic: str) -> str:
    """Generate a structured prompt from a parameter dictionary and topic."""
    return f"""
You are acting as a {params['role']}.

Task topic: {topic}

Audience:
- The audience is a {params['audience']}.

Response style requirements:
- Style: {params['style']}
- Tone: {params['tone']}
- Depth: {params['depth']}
- Target length: {params['length']}
- Structure: {params['structure']}
- Include examples: {params['examples']}
- Use analogies: {params['analogies']}
- Step-by-step explanation: {params['step_by_step']}
- Clarity requirement: {params['clarity']}
- Constraints: {params['constraints']}
- Output format: {params['format']}

Quality bar:
- Be factually correct and useful.
- Prioritize clarity and practical understanding.
- Avoid unnecessary filler.
""".strip()
