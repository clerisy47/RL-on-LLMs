"""Prompt synthesis from parameter combinations."""

from __future__ import annotations

from typing import Any


def build_prompt(topic: str, params: dict[str, Any]) -> str:
    """Construct a concrete prompt from prompt-parameter assignments."""
    role = params.get("role", "teacher")
    audience = params.get("audience", "student")
    style = params.get("style", "conversational")
    tone = params.get("tone", "friendly")
    depth = params.get("depth", "moderate")
    length = params.get("length", "short paragraph")
    structure = params.get("structure", "paragraph")
    examples = params.get("examples", "one")
    analogies = params.get("analogies", "none")
    step_by_step = params.get("step_by_step", "yes")
    clarity = params.get("clarity", "define terms")
    constraints = params.get("constraints", "none")
    out_format = params.get("format", "plain text")

    return (
        f"You are a {role}. Explain '{topic}' for a {audience}.\n"
        f"Style: {style}. Tone: {tone}. Depth: {depth}. Length: {length}.\n"
        f"Structure: {structure}. Examples: {examples}. Analogies: {analogies}.\n"
        f"Step-by-step: {step_by_step}. Clarity: {clarity}. Constraints: {constraints}.\n"
        f"Output format: {out_format}."
    )
