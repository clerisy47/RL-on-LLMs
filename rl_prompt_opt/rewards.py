"""Reward computation primitives."""

from __future__ import annotations

import re


def heuristic_reward(response: str) -> float:
    """Score response quality using light text heuristics in range [0, 10]."""
    text = response.strip()
    if not text:
        return 0.0

    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    if word_count <= 30:
        conciseness = 6.0
    elif word_count <= 220:
        conciseness = 10.0
    elif word_count <= 420:
        conciseness = 7.0
    else:
        conciseness = 4.0

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_like = sum(
        1
        for line in lines
        if line.startswith(("-", "*")) or bool(re.match(r"^\d+[\.)]\s+", line))
    )
    structure = 10.0 if bullet_like >= 2 else 6.0 if len(lines) >= 2 else 4.0

    lowered = [word.lower() for word in words]
    unique_ratio = len(set(lowered)) / max(1, len(lowered))
    repetition = 10.0 * unique_ratio

    score = (0.4 * conciseness) + (0.3 * structure) + (0.3 * repetition)
    return float(max(0.0, min(10.0, score)))


def argmax(values: list[float]) -> int:
    """Index of max value (stable on ties)."""
    return max(range(len(values)), key=lambda idx: values[idx])
