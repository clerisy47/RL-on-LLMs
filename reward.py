"""Reward functions using heuristic and human feedback signals only."""

from __future__ import annotations

import re
from typing import List, Optional


def heuristic_reward(response: str) -> float:
    """Score response quality with structure, conciseness, and repetition checks."""
    text = response.strip()
    if not text:
        return 0.0

    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    # Conciseness: prefer informative but compact outputs.
    if word_count <= 25:
        conciseness = 5.0
    elif word_count <= 160:
        conciseness = 10.0
    elif word_count <= 280:
        conciseness = 7.0
    else:
        conciseness = 4.0

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_like = sum(
        1
        for line in lines
        if line.startswith(("-", "*"))
        or bool(re.match(r"^\d+[\.)]\s+", line))
    )
    structure = 10.0 if bullet_like >= 2 else 6.0 if len(lines) >= 2 else 4.0

    lowered = [w.lower() for w in words]
    unique_ratio = len(set(lowered)) / max(len(lowered), 1)
    repetition = 10.0 * unique_ratio

    score = (0.4 * conciseness) + (0.3 * structure) + (0.3 * repetition)
    return float(max(0.0, min(10.0, score)))


def compute_final_reward(
    mode: str,
    heuristic_score: float,
    human_score: Optional[float] = None,
) -> float:
    """Combine available reward signals according to selected mode."""
    if mode == "heuristics":
        return float(heuristic_score)

    if mode == "human":
        if human_score is None:
            raise ValueError("Human-only mode requires human feedback.")
        return float(human_score)

    if mode == "both":
        if human_score is None:
            return float(heuristic_score)
        return float((0.5 * heuristic_score) + (0.5 * human_score))

    raise ValueError(f"Unsupported reward mode: {mode}")


def best_index_from_scores(scores: List[float]) -> int:
    """Return index of maximum score, breaking ties by earliest index."""
    return max(range(len(scores)), key=lambda idx: scores[idx])
