from __future__ import annotations

import re
from collections import Counter


def heuristic_reward(response: str) -> float:
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
    diversity = 10.0 * unique_ratio

    counts = Counter(lowered)
    repeated_tokens = sum(max(0, count - 1) for count in counts.values())
    repetition_ratio = repeated_tokens / max(1, len(lowered))
    non_repetition = 10.0 * (1.0 - repetition_ratio)

    score = (
        (0.35 * conciseness)
        + (0.25 * structure)
        + (0.20 * diversity)
        + (0.20 * non_repetition)
    )
    return float(max(0.0, min(10.0, score)))


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])
