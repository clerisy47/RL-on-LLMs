from __future__ import annotations

def collect_generation_ranking_cli(items: list[tuple[str, str]]) -> list[float]:
    print("\n=== Generation Ranking (One Choice) ===")
    for idx, (label, response_text) in enumerate(items):
        print(f"\nCandidate {idx + 1} | {label}\n{'-' * 72}\n{response_text}\n")

    raw = input(
        "Enter ranking over all candidates as space-separated indices (1-based), e.g. '2 1 4 3': "
    ).strip()
    ranking = [int(token) - 1 for token in raw.split()]
    if sorted(ranking) != list(range(len(items))):
        raise ValueError("Ranking must contain each candidate index exactly once.")

    total = len(items)
    scores = [1.0] * total
    for rank_position, idx in enumerate(ranking):
        rank_value = total - rank_position
        scores[idx] = 10.0 * (rank_value / total)
    return scores
