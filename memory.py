"""Persistent memory store and retrieval utilities for optimization history."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class MemoryEntry:
    """One optimization record stored in JSON memory."""

    topic: str
    params: Dict[str, Any]
    responses: List[str]
    reward: float
    selected_response_index: int
    human_choice: Optional[dict[str, Any]]
    heuristic_scores: List[float]
    final_scores: List[float]
    created_at: str


class MemoryStore:
    """JSON-backed storage for optimization trajectories and retrieval."""

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]\n", encoding="utf-8")

    def _load(self) -> List[dict[str, Any]]:
        raw = self.file_path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def _save(self, data: List[dict[str, Any]]) -> None:
        self.file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add(self, entry: MemoryEntry) -> None:
        """Append a memory entry to disk."""
        data = self._load()
        data.append(asdict(entry))
        self._save(data)

    def all(self) -> List[dict[str, Any]]:
        """Return all memory entries as dictionaries."""
        return self._load()

    def similar_topics(self, topic: str, top_k: int = 5) -> List[dict[str, Any]]:
        """Retrieve top-k entries by fuzzy topic similarity and reward."""
        entries = self._load()
        scored: List[tuple[float, dict[str, Any]]] = []

        for entry in entries:
            source_topic = str(entry.get("topic", ""))
            similarity = SequenceMatcher(None, topic.lower(), source_topic.lower()).ratio()
            reward = float(entry.get("reward", 0.0))
            combined_score = (0.7 * similarity) + (0.3 * (reward / 10.0))
            scored.append((combined_score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def suggest_parameters(self, topic: str) -> Optional[Dict[str, Any]]:
        """Suggest likely-good parameters based on similar historical topics."""
        neighbors = self.similar_topics(topic, top_k=6)
        if not neighbors:
            return None

        best = max(neighbors, key=lambda x: float(x.get("reward", 0.0)))
        params = best.get("params")
        if not isinstance(params, dict):
            return None

        return {str(k): v for k, v in params.items()}


def build_memory_entry(
    topic: str,
    params: Dict[str, Any],
    responses: List[str],
    reward: float,
    selected_response_index: int,
    human_choice: Optional[dict[str, Any]],
    heuristic_scores: List[float],
    final_scores: List[float],
) -> MemoryEntry:
    """Create a memory entry with current UTC timestamp."""
    return MemoryEntry(
        topic=topic,
        params=params,
        responses=responses,
        reward=reward,
        selected_response_index=selected_response_index,
        human_choice=human_choice,
        heuristic_scores=heuristic_scores,
        final_scores=final_scores,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
