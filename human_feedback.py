"""Human-in-the-loop feedback collection for CLI and Streamlit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple


@dataclass(slots=True)
class HumanFeedback:
    """Container for different feedback signal types."""

    mode: str
    best_index: Optional[int] = None
    ranking: Optional[List[int]] = None
    pairwise_preferences: Optional[List[Tuple[int, int]]] = None

    def to_dict(self) -> dict:
        """Convert feedback to serializable dictionary."""
        return asdict(self)

    def to_scores(self, num_responses: int) -> List[float]:
        """Convert feedback into per-response 0-10 scores."""
        if self.mode == "best" and self.best_index is not None:
            scores = [2.0] * num_responses
            if 0 <= self.best_index < num_responses:
                scores[self.best_index] = 10.0
            return scores

        if self.mode == "ranking" and self.ranking:
            scores = [1.0] * num_responses
            total = len(self.ranking)
            for rank_position, idx in enumerate(self.ranking):
                if 0 <= idx < num_responses:
                    rank_value = total - rank_position
                    scores[idx] = 10.0 * (rank_value / total)
            return scores

        if self.mode == "pairwise" and self.pairwise_preferences:
            wins = [0.0] * num_responses
            counts = [0.0] * num_responses
            for winner, loser in self.pairwise_preferences:
                if 0 <= winner < num_responses and 0 <= loser < num_responses:
                    wins[winner] += 1.0
                    counts[winner] += 1.0
                    counts[loser] += 1.0
            scores = []
            for i in range(num_responses):
                if counts[i] == 0:
                    scores.append(5.0)
                else:
                    scores.append(10.0 * (wins[i] / counts[i]))
            return scores

        return [5.0] * num_responses

    def primary_choice(self) -> Optional[int]:
        """Return human-prioritized selected response index when available."""
        if self.mode == "best":
            return self.best_index
        if self.mode == "ranking" and self.ranking:
            return self.ranking[0]
        if self.mode == "pairwise" and self.pairwise_preferences:
            tally: dict[int, int] = {}
            for winner, _ in self.pairwise_preferences:
                tally[winner] = tally.get(winner, 0) + 1
            if not tally:
                return None
            return max(tally, key=tally.get)
        return None


def collect_human_feedback_cli(
    responses: List[str],
    choice_mode: str,
) -> Optional[HumanFeedback]:
    """Collect human preference via terminal input using a fixed choice mode."""
    print("\n=== Human Feedback ===")
    for idx, response in enumerate(responses):
        print(f"\nResponse {idx + 1}:\n{'-' * 60}\n{response}\n")

    mode = choice_mode.strip().lower()

    if mode == "best":
        raw = input(f"Select best response index (1-{len(responses)}): ").strip()
        if not raw.isdigit():
            return None
        chosen = int(raw) - 1
        return HumanFeedback(mode="best", best_index=chosen)

    if mode == "ranking":
        raw = input(
            "Enter ranking as space-separated indices (1-based), e.g. '2 1 3': "
        ).strip()
        try:
            ranking = [int(x) - 1 for x in raw.split()]
        except ValueError:
            return None
        if len(ranking) != len(responses):
            return None
        return HumanFeedback(mode="ranking", ranking=ranking)

    return None


def render_streamlit_feedback(
    topic: str,
    responses: List[str],
    key_prefix: str = "feedback",
    forced_mode: str | None = None,
):
    """Render streamlit widgets and return feedback on submit; None otherwise."""
    import streamlit as st

    st.subheader(f"Human Feedback for Topic: {topic}")
    cols = st.columns(len(responses))
    for idx, col in enumerate(cols):
        with col:
            st.markdown(f"**Response {idx + 1}**")
            st.text_area(
                label=f"r_{idx}",
                value=responses[idx],
                key=f"{key_prefix}_resp_{idx}",
                height=250,
                disabled=True,
                label_visibility="collapsed",
            )

    mode = (forced_mode or "best").strip().lower()

    if forced_mode is None:
        mode = st.radio(
            "Feedback Mode",
            options=["skip", "best", "ranking", "pairwise"],
            key=f"{key_prefix}_mode",
            horizontal=True,
        )

    if mode == "best":
        choice = st.selectbox(
            "Best response",
            options=list(range(1, len(responses) + 1)),
            key=f"{key_prefix}_best",
        )
        if st.button("Submit Feedback", key=f"{key_prefix}_submit"):
            return HumanFeedback(mode="best", best_index=choice - 1)

    if mode == "ranking":
        raw = st.text_input(
            "Ranking (space-separated, 1-based), e.g. '2 1 3'",
            key=f"{key_prefix}_ranking",
        )
        if st.button("Submit Feedback", key=f"{key_prefix}_submit"):
            try:
                ranking = [int(x) - 1 for x in raw.split()]
            except ValueError:
                st.error("Invalid ranking format.")
                return None
            if len(ranking) != len(responses):
                st.error("Ranking must include all responses exactly once.")
                return None
            return HumanFeedback(mode="ranking", ranking=ranking)

    if mode == "pairwise":
        raw = st.text_input(
            "Pairwise preferences, e.g. '1>2,2>3'",
            key=f"{key_prefix}_pairwise",
        )
        if st.button("Submit Feedback", key=f"{key_prefix}_submit"):
            pairs: List[Tuple[int, int]] = []
            for token in raw.split(","):
                token = token.strip()
                if ">" not in token:
                    continue
                left, right = token.split(">", maxsplit=1)
                if not left.strip().isdigit() or not right.strip().isdigit():
                    continue
                pairs.append((int(left.strip()) - 1, int(right.strip()) - 1))
            if not pairs:
                st.error("Could not parse pairwise preferences.")
                return None
            return HumanFeedback(mode="pairwise", pairwise_preferences=pairs)

    if mode == "skip" and st.button("Skip Feedback", key=f"{key_prefix}_skip"):
        return None

    return None
