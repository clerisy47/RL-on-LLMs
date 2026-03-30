"""Streamlit UI for evolutionary population-based prompt optimization."""

from __future__ import annotations

import streamlit as st

from rl_prompt_opt import AppSettings, EvolutionaryPromptOptimizer, load_hyperparams


def _show_generation_error(exc: Exception) -> None:
    """Render user-friendly error guidance for generation failures."""
    message = str(exc)
    st.error(message)
    if "RESOURCE_EXHAUSTED" in message or "quota" in message.lower() or "429" in message:
        st.info(
            "All available keys may be out of quota (often shared per project on free tier). "
            "Try fewer iterations/samples, a different model/project, or wait for quota reset."
        )


def _reset_human_session() -> None:
    """Clear Streamlit human-session state keys."""
    for key in list(st.session_state.keys()):
        if key.startswith("human_"):
            del st.session_state[key]


def _start_human_session(topic: str, generations: int, population_size: int) -> None:
    """Initialize optimizer and first generation for in-website human ranking."""
    settings = AppSettings.from_env()
    hyperparams = load_hyperparams("hyperparams.json")
    hyperparams.generations = int(generations)
    hyperparams.population_size = int(population_size)

    optimizer = EvolutionaryPromptOptimizer(settings=settings, hyperparams=hyperparams)
    evaluations = optimizer.evaluate_generation(topic=topic, reward_mode="human")

    st.session_state.human_active = True
    st.session_state.human_topic = topic
    st.session_state.human_total_generations = int(generations)
    st.session_state.human_current_generation = 1
    st.session_state.human_optimizer = optimizer
    st.session_state.human_evaluations = evaluations
    st.session_state.human_result = None


def _parse_generation_ranking(raw: str, candidate_count: int) -> list[float]:
    """Parse ranking input like '2 1 4 3' into per-candidate scores."""
    ranking = [int(token) - 1 for token in raw.split()]
    if sorted(ranking) != list(range(candidate_count)):
        raise ValueError("Ranking must contain each candidate index exactly once.")

    total = candidate_count
    scores = [1.0] * total
    for rank_position, idx in enumerate(ranking):
        rank_value = total - rank_position
        scores[idx] = 10.0 * (rank_value / total)
    return scores


def main() -> None:
    """Render and run evolutionary population optimization in Streamlit."""
    st.set_page_config(page_title="RL Prompt Optimizer", layout="wide")
    st.title("Evolutionary Prompt Optimizer")
    st.caption("Population -> selection -> crossover -> mutation -> evaluate only new candidates")

    topic = st.text_input("Topic", value="Explain recursion")
    col1, col2 = st.columns(2)
    with col1:
        generations = st.number_input("Generations", min_value=1, max_value=200, value=5, step=1)
    with col2:
        population_size = st.number_input("Population Size", min_value=4, max_value=64, value=8, step=1)

    reward_mode = st.radio(
        "Reward Mode",
        options=["heuristics", "llm_judge", "human"],
        index=0,
        horizontal=True,
        help="Human mode uses one ranking input per generation across all candidates.",
    )

    if reward_mode != "human" and st.button("Run Optimization"):
        try:
            settings = AppSettings.from_env()
            hyperparams = load_hyperparams("hyperparams.json")
            hyperparams.generations = int(generations)
            hyperparams.population_size = int(population_size)

            optimizer = EvolutionaryPromptOptimizer(settings=settings, hyperparams=hyperparams)
            result = optimizer.optimize(
                topic=topic,
                reward_mode=reward_mode,
                generations=int(generations),
            )

            st.success("Optimization complete.")
            st.json(result)
        except Exception as exc:
            _show_generation_error(exc)

    if reward_mode == "human":
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            start_human = st.button("Start Human Session")
        with action_col2:
            reset_human = st.button("Reset Human Session")

        if reset_human:
            _reset_human_session()
            st.success("Human session reset.")

        if start_human:
            try:
                _start_human_session(topic=topic, generations=int(generations), population_size=int(population_size))
                st.success("Human session started.")
            except Exception as exc:
                _show_generation_error(exc)

        if st.session_state.get("human_active"):
            current_generation = int(st.session_state.human_current_generation)
            total_generations = int(st.session_state.human_total_generations)
            evaluations = st.session_state.human_evaluations

            st.subheader(f"Generation {current_generation} / {total_generations}")
            st.caption("Provide one ranking over all candidates (one choice per generation).")

            for idx, evaluation in enumerate(evaluations):
                params = evaluation.candidate.model_params
                label = (
                    f"Candidate {idx + 1} | signature={evaluation.candidate.signature[:8]} "
                    f"| temp={params.get('temperature')} top_p={params.get('top_p')}"
                )
                with st.expander(label):
                    st.write(evaluation.selected_response)

            ranking_input = st.text_input(
                "Generation ranking (space-separated candidate indices, 1-based), e.g. '2 1 4 3'",
                key=f"human_ranking_gen_{current_generation}",
            )

            if st.button("Submit Generation Ranking"):
                try:
                    scores = _parse_generation_ranking(ranking_input.strip(), len(evaluations))
                    optimizer = st.session_state.human_optimizer
                    optimizer.apply_human_scores(evaluations, scores)
                    best = optimizer.complete_generation(
                        generation_index=current_generation,
                        evaluations=evaluations,
                    )

                    st.info(
                        f"Generation {current_generation} complete. "
                        f"Best reward={best.reward:.3f}, signature={best.candidate.signature[:8]}"
                    )

                    next_generation = current_generation + 1
                    if next_generation > total_generations:
                        result = optimizer.build_result(
                            topic=st.session_state.human_topic,
                            reward_mode="human",
                            generations=total_generations,
                        )
                        st.session_state.human_result = result
                        st.session_state.human_active = False
                        st.success("Human optimization complete.")
                        st.json(result)
                    else:
                        st.session_state.human_current_generation = next_generation
                        st.session_state.human_evaluations = optimizer.evaluate_generation(
                            topic=st.session_state.human_topic,
                            reward_mode="human",
                        )
                        st.rerun()
                except Exception as exc:
                    _show_generation_error(exc)

        if st.session_state.get("human_result") is not None and not st.session_state.get("human_active"):
            st.subheader("Last Human Session Result")
            st.json(st.session_state.human_result)


if __name__ == "__main__":
    main()
