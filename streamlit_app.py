"""Streamlit UI that mirrors CLI optimization capabilities."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config import AppSettings, load_hyperparams
from human_feedback import HumanFeedback, render_streamlit_feedback
from memory import build_memory_entry
from optimizer import PromptOptimizer
from reward import compute_final_reward


def _init_optimizer() -> PromptOptimizer:
    """Initialize optimizer once and store in Streamlit session state."""
    if "optimizer" not in st.session_state:
        settings = AppSettings.from_env()
        hyperparams = load_hyperparams("hyperparams.json")
        st.session_state.optimizer = PromptOptimizer(settings=settings, hyperparams=hyperparams)
    return st.session_state.optimizer


def _reset_session() -> None:
    """Clear optimization session state from Streamlit memory."""
    for key in list(st.session_state.keys()):
        if key.startswith("opt_"):
            del st.session_state[key]


def _compute_reward_from_feedback(
    reward_mode: str,
    heuristic_scores: list[float],
    feedback: HumanFeedback,
) -> tuple[float, int, list[float]]:
    """Compute selected reward from feedback according to active mode."""
    human_scores = feedback.to_scores(len(heuristic_scores))
    final_scores: list[float] = []
    for idx, heuristic in enumerate(heuristic_scores):
        final_scores.append(
            compute_final_reward(
                mode=reward_mode,
                heuristic_score=heuristic,
                human_score=human_scores[idx],
            )
        )

    selected = feedback.primary_choice()
    if selected is None:
        selected = max(range(len(final_scores)), key=lambda i: final_scores[i])
    return final_scores[selected], selected, final_scores


def _prepare_iteration(optimizer: PromptOptimizer) -> None:
    """Create candidate parameter sets and evaluate their responses once per iteration."""
    topic = st.session_state.opt_topic
    samples = st.session_state.opt_samples

    key_index = optimizer._configure_client_for_iteration(st.session_state.opt_iteration)
    st.session_state.opt_key_slot = key_index + 1
    st.session_state.opt_key_total = len(optimizer.settings.gemini_api_keys)

    base_params = optimizer._blend_with_memory(topic, optimizer.bandit.sample_parameters())
    mutant_params = optimizer.evolution.mutate(base_params)

    base_eval = optimizer.evaluate_params(
        topic=topic,
        params=base_params,
        num_samples=samples,
        reward_mode="heuristics",
        feedback_mode="none",
    )
    mutant_eval = optimizer.evaluate_params(
        topic=topic,
        params=mutant_params,
        num_samples=samples,
        reward_mode="heuristics",
        feedback_mode="none",
    )

    st.session_state.opt_base_params = base_params
    st.session_state.opt_mutant_params = mutant_params
    st.session_state.opt_base_eval = base_eval
    st.session_state.opt_mutant_eval = mutant_eval
    st.session_state.opt_base_feedback = None
    st.session_state.opt_mutant_feedback = None
    st.session_state.opt_base_reward = None
    st.session_state.opt_mutant_reward = None
    st.session_state.opt_base_selected = None
    st.session_state.opt_mutant_selected = None
    st.session_state.opt_base_final_scores = list(base_eval.heuristic_scores)
    st.session_state.opt_mutant_final_scores = list(mutant_eval.heuristic_scores)


def _apply_automatic_candidate_rewards() -> None:
    """Set candidate rewards when human feedback is not required."""
    if st.session_state.opt_base_reward is None:
        base_scores = st.session_state.opt_base_eval.heuristic_scores
        base_selected = max(range(len(base_scores)), key=lambda i: base_scores[i])
        st.session_state.opt_base_reward = base_scores[base_selected]
        st.session_state.opt_base_selected = base_selected

    if st.session_state.opt_mutant_reward is None:
        mutant_scores = st.session_state.opt_mutant_eval.heuristic_scores
        mutant_selected = max(range(len(mutant_scores)), key=lambda i: mutant_scores[i])
        st.session_state.opt_mutant_reward = mutant_scores[mutant_selected]
        st.session_state.opt_mutant_selected = mutant_selected


def _finalize_iteration(optimizer: PromptOptimizer) -> None:
    """Choose winner, update learner state and persist memory for one iteration."""
    base_reward = float(st.session_state.opt_base_reward)
    mutant_reward = float(st.session_state.opt_mutant_reward)

    if mutant_reward > base_reward:
        winner_source = "mutated"
        winner_params = st.session_state.opt_mutant_params
        winner_eval = st.session_state.opt_mutant_eval
        winner_reward = mutant_reward
        winner_selected = int(st.session_state.opt_mutant_selected)
        winner_feedback = st.session_state.opt_mutant_feedback
        winner_final_scores = st.session_state.opt_mutant_final_scores
    else:
        winner_source = "current"
        winner_params = st.session_state.opt_base_params
        winner_eval = st.session_state.opt_base_eval
        winner_reward = base_reward
        winner_selected = int(st.session_state.opt_base_selected)
        winner_feedback = st.session_state.opt_base_feedback
        winner_final_scores = st.session_state.opt_base_final_scores

    optimizer.bandit.update(winner_params, winner_reward)
    if winner_reward > optimizer.best_reward:
        optimizer.best_reward = winner_reward
        optimizer.best_params = dict(winner_params)

    entry = build_memory_entry(
        topic=st.session_state.opt_topic,
        params=winner_params,
        responses=winner_eval.responses,
        reward=winner_reward,
        selected_response_index=winner_selected,
        human_choice=(winner_feedback.to_dict() if winner_feedback is not None else None),
        heuristic_scores=winner_eval.heuristic_scores,
        final_scores=winner_final_scores,
    )
    optimizer.memory.add(entry)

    optimizer.logger.info(
        "iteration=%s key_slot=%s/%s winner=%s reward=%.3f selected_response=%s params=%s",
        st.session_state.opt_iteration,
        st.session_state.opt_key_slot,
        st.session_state.opt_key_total,
        winner_source,
        winner_reward,
        winner_selected + 1,
        winner_params,
    )

    st.session_state.opt_last_summary = (
        f"Iteration {st.session_state.opt_iteration} | key_slot={st.session_state.opt_key_slot}/{st.session_state.opt_key_total} "
        f"| winner={winner_source} | reward={winner_reward:.3f} | selected_response={winner_selected + 1}"
    )

    st.session_state.opt_iteration += 1
    for key in [
        "opt_base_params",
        "opt_mutant_params",
        "opt_base_eval",
        "opt_mutant_eval",
        "opt_base_feedback",
        "opt_mutant_feedback",
        "opt_base_reward",
        "opt_mutant_reward",
        "opt_base_selected",
        "opt_mutant_selected",
        "opt_base_final_scores",
        "opt_mutant_final_scores",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def _show_generation_error(exc: Exception) -> None:
    """Render user-friendly error guidance for generation failures."""
    message = str(exc)
    st.error(message)
    if "RESOURCE_EXHAUSTED" in message or "quota" in message.lower() or "429" in message:
        st.info(
            "All available keys may be out of quota (often shared per project on free tier). "
            "Try fewer iterations/samples, a different model/project, or wait for quota reset."
        )


def main() -> None:
    """Render and run a CLI-equivalent optimization workflow in Streamlit."""
    st.set_page_config(page_title="RL Prompt Optimizer", layout="wide")
    st.title("Prompt Optimizer Console")
    st.caption("CLI-equivalent controls: topic, iterations, samples, reward mode, and iterative learning")

    optimizer = _init_optimizer()

    topic = st.text_input("Topic", value="Explain recursion")
    col1, col2 = st.columns(2)
    with col1:
        iterations = st.number_input("Iterations", min_value=1, max_value=200, value=5, step=1)
    with col2:
        samples = st.number_input("Responses per prompt", min_value=1, max_value=8, value=3, step=1)

    reward_mode = st.radio(
        "Reward Mode",
        options=["heuristics", "human", "both"],
        index=2,
        horizontal=True,
        help="Select the same reward strategy available in CLI.",
    )
    human_choice_mode: str | None = None
    if reward_mode in {"human", "both"}:
        human_choice_mode = st.radio(
            "Human Choice Type",
            options=["best", "ranking"],
            index=0,
            horizontal=True,
            help="Select once, then this same feedback type is used for every iteration.",
        )

    run_col, reset_col = st.columns(2)
    with run_col:
        run_clicked = st.button("Run Optimization")
    with reset_col:
        reset_clicked = st.button("Reset Session")

    if reset_clicked:
        _reset_session()
        st.success("Session reset.")

    if run_clicked:
        if reward_mode == "heuristics":
            try:
                result = optimizer.optimize(
                    topic=topic,
                    iterations=int(iterations),
                    num_samples=int(samples),
                    reward_mode="heuristics",
                    feedback_mode="none",
                    human_choice_mode=None,
                )
                st.success("Optimization complete.")
                st.json(result)
            except Exception as exc:
                _show_generation_error(exc)
        else:
            st.session_state.opt_active = True
            st.session_state.opt_topic = topic
            st.session_state.opt_iterations = int(iterations)
            st.session_state.opt_samples = int(samples)
            st.session_state.opt_reward_mode = reward_mode
            st.session_state.opt_collect_human = True
            st.session_state.opt_human_choice_mode = human_choice_mode or "best"
            st.session_state.opt_iteration = 1
            st.session_state.opt_last_summary = None

    if not st.session_state.get("opt_active"):
        return

    if st.session_state.opt_last_summary:
        st.info(st.session_state.opt_last_summary)

    if st.session_state.opt_iteration > st.session_state.opt_iterations:
        st.success("Interactive optimization session complete.")
        st.json(
            {
                "topic": st.session_state.opt_topic,
                "best_params": optimizer.best_params,
                "best_reward": optimizer.best_reward,
                "bandit_snapshot": optimizer.bandit.snapshot(),
            }
        )
        return

    st.subheader(
        f"Iteration {st.session_state.opt_iteration} / {st.session_state.opt_iterations}"
    )

    if "opt_base_eval" not in st.session_state or "opt_mutant_eval" not in st.session_state:
        try:
            _prepare_iteration(optimizer)
        except Exception as exc:
            _show_generation_error(exc)
            st.session_state.opt_active = False
            return

    st.caption(
        f"Using API key slot {st.session_state.opt_key_slot}/{st.session_state.opt_key_total} for this iteration"
    )

    base_eval = st.session_state.opt_base_eval
    mutant_eval = st.session_state.opt_mutant_eval

    cand1, cand2 = st.columns(2)
    with cand1:
        st.markdown("### Candidate A (current)")
        st.json(st.session_state.opt_base_params)
        if st.session_state.opt_collect_human and st.session_state.opt_reward_mode in {"human", "both"}:
            feedback = render_streamlit_feedback(
                topic=f"{st.session_state.opt_topic} - Candidate A",
                responses=base_eval.responses,
                key_prefix=f"iter_{st.session_state.opt_iteration}_base",
                forced_mode=st.session_state.opt_human_choice_mode,
            )
            if feedback is not None:
                reward, selected, final_scores = _compute_reward_from_feedback(
                    st.session_state.opt_reward_mode,
                    base_eval.heuristic_scores,
                    feedback,
                )
                st.session_state.opt_base_feedback = feedback
                st.session_state.opt_base_reward = reward
                st.session_state.opt_base_selected = selected
                st.session_state.opt_base_final_scores = final_scores
                st.success(
                    f"Candidate A feedback saved. selected={selected + 1}, reward={reward:.3f}"
                )

    with cand2:
        st.markdown("### Candidate B (mutated)")
        st.json(st.session_state.opt_mutant_params)
        if st.session_state.opt_collect_human and st.session_state.opt_reward_mode in {"human", "both"}:
            feedback = render_streamlit_feedback(
                topic=f"{st.session_state.opt_topic} - Candidate B",
                responses=mutant_eval.responses,
                key_prefix=f"iter_{st.session_state.opt_iteration}_mutant",
                forced_mode=st.session_state.opt_human_choice_mode,
            )
            if feedback is not None:
                reward, selected, final_scores = _compute_reward_from_feedback(
                    st.session_state.opt_reward_mode,
                    mutant_eval.heuristic_scores,
                    feedback,
                )
                st.session_state.opt_mutant_feedback = feedback
                st.session_state.opt_mutant_reward = reward
                st.session_state.opt_mutant_selected = selected
                st.session_state.opt_mutant_final_scores = final_scores
                st.success(
                    f"Candidate B feedback saved. selected={selected + 1}, reward={reward:.3f}"
                )

    if not st.session_state.opt_collect_human or st.session_state.opt_reward_mode == "heuristics":
        _apply_automatic_candidate_rewards()

    ready_to_finalize = (
        st.session_state.opt_base_reward is not None
        and st.session_state.opt_mutant_reward is not None
    )

    if ready_to_finalize:
        if st.button("Complete Iteration"):
            _finalize_iteration(optimizer)
            st.rerun()
    else:
        st.warning("Provide rewards for both candidates to continue this iteration.")


if __name__ == "__main__":
    main()
