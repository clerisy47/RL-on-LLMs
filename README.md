# Evolutionary Prompt Optimization (Gemini)

Population-based prompt optimization over Gemini with a clean package structure.

## Algorithm

Each generation follows:

1. Maintain a population of candidates.
2. Candidate = prompt parameter combination + model parameter combination.
3. Evaluate only candidates not already in cache/memory.
4. Rank by reward and select stronger parents.
5. Randomly eliminate weaker candidates.
6. Create children with crossover.
7. Apply random mutation.
8. Repeat for next generation.

In `human` mode, you provide exactly one ranking per generation across all candidates.

Reward modes:

- `heuristics`
- `human`
- `llm_judge`

## Folder Structure

- `rl_prompt_opt/`
	- `__init__.py`
	- `settings.py`
	- `search_space.py`
	- `candidate.py`
	- `prompting.py`
	- `feedback.py`
	- `rewards.py`
	- `gemini.py`
	- `engine.py`
- `main.py` (CLI entrypoint)
- `streamlit_app.py` (Streamlit entrypoint)
- `hyperparams.json`
- `data/memory.json`
- `logs/optimizer.log`

## Setup

1. Create a Python virtual environment (required for dependency isolation):

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or on Windows:
# venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses the current Gemini SDK package: `google-genai`. Note: The system Python may be externally managed; using a virtual environment is strongly recommended.

3. Create a `.env` file in the project root with:

Expected `.env` keys:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
MEMORY_PATH=data/memory.json
LOG_PATH=logs/optimizer.log
```

Multiple keys are supported by setting `GEMINI_API_KEY` as comma-separated values.
The engine automatically fails over to the next key on API errors.

Example:

```dotenv
GEMINI_API_KEY=key_one,key_two,key_three
```

Security note: keep `.env` local and never commit real API keys.

## CLI Usage

```bash
python main.py --topic "Explain recursion"
```

At startup, the CLI asks which reward mode to use:

- `heuristics`
- `human`
- `llm_judge`

Optional flags:

- `--generations 10`
- `--hyperparams hyperparams.json`
- `--reward-mode heuristics|human|llm_judge`

Default generations: 5.

Model parameters in the search space:

```python
MODEL_PARAMS = {
    "temperature": [0.2, 0.5, 0.7, 1.0],
    "top_p": [0.7, 0.9, 1.0],
}
```

## Streamlit

After activating your virtual environment:

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

The app runs the same evolutionary engine with generation/population controls.

In Streamlit `human` mode, start a human session and provide one generation-level ranking per generation across all candidates.

Hyperparameter knobs for reproduction in `hyperparams.json`:

- `parent_fraction`: top candidates prioritized as parents.
- `elimination_fraction`: weakest candidates prioritized for elimination.
- `crossover_fraction`: portion of population replaced by crossover children.
- `mutation_fraction`: portion of population replaced by mutated variants.

## Notes

- Memory is persisted in `data/memory.json` and reused to avoid re-evaluating known candidates.
- Generation logs are written to `logs/optimizer.log`.
