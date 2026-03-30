# RL Prompt Optimization System (Gemini + RLHF-style)

Production-quality Python project for black-box prompt optimization over Gemini APIs using:

- epsilon-greedy parameter bandits
- mutation-based exploration (Evolution Strategy)
- configurable rewards (heuristics, human feedback, or both)
- mixed optimization over prompt + model generation parameters
- persistent memory and retrieval of successful settings
- CLI and Streamlit interfaces

## Project Structure

- `config.py`
- `prompt_generator.py`
- `gemini_client.py`
- `reward.py`
- `human_feedback.py`
- `bandit.py`
- `evolution.py`
- `memory.py`
- `optimizer.py`
- `main.py`
- `streamlit_app.py`
- `hyperparams.json`

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
The optimizer uses one key per iteration and rotates via modulo indexing.

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
- `both`

Optional flags:

- `--iterations 10`
- `--samples 3`
- `--feedback-mode cli`
- `--hyperparams hyperparams.json`
- `--reward-mode heuristics|human|both`

Default optimization iterations: 5.

Model generation parameters are included in the optimizer mixer and learned jointly:

```python
MODEL_PARAMS = {
	"temperature": [0.2, 0.5, 0.7, 1.0],
	"top_p": [0.7, 0.9, 1.0],
	"presence_penalty": [0.0, 0.5, 1.0],
	"frequency_penalty": [0.0, 0.5, 1.0],
	"max_tokens": [64, 128, 256, 512],
}
```

## Streamlit Human Interface

After activating your virtual environment:

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

The app shows multiple responses side-by-side and supports:

- best-response selection
- full ranking
- pairwise preferences
- full interactive optimization loop with CLI-equivalent controls (topic, iterations, samples, reward mode)
- automatic run mode for non-human reward workflows

## Notes

- Human feedback is treated as primary when provided.
- Memory is persisted in `data/memory.json`.
- Training and iteration logs are saved in `logs/optimizer.log`.
