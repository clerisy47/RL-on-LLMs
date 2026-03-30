"""Configuration and shared constants for the RL prompt optimization system."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

PROMPT_PARAMETER_SPACE: Dict[str, list[str]] = {
    "role": ["teacher", "engineer", "researcher", "friendly tutor"],
    "audience": ["beginner", "student", "expert"],
    "style": ["intuitive", "technical", "conversational"],
    "tone": ["formal", "friendly", "concise"],
    "depth": ["shallow", "moderate", "deep"],
    "length": ["1 sentence", "short paragraph", "detailed"],
    "structure": ["paragraph", "bullet points", "numbered steps"],
    "examples": ["none", "one", "multiple"],
    "analogies": ["none", "simple", "real-world"],
    "step_by_step": ["yes", "no"],
    "clarity": ["avoid jargon", "define terms"],
    "constraints": ["none", "avoid long sentences", "avoid repetition"],
    "format": ["plain text", "markdown"],
}

MODEL_PARAMS: Dict[str, list[int | float]] = {
    "temperature": [0.2, 0.5, 0.7, 1.0],
    "top_p": [0.7, 0.9, 1.0],
    "presence_penalty": [0.0, 0.5, 1.0],
    "frequency_penalty": [0.0, 0.5, 1.0],
    "max_tokens": [64, 128, 256, 512],
}

SEARCH_PARAMETER_SPACE: Dict[str, list[Any]] = {
    **PROMPT_PARAMETER_SPACE,
    **MODEL_PARAMS,
}


@dataclass(slots=True)
class AppSettings:
    """Runtime settings loaded from environment variables."""

    gemini_api_key: str
    gemini_api_keys: tuple[str, ...]
    gemini_model: str = "gemini-2.5-flash"
    memory_path: str = "data/memory.json"
    log_path: str = "logs/optimizer.log"

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Load .env values and build app settings."""
        load_dotenv()
        raw_api_keys = os.getenv("GEMINI_API_KEY", "").strip()
        api_keys = tuple(key.strip() for key in raw_api_keys.split(",") if key.strip())
        if not api_keys:
            raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

        return cls(
            gemini_api_key=api_keys[0],
            gemini_api_keys=api_keys,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            memory_path=os.getenv("MEMORY_PATH", "data/memory.json").strip(),
            log_path=os.getenv("LOG_PATH", "logs/optimizer.log").strip(),
        )


@dataclass(slots=True)
class HyperParams:
    """Train-time hyperparameters."""

    iterations: int = 5
    num_samples: int = 3
    epsilon: float = 0.25
    mutation_rate: float = 0.25
    temperature: float = 0.9
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "HyperParams":
        """Create hyperparameters from a dictionary."""
        return cls(
            iterations=int(values.get("iterations", cls.iterations)),
            num_samples=int(values.get("num_samples", cls.num_samples)),
            epsilon=float(values.get("epsilon", cls.epsilon)),
            mutation_rate=float(values.get("mutation_rate", cls.mutation_rate)),
            temperature=float(values.get("temperature", cls.temperature)),
            max_retries=int(values.get("max_retries", cls.max_retries)),
            retry_backoff_seconds=float(
                values.get("retry_backoff_seconds", cls.retry_backoff_seconds)
            ),
        )


def load_hyperparams(path: str = "hyperparams.json") -> HyperParams:
    """Load hyperparameters from a JSON file, falling back to defaults."""
    file_path = Path(path)
    if not file_path.exists():
        return HyperParams()

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return HyperParams.from_dict(data)


def setup_logging(log_path: str) -> logging.Logger:
    """Configure and return the optimizer logger."""
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("prompt_optimizer")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
