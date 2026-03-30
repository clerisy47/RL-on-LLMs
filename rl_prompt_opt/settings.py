"""Application settings and hyperparameters."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class AppSettings:
    """Runtime app settings loaded from environment variables."""

    gemini_api_keys: tuple[str, ...]
    gemini_model: str = "gemini-2.5-flash"
    memory_path: str = "data/memory.json"
    log_path: str = "logs/optimizer.log"

    @property
    def primary_api_key(self) -> str:
        return self.gemini_api_keys[0]

    @classmethod
    def from_env(cls) -> "AppSettings":
        load_dotenv()
        raw_api_keys = os.getenv("GEMINI_API_KEY", "").strip()
        keys = tuple(key.strip() for key in raw_api_keys.split(",") if key.strip())
        if not keys:
            raise ValueError("GEMINI_API_KEY is missing. Add one or more keys in .env.")

        return cls(
            gemini_api_keys=keys,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            memory_path=os.getenv("MEMORY_PATH", "data/memory.json").strip(),
            log_path=os.getenv("LOG_PATH", "logs/optimizer.log").strip(),
        )


@dataclass(slots=True)
class HyperParams:
    """Evolution algorithm hyperparameters."""

    generations: int = 5
    population_size: int = 8
    parent_fraction: float = 0.4
    elimination_fraction: float = 0.5
    crossover_fraction: float = 0.3
    mutation_fraction: float = 0.2
    mutation_rate: float = 0.25
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    @classmethod
    def from_dict(cls, values: dict) -> "HyperParams":
        return cls(
            generations=int(values.get("generations", cls.generations)),
            population_size=int(values.get("population_size", cls.population_size)),
            parent_fraction=float(values.get("parent_fraction", cls.parent_fraction)),
            elimination_fraction=float(values.get("elimination_fraction", cls.elimination_fraction)),
            crossover_fraction=float(values.get("crossover_fraction", cls.crossover_fraction)),
            mutation_fraction=float(values.get("mutation_fraction", cls.mutation_fraction)),
            mutation_rate=float(values.get("mutation_rate", cls.mutation_rate)),
            max_retries=int(values.get("max_retries", cls.max_retries)),
            retry_backoff_seconds=float(values.get("retry_backoff_seconds", cls.retry_backoff_seconds)),
        )


def load_hyperparams(path: str = "hyperparams.json") -> HyperParams:
    """Load hyperparameters from JSON, fallback to defaults."""
    file_path = Path(path)
    if not file_path.exists():
        return HyperParams()

    with file_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return HyperParams.from_dict(raw)


def setup_logger(log_path: str) -> logging.Logger:
    """Configure logger once."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("evolutionary_prompt_optimizer")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
