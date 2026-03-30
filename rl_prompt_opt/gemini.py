"""Gemini client wrapper with key failover and response sampling."""

from __future__ import annotations

import asyncio
import random
import time
from hashlib import sha256
from typing import Optional

import google.genai as genai


def _extract_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text.strip())

    return "\n".join(chunks).strip()


class GeminiClient:
    """Thin generation/scoring wrapper with API-key failover."""

    def __init__(
        self,
        api_keys: tuple[str, ...],
        model_name: str,
        start_key_index: int = 0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        enable_cache: bool = True,
    ) -> None:
        if not api_keys:
            raise ValueError("At least one Gemini API key must be provided.")

        self.api_keys = tuple(key.strip() for key in api_keys if key.strip())
        self.active_key_index = start_key_index % len(self.api_keys)
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.enable_cache = enable_cache
        self.cache: dict[str, str] = {}

    def _client_for_key(self, key_index: int) -> genai.Client:
        return genai.Client(api_key=self.api_keys[key_index])

    def _cache_key(self, prompt: str, cfg: dict[str, float]) -> str:
        serialized = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))
        payload = f"{self.model_name}|{serialized}|{prompt}".encode("utf-8")
        return sha256(payload).hexdigest()

    def generate(self, prompt: str, temperature: float, top_p: float, use_cache: bool = True) -> str:
        """Generate one response using failover across available keys."""
        cfg = {"temperature": temperature, "top_p": top_p}
        cache_enabled = self.enable_cache and use_cache
        key = self._cache_key(prompt, cfg)
        if cache_enabled and key in self.cache:
            return self.cache[key]

        last_error: Optional[Exception] = None
        key_count = len(self.api_keys)
        attempts_per_key = 1 if key_count > 1 else max(1, self.max_retries)
        total_attempts = key_count * attempts_per_key
        key_index = self.active_key_index

        for attempt in range(1, total_attempts + 1):
            try:
                client = self._client_for_key(key_index)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=cfg,
                )
                text = _extract_text(response)
                if not text:
                    raise ValueError("Gemini returned an empty response.")
                if cache_enabled:
                    self.cache[key] = text
                self.active_key_index = key_index
                return text
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt >= total_attempts:
                    break
                next_index = (key_index + 1) % key_count
                if next_index == key_index:
                    backoff = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    jitter = random.uniform(0.0, 0.2)
                    time.sleep(backoff + jitter)
                key_index = next_index

        raise RuntimeError(
            f"Gemini call failed after trying {total_attempts} attempt(s) across {key_count} key(s): {last_error}"
        )

    async def generate_async(self, prompt: str, temperature: float, top_p: float, use_cache: bool = True) -> str:
        return await asyncio.to_thread(self.generate, prompt, temperature, top_p, use_cache)

    async def sample_async(self, prompt: str, n: int, temperature: float, top_p: float) -> list[str]:
        tasks = [
            self.generate_async(prompt=prompt, temperature=temperature, top_p=top_p, use_cache=False)
            for _ in range(n)
        ]
        return await asyncio.gather(*tasks)

    def score_with_llm_judge(self, topic: str, response: str) -> float:
        """Score a response on a 0-10 scale using Gemini as judge."""
        judge_prompt = (
            "You are an evaluator. Score the candidate answer from 0 to 10. "
            "Return only a number with optional decimal.\n\n"
            f"Topic: {topic}\n"
            f"Candidate answer:\n{response}\n"
        )
        raw = self.generate(judge_prompt, temperature=0.0, top_p=1.0, use_cache=False)
        cleaned = raw.strip().split()[0]
        try:
            value = float(cleaned)
        except ValueError:
            # Fallback if judge output is verbose.
            import re

            match = re.search(r"(\d+(?:\.\d+)?)", raw)
            if not match:
                return 5.0
            value = float(match.group(1))
        return max(0.0, min(10.0, value))
