"""Gemini API client wrapper with retries, caching, and async helpers."""

from __future__ import annotations

import asyncio
import random
import time
from hashlib import sha256
from typing import Dict, Optional

import google.genai as genai


def _extract_text(response: object) -> str:
    """Extract text from Gemini responses across SDK variants."""
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
    """High-level wrapper around the Gemini SDK for generation calls."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_keys: tuple[str, ...] | None = None,
        start_key_index: int = 0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        enable_cache: bool = True,
    ) -> None:
        normalized_keys = tuple(key.strip() for key in (api_keys or (api_key,)) if key.strip())
        if not normalized_keys:
            raise ValueError("At least one Gemini API key must be provided.")

        self.api_keys = normalized_keys
        self.active_key_index = start_key_index % len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self.active_key_index])
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.enable_cache = enable_cache
        self.cache: Dict[str, str] = {}

    def _client_for_key_index(self, key_index: int) -> genai.Client:
        """Build a Gemini client for a specific key index."""
        return genai.Client(api_key=self.api_keys[key_index])

    def _cache_key(self, prompt: str, generation_config: dict[str, float | int]) -> str:
        cfg = "|".join(f"{k}={generation_config[k]}" for k in sorted(generation_config))
        payload = f"{self.model_name}|{cfg}|{prompt}".encode("utf-8")
        return sha256(payload).hexdigest()

    def call_gemini(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_tokens: int = 256,
        use_cache: bool = True,
    ) -> str:
        """Call Gemini with retry and optional in-memory caching."""
        cache_enabled = self.enable_cache and use_cache
        generation_config: dict[str, float | int] = {
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_output_tokens": max_tokens,
        }
        key = self._cache_key(prompt, generation_config)
        if cache_enabled and key in self.cache:
            return self.cache[key]

        last_error: Optional[Exception] = None
        key_count = len(self.api_keys)
        # With multiple keys, fail over through the whole key pool before giving up.
        attempts_per_key = 1 if key_count > 1 else max(1, self.max_retries)
        total_attempts = key_count * attempts_per_key
        key_index = self.active_key_index

        for attempt in range(1, total_attempts + 1):
            try:
                current_client = self._client_for_key_index(key_index)
                response = current_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generation_config,
                )
                text = _extract_text(response)
                if not text:
                    raise ValueError("Gemini returned an empty response.")

                if cache_enabled:
                    self.cache[key] = text
                self.active_key_index = key_index
                self.client = current_client
                return text
            except Exception as exc:  # pragma: no cover - network dependency
                last_error = exc
                if attempt >= total_attempts:
                    break
                next_key_index = (key_index + 1) % key_count

                # If only one key is available, keep classic exponential backoff retries.
                if next_key_index == key_index:
                    backoff = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    jitter = random.uniform(0.0, 0.2)
                    time.sleep(backoff + jitter)

                key_index = next_key_index

        raise RuntimeError(
            f"Gemini call failed after trying {total_attempts} attempt(s) across {key_count} key(s): {last_error}"
        )

    async def call_gemini_async(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_tokens: int = 256,
        use_cache: bool = True,
    ) -> str:
        """Async wrapper over the synchronous Gemini call."""
        return await asyncio.to_thread(
            self.call_gemini,
            prompt,
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            max_tokens,
            use_cache,
        )

    async def sample_responses_async(
        self,
        prompt: str,
        n: int = 3,
        temperature: float = 0.9,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_tokens: int = 256,
    ) -> list[str]:
        """Sample multiple outputs for a single prompt in parallel."""
        tasks = [
            self.call_gemini_async(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                use_cache=False,
            )
            for _ in range(n)
        ]
        return await asyncio.gather(*tasks)


def call_gemini(
    prompt: str,
    client: GeminiClient,
    temperature: float = 0.8,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """Convenience function matching the requested API shape."""
    return client.call_gemini(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        max_tokens=max_tokens,
    )
