from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx


def parse_aih_command(
    text: str,
    default_tokens: int = 100,
    max_tokens: int = 2000,
) -> tuple[bool, int, str]:
    """Parse an !aih command string into (is_match, token_count, prompt)."""
    if not text:
        return False, default_tokens, ""

    match = re.match(
        r"(?i)^\s*!?aih(\d+)?(?::?\s*|\s+)?(.*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return False, default_tokens, ""

    digits_str = match.group(1)
    prompt = (match.group(2) or "").strip()

    if digits_str:
        token_count = int(digits_str)
    else:
        # Check if prompt starts with a number (e.g. "!aih 400: prompt" or "!aih 400 prompt")
        leading_number_match = re.match(r"^(\d+)(?::?\s*|\s+)(.*)$", prompt, re.DOTALL)
        if leading_number_match:
            token_count = int(leading_number_match.group(1))
            prompt = leading_number_match.group(2).strip()
        else:
            token_count = default_tokens

    # Clamp token count between 1 and max_tokens
    token_count = max(1, min(token_count, max_tokens))

    return True, token_count, prompt


async def stream_ollama_completion(
    base_url: str,
    model: str,
    prompt: str,
    num_predict: int,
    timeout_seconds: int = 120,
    system_prompt: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncGenerator[str, None]:
    """Stream text completion chunks from an Ollama instance."""
    target_url = f"{base_url.rstrip('/')}/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": num_predict,
        },
    }

    if system_prompt is not None and system_prompt.strip():
        payload["system"] = system_prompt
    else:
        payload["system"] = "Vastaa aina suomeksi suoraan ja tiiviisti."

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds)
        close_client = True

    try:
        async with client.stream("POST", target_url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                chunk = data.get("response", "")
                if chunk:
                    yield chunk
    finally:
        if close_client:
            await client.aclose()
