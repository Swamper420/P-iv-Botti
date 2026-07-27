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
    default_strip_thinking: bool = True,
) -> tuple[bool, int, bool, str]:
    """Parse an !aih command string into (is_match, token_count, strip_thinking, prompt)."""
    if not text:
        return False, default_tokens, default_strip_thinking, ""

    match = re.match(
        r"(?i)^\s*!?aih(nothink)?(\d+)?(nothink)?(?::?\s*|\s+)?(.*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return False, default_tokens, default_strip_thinking, ""

    has_nothink = bool(match.group(1) or match.group(3))
    strip_thinking = True if has_nothink else default_strip_thinking

    digits_str = match.group(2)
    prompt = (match.group(4) or "").strip()

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

    return True, token_count, strip_thinking, prompt


async def filter_thinking_tags(
    source: AsyncGenerator[str, None]
) -> AsyncGenerator[str, None]:
    """Filters out <think>...</think> blocks from a text stream."""
    buffer = ""
    in_think = False

    async for chunk in source:
        buffer += chunk

        while buffer:
            if not in_think:
                think_start = buffer.lower().find("<think>")
                if think_start != -1:
                    pre_think = buffer[:think_start]
                    if pre_think:
                        yield pre_think
                    buffer = buffer[think_start + len("<think>") :]
                    in_think = True
                else:
                    partial_len = 0
                    for i in range(1, len("<think>")):
                        if buffer.lower().endswith("<think>"[:i]):
                            partial_len = i
                            break

                    if partial_len > 0:
                        yield_text = buffer[:-partial_len]
                        if yield_text:
                            yield yield_text
                        buffer = buffer[-partial_len:]
                        break
                    else:
                        yield buffer
                        buffer = ""
            else:
                think_end = buffer.lower().find("</think>")
                if think_end != -1:
                    buffer = buffer[think_end + len("</think>") :]
                    in_think = False
                else:
                    partial_len = 0
                    for i in range(1, len("</think>")):
                        if buffer.lower().endswith("</think>"[:i]):
                            partial_len = i
                            break
                    if partial_len > 0:
                        buffer = buffer[-partial_len:]
                    else:
                        buffer = ""
                    break

    if not in_think and buffer:
        yield buffer


async def _raw_stream_ollama(
    base_url: str,
    model: str,
    prompt: str,
    num_predict: int,
    timeout_seconds: int,
    strip_thinking: bool,
    system_prompt: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncGenerator[str, None]:
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
    elif strip_thinking:
        payload["system"] = (
            "Do not include internal thinking processes, reasoning steps, or <think> tags. "
            "Answer directly and concisely."
        )

    if strip_thinking:
        payload["options"]["think"] = False

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


async def stream_ollama_completion(
    base_url: str,
    model: str,
    prompt: str,
    num_predict: int,
    timeout_seconds: int = 120,
    strip_thinking: bool = True,
    system_prompt: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncGenerator[str, None]:
    """Stream text completion chunks from an Ollama instance, stripping thinking blocks if configured."""
    raw_stream = _raw_stream_ollama(
        base_url=base_url,
        model=model,
        prompt=prompt,
        num_predict=num_predict,
        timeout_seconds=timeout_seconds,
        strip_thinking=strip_thinking,
        system_prompt=system_prompt,
        client=client,
    )

    if strip_thinking:
        async for chunk in filter_thinking_tags(raw_stream):
            yield chunk
    else:
        async for chunk in raw_stream:
            yield chunk

