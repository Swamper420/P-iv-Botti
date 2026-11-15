from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx


_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL
)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning paragraphs (e.g. lfm3) from text.

    Handles both the regular case where the model generates the opening tag
    and the prefilled-template case where the response starts directly with
    reasoning text terminated by a lone ``</think>``. Stray or truncated tags
    are removed as well. Text without think tags is returned unchanged.
    """
    if not text or "think" not in text.lower():
        return text
    cleaned = text
    for _ in range(10):
        new_text = _THINK_BLOCK_RE.sub("", cleaned)
        close_match = _THINK_CLOSE_RE.search(new_text)
        if close_match:
            after = new_text[close_match.end():]
            if after.strip():
                # Prefilled template: reasoning prefix terminated by a
                # lone </think>, the answer follows it.
                new_text = after
            else:
                # Trailing stray tag after answer text, keep the prefix.
                new_text = new_text[: close_match.start()]
        if new_text == cleaned:
            break
        cleaned = new_text
    open_match = _THINK_OPEN_RE.search(cleaned)
    if open_match and not cleaned[: open_match.start()].strip():
        # Truncated leading thinking with no closing tag and no answer
        # before it: the model was cut off mid-thought.
        return ""
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    cleaned = _THINK_CLOSE_RE.sub("", cleaned)
    return cleaned


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
    num_ctx: int = 2048,
    timeout_seconds: int = 120,
    system_prompt: str | None = None,
    client: httpx.AsyncClient | None = None,
    think: bool | None = False,
) -> AsyncGenerator[str, None]:
    """Stream text completion chunks from an Ollama instance.

    Thinking models (e.g. lfm3) are asked for a direct answer
    (``think=False``) so reasoning tokens don't consume the ``num_predict``
    budget. Ollama's native ``thinking`` stream field is ignored and any
    ``<think>...</think>`` reasoning paragraphs still present in the
    ``response`` field are stripped, so callers only see the final answer.
    Pass ``think=None`` to use the model default or ``think=True`` to
    explicitly request thinking output (still stripped from ``response``).
    """
    target_url = f"{base_url.rstrip('/')}/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }

    if think is not None:
        payload["think"] = think

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
            # Stateful filter for <think>...</think> reasoning paragraphs.
            # Tags may split across stream lines, so incomplete trailing
            # "<..." fragments are held back until the next chunk arrives.
            pending = ""
            in_think = False
            async for line in response.aiter_lines():
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                # Native thinking channel (lfm3, qwen3, deepseek-r1, ...):
                # never forwarded to callers.
                data.pop("thinking", None)
                chunk = data.get("response", "")
                if not chunk:
                    continue

                pending += chunk
                answer_parts: list[str] = []
                while True:
                    if not in_think:
                        open_match = _THINK_OPEN_RE.search(pending)
                        close_match = _THINK_CLOSE_RE.search(pending)
                        if open_match and (
                            close_match is None
                            or open_match.start() <= close_match.start()
                        ):
                            if open_match.start() > 0:
                                answer_parts.append(pending[: open_match.start()])
                            pending = pending[open_match.end():]
                            in_think = True
                            continue
                        if close_match is not None:
                            # Prefilled template: reasoning without an
                            # opening tag, drop everything through </think>.
                            pending = pending[close_match.end():]
                            continue
                        tail_start = pending.rfind("<")
                        if tail_start != -1 and ">" not in pending[tail_start:]:
                            if tail_start > 0:
                                answer_parts.append(pending[:tail_start])
                                pending = pending[tail_start:]
                            break
                        answer_parts.append(pending)
                        pending = ""
                        break
                    else:
                        close_match = _THINK_CLOSE_RE.search(pending)
                        if close_match is not None:
                            pending = pending[close_match.end():]
                            in_think = False
                            continue
                        tail_start = pending.rfind("<")
                        if tail_start != -1 and ">" not in pending[tail_start:]:
                            pending = pending[tail_start:]
                        else:
                            pending = ""
                        break
                for part in answer_parts:
                    if part:
                        yield part
            if pending and not in_think:
                tail_start = pending.rfind("<")
                if tail_start != -1 and ">" not in pending[tail_start:]:
                    pending = pending[:tail_start]
                cleaned = strip_think_blocks(pending)
                if cleaned:
                    yield cleaned
            # If still in_think at end of stream the model was cut off
            # mid-thought: drop the truncated reasoning, yield nothing.
    finally:
        if close_client:
            await client.aclose()
