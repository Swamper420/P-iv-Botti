from __future__ import annotations

import re
from typing import Any

import httpx


def parse_tts_command(text: str) -> tuple[bool, str | None, str]:
    """Parse a TTS command string into (is_match, voice, text).

    Syntax: [voice] puhuu: <text>
    If voice is omitted, returns voice=None.
    """
    if not text:
        return False, None, ""

    match = re.match(
        r"(?i)^\s*!?(?:([^\n:]+)\s+)?puhuu:\s*(.*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return False, None, ""

    voice_raw = match.group(1)
    prompt_text = (match.group(2) or "").strip()

    voice: str | None = None
    if voice_raw:
        voice_clean = voice_raw.strip().lstrip("!")
        if voice_clean:
            voice = voice_clean

    return True, voice, prompt_text


async def synthesize_speech(
    base_url: str,
    text: str,
    voice: str | None = None,
    fmt: str = "ogg",
    model: str | None = None,
    timeout_seconds: int = 60,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    """Synthesize speech using the Chatterbox TTS API."""
    target_url = f"{base_url.rstrip('/')}/api/tts"
    payload: dict[str, Any] = {
        "text": text,
        "format": fmt,
    }
    if voice:
        payload["voice"] = voice
    if model:
        payload["model"] = model

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds)
        close_client = True

    try:
        response = await client.post(target_url, json=payload)
        response.raise_for_status()
        return response.content
    finally:
        if close_client:
            await client.aclose()
