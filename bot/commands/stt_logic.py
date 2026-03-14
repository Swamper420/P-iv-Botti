from __future__ import annotations

import json
from typing import Any


def is_audio_duration_allowed(duration_seconds: int | None, max_seconds: int) -> bool:
    if duration_seconds is None:
        return False
    return 0 < duration_seconds <= max_seconds


def parse_transcription_response(response_body: str) -> str | None:
    try:
        parsed: Any = json.loads(response_body)
    except json.JSONDecodeError:
        return response_body.strip() or None

    if isinstance(parsed, str):
        return parsed.strip() or None

    if isinstance(parsed, dict):
        for key in ("text", "transcript", "transcription", "result", "message"):
            value = parsed.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    return cleaned
    return None
