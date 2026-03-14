from __future__ import annotations


def get_aih_prompt(text: str | None) -> str | None:
    if text is None:
        return None

    stripped = text.strip()
    if not stripped.casefold().startswith("aih:"):
        return None

    prompt = stripped[4:].strip()
    if not prompt:
        return None

    return prompt
