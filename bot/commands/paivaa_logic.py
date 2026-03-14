from __future__ import annotations


def get_paivaa_reply(text: str | None) -> str | None:
    if text is None:
        return None

    if text.strip().casefold() == "päivää".casefold():
        return "Päivää *tips fedora*"

    return None


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


def split_message(text: str, max_length: int = 5000) -> list[str]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    if not text:
        return []

    return [text[i : i + max_length] for i in range(0, len(text), max_length)]
