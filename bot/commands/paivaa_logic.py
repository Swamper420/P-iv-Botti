from __future__ import annotations


def get_paivaa_reply(text: str | None) -> str | None:
    if text is None:
        return None

    if text.strip().casefold() == "päivää".casefold():
        return "Päivää *tips fedora*"

    return None
