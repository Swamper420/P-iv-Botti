from __future__ import annotations

from bot.commands.message_utils import split_message


def get_paivaa_reply(text: str | None) -> str | None:
    if text is None:
        return None

    if text.strip().casefold() == "päivää".casefold():
        return "Päivää *tips fedora*"

    return None
