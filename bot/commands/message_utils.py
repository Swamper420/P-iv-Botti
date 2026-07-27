from __future__ import annotations

from telegram import Update


def split_message(text: str, max_length: int = 5000) -> list[str]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    if not text:
        return []

    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


async def reply_in_chunks(
    update: Update,
    reply: str,
    max_reply_length: int,
    parse_mode: str | None = None,
) -> None:
    message = update.effective_message
    if message is None:
        return

    for chunk in split_message(reply, max_reply_length):
        if parse_mode:
            await message.reply_text(chunk, parse_mode=parse_mode)
        else:
            await message.reply_text(chunk)

