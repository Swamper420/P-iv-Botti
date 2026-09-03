from __future__ import annotations

import asyncio
import logging
from io import BytesIO

from telegram import InputFile, Update

from bot.rendering.engine import render_card
from bot.rendering.models import Card, Theme

LOGGER = logging.getLogger(__name__)


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


async def reply_with_image(
    update: Update,
    image: bytes | bytearray | BytesIO,
    caption: str | None = None,
    fallback_text: str | None = None,
    filename: str = "card.png",
    max_reply_length: int = 5000,
    parse_mode: str | None = None,
) -> None:
    """Replies to update with an image. If image sending fails, gracefully falls back to text."""
    message = update.effective_message
    if message is None:
        return

    bio = BytesIO(image) if isinstance(image, (bytes, bytearray)) else image
    bio.seek(0)
    photo_file = InputFile(bio, filename=filename)

    try:
        await message.reply_photo(photo=photo_file, caption=caption, parse_mode=parse_mode)
    except Exception as exc:
        LOGGER.warning("Failed to send image reply (%s). Falling back to text.", exc)
        text_to_send = fallback_text or caption
        if text_to_send:
            await reply_in_chunks(
                update,
                text_to_send,
                max_reply_length=max_reply_length,
                parse_mode=parse_mode,
            )
        else:
            raise


async def reply_with_card(
    update: Update,
    card: Card,
    theme: Theme | None = None,
    caption: str | None = None,
    fallback_text: str | None = None,
    filename: str = "card.png",
    max_reply_length: int = 5000,
    parse_mode: str | None = None,
) -> None:
    """Renders a card asynchronously and replies with the generated image."""
    if theme is not None:
        img_bytes = await asyncio.to_thread(render_card, card, theme)
    else:
        img_bytes = await asyncio.to_thread(render_card, card)

    await reply_with_image(
        update=update,
        image=img_bytes,
        caption=caption,
        fallback_text=fallback_text,
        filename=filename,
        max_reply_length=max_reply_length,
        parse_mode=parse_mode,
    )
