from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from socket import timeout as socket_timeout
from urllib.error import URLError
from urllib.request import Request, urlopen

from telegram import PhotoSize, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.analysoi_logic import build_analysoi_prompt, encode_image_base64
from bot.commands.message_utils import reply_in_chunks
from bot.config import BotConfig

COMMAND_USAGE = "!analysoi [kuva tai vastaus kuvaan]"

LOGGER = logging.getLogger(__name__)
_ANALYSOI_REGEX = r"(?i)^\s*!analysoi\s*$"


def _extract_target_photo(message: object) -> PhotoSize | None:
    """Return the largest photo from *message* when triggered by ``!analysoi``."""
    photos = getattr(message, "photo", None)
    caption = getattr(message, "caption", None)
    if photos and isinstance(caption, str) and caption.strip().lower() == "!analysoi":
        return photos[-1]

    text = getattr(message, "text", None)
    reply_message = getattr(message, "reply_to_message", None)
    reply_photos = getattr(reply_message, "photo", None)
    if (
        isinstance(text, str)
        and text.strip().lower() == "!analysoi"
        and reply_photos
    ):
        return reply_photos[-1]

    return None


def _query_ai_with_image(image_b64: str, prompt: str, config: BotConfig) -> str:
    """Send *image_b64* and *prompt* to the AI backend and return the reply."""
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": config.ai_max_tokens,
            "image": image_b64,
        }
    ).encode("utf-8")

    request = Request(
        config.ai_backend_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=config.ai_backend_timeout_seconds) as response:
        response_body = response.read().decode("utf-8")

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        LOGGER.warning("AI backend returned non-JSON response")
        return response_body.strip()

    if isinstance(parsed, str):
        return parsed.strip()

    if isinstance(parsed, dict):
        for key in ("response", "text", "answer", "result", "message"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value.strip()

    return json.dumps(parsed, ensure_ascii=False)


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_analysoi(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        if message is None:
            return

        track_active_chat(update, config.storage_dir)

        target_photo = _extract_target_photo(message)
        if target_photo is None:
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

        tg_file = await context.bot.get_file(target_photo.file_id)
        photo_data = await tg_file.download_as_bytearray()
        photo_bytes = bytes(photo_data)

        if len(photo_bytes) > config.analysoi_max_image_bytes:
            await reply_in_chunks(
                update,
                "Kuva on liian suuri !analysoi-käsittelyyn.",
                config.max_reply_length,
            )
            return

        image_b64 = encode_image_base64(photo_bytes)
        prompt = build_analysoi_prompt()

        try:
            ai_reply = await asyncio.to_thread(
                _query_ai_with_image, image_b64, prompt, config
            )
        except (URLError, socket_timeout, TimeoutError, OSError):
            LOGGER.exception("Failed to query AI backend for !analysoi")
            ai_reply = (
                "AI-taustajärjestelmä ei ole juuri nyt käytettävissä. "
                "Yritä myöhemmin uudelleen."
            )

        if not ai_reply:
            ai_reply = "AI-taustajärjestelmä palautti tyhjän vastauksen."

        await reply_in_chunks(update, ai_reply, config.max_reply_length)

    return handle_analysoi


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            (filters.PHOTO & filters.CaptionRegex(_ANALYSOI_REGEX))
            | (filters.REPLY & filters.Regex(_ANALYSOI_REGEX)),
            _build_handler(config),
        )
    )
