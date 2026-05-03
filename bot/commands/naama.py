from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from io import BytesIO
from typing import Literal

from telegram import InputFile, PhotoSize, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.message_utils import reply_in_chunks
from bot.commands.naama_logic import compose_naama_image
from bot.config import BotConfig

COMMAND_USAGE = "!naama / !naamao / !naamav / !naamatarra / !naamatarrao / !naamatarrav [kuva tai vastaus kuvaan]"

LOGGER = logging.getLogger(__name__)
_NAAMA_REGEX = r"(?i)^\s*!(naama(?:o|v)?|naamatarra(?:o|v)?)\s*$"


def _extract_target_photo(message: object) -> tuple[PhotoSize | None, str | None]:
    photos = getattr(message, "photo", None)
    caption = getattr(message, "caption", None)
    if photos and isinstance(caption, str):
        match = re.match(_NAAMA_REGEX, caption)
        if match:
            return photos[-1], match.group(1).lower()

    text = getattr(message, "text", None)
    reply_message = getattr(message, "reply_to_message", None)
    reply_photos = getattr(reply_message, "photo", None)
    if isinstance(text, str) and reply_photos:
        match = re.match(_NAAMA_REGEX, text)
        if match:
            return reply_photos[-1], match.group(1).lower()

    return None, None


def _parse_command(cmd: str) -> tuple[Literal["mirror", "sticker"], Literal["auto", "left", "right"]]:
    action: Literal["mirror", "sticker"] = "mirror"
    side: Literal["auto", "left", "right"] = "auto"

    if cmd.startswith("naamatarra"):
        action = "sticker"
        suffix = cmd[len("naamatarra"):]
    else:
        action = "mirror"
        suffix = cmd[len("naama"):]

    if suffix == "o":
        side = "right"
    elif suffix == "v":
        side = "left"

    return action, side


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_naama(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return

        track_active_chat(update, config.storage_dir)

        target_photo, cmd_text = _extract_target_photo(message)
        if target_photo is None or cmd_text is None:
            return

        action, side = _parse_command(cmd_text)

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
            )

        tg_file = await context.bot.get_file(target_photo.file_id)
        photo_data = await tg_file.download_as_bytearray()
        photo_bytes = bytes(photo_data)

        if len(photo_bytes) > config.naama_max_image_bytes:
            await reply_in_chunks(
                update,
                "Kuva on liian suuri käsittelyyn.",
                config.max_reply_length,
            )
            return

        processed = await asyncio.to_thread(
            compose_naama_image,
            photo_bytes,
            model_name=config.naama_model_name,
            confidence_threshold=config.naama_confidence_threshold,
            mask_threshold=config.naama_mask_threshold,
            action=action,
            side=side,
        )

        if not processed:
            LOGGER.error("!naama image processing failed")
            await reply_in_chunks(
                update,
                "Kuvan käsittely epäonnistui. Kuvasta ei todennäköisesti löytynyt mitään tunnistettavaa.",
                config.max_reply_length,
            )
            return

        if action == "sticker":
            await message.reply_sticker(sticker=InputFile(BytesIO(processed), filename="naama.webp"))
        else:
            await message.reply_photo(photo=InputFile(BytesIO(processed), filename="naama.png"))

    return handle_naama


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            (filters.PHOTO & filters.CaptionRegex(_NAAMA_REGEX))
            | (filters.REPLY & filters.Regex(_NAAMA_REGEX)),
            _build_handler(config),
        )
    )
