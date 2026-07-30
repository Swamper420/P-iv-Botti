from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from io import BytesIO

from telegram import InputFile, PhotoSize, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.paranna_logic import upscale_image
from bot.config import BotConfig

COMMAND_USAGE = "!paranna [kuva tai vastaus kuvaan]"

LOGGER = logging.getLogger(__name__)
_PARANNA_REGEX = r"(?i)^\s*!paranna\s*$"


def _extract_target_photo(message: object) -> PhotoSize | None:
    photos = getattr(message, "photo", None)
    caption = getattr(message, "caption", None)
    if photos and isinstance(caption, str):
        if re.match(_PARANNA_REGEX, caption):
            return photos[-1]

    text = getattr(message, "text", None)
    reply_message = getattr(message, "reply_to_message", None)
    reply_photos = getattr(reply_message, "photo", None)
    if isinstance(text, str) and reply_photos:
        if re.match(_PARANNA_REGEX, text):
            return reply_photos[-1]

    return None


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_paranna(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return

        target_photo = _extract_target_photo(message)
        if target_photo is None:
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
            )

        tg_file = await context.bot.get_file(target_photo.file_id)
        photo_data = await tg_file.download_as_bytearray()
        photo_bytes = bytes(photo_data)

        paranna_cfg = config.paranna
        if len(photo_bytes) > paranna_cfg.max_image_bytes:
            await reply_in_chunks(
                update,
                "Kuva on liian suuri käsittelyyn.",
                config.max_reply_length,
            )
            return

        processed = await asyncio.to_thread(
            upscale_image,
            photo_bytes,
            model_path=paranna_cfg.model_path,
            model_url=paranna_cfg.model_url,
            tile_size=paranna_cfg.tile_size,
            tile_pad=paranna_cfg.tile_pad,
            max_input_dimension=paranna_cfg.max_input_dimension,
            max_output_dimension=paranna_cfg.max_output_dimension,
            jpeg_quality=paranna_cfg.jpeg_quality,
        )

        if not processed:
            LOGGER.error("!paranna image processing failed")
            await reply_in_chunks(
                update,
                "Kuvan parantaminen epäonnistui.",
                config.max_reply_length,
            )
            return

        await message.reply_photo(
            photo=InputFile(BytesIO(processed), filename="paranna.jpg")
        )

    return handle_paranna


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            (filters.PHOTO & filters.CaptionRegex(_PARANNA_REGEX))
            | (filters.REPLY & filters.Regex(_PARANNA_REGEX)),
            _build_handler(config),
        )
    )
