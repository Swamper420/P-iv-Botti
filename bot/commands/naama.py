from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from io import BytesIO

from telegram import InputFile, PhotoSize, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.message_utils import reply_in_chunks
from bot.commands.naama_logic import compose_naama_image
from bot.config import BotConfig

COMMAND_USAGE = "!naama [kuva tai vastaus kuvaan]"

LOGGER = logging.getLogger(__name__)
_NAAMA_REGEX = r"(?i)^\s*!naama\s*$"


def _extract_target_photo(message: object) -> PhotoSize | None:
    photos = getattr(message, "photo", None)
    caption = getattr(message, "caption", None)
    if photos and isinstance(caption, str) and caption.strip().lower() == "!naama":
        return photos[-1]

    text = getattr(message, "text", None)
    reply_message = getattr(message, "reply_to_message", None)
    reply_photos = getattr(reply_message, "photo", None)
    if isinstance(text, str) and text.strip().lower() == "!naama" and reply_photos:
        return reply_photos[-1]

    return None


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_naama(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return

        track_active_chat(update, config.storage_dir)

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

        if len(photo_bytes) > config.naama_max_image_bytes:
            await reply_in_chunks(
                update,
                "Kuva on liian suuri !naama-käsittelyyn.",
                config.max_reply_length,
            )
            return

        processed = await asyncio.to_thread(
            compose_naama_image,
            photo_bytes,
            assets_dir=config.storage_dir / "naama",
            model_name=config.naama_model_name,
            confidence_threshold=config.naama_confidence_threshold,
            mask_threshold=config.naama_mask_threshold,
        )

        if not processed:
            LOGGER.error("!naama image processing failed")
            await reply_in_chunks(
                update,
                (
                    "Naama-kuvan käsittely epäonnistui. Varmista, että storage/naama sisältää "
                    "background*, hat*, suit*, gloves*, cigar* ja sun* -kuvia."
                ),
                config.max_reply_length,
            )
            return

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
