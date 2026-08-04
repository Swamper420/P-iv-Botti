from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.stt_logic import format_stt_response, transcribe_audio
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_stt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return

        media = None
        filename = "audio.ogg"

        if message.voice:
            media = message.voice
            filename = "voice.ogg"
        elif message.video_note:
            media = message.video_note
            filename = "video_note.mp4"
        elif message.video:
            media = message.video
            filename = getattr(message.video, "file_name", None) or "video.mp4"

        if media is None:
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

        tg_file = await context.bot.get_file(media.file_id)
        file_data = await tg_file.download_as_bytearray()
        file_bytes = bytes(file_data)

        transcribed_text = await transcribe_audio(
            file_bytes=file_bytes,
            filename=filename,
            config=config.stt,
        )

        if not transcribed_text:
            LOGGER.debug("No STT text generated for incoming media message.")
            return

        reply_text = format_stt_response(transcribed_text)
        if reply_text:
            await reply_in_chunks(update, reply_text, config.max_reply_length)

    return handle_stt


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.VOICE | filters.VIDEO_NOTE | filters.VIDEO,
            _build_handler(config),
        )
    )
