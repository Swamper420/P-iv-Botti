from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from io import BytesIO

from telegram import InputFile, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.tts_logic import (
    parse_tts_command,
    reencode_audio_for_telegram,
    synthesize_speech,
)
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)

COMMAND_USAGE = "*ääni* puhuu: <teksti> | puhuu: <teksti> | *ääni* puhuu selkeästi: <teksti>"


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_tts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not message.text:
            return

        is_match, voice, text = parse_tts_command(message.text)
        if not is_match:
            return

        if not text:
            await reply_in_chunks(
                update,
                "Käyttö: [ääni] puhuu: <teksti>\nEsimerkki: Matti puhuu: Terve maailma!",
                config.max_reply_length,
            )
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE
            )

        try:
            audio_bytes = await synthesize_speech(
                base_url=config.tts.base_url,
                text=text,
                voice=voice,
                fmt=config.tts.format,
                language=config.tts.language,
                speed=config.tts.speed,
                num_step=config.tts.num_step,
                guidance_scale=config.tts.guidance_scale,
                timeout_seconds=config.tts.timeout_seconds,
            )
            if not audio_bytes:
                LOGGER.error("TTS synthesis returned empty audio bytes.")
                await reply_in_chunks(
                    update, config.tts.error_message, config.max_reply_length
                )
                return

            if config.tts.reencode_audio:
                audio_bytes = await reencode_audio_for_telegram(
                    audio_bytes,
                    ffmpeg_path=config.tts.ffmpeg_path,
                    sample_rate=config.tts.audio_sample_rate,
                    channels=config.tts.audio_channels,
                    bitrate=config.tts.audio_bitrate,
                    timeout_seconds=config.tts.timeout_seconds,
                )

            voice_file = InputFile(BytesIO(audio_bytes), filename="speech.ogg")
            await message.reply_voice(voice=voice_file)
        except Exception:
            LOGGER.exception("Error during TTS speech synthesis")
            await reply_in_chunks(
                update, config.tts.error_message, config.max_reply_length
            )

    return handle_tts


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!?(?:[^\n:]+\s+)?puhuu(?:\s+selkeästi)?:"),
            _build_handler(config),
        )
    )

