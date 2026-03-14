from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
from collections.abc import Awaitable, Callable
from socket import timeout as socket_timeout
from urllib.error import URLError
from urllib.request import Request, urlopen

from telegram import Audio, Update, Voice
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.message_utils import reply_in_chunks
from bot.commands.stt_logic import (
    is_audio_duration_allowed,
    parse_transcription_response,
)
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)


def _convert_to_pcm_base64(audio_bytes: bytes) -> str | None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "48000",
        "pipe:1",
    ]

    try:
        completed = subprocess.run(
            command,
            input=audio_bytes,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        LOGGER.exception("Audio conversion to PCM failed")
        return None

    if not completed.stdout:
        return None

    return base64.b64encode(completed.stdout).decode("ascii")


def _transcribe_audio(audio_bytes: bytes, config: BotConfig) -> str | None:
    pcm_base64 = _convert_to_pcm_base64(audio_bytes)
    if not pcm_base64:
        return None

    payload = json.dumps({"pcm_base64": pcm_base64}).encode("utf-8")
    request = Request(
        config.stt_backend_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=config.stt_timeout_seconds) as response:
        response_body = response.read().decode("utf-8")

    return parse_transcription_response(response_body)


def _extract_media(message: object) -> Voice | Audio | None:
    voice = getattr(message, "voice", None)
    if voice is not None:
        return voice
    return getattr(message, "audio", None)


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_transcription(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        if message is None:
            return

        track_active_chat(update, config.storage_dir)

        media = _extract_media(message)
        if media is None:
            return

        if not is_audio_duration_allowed(media.duration, config.stt_max_audio_seconds):
            return

        try:
            tg_file = await context.bot.get_file(media.file_id)
            audio_data = await tg_file.download_as_bytearray()
            transcript = await asyncio.to_thread(_transcribe_audio, bytes(audio_data), config)
        except (URLError, socket_timeout, TimeoutError, OSError):
            LOGGER.exception("Voice transcription failed")
            return

        if not transcript:
            return

        await reply_in_chunks(update, transcript, config.max_reply_length)

    return handle_transcription


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler((filters.VOICE | filters.AUDIO) & ~filters.COMMAND, _build_handler(config))
    )
