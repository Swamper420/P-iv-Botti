from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from io import BytesIO

from telegram import InputFile, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.tiivista_logic import (
    create_3_word_caption,
    extract_urls,
    fetch_webpage_text,
    parse_tiivista_command,
    summarize_text_with_ollama,
)
from bot.commands.tts_logic import reencode_audio_for_telegram, synthesize_speech
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)

COMMAND_USAGE = "!tiivistä <URL|teksti> | !tiivistä [ääni] <URL|teksti>"


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_tiivista(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not message.text:
            return

        is_match, voice, content_or_url = parse_tiivista_command(message.text)
        if not is_match:
            return

        target_url: str | None = None
        raw_text: str = ""

        urls = extract_urls(content_or_url)
        if urls:
            target_url = urls[0]
        elif content_or_url:
            raw_text = content_or_url

        # Check reply_to_message if no direct content or URL provided
        if not target_url and not raw_text and message.reply_to_message:
            replied_msg = message.reply_to_message
            replied_text = replied_msg.text or replied_msg.caption or ""
            replied_urls = extract_urls(replied_text)
            if replied_urls:
                target_url = replied_urls[0]
            elif replied_text.strip():
                raw_text = replied_text.strip()

        if not target_url and not raw_text:
            await reply_in_chunks(
                update,
                "Käyttö: !tiivistä <URL|teksti> tai vastaa viestiin komennolla !tiivistä.\n"
                "Esimerkki: !tiivistä https://yle.fi/a/74-20000000",
                config.max_reply_length,
            )
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

        source_text: str = ""
        if target_url:
            try:
                source_text = await fetch_webpage_text(
                    url=target_url,
                    user_agent=config.tiivista.user_agent,
                    timeout_seconds=config.tiivista.timeout_seconds,
                    max_bytes=config.tiivista.max_web_page_bytes,
                )
            except Exception:
                LOGGER.exception("Failed to fetch webpage text from URL '%s'", target_url)
                await reply_in_chunks(
                    update, "Virhe haettaessa verkkosivua.", config.max_reply_length
                )
                return
        else:
            source_text = raw_text

        if not source_text or not source_text.strip():
            await reply_in_chunks(
                update,
                "Ei löydetty tekstiä tiivistettäväksi.",
                config.max_reply_length,
            )
            return

        # Truncate source text if it exceeds maximum context length
        if len(source_text) > config.tiivista.max_text_length:
            source_text = source_text[: config.tiivista.max_text_length]

        try:
            summary = await summarize_text_with_ollama(
                base_url=config.ollama.base_url,
                model=config.ollama.model,
                text=source_text,
                num_predict=config.tiivista.summary_num_predict,
                num_ctx=config.ollama.num_ctx,
                timeout_seconds=config.ollama.timeout_seconds,
                system_prompt=config.tiivista.system_prompt,
            )
        except Exception:
            LOGGER.exception("Error summarizing text with Ollama")
            await reply_in_chunks(
                update, "Virhe tiivistelmän luonnissa.", config.max_reply_length
            )
            return

        if not summary or not summary.strip():
            await reply_in_chunks(
                update, "Virhe: tiivistelmä on tyhjä.", config.max_reply_length
            )
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE
            )

        try:
            audio_bytes = await synthesize_speech(
                base_url=config.tts.base_url,
                text=summary,
                voice=voice,
                fmt=config.tts.format,
                language=config.tts.language,
                speed=config.tts.speed,
                num_step=config.tts.num_step,
                guidance_scale=config.tts.guidance_scale,
                timeout_seconds=config.tts.timeout_seconds,
            )
            if not audio_bytes:
                LOGGER.error("TTS synthesis returned empty audio bytes for tiivistä command")
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

            caption = create_3_word_caption(summary, config.tiivista.caption_num_words)
            voice_file = InputFile(BytesIO(audio_bytes), filename="summary.ogg")
            await message.reply_voice(voice=voice_file, caption=caption)
        except Exception:
            LOGGER.exception("Error during TTS synthesis for tiivistä command")
            await reply_in_chunks(
                update, config.tts.error_message, config.max_reply_length
            )

    return handle_tiivista


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!?tiivist(?:ä|a)\b"),
            _build_handler(config),
        )
    )
