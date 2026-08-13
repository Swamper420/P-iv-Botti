from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from io import BytesIO

from telegram import InputFile, PhotoSize, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.tiivista_logic import (
    create_3_word_caption,
    extract_text_with_ocr,
    extract_urls,
    fetch_webpage_text,
    format_image_analysis_text,
    parse_tiivista_command,
    recognize_objects_with_yolo,
    summarize_text_with_ollama,
)
from bot.commands.tts_logic import reencode_audio_for_telegram, synthesize_speech
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)

COMMAND_USAGE = "!tiivistä <URL|teksti|kuva> | !tiivistä [ääni] <URL|teksti|kuva>"
_TIIVISTA_REGEX = r"(?i)^\s*!?tiivist(?:ä|a)\b"


def _extract_target_photo(message: object) -> PhotoSize | None:
    photos = getattr(message, "photo", None)
    if photos:
        return photos[-1]

    reply_message = getattr(message, "reply_to_message", None)
    reply_photos = getattr(reply_message, "photo", None)
    if reply_photos:
        return reply_photos[-1]

    return None


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_tiivista(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return

        cmd_text = message.text or message.caption or ""
        is_match, voice, content_or_url = parse_tiivista_command(cmd_text)
        if not is_match:
            return

        target_photo = _extract_target_photo(message)
        target_url: str | None = None
        raw_text: str = ""

        urls = extract_urls(content_or_url)
        if urls:
            target_url = urls[0]
        elif content_or_url:
            raw_text = content_or_url

        # Check reply_to_message if no photo/direct content/URL provided
        if not target_photo and not target_url and not raw_text and message.reply_to_message:
            replied_msg = message.reply_to_message
            replied_text = replied_msg.text or replied_msg.caption or ""
            replied_urls = extract_urls(replied_text)
            if replied_urls:
                target_url = replied_urls[0]
            elif replied_text.strip():
                raw_text = replied_text.strip()

        if not target_photo and not target_url and not raw_text:
            await reply_in_chunks(
                update,
                "Käyttö: !tiivistä <URL|teksti|kuva> tai vastaa viestiin komennolla !tiivistä.\n"
                "Esimerkki: !tiivistä https://yle.fi/a/74-20000000",
                config.max_reply_length,
            )
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

        source_text: str = ""
        if target_photo:
            try:
                tg_file = await context.bot.get_file(target_photo.file_id)
                photo_data = await tg_file.download_as_bytearray()
                photo_bytes = bytes(photo_data)
            except Exception:
                LOGGER.exception("Failed to download photo for tiivistä command")
                await reply_in_chunks(
                    update, "Virhe kuvan lataamisessa.", config.max_reply_length
                )
                return

            if len(photo_bytes) > config.tiivista.max_image_bytes:
                await reply_in_chunks(
                    update, "Kuva on liian suuri käsittelyyn.", config.max_reply_length
                )
                return

            yolo_task = asyncio.to_thread(
                recognize_objects_with_yolo,
                photo_bytes,
                model_name=config.tiivista.yolo_model,
                confidence_threshold=config.tiivista.yolo_confidence_threshold,
            )

            if config.tiivista.ocr_enabled:
                ocr_task = asyncio.to_thread(
                    extract_text_with_ocr,
                    photo_bytes,
                    tesseract_cmd=config.tiivista.ocr_tesseract_cmd,
                    lang=config.tiivista.ocr_language,
                    tessdata_dir=config.tiivista.ocr_tessdata_dir,
                    timeout_seconds=config.tiivista.ocr_timeout_seconds,
                )
                yolo_res, ocr_res = await asyncio.gather(
                    yolo_task, ocr_task, return_exceptions=True
                )
            else:
                try:
                    yolo_res = await yolo_task
                except Exception as exc:
                    yolo_res = exc
                ocr_res = ""

            if isinstance(yolo_res, Exception):
                LOGGER.exception(
                    "Error performing YOLO object recognition for tiivistä command",
                    exc_info=yolo_res,
                )
                image_description = ""
            else:
                image_description = yolo_res or ""

            if isinstance(ocr_res, Exception):
                LOGGER.exception(
                    "Error performing OCR for tiivistä command",
                    exc_info=ocr_res,
                )
                ocr_text = ""
            else:
                ocr_text = ocr_res or ""

            if not image_description and not ocr_text and not raw_text:
                await reply_in_chunks(
                    update, "Virhe kuvan tunnistuksessa.", config.max_reply_length
                )
                return

            source_text = format_image_analysis_text(
                image_description=image_description,
                ocr_text=ocr_text,
                additional_text=raw_text,
            )
        elif target_url:
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
            filters.Regex(_TIIVISTA_REGEX)
            | (filters.PHOTO & filters.CaptionRegex(_TIIVISTA_REGEX))
            | (filters.REPLY & filters.Regex(_TIIVISTA_REGEX)),
            _build_handler(config),
        )
    )
