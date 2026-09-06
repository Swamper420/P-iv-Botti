from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from telegram import PhotoSize, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.aih_logic import stream_ollama_completion
from bot.commands.common import command_handler
from bot.commands.meme_logic import (
    build_meme_prompt,
    fallback_caption,
    parse_meme_caption,
    parse_meme_command,
    render_meme_on_photo,
    sanitize_caption_line,
)
from bot.commands.message_utils import reply_in_chunks, reply_with_image
from bot.commands.tiivista_logic import extract_text_with_ocr, recognize_objects_with_yolo
from bot.config import BotConfig

COMMAND_USAGE = (
    "!meme [vihje] - Luo meemi kuvasta. Lähetä kuva captionilla "
    "!meme [vihje] tai vastaa kuvaan komennolla !meme [vihje].\n"
    "Botti tunnistaa kuvan sisällön (YOLO + OCR) ja keksii siihen "
    "paikallisella tekoälyllä (Ollama) hauskan ylä- ja alatekstin."
)

LOGGER = logging.getLogger(__name__)
_MEME_REGEX = r"(?i)^\s*!meme\b"


def _extract_target_photo(message: object) -> tuple[PhotoSize | None, str]:
    """Return (photo, hint) from a photo caption or a reply to a photo."""
    photos = getattr(message, "photo", None)
    caption = getattr(message, "caption", None)
    if photos and isinstance(caption, str):
        is_match, hint = parse_meme_command(caption)
        if is_match:
            return photos[-1], hint

    text = getattr(message, "text", None)
    reply_message = getattr(message, "reply_to_message", None)
    reply_photos = getattr(reply_message, "photo", None)
    if isinstance(text, str) and reply_photos:
        is_match, hint = parse_meme_command(text)
        if is_match:
            return reply_photos[-1], hint

    return None, ""


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_meme(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return

        target_photo, hint = _extract_target_photo(message)
        if target_photo is None:
            await reply_in_chunks(update, COMMAND_USAGE, config.max_reply_length)
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
            )

        try:
            tg_file = await context.bot.get_file(target_photo.file_id)
            photo_data = await tg_file.download_as_bytearray()
            photo_bytes = bytes(photo_data)
        except Exception:
            LOGGER.exception("!meme: failed to download photo")
            await reply_in_chunks(
                update, "Virhe kuvan lataamisessa.", config.max_reply_length
            )
            return

        if len(photo_bytes) > config.meme.max_image_bytes:
            await reply_in_chunks(
                update,
                "Kuva on liian suuri käsittelyyn.",
                config.max_reply_length,
            )
            return

        # Local perception: YOLO object detection + Tesseract OCR (same stack as !tiivistä).
        yolo_task = asyncio.to_thread(
            recognize_objects_with_yolo,
            photo_bytes,
            model_name=config.meme.yolo_model,
            confidence_threshold=config.meme.yolo_confidence_threshold,
        )
        if config.meme.ocr_enabled:
            ocr_task = asyncio.to_thread(
                extract_text_with_ocr,
                photo_bytes,
                tesseract_cmd=config.meme.ocr_tesseract_cmd,
                lang=config.meme.ocr_language,
                tessdata_dir=config.meme.ocr_tessdata_dir,
                timeout_seconds=config.meme.ocr_timeout_seconds,
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
            LOGGER.exception("!meme: YOLO recognition failed", exc_info=yolo_res)
            image_description = ""
        else:
            image_description = yolo_res or ""

        if isinstance(ocr_res, Exception):
            LOGGER.exception("!meme: OCR failed", exc_info=ocr_res)
            ocr_text = ""
        else:
            ocr_text = ocr_res or ""

        prompt = build_meme_prompt(
            image_description=image_description,
            ocr_text=ocr_text,
            user_hint=hint,
        )
        if not prompt:
            await reply_in_chunks(
                update,
                "Kuvasta ei saatu tunnistetta (ei kohteita, ei tekstiä) "
                "eikä vihjettä annettu. Yritä toista kuvaa tai lisää vihje: "
                "!meme <vihje>.",
                config.max_reply_length,
            )
            return

        # Local captioning via Ollama; fall back to perception-only caption.
        raw_caption = ""
        try:
            chunks: list[str] = []
            async for chunk in stream_ollama_completion(
                base_url=config.ollama.base_url,
                model=config.ollama.model,
                prompt=prompt,
                num_predict=config.meme.caption_num_predict,
                num_ctx=config.ollama.num_ctx,
                timeout_seconds=config.ollama.timeout_seconds,
                system_prompt=config.meme.system_prompt,
            ):
                chunks.append(chunk)
            raw_caption = "".join(chunks).strip()
        except Exception:
            LOGGER.exception("!meme: Ollama caption generation failed")

        top_text, bottom_text = parse_meme_caption(raw_caption)
        top_text = sanitize_caption_line(top_text, config.meme.max_top_chars)
        bottom_text = sanitize_caption_line(bottom_text, config.meme.max_bottom_chars)
        if not top_text and not bottom_text:
            if raw_caption:
                LOGGER.warning("!meme: unparsable Ollama caption, using fallback")
            else:
                LOGGER.warning("!meme: empty Ollama caption, using fallback")
            top_text, bottom_text = fallback_caption(
                image_description=image_description,
                ocr_text=ocr_text,
                user_hint=hint,
                max_top_chars=config.meme.max_top_chars,
                max_bottom_chars=config.meme.max_bottom_chars,
            )

        meme_bytes = await asyncio.to_thread(
            render_meme_on_photo,
            photo_bytes,
            top_text,
            bottom_text,
            output_max_width=config.meme.output_max_width,
            jpeg_quality=config.meme.jpeg_quality,
        )
        if not meme_bytes:
            LOGGER.error("!meme: rendering failed")
            await reply_in_chunks(
                update,
                "Meemin renderöinti epäonnistui.",
                config.max_reply_length,
            )
            return

        caption = f"{top_text} / {bottom_text}" if bottom_text else top_text
        fallback_lines = [caption]
        if image_description:
            fallback_lines.append(image_description)
        if ocr_text:
            fallback_lines.append(ocr_text)
        await reply_with_image(
            update,
            image=meme_bytes,
            caption=caption or None,
            fallback_text="\n".join(fallback_lines),
            filename="meme.jpg",
            max_reply_length=config.max_reply_length,
        )

    return handle_meme


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(_MEME_REGEX)
            | (filters.PHOTO & filters.CaptionRegex(_MEME_REGEX))
            | (filters.REPLY & filters.Regex(_MEME_REGEX)),
            _build_handler(config),
        )
    )
