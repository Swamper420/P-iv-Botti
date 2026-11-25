from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.niilo_logic import (
    detect_person_in_image,
    detect_person_in_video,
    format_niilo_detection_reply,
)
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)

# Run in a separate handler group so passive detection never blocks !naama, !meme,
# !tiivista, !paranna, or passive STT handlers in group 0.
NIILO_HANDLER_GROUP = 1

MediaKind = Literal["image", "video"]


def _guess_suffix(file_name: str | None, default: str) -> str:
    if file_name:
        suffix = Path(file_name).suffix.lower()
        if suffix and len(suffix) <= 10:
            return suffix
    return default


def _extract_media_target(message: Any) -> tuple[MediaKind, Any, str] | None:
    photos = getattr(message, "photo", None)
    if photos:
        return "image", photos[-1], ".jpg"

    video = getattr(message, "video", None)
    if video is not None:
        return "video", video, _guess_suffix(getattr(video, "file_name", None), ".mp4")

    video_note = getattr(message, "video_note", None)
    if video_note is not None:
        return "video", video_note, ".mp4"

    animation = getattr(message, "animation", None)
    if animation is not None:
        return (
            "video",
            animation,
            _guess_suffix(getattr(animation, "file_name", None), ".mp4"),
        )

    document = getattr(message, "document", None)
    if document is not None:
        mime = (getattr(document, "mime_type", None) or "").lower()
        file_name = getattr(document, "file_name", None)
        if mime == "image/gif":
            return "video", document, _guess_suffix(file_name, ".gif")
        if mime.startswith("image/"):
            return "image", document, _guess_suffix(file_name, ".jpg")
        if mime.startswith("video/"):
            return "video", document, _guess_suffix(file_name, ".mp4")

    return None


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    # Serialize CPU face-recognition jobs so bursts/albums don't saturate CPU/RAM.
    semaphore = asyncio.Semaphore(1)

    @command_handler(config)
    async def handle_niilo_detection(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        niilo_cfg = config.niilo
        if not niilo_cfg.enabled:
            return

        message = update.effective_message
        if message is None:
            return

        target = _extract_media_target(message)
        if target is None:
            return

        media_kind, media, suffix = target
        max_bytes = (
            niilo_cfg.max_image_bytes
            if media_kind == "image"
            else niilo_cfg.max_video_bytes
        )

        declared_size = getattr(media, "file_size", None)
        if isinstance(declared_size, int) and declared_size > max_bytes:
            LOGGER.debug(
                "Skipping %s for %s detection: declared size %d exceeds limit %d.",
                media_kind,
                niilo_cfg.person_name,
                declared_size,
                max_bytes,
            )
            return

        tg_file = await context.bot.get_file(media.file_id)
        file_bytes = bytes(await tg_file.download_as_bytearray())
        if len(file_bytes) > max_bytes:
            LOGGER.debug(
                "Skipping %s for %s detection: downloaded size %d exceeds limit %d.",
                media_kind,
                niilo_cfg.person_name,
                len(file_bytes),
                max_bytes,
            )
            return

        try:
            async with semaphore:
                if media_kind == "image":
                    result = await asyncio.to_thread(
                        detect_person_in_image,
                        file_bytes,
                        niilo_cfg,
                    )
                else:
                    result = await asyncio.to_thread(
                        detect_person_in_video,
                        file_bytes,
                        niilo_cfg,
                        suffix=suffix,
                    )
        except ImportError:
            LOGGER.error(
                "Face recognition dependencies (insightface, onnxruntime, opencv-python-headless) "
                "are not installed; skipping %s detection.",
                niilo_cfg.person_name,
            )
            return
        except Exception:
            LOGGER.exception(
                "Failed during %s face detection on %s.",
                niilo_cfg.person_name,
                media_kind,
            )
            return

        if result.found:
            LOGGER.info(
                "Detected %s in %s (similarity=%.3f, frame=%s, t=%s).",
                niilo_cfg.person_name,
                media_kind,
                result.similarity,
                result.frame_index,
                result.timestamp_seconds,
            )
            reply_text = format_niilo_detection_reply(niilo_cfg.person_name, result)
            await message.reply_text(reply_text)

    return handle_niilo_detection


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.PHOTO
            | filters.VIDEO
            | filters.VIDEO_NOTE
            | filters.ANIMATION
            | filters.Document.IMAGE
            | filters.Document.VIDEO,
            _build_handler(config),
        ),
        group=NIILO_HANDLER_GROUP,
    )
