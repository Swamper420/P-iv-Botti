from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from io import BytesIO

from telegram import InputFile, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks, reply_with_card
from bot.commands.weather_logic import (
    build_weather_card,
    build_weather_error_card,
    build_weather_fallback_text,
    get_openweather_details,
    get_weather_cam_details,
    parse_weather_camera_location,
)
from bot.config import BotConfig

COMMAND_USAGE = "!sääkuva <kaupunki> [kulma]"

LOGGER = logging.getLogger(__name__)


def _safe_filename(camera_id: str | None, fallback: str = "saakuva") -> str:
    if camera_id:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", camera_id).strip("_")
        if cleaned:
            return f"saakuva-{cleaned}.png"
    safe_query = re.sub(r"[^A-Za-z0-9_-]+", "_", fallback).strip("_")
    if safe_query:
        return f"saakuva-{safe_query}.png"
    return "saakuva.png"


def _display_query(location: str, angle: int | None) -> str:
    if angle is not None:
        return f"{location} {angle}"
    return location


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_weather(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return

        matched, location, angle = parse_weather_camera_location(message.text)
        if not matched:
            return

        if not location:
            await reply_in_chunks(
                update,
                "Käyttö: `!sääkuva <kaupunki> [kulma]` (esim. `!sääkuva Helsinki`, `!sääkuva Helsinki 2`)",
                config.max_reply_length,
            )
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
            )

        # Fetch camera(s) + weather in parallel (both are blocking urllib calls).
        cam_result, weather_info = await asyncio.gather(
            asyncio.to_thread(
                get_weather_cam_details, location, config.weather, angle
            ),
            asyncio.to_thread(get_openweather_details, location, config.weather),
        )

        query_for_text = _display_query(location, angle)

        # Both missing: render a compact error card (picture pipeline stays consistent).
        if cam_result.image_bytes is None and weather_info is None:
            error_text = cam_result.error or "Sijaintia ei löytynyt"
            try:
                error_card = build_weather_error_card(error_text, query_for_text)
                await reply_with_card(
                    update,
                    card=error_card,
                    fallback_text=f"⚠️ {error_text}",
                    filename=_safe_filename(cam_result.camera_id, query_for_text),
                    max_reply_length=config.max_reply_length,
                )
            except Exception:
                LOGGER.exception("Weather error-card rendering failed")
                await reply_in_chunks(
                    update, f"⚠️ {error_text}", config.max_reply_length
                )
            return

        # Build the pleasing informative composite card (photo grid + weather facts).
        try:
            card = build_weather_card(location, cam=cam_result, weather=weather_info)
            fallback_text = build_weather_fallback_text(
                location, cam=cam_result, weather=weather_info
            )
        except Exception:
            LOGGER.exception("Weather card building failed")
            # Graceful degradation to legacy behaviour: raw photo + error note.
            if cam_result.image_bytes is not None:
                photo = InputFile(
                    BytesIO(cam_result.image_bytes),
                    filename=f"{cam_result.camera_id or 'saakuva'}.jpg",
                )
                await message.reply_photo(photo=photo)
            else:
                await reply_in_chunks(
                    update,
                    f"⚠️ {cam_result.error or 'Sääkuvan haku epäonnistui'}",
                    config.max_reply_length,
                )
            return

        try:
            await reply_with_card(
                update,
                card=card,
                fallback_text=fallback_text,
                filename=_safe_filename(cam_result.camera_id, query_for_text),
                max_reply_length=config.max_reply_length,
            )
        except Exception:
            LOGGER.exception("Weather card sending failed, falling back to raw photo")
            # Last-resort fallback: raw camera photo so the user still gets the image.
            if cam_result.image_bytes is not None:
                try:
                    photo = InputFile(
                        BytesIO(cam_result.image_bytes),
                        filename=f"{cam_result.camera_id or 'saakuva'}.jpg",
                    )
                    await message.reply_photo(photo=photo, caption=fallback_text)
                except Exception:
                    LOGGER.exception("Weather raw-photo fallback also failed")
                    await reply_in_chunks(update, fallback_text, config.max_reply_length)
            else:
                await reply_in_chunks(update, fallback_text, config.max_reply_length)

    return handle_weather


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!(?:sääkuva|saakuva)(?:\s|$)"),
            _build_handler(config),
        )
    )
