from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from io import BytesIO

from telegram import InputFile, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.message_utils import reply_in_chunks
from bot.commands.weather_logic import (
    get_openweather_summary,
    get_weather_cam_data,
    parse_weather_camera_location,
)
from bot.config import BotConfig

COMMAND_USAGE = "!sääkuva <kaupunki>"

def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_weather(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return

        matched, location = parse_weather_camera_location(message.text)
        if not matched:
            return

        if not location:
            await reply_in_chunks(
                update, "Käyttö: `!sääkuva <kaupunki>`", config.max_reply_length
            )
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

        weather_summary = await asyncio.to_thread(
            get_openweather_summary, location, config
        )
        img_data, result = await asyncio.to_thread(get_weather_cam_data, location, config)

        if img_data is None:
            await reply_in_chunks(update, f"⚠️ {result}", config.max_reply_length)
            return

        photo = InputFile(BytesIO(img_data), filename=result)
        await message.reply_photo(photo=photo)
        if weather_summary:
            await reply_in_chunks(update, weather_summary, config.max_reply_length)

    return handle_weather


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!(?:sääkuva|saakuva)(?:\s|$)"),
            _build_handler(config),
        )
    )
