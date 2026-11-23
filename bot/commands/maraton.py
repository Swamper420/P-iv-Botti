from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.maraton_logic import (
    handle_maraton_card_command,
    parse_maraton_command,
    render_maraton_command_image,
)
from bot.commands.message_utils import reply_in_chunks, reply_with_image
from bot.config import BotConfig

COMMAND_USAGE = "!maraton | !maraton osuudet | !maraton treeni"


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_maraton(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        if message is None or not message.text:
            return

        is_match, _ = parse_maraton_command(message.text)
        if not is_match:
            return

        if config.maraton.use_card:
            if update.effective_chat is not None:
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
                )
            reply_text, image_bytes = await asyncio.to_thread(
                render_maraton_command_image, config.maraton, message.text
            )
            if image_bytes is not None:
                await reply_with_image(
                    update,
                    image=image_bytes,
                    fallback_text=reply_text,
                    filename="maraton.png",
                    max_reply_length=config.max_reply_length,
                    parse_mode="HTML",
                )
                return

        reply_text, _ = handle_maraton_card_command(config.maraton, message.text)
        if reply_text:
            await reply_in_chunks(
                update, reply_text, config.max_reply_length, parse_mode="HTML"
            )

    return handle_maraton


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!maraton(?:\s+.*|$)"),
            _build_handler(config),
        )
    )
