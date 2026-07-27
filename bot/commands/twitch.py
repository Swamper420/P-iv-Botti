from __future__ import annotations

from collections.abc import Awaitable, Callable
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.twitch_logic import fetch_twitch_status_reply, parse_twitch_command
from bot.config import BotConfig

COMMAND_USAGE = "!twitch"


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_twitch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return

        matched, _ = parse_twitch_command(message.text)
        if not matched:
            return

        reply_text = await fetch_twitch_status_reply(config.twitch)
        await reply_in_chunks(update, reply_text, config.max_reply_length, parse_mode="HTML")

    return handle_twitch


def register(application: Application, config: BotConfig) -> None:
    handler = _build_handler(config)
    application.add_handler(
        MessageHandler(filters.Regex(r"^\s*!twitch(?:\s+|$)"), handler)
    )
