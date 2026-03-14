from __future__ import annotations

from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.help_logic import build_help_reply
from bot.commands.message_utils import reply_in_chunks
from bot.config import BotConfig

COMMAND_USAGE = "!help"


def _build_handler(
    config: BotConfig, command_usages: tuple[str, ...]
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context

        track_active_chat(update, config.storage_dir)

        await reply_in_chunks(
            update, build_help_reply(command_usages), config.max_reply_length
        )

    return handle_help


def register(
    application: Application, config: BotConfig, command_usages: tuple[str, ...]
) -> None:
    application.add_handler(
        MessageHandler(filters.Regex(r"(?i)^\s*!help\s*$"), _build_handler(config, command_usages))
    )
