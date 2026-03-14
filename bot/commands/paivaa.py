from __future__ import annotations

from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.message_utils import reply_in_chunks
from bot.commands.paivaa_logic import get_paivaa_reply
from bot.config import BotConfig

COMMAND_USAGE = "päivää"


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_paivaa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context

        message = update.effective_message
        if message is None:
            return

        track_active_chat(update, config.storage_dir)

        reply = get_paivaa_reply(message.text)
        if reply is not None:
            await reply_in_chunks(update, reply, config.max_reply_length)

    return handle_paivaa


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(filters.Regex(r"(?i)^\s*päivää\s*$"), _build_handler(config))
    )
