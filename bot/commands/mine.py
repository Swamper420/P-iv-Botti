from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.mine_logic import fetch_mine_status, parse_mine_command
from bot.config import BotConfig

COMMAND_USAGE = "!mine | !mine <palvelin>"


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_mine(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        if message is None or not message.text:
            return

        is_match, server_query = parse_mine_command(message.text)
        if not is_match:
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

        reply_text = await asyncio.to_thread(
            fetch_mine_status, config.crafty, server_query
        )

        if reply_text:
            await reply_in_chunks(update, reply_text, config.max_reply_length)

    return handle_mine


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!mine(?:\s+.*|$)"),
            _build_handler(config),
        )
    )
