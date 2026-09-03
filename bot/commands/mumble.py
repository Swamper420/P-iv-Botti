from __future__ import annotations

from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.mumble_logic import handle_mumble_command, parse_mumble_command
from bot.config import BotConfig
from bot.tasks.mumble import get_mumble_manager

COMMAND_USAGE = "!mumble | !mumble <käyttäjä>"


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_mumble(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        if message is None or not message.text:
            return

        is_match, _, _ = parse_mumble_command(message.text)
        if not is_match:
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

        manager = get_mumble_manager(context.application)
        reply_text = await handle_mumble_command(manager, config.mumble, message.text)

        if reply_text:
            await reply_in_chunks(
                update, reply_text, config.max_reply_length, parse_mode="HTML"
            )

    return handle_mumble


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!mumble(?:\s+.*|$)"),
            _build_handler(config),
        )
    )
