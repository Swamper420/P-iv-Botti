from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.telkkari_logic import get_channel_day_schedule, get_next_hour_schedule
from bot.config import BotConfig

COMMAND_USAGE = "!telkkari | !telkkari <kanavanumero>"


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_telkkari(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        if message is None or not message.text:
            return

        text = message.text.strip()
        match = re.match(r"(?i)^!telkkari(?:\s+(.+))?$", text)
        if not match:
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

        arg_str = match.group(1)

        if arg_str:
            cleaned_arg = arg_str.strip()
            if cleaned_arg.isdigit():
                ch_num = int(cleaned_arg)
                reply_text = await asyncio.to_thread(
                    get_channel_day_schedule, ch_num, config.telkkari
                )
            else:
                reply_text = (
                    f"⚠️ Virheellinen kanavanumero: '{cleaned_arg}'. "
                    "Anna kanavanumero pelkkänä lukuna (esim. !telkkari 1)."
                )
        else:
            reply_text = await asyncio.to_thread(
                get_next_hour_schedule, config.telkkari
            )

        if reply_text:
            await reply_in_chunks(update, reply_text, config.max_reply_length)

    return handle_telkkari


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!telkkari(?:\s+.*|$)"),
            _build_handler(config),
        )
    )
