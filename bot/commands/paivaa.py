from __future__ import annotations

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.paivaa_logic import get_paivaa_reply, split_message
from bot.config import BotConfig


async def _reply_in_chunks(update: Update, reply: str, max_reply_length: int) -> None:
    message = update.effective_message
    if message is None:
        return

    for chunk in split_message(reply, max_reply_length):
        await message.reply_text(chunk)


def _build_handler(config: BotConfig):
    async def handle_paivaa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context

        message = update.effective_message
        if message is None:
            return

        reply = get_paivaa_reply(message.text)
        if reply is not None:
            await _reply_in_chunks(update, reply, config.max_reply_length)

    return handle_paivaa


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _build_handler(config))
    )
