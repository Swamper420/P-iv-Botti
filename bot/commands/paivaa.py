from __future__ import annotations

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.paivaa_logic import get_paivaa_reply


async def handle_paivaa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context

    message = update.effective_message
    if message is None:
        return

    reply = get_paivaa_reply(message.text)
    if reply is None:
        return

    await message.reply_text(reply)


def register(application: Application) -> None:
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_paivaa))
