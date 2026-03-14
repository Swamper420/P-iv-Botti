from __future__ import annotations

from telegram import Update

from bot.commands.paivaa_logic import split_message


async def reply_in_chunks(update: Update, reply: str, max_reply_length: int) -> None:
    message = update.effective_message
    if message is None:
        return

    for chunk in split_message(reply, max_reply_length):
        await message.reply_text(chunk)
