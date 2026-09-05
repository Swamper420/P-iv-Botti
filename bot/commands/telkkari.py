from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks, reply_with_card
from bot.commands.telkkari_logic import (
    build_invalid_channel_arg_card,
    get_channel_day_schedule_card,
    get_next_hour_schedule_card,
)
from bot.config import BotConfig

COMMAND_USAGE = "!telkkari | !telkkari <kanavanumero>"

LOGGER = logging.getLogger(__name__)


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
                chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
            )

        arg_str = match.group(1)

        try:
            if arg_str:
                cleaned_arg = arg_str.strip()
                if cleaned_arg.isdigit():
                    ch_num = int(cleaned_arg)
                    reply_text, card = await asyncio.to_thread(
                        get_channel_day_schedule_card, ch_num, config.telkkari
                    )
                else:
                    reply_text = (
                        f"⚠️ Virheellinen kanavanumero: '{cleaned_arg}'. "
                        "Anna kanavanumero pelkkänä lukuna (esim. !telkkari 1)."
                    )
                    card = build_invalid_channel_arg_card(cleaned_arg)
            else:
                reply_text, card = await asyncio.to_thread(
                    get_next_hour_schedule_card, config.telkkari
                )
        except Exception:
            LOGGER.exception("Telkkari card building failed, falling back to text")
            # Last-resort: plain-text replies preserve previous behaviour.
            try:
                from bot.commands.telkkari_logic import (
                    get_channel_day_schedule,
                    get_next_hour_schedule,
                )

                if arg_str:
                    cleaned_arg = arg_str.strip()
                    if cleaned_arg.isdigit():
                        fallback = await asyncio.to_thread(
                            get_channel_day_schedule, int(cleaned_arg), config.telkkari
                        )
                    else:
                        fallback = (
                            f"⚠️ Virheellinen kanavanumero: '{cleaned_arg}'. "
                            "Anna kanavanumero pelkkänä lukuna (esim. !telkkari 1)."
                        )
                else:
                    fallback = await asyncio.to_thread(
                        get_next_hour_schedule, config.telkkari
                    )
            except Exception:
                LOGGER.exception("Telkkari text fallback also failed")
                return
            if fallback:
                await reply_in_chunks(update, fallback, config.max_reply_length)
            return

        if card is not None:
            await reply_with_card(
                update,
                card=card,
                fallback_text=reply_text,
                filename="telkkari.png",
                max_reply_length=config.max_reply_length,
            )
        elif reply_text:
            await reply_in_chunks(update, reply_text, config.max_reply_length)

    return handle_telkkari


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!telkkari(?:\s+.*|$)"),
            _build_handler(config),
        )
    )
