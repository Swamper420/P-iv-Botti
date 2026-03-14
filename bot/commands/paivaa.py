from __future__ import annotations

import asyncio
import logging
from socket import timeout as socket_timeout
from urllib.error import URLError
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.aih import _query_ai_backend
from bot.commands.message_utils import reply_in_chunks
from bot.commands.paivaa_logic import (
    build_paivaa_ai_prompt,
    ensure_unique_paivaa_reply,
    get_paivaa_reply,
    load_recent_paivaa_replies,
    store_recent_paivaa_reply,
)
from bot.config import BotConfig

COMMAND_USAGE = "päivää"
LOGGER = logging.getLogger(__name__)


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_paivaa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context

        message = update.effective_message
        if message is None:
            return

        track_active_chat(update, config.storage_dir)

        trigger = get_paivaa_reply(message.text)
        if trigger is None:
            return

        recent_replies = load_recent_paivaa_replies(config.storage_dir)
        prompt = build_paivaa_ai_prompt(recent_replies)

        try:
            ai_reply = await asyncio.to_thread(_query_ai_backend, prompt, config)
        except (URLError, socket_timeout, TimeoutError, OSError):
            LOGGER.exception("Failed to query AI backend for päivä greeting")
            ai_reply = "Päiwää~ muru-kisu uwu, oon niin cringe et posket palaa nyaa >w<"

        reply = ensure_unique_paivaa_reply(ai_reply, recent_replies)
        try:
            store_recent_paivaa_reply(config.storage_dir, reply)
        except OSError:
            LOGGER.exception("Failed to persist päivä reply history")

        await reply_in_chunks(update, reply, config.max_reply_length)

    return handle_paivaa


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(filters.Regex(r"(?i)^\s*päivää\s*$"), _build_handler(config))
    )
