from __future__ import annotations

import asyncio
import json
import logging
from socket import timeout as socket_timeout
from urllib.error import URLError
from urllib.request import Request, urlopen

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.aih_logic import get_aih_prompt
from bot.commands.paivaa_logic import split_message
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)


async def _reply_in_chunks(update: Update, reply: str, max_reply_length: int) -> None:
    message = update.effective_message
    if message is None:
        return

    for chunk in split_message(reply, max_reply_length):
        await message.reply_text(chunk)


def _query_ai_backend(prompt: str, config: BotConfig) -> str:
    payload = json.dumps({"prompt": prompt, "max_tokens": config.ai_max_tokens}).encode(
        "utf-8"
    )
    request = Request(
        config.ai_backend_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=config.ai_backend_timeout_seconds) as response:
        response_body = response.read().decode("utf-8")

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        LOGGER.warning("AI backend returned non-JSON response")
        return response_body.strip()

    if isinstance(parsed, str):
        return parsed.strip()

    if isinstance(parsed, dict):
        for key in ("response", "text", "answer", "result", "message"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value.strip()

    return json.dumps(parsed, ensure_ascii=False)


def _build_handler(config: BotConfig):
    async def handle_aih(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context

        message = update.effective_message
        if message is None:
            return

        prompt = get_aih_prompt(message.text)
        if prompt is None:
            return

        try:
            ai_reply = await asyncio.to_thread(_query_ai_backend, prompt, config)
        except (URLError, socket_timeout, TimeoutError, OSError):
            LOGGER.exception("Failed to query AI backend")
            ai_reply = "AI backend is temporarily unavailable. Please try again later."

        if not ai_reply:
            ai_reply = "AI backend returned an empty response."

        await _reply_in_chunks(update, ai_reply, config.max_reply_length)

    return handle_aih


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _build_handler(config))
    )
