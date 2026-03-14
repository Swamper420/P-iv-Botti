from __future__ import annotations

import asyncio
import json
import logging
from socket import timeout as socket_timeout
from urllib.error import URLError
from urllib.request import Request, urlopen

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.paivaa_logic import get_aih_prompt, get_paivaa_reply, split_message

LOGGER = logging.getLogger(__name__)
AI_BACKEND_URL = "http://127.0.0.1:8080/query"
MAX_REPLY_LENGTH = 5000
AI_MAX_TOKENS = 650
AI_BACKEND_TIMEOUT_SECONDS = 30


async def _reply_in_chunks(update: Update, reply: str) -> None:
    message = update.effective_message
    if message is None:
        return

    for chunk in split_message(reply, MAX_REPLY_LENGTH):
        await message.reply_text(chunk)


def _query_ai_backend(prompt: str) -> str:
    payload = json.dumps({"prompt": prompt, "max_tokens": AI_MAX_TOKENS}).encode("utf-8")
    request = Request(
        AI_BACKEND_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=AI_BACKEND_TIMEOUT_SECONDS) as response:
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


async def handle_paivaa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context

    message = update.effective_message
    if message is None:
        return

    reply = get_paivaa_reply(message.text)
    if reply is not None:
        await _reply_in_chunks(update, reply)
        return

    prompt = get_aih_prompt(message.text)
    if prompt is None:
        return

    try:
        ai_reply = await asyncio.to_thread(_query_ai_backend, prompt)
    except (URLError, socket_timeout, TimeoutError, OSError):
        LOGGER.exception("Failed to query AI backend")
        ai_reply = "AI backend is temporarily unavailable. Please try again later."

    if not ai_reply:
        ai_reply = "AI backend returned an empty response."

    await _reply_in_chunks(update, ai_reply)


def register(application: Application) -> None:
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_paivaa))
