from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.aih_logic import parse_aih_command, stream_ollama_completion
from bot.commands.common import command_handler
from bot.commands.message_utils import split_message
from bot.config import BotConfig

COMMAND_USAGE = "!aih | !aih<tokenit>: <prompti> | !aih <prompti>"
EDIT_INTERVAL_SECONDS = 0.8


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_aih(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context

        message = update.effective_message
        if not message or not message.text:
            return

        is_match, num_predict, strip_thinking, prompt = parse_aih_command(
            message.text,
            default_tokens=config.ollama.default_num_predict,
            max_tokens=config.ollama.max_num_predict,
            default_strip_thinking=config.ollama.strip_thinking,
        )

        if not is_match:
            return

        if not prompt:
            await message.reply_text(
                "Käyttö: !aih[tokenit]: <prompti>\nEsimerkki: !aih400: Kerro tarina"
            )
            return

        placeholder = await message.reply_text("⏳ Generoidaan vastausta...")
        accumulated_text = ""
        last_edit_time = asyncio.get_running_loop().time()
        last_displayed_text = ""

        try:
            async for chunk in stream_ollama_completion(
                base_url=config.ollama.base_url,
                model=config.ollama.model,
                prompt=prompt,
                num_predict=num_predict,
                timeout_seconds=config.ollama.timeout_seconds,
                strip_thinking=strip_thinking,
                system_prompt=config.ollama.system_prompt,
            ):


                accumulated_text += chunk
                now = asyncio.get_running_loop().time()

                if now - last_edit_time >= EDIT_INTERVAL_SECONDS:
                    display_text = accumulated_text[: config.max_reply_length]
                    if display_text != last_displayed_text:
                        await placeholder.edit_text(display_text)
                        last_displayed_text = display_text
                        last_edit_time = now

            # Stream finished - render final complete response
            if not accumulated_text.strip():
                await placeholder.edit_text("Ollama ei palauttanut vastausta.")
                return

            chunks = split_message(accumulated_text, config.max_reply_length)
            if chunks:
                if chunks[0] != last_displayed_text:
                    await placeholder.edit_text(chunks[0])
                for overflow_chunk in chunks[1:]:
                    await message.reply_text(overflow_chunk)

        except Exception as exc:
            error_msg = f"Virhe LLM-vastauksen haussa: {exc}"
            if last_displayed_text:
                await message.reply_text(error_msg)
            else:
                await placeholder.edit_text(error_msg)

    return handle_aih


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!?aih\b"),
            _build_handler(config),
        )
    )
