from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from io import BytesIO
from typing import Literal

from telegram import InputFile, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.meme_logic import generate_custom_meme, generate_meme_image, list_templates
from bot.config import BotConfig

COMMAND_USAGE = (
    "!meme [mallikey] [yläteksti] | [alateksti] - luo klassinen meme\n"
    "!meme list - listaa mallit\n"
    "!meme dev - kehittäjä-huumori\n"
    "!meme motivaatio - motivaatiomainen\n"
    "!meme custom <teksti> - vapaamuotoinen tekstikuva\n"
    "!memehelp - näytä tämä ohje"
)

LOGGER = logging.getLogger(__name__)
_MEME_REGEX = r"(?i)^\s*!(meme|memehelp)\b(.*)$"


def _parse_args(arg_string: str) -> tuple[str, list[str]]:
    arg_string = arg_string.strip()
    if not arg_string:
        return "random", []
    parts = arg_string.split()
    cmd = parts[0].lower()
    args = parts[1:]
    return cmd, args


def _build_handler(config: BotConfig) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_meme(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or message.text is None:
            return

        match = re.match(_MEME_REGEX, message.text)
        if not match:
            return

        cmd, args = _parse_args(match.group(2))

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
            )

        try:
            if cmd in ("help", "memehelp", ""):
                await reply_in_chunks(update, COMMAND_USAGE, config.max_reply_length)
                return

            if cmd == "list":
                templates = list_templates()
                text = "Saatavilla olevat meme-mallit:\n" + "\n".join(f"  {t}" for t in templates)
                await reply_in_chunks(update, text, config.max_reply_length)
                return

            style: Literal["classic", "motivational", "dev", "random"] = "random"
            template = "random"
            top_text = None
            bottom_text = None
            custom_text = None

            if cmd == "dev":
                style = "dev"
            elif cmd == "motivaatio" or cmd == "motivation":
                style = "motivational"
            elif cmd == "custom":
                custom_text = " ".join(args) if args else "Anna tekstiä!"
            elif cmd in _MEME_REGEX:
                template = cmd
            else:
                template = cmd
                if args:
                    full_text = " ".join(args)
                    if "|" in full_text:
                        top_text, bottom_text = full_text.split("|", 1)
                        top_text = top_text.strip()
                        bottom_text = bottom_text.strip()
                    else:
                        top_text = full_text

            if custom_text:
                img_bytes = await asyncio.to_thread(
                    generate_custom_meme,
                    custom_text,
                    width=800,
                    height=500,
                    style="impact",
                )
            else:
                img_bytes = await asyncio.to_thread(
                    generate_meme_image,
                    template=template,
                    top_text=top_text,
                    bottom_text=bottom_text,
                    style=style,
                    width=800,
                )

            await message.reply_photo(photo=InputFile(BytesIO(img_bytes), filename="meme.png"))

        except Exception as e:
            LOGGER.exception("Meme generation failed")
            await reply_in_chunks(
                update,
                f"Meme-generointi epäonnistui: {e}",
                config.max_reply_length,
            )

    return handle_meme


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(_MEME_REGEX),
            _build_handler(config),
        )
    )