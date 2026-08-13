from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from telegram import Message, Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.common import command_handler
from bot.commands.message_utils import reply_in_chunks
from bot.commands.muistuta_logic import (
    add_reminder,
    cancel_reminder,
    list_reminders,
    parse_reminder_args,
)
from bot.config import BotConfig

COMMAND_USAGE = "!muistuta <aika> [päivä] [@käyttäjä] [viesti] | !muistutukset | !peruuta <id>"
_COMMAND_REGEX = r"(?i)^\s*!(muistuta|muistutukset|peruuta)(?:\s+(.+))?$"


def _extract_media(message: Message) -> dict[str, str] | None:
    targets = [message]
    if message.reply_to_message is not None:
        targets.append(message.reply_to_message)

    for msg in targets:
        if msg.photo:
            return {"file_id": msg.photo[-1].file_id, "media_type": "photo"}
        if msg.video:
            return {"file_id": msg.video.file_id, "media_type": "video"}
        if msg.document:
            return {"file_id": msg.document.file_id, "media_type": "document"}
        if msg.animation:
            return {"file_id": msg.animation.file_id, "media_type": "animation"}
        if msg.voice:
            return {"file_id": msg.voice.file_id, "media_type": "voice"}
        if msg.audio:
            return {"file_id": msg.audio.file_id, "media_type": "audio"}
        if msg.sticker:
            return {"file_id": msg.sticker.file_id, "media_type": "sticker"}
        if msg.video_note:
            return {"file_id": msg.video_note.file_id, "media_type": "video_note"}

    return None


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    @command_handler(config)
    async def handle_muistuta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context

        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return

        raw_text = message.text or message.caption or ""
        match = re.match(_COMMAND_REGEX, raw_text.strip())
        if not match:
            return

        cmd_name = match.group(1).lower()
        args_str = (match.group(2) or "").strip()

        if cmd_name == "muistutukset" or args_str.lower() in ("list", "lista"):
            reply_text = list_reminders(config.storage_dir, chat.id)
            await reply_in_chunks(update, reply_text, config.max_reply_length)
            return

        if cmd_name == "peruuta":
            if not args_str.isdigit():
                await reply_in_chunks(update, "Käyttö: !peruuta <id>", config.max_reply_length)
                return
            reply_text = cancel_reminder(config.storage_dir, chat.id, int(args_str))
            await reply_in_chunks(update, reply_text, config.max_reply_length)
            return

        # Check for subcommand !muistuta poista <id> or !muistuta del <id>
        sub_cancel_match = re.match(r"^(?:poista|del|cancel)\s+(\d+)$", args_str, re.IGNORECASE)
        if sub_cancel_match:
            reminder_id = int(sub_cancel_match.group(1))
            reply_text = cancel_reminder(config.storage_dir, chat.id, reminder_id)
            await reply_in_chunks(update, reply_text, config.max_reply_length)
            return

        due_at, targets, msg_text, err = parse_reminder_args(args_str)
        if err or due_at is None:
            await reply_in_chunks(update, err or "Virhe muistutuksen jäsennyksessä.", config.max_reply_length)
            return

        creator = ""
        if message.from_user:
            if message.from_user.username:
                creator = f"@{message.from_user.username}"
            else:
                creator = message.from_user.first_name or "Käyttäjä"

        media = _extract_media(message)

        _, reply_text = add_reminder(
            storage_dir=config.storage_dir,
            chat_id=chat.id,
            creator=creator,
            due_at=due_at,
            targets=targets,
            message=msg_text,
            media=media,
            max_per_chat=config.reminder.max_per_chat,
        )

        await reply_in_chunks(update, reply_text, config.max_reply_length)

    return handle_muistuta


def register(application: Application, config: BotConfig) -> None:
    pattern = r"(?i)^\s*!(muistuta|muistutukset|peruuta)\b"
    application.add_handler(
        MessageHandler(
            (filters.TEXT & filters.Regex(pattern)) | (filters.CAPTION & filters.CaptionRegex(pattern)),
            _build_handler(config),
        )
    )
