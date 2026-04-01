from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.message_utils import reply_in_chunks
from bot.config import BotConfig
from bot.commands.hoi_logic import add_users, list_all, ping_list, remove_users

COMMAND_USAGE = "!hoi | !hoi <lista> | !hoi @kayttaja <lista> | !hoijaa @kayttaja <lista>"


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_hoi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context

        message = update.effective_message
        chat = update.effective_chat
        if not message or not message.text or not chat:
            return

        track_active_chat(update, config.storage_dir)

        text = message.text.strip()
        match = re.match(r"(?i)^!(hoi|hoijaa)(?:\s+(.+))?$", text)
        if not match:
            return

        command = match.group(1).lower()
        args_str = match.group(2)

        chat_id = chat.id
        reply_text = ""

        if command == "hoi":
            if not args_str:
                reply_text = list_all(config.storage_dir, chat_id)
            else:
                args = args_str.split()
                if len(args) == 1:
                    # e.g., !hoi listname
                    reply_text = ping_list(config.storage_dir, chat_id, args[0])
                else:
                    # e.g., !hoi @user listname or !hoi @user1 @user2 listname
                    list_name = args[-1]
                    users = args[:-1]
                    reply_text = add_users(config.storage_dir, chat_id, list_name, users)

        elif command == "hoijaa":
            if not args_str:
                reply_text = "Käyttö: !hoijaa @kayttaja [lista]"
            else:
                args = args_str.split()
                if len(args) < 2:
                    reply_text = (
                        "Virhe: Määritä vähintään yksi poistettava käyttäjä ja lista.\n"
                        "Esim: !hoijaa @kayttaja listanimi"
                    )
                else:
                    list_name = args[-1]
                    users = args[:-1]
                    reply_text = remove_users(config.storage_dir, chat_id, list_name, users)

        if reply_text:
            await reply_in_chunks(update, reply_text, config.max_reply_length)

    return handle_hoi


def register(application: Application, config: BotConfig) -> None:
    # Captures !hoi and !hoijaa along with any trailing arguments
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!(hoi|hoijaa)\b"),
            _build_handler(config)
        )
    )
