from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from telegram import Update
from telegram.ext import ContextTypes

from bot.active_chats import track_active_chat
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)
HandlerFunc = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[Any]]
F = TypeVar("F", bound=HandlerFunc)


def command_handler(config: BotConfig, track_chat: bool = True) -> Callable[[F], F]:
    """Decorator to attach active chat tracking and exception handling to command handlers."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
            if track_chat:
                track_active_chat(update, config.storage_dir)
            try:
                return await func(update, context)
            except Exception:
                LOGGER.exception("Unhandled error executing command handler %s", func.__name__)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator
