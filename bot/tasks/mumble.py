from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bot.config import BotConfig
from bot.tasks.mumble_logic import MumbleManager

if TYPE_CHECKING:
    from telegram.ext import Application

LOGGER = logging.getLogger(__name__)

_GLOBAL_MUMBLE_MANAGER: MumbleManager | None = None


def get_mumble_manager(application: Application | None = None) -> MumbleManager | None:
    """
    Retrieve the active MumbleManager instance, either from application.bot_data
    or the module-level fallback.
    """
    global _GLOBAL_MUMBLE_MANAGER
    if (
        application is not None
        and hasattr(application, "bot_data")
        and isinstance(application.bot_data, dict)
        and "mumble_manager" in application.bot_data
    ):
        return application.bot_data["mumble_manager"]
    return _GLOBAL_MUMBLE_MANAGER


def set_mumble_manager(manager: MumbleManager | None) -> None:
    """Set global MumbleManager (useful for testing or direct initialization)."""
    global _GLOBAL_MUMBLE_MANAGER
    _GLOBAL_MUMBLE_MANAGER = manager


class MumbleTask:
    def __init__(self, application: Application, config: BotConfig) -> None:
        self.application = application
        self.config = config
        self.manager = MumbleManager(config.mumble)

        # Store in bot_data if available and global reference
        if hasattr(self.application, "bot_data") and isinstance(self.application.bot_data, dict):
            self.application.bot_data["mumble_manager"] = self.manager
        set_mumble_manager(self.manager)

    def start(self) -> None:
        if self.config.mumble.is_configured:
            self.manager.start()
            LOGGER.info("Mumble background task started.")
        else:
            LOGGER.info("Mumble is not configured (MUMBLE_HOST empty). Task disabled.")

    async def stop(self) -> None:
        self.manager.stop()
        LOGGER.info("Mumble background task stopped.")


def register(application: Application, config: BotConfig) -> None:
    task = MumbleTask(application, config)

    prev_init = application.post_init
    prev_shutdown = application.post_shutdown

    async def post_init(app: Application) -> None:
        if prev_init is not None:
            await prev_init(app)
        task.start()

    async def post_shutdown(app: Application) -> None:
        if prev_shutdown is not None:
            await prev_shutdown(app)
        await task.stop()

    application.post_init = post_init
    application.post_shutdown = post_shutdown
