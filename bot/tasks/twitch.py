from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from bot.active_chats import load_active_chat_ids
from bot.config import BotConfig
from bot.tasks.twitch_logic import TwitchEventSubNotifier, TwitchStreamNotification

if TYPE_CHECKING:
    from telegram.ext import Application

LOGGER = logging.getLogger(__name__)


class TwitchTask:
    def __init__(self, application: Application, config: BotConfig) -> None:
        self.application = application
        self.config = config
        self.notifier = TwitchEventSubNotifier(
            config=config.twitch,
            on_stream_online=self._handle_stream_online,
        )
        self._task: asyncio.Task[None] | None = None

    async def _handle_stream_online(self, notification: TwitchStreamNotification) -> None:
        active_chat_ids = load_active_chat_ids(self.config.storage_dir)
        if not active_chat_ids:
            LOGGER.info("No active chat IDs stored. Skipping Twitch stream notification.")
            return

        message = notification.format_telegram_message()

        for chat_id in active_chat_ids:
            try:
                if notification.thumbnail_url:
                    await self.application.bot.send_photo(
                        chat_id=chat_id,
                        photo=notification.thumbnail_url,
                        caption=message,
                        parse_mode="HTML",
                    )
                else:
                    await self.application.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode="HTML",
                        disable_web_page_preview=False,
                    )
            except Exception as err:
                LOGGER.error("Failed to send Twitch live notification to chat %s: %s", chat_id, err)
                # Fallback to text message if photo sending fails
                if notification.thumbnail_url:
                    try:
                        await self.application.bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode="HTML",
                        )
                    except Exception as fallback_err:
                        LOGGER.error("Fallback text notification also failed for chat %s: %s", chat_id, fallback_err)

    def start(self) -> None:
        if self.config.twitch.is_configured:
            self._task = asyncio.create_task(self.notifier.start())
            LOGGER.info("Twitch EventSub background task started.")
        else:
            LOGGER.info("Twitch is not configured (TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, TWITCH_CHANNELS). Task disabled.")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


def register(application: Application, config: BotConfig) -> None:
    task = TwitchTask(application, config)

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
