from __future__ import annotations

import asyncio
from datetime import datetime
from io import BytesIO
import logging
from typing import TYPE_CHECKING

from bot.config import BotConfig
from bot.tasks._maraton_render import render_maraton_briefing_image
from bot.tasks.maraton_logic import (
    build_maraton_briefing,
    format_maraton_caption,
    format_maraton_text,
    get_local_now,
    mark_reminder_sent,
    should_send_daily_reminder,
)

if TYPE_CHECKING:
    from telegram.ext import Application

LOGGER = logging.getLogger(__name__)


class MaratonNotifier:
    """Background task delivering the daily Tampere Maraton 2027 maratonviesti reminder."""

    def __init__(self, config: BotConfig) -> None:
        self._config = config
        self._maraton_config = config.maraton
        self._storage_dir = config.storage_dir
        self._task: asyncio.Task[None] | None = None

    def start(self, app: Application) -> None:
        if not self._maraton_config.enabled:
            LOGGER.info("Maraton daily reminder is disabled (MARATON_ENABLED=false).")
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop(app))
        LOGGER.info(
            "MaratonNotifier started (event=%s, daily=%s %s, interval=%ds)",
            self._maraton_config.event_date,
            self._maraton_config.send_time,
            self._maraton_config.timezone,
            self._maraton_config.check_interval_seconds,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run_loop(self, app: Application) -> None:
        while True:
            try:
                await self.check_and_send(app)
            except asyncio.CancelledError:
                LOGGER.info("MaratonNotifier loop cancelled")
                break
            except Exception:
                LOGGER.exception("Error while checking/sending daily maraton reminder")

            await asyncio.sleep(self._maraton_config.check_interval_seconds)

    def _get_target_chat_ids(self) -> list[int]:
        if self._maraton_config.chat_ids:
            return list(self._maraton_config.chat_ids)
        from bot.active_chats import load_active_chat_ids

        return sorted(load_active_chat_ids(self._storage_dir))

    async def check_and_send(
        self,
        app: Application,
        now: datetime | None = None,
    ) -> bool:
        """Checks whether today's reminder is due and broadcasts it to target chats."""
        if not should_send_daily_reminder(
            self._maraton_config, self._storage_dir, now=now
        ):
            return False

        chat_ids = self._get_target_chat_ids()
        if not chat_ids:
            LOGGER.info(
                "No target chat IDs available for maraton daily reminder; skipping for now."
            )
            return False

        local_now = get_local_now(self._maraton_config.timezone, now=now)
        briefing = build_maraton_briefing(self._maraton_config, today=local_now.date())
        full_text = format_maraton_text(briefing)
        caption = format_maraton_caption(briefing)

        image_bytes: bytes | None = None
        if self._maraton_config.use_card:
            try:
                image_bytes = await asyncio.to_thread(
                    render_maraton_briefing_image, briefing
                )
            except Exception:
                LOGGER.exception(
                    "Failed to render maraton daily poster; falling back to text."
                )
                image_bytes = None

        delivered_count = 0
        for chat_id in chat_ids:
            try:
                if image_bytes is not None:
                    photo_buf = BytesIO(image_bytes)
                    photo_buf.name = "maraton.png"
                    await app.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo_buf,
                        caption=caption,
                        parse_mode="HTML",
                    )
                else:
                    await app.bot.send_message(
                        chat_id=chat_id,
                        text=full_text,
                        parse_mode="HTML",
                    )
                delivered_count += 1
            except Exception:
                LOGGER.exception(
                    "Failed to send maraton daily reminder to chat %s", chat_id
                )
                if image_bytes is not None:
                    try:
                        await app.bot.send_message(
                            chat_id=chat_id,
                            text=full_text,
                            parse_mode="HTML",
                        )
                        delivered_count += 1
                    except Exception:
                        LOGGER.exception(
                            "Fallback text maraton reminder also failed for chat %s",
                            chat_id,
                        )

        if delivered_count > 0:
            mark_reminder_sent(self._storage_dir, local_now.date(), now=local_now)
            LOGGER.info(
                "Sent Tampere Maraton 2027 daily reminder (%d days left) to %d/%d chats.",
                briefing.days_remaining,
                delivered_count,
                len(chat_ids),
            )
            return True

        return False


def register(application: Application, config: BotConfig) -> None:
    notifier = MaratonNotifier(config)

    prev_init = application.post_init
    prev_shutdown = application.post_shutdown

    async def post_init(app: Application) -> None:
        if prev_init is not None:
            await prev_init(app)
        notifier.start(app)

    async def post_shutdown(app: Application) -> None:
        if prev_shutdown is not None:
            await prev_shutdown(app)
        await notifier.stop()

    application.post_init = post_init
    application.post_shutdown = post_shutdown
