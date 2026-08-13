from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from bot.commands.muistuta_logic import get_due_reminders, remove_reminders
from bot.config import BotConfig

if TYPE_CHECKING:
    from telegram.ext import Application

LOGGER = logging.getLogger(__name__)


class ReminderNotifier:
    def __init__(self, config: BotConfig) -> None:
        self._config = config
        self._storage_dir = config.storage_dir
        self._check_interval = config.reminder.check_interval_seconds
        self._task: asyncio.Task[None] | None = None

    def start(self, app: Application) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop(app))

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self, app: Application) -> None:
        LOGGER.info("ReminderNotifier loop started (interval: %ds)", self._check_interval)
        while True:
            try:
                await self._check_and_send_due_reminders(app)
            except asyncio.CancelledError:
                LOGGER.info("ReminderNotifier loop cancelled")
                break
            except Exception:
                LOGGER.exception("Error checking/sending due reminders")

            await asyncio.sleep(self._check_interval)

    async def _check_and_send_due_reminders(self, app: Application) -> None:
        due_items = get_due_reminders(self._storage_dir)
        if not due_items:
            return

        processed_ids: set[int] = set()

        for item in due_items:
            rid = int(item["id"])
            chat_id = int(item["chat_id"])
            targets = item.get("targets", [])
            msg_text = item.get("message", "")
            media = item.get("media")

            parts = ["⏰ MUISTUTUS!"]
            if targets:
                parts.append(" ".join(targets))
            if msg_text:
                parts.append(msg_text)

            full_text = "\n".join(parts)

            try:
                if media and isinstance(media, dict) and media.get("file_id"):
                    file_id = media["file_id"]
                    media_type = media.get("media_type", "photo")
                    await self._send_media_reminder(app, chat_id, file_id, media_type, full_text)
                else:
                    await app.bot.send_message(chat_id=chat_id, text=full_text)
            except Exception:
                LOGGER.exception("Failed to send reminder #%d to chat %d", rid, chat_id)
            finally:
                processed_ids.add(rid)

        remove_reminders(self._storage_dir, processed_ids)

    async def _send_media_reminder(
        self,
        app: Application,
        chat_id: int,
        file_id: str,
        media_type: str,
        caption: str,
    ) -> None:
        bot = app.bot
        if media_type == "photo":
            await bot.send_photo(chat_id=chat_id, photo=file_id, caption=caption)
        elif media_type == "video":
            await bot.send_video(chat_id=chat_id, video=file_id, caption=caption)
        elif media_type == "document":
            await bot.send_document(chat_id=chat_id, document=file_id, caption=caption)
        elif media_type == "animation":
            await bot.send_animation(chat_id=chat_id, animation=file_id, caption=caption)
        elif media_type == "voice":
            await bot.send_voice(chat_id=chat_id, voice=file_id, caption=caption)
        elif media_type == "audio":
            await bot.send_audio(chat_id=chat_id, audio=file_id, caption=caption)
        elif media_type == "sticker":
            await bot.send_sticker(chat_id=chat_id, sticker=file_id)
            await bot.send_message(chat_id=chat_id, text=caption)
        elif media_type == "video_note":
            await bot.send_video_note(chat_id=chat_id, video_note=file_id)
            await bot.send_message(chat_id=chat_id, text=caption)
        else:
            await bot.send_message(chat_id=chat_id, text=caption)


def register(application: Application, config: BotConfig) -> None:
    notifier = ReminderNotifier(config)

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
