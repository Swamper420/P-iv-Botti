from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.link_video_logic import (
    build_yt_dlp_format_selector,
    extract_urls,
    get_url_regex,
)
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)
BYTES_PER_MB = 1024 * 1024


def _download_video(url: str, config: BotConfig, download_dir: Path) -> Path | None:
    format_selector = build_yt_dlp_format_selector(config.link_video_download_max_height)
    output_template = str(download_dir / "video.%(ext)s")
    command = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--no-overwrites",
        "--max-filesize",
        f"{config.link_video_download_max_filesize_mb}M",
        "--output",
        output_template,
        "--format",
        format_selector,
        "--",
        url,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=config.link_video_download_timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        LOGGER.exception("yt-dlp download failed")
        return None

    if completed.returncode != 0:
        return None

    return next(
        (
            candidate
            for candidate in sorted(download_dir.glob("video.*"))
            if candidate.suffix != ".part" and candidate.is_file()
        ),
        None,
    )


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_link_video(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        del context

        if not config.link_video_download_enabled:
            return

        message = update.effective_message
        if message is None:
            return

        text = message.text or message.caption
        urls = extract_urls(text)
        if not urls:
            return

        track_active_chat(update, config.storage_dir)

        for url in urls:
            with TemporaryDirectory(prefix="link-video-") as tmp_dir:
                downloaded_path = await asyncio.to_thread(
                    _download_video, url, config, Path(tmp_dir)
                )
                if downloaded_path is None:
                    continue

                try:
                    if (
                        downloaded_path.stat().st_size
                        > config.link_video_download_max_filesize_mb * BYTES_PER_MB
                    ):
                        continue
                    with downloaded_path.open("rb") as video_file:
                        await message.reply_video(
                            video=video_file,
                            disable_notification=True,
                        )
                except OSError:
                    LOGGER.exception("Failed to send downloaded video")

    return handle_link_video


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            (
                filters.Regex(get_url_regex()) | filters.CaptionRegex(get_url_regex())
            )
            & ~filters.COMMAND,
            _build_handler(config),
        )
    )
