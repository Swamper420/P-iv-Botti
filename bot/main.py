from __future__ import annotations

import logging

from telegram.ext import Application

from bot.commands import register_commands
from bot.config import BotConfig
from bot.cs2_rss import Cs2RssNotifier


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )


def main() -> None:
    configure_logging()
    config = BotConfig.from_environment()

    cs2_notifier = Cs2RssNotifier(config)

    async def _start_background_tasks(application: Application) -> None:
        cs2_notifier.start(application)

    async def _stop_background_tasks(_application: Application) -> None:
        await cs2_notifier.stop()

    app = (
        Application.builder()
        .token(config.telegram_bot_token)
        .post_init(_start_background_tasks)
        .post_shutdown(_stop_background_tasks)
        .build()
    )
    register_commands(app, config)
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
