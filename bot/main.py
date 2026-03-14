from __future__ import annotations

import logging

from telegram.ext import Application

from bot.commands import register_commands
from bot.config import BotConfig


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )


def main() -> None:
    configure_logging()
    config = BotConfig.from_environment()

    app = Application.builder().token(config.telegram_bot_token).build()
    register_commands(app, config)
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
