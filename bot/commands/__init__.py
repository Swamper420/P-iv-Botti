from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram.ext import Application

    from bot.config import BotConfig


def register_commands(application: "Application", config: "BotConfig") -> None:
    from bot.commands.aih import register as register_aih
    from bot.commands.paivaa import register as register_paivaa
    from bot.commands.weather import register as register_weather

    register_paivaa(application, config)
    register_aih(application, config)
    register_weather(application, config)
