from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram.ext import Application


def register_commands(application: "Application") -> None:
    from bot.commands.paivaa import register as register_paivaa

    register_paivaa(application)
