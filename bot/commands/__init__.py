from __future__ import annotations


def register_commands(application) -> None:
    from bot.commands.paivaa import register as register_paivaa

    register_paivaa(application)
