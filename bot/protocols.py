from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from telegram.ext import Application
    from bot.config import BotConfig


@runtime_checkable
class CommandModule(Protocol):
    def register(self, application: Application, config: BotConfig, **kwargs: Any) -> None:
        ...


@runtime_checkable
class TaskModule(Protocol):
    def register(self, application: Application, config: BotConfig) -> None:
        ...
