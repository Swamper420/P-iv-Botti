from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType
    from telegram.ext import Application
    from bot.config import BotConfig


def _discover_task_modules() -> list[ModuleType]:
    modules: list[ModuleType] = []
    for module_info in iter_modules(__path__):
        module_name = module_info.name
        if module_name.startswith("_") or module_name.endswith("_logic"):
            continue

        module = import_module(f"{__name__}.{module_name}")
        if callable(getattr(module, "register", None)):
            modules.append(module)

    return modules


def register_tasks(application: Application, config: BotConfig) -> None:
    modules = _discover_task_modules()
    for module in modules:
        register = getattr(module, "register", None)
        if callable(register):
            register(application, config)
