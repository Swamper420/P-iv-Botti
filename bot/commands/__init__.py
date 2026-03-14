from __future__ import annotations

from inspect import signature
from importlib import import_module
from typing import TYPE_CHECKING
from pkgutil import iter_modules

if TYPE_CHECKING:
    from types import ModuleType

    from telegram.ext import Application

    from bot.config import BotConfig


def _discover_command_modules() -> list["ModuleType"]:
    modules: list["ModuleType"] = []
    for module_info in iter_modules(__path__):
        module_name = module_info.name
        if module_name.startswith("_") or module_name.endswith("_logic"):
            continue

        module = import_module(f"{__name__}.{module_name}")
        if callable(getattr(module, "register", None)):
            modules.append(module)

    return modules


def _discover_command_usages(modules: list["ModuleType"]) -> tuple[str, ...]:
    usages: list[str] = []
    for module in modules:
        usage = getattr(module, "COMMAND_USAGE", None)
        if isinstance(usage, str):
            usages.append(usage)
    return tuple(sorted(set(usages), key=str.casefold))


def register_commands(application: "Application", config: "BotConfig") -> None:
    modules = _discover_command_modules()
    command_usages = _discover_command_usages(modules)

    for module in modules:
        register = getattr(module, "register", None)
        if not callable(register):
            continue
        register_parameters = signature(register).parameters
        if "command_usages" in register_parameters:
            register(application, config, command_usages=command_usages)
        else:
            register(application, config)
