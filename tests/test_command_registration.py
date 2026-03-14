import unittest
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

from telegram.ext import MessageHandler, filters

from bot.commands import register_commands
from bot.config import BotConfig


class _DummyApplication:
    def __init__(self) -> None:
        self.handlers: list[MessageHandler] = []

    def add_handler(self, handler: MessageHandler) -> None:
        self.handlers.append(handler)


class CommandRegistrationTests(unittest.TestCase):
    def _config(self) -> BotConfig:
        return BotConfig(
            telegram_bot_token="token",
            storage_dir=Path("."),
            ai_backend_url="http://example.invalid/query",
            ai_max_tokens=650,
            ai_backend_timeout_seconds=30,
            openweather_api_key="",
            weathercam_stations_url="https://tie.digitraffic.fi/api/weathercam/v1/stations",
            weathercam_image_base_url="https://weathercam.digitraffic.fi",
            openweather_current_url="https://api.openweathermap.org/data/2.5/weather",
            weather_api_timeout_seconds=30,
            digitraffic_user="telegram-bot-1.0",
            max_reply_length=5000,
        )

    def test_registers_all_command_modules_with_message_filters(self) -> None:
        expected_count = 0
        for module_info in iter_modules(["bot/commands"]):
            module_name = module_info.name
            if module_name.startswith("_") or module_name.endswith("_logic"):
                continue

            module = import_module(f"bot.commands.{module_name}")
            if callable(getattr(module, "register", None)):
                expected_count += 1

        app = _DummyApplication()
        register_commands(app, self._config())

        self.assertEqual(len(app.handlers), expected_count)
        self.assertTrue(all(isinstance(handler, MessageHandler) for handler in app.handlers))
        self.assertTrue(all(isinstance(handler.filters, filters.Regex) for handler in app.handlers))

    def test_registers_specific_message_filters(self) -> None:
        app = _DummyApplication()
        register_commands(app, self._config())

        self.assertTrue(all(isinstance(handler, MessageHandler) for handler in app.handlers))
        self.assertTrue(all(isinstance(handler.filters, filters.Regex) for handler in app.handlers))


if __name__ == "__main__":
    unittest.main()
