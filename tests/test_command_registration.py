import unittest
from pathlib import Path

from telegram.ext import MessageHandler, filters

from bot.commands import _discover_command_modules, register_commands
from bot.config import BotConfig, Cs2RssConfig, NaamaConfig, WeatherConfig


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
            max_reply_length=5000,
            weather=WeatherConfig(
                openweather_api_key="",
                weathercam_stations_url="https://tie.digitraffic.fi/api/weathercam/v1/stations",
                weathercam_image_base_url="https://weathercam.digitraffic.fi",
                openweather_current_url="https://api.openweathermap.org/data/2.5/weather",
                timeout_seconds=30,
                digitraffic_user="telegram-bot-1.0",
            ),
            cs2_rss=Cs2RssConfig(
                url="https://steamcommunity.com/games/csgo/rss/",
                poll_interval_seconds=300,
                request_timeout_seconds=30,
            ),
            naama=NaamaConfig(),
        )

    def test_registers_all_command_modules_with_message_filters(self) -> None:
        discovered_modules = _discover_command_modules()
        discovered_names = {module.__name__.split(".")[-1] for module in discovered_modules}
        expected_count = len(discovered_modules)

        app = _DummyApplication()
        register_commands(app, self._config())

        self.assertTrue(
            {"help", "hoi", "naama", "weather"}.issubset(
                discovered_names
            )
        )
        self.assertEqual(len(app.handlers), expected_count)
        self.assertTrue(all(isinstance(handler, MessageHandler) for handler in app.handlers))
        has_regex = any(isinstance(handler.filters, filters.Regex) for handler in app.handlers)
        self.assertTrue(has_regex)
        self.assertGreaterEqual(len(app.handlers), 1)

    def test_registers_specific_message_filters(self) -> None:
        app = _DummyApplication()
        register_commands(app, self._config())

        self.assertTrue(all(isinstance(handler, MessageHandler) for handler in app.handlers))
        regex_handlers = [
            handler for handler in app.handlers if isinstance(handler.filters, filters.Regex)
        ]
        non_regex_handlers = [
            handler
            for handler in app.handlers
            if not isinstance(handler.filters, filters.Regex)
        ]
        self.assertGreaterEqual(len(regex_handlers), 1)
        self.assertGreaterEqual(len(non_regex_handlers), 1)


if __name__ == "__main__":
    unittest.main()
