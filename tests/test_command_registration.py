import unittest
from pathlib import Path

from telegram.ext import MessageHandler, filters

from bot.commands import _discover_command_modules, register_commands
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
            steam_cs2_rss_url="https://steamcommunity.com/games/csgo/rss/",
            steam_rss_poll_interval_seconds=300,
            steam_rss_request_timeout_seconds=30,
            stt_backend_url="http://127.0.0.1:8081/transcribe",
            stt_timeout_seconds=30,
            stt_max_audio_seconds=600,
            mumble_host="127.0.0.1",
            mumble_port=64738,
            mumble_username="status-bot",
            mumble_password="secret",
            mumble_connect_timeout_seconds=10,
            mumble_status_wait_seconds=1,
            mumble_monitor_interval_seconds=10,
        )

    def test_registers_all_command_modules_with_message_filters(self) -> None:
        discovered_modules = _discover_command_modules()
        discovered_names = {module.__name__.split(".")[-1] for module in discovered_modules}
        expected_count = len(discovered_modules)

        app = _DummyApplication()
        register_commands(app, self._config())

        self.assertTrue(
            {"aih", "help", "mumble", "paivaa", "stt", "weather"}.issubset(discovered_names)
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
