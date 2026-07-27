import unittest
from pathlib import Path

from bot.config import BotConfig, Cs2RssConfig, NaamaConfig, WeatherConfig
from bot.tasks import _discover_task_modules, register_tasks


class _DummyAppWithHooks:
    def __init__(self) -> None:
        self.post_init = None
        self.post_shutdown = None


class TaskRegistrationTests(unittest.TestCase):
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

    def test_discovers_task_modules(self) -> None:
        modules = _discover_task_modules()
        module_names = {module.__name__.split(".")[-1] for module in modules}
        self.assertIn("cs2_rss", module_names)
        self.assertIn("twitch", module_names)


    def test_register_tasks_attaches_post_init_and_post_shutdown_callbacks(self) -> None:
        app = _DummyAppWithHooks()
        register_tasks(app, self._config())

        self.assertIsNotNone(app.post_init)
        self.assertIsNotNone(app.post_shutdown)


if __name__ == "__main__":
    unittest.main()
