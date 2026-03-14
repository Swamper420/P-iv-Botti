import unittest
import asyncio
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from telegram.ext import MessageHandler

from bot.commands import mumble
from bot.config import BotConfig


class _DummyJobQueue:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run_repeating(
        self,
        callback,
        *,
        interval: int,
        first: int,
        name: str,
    ) -> None:
        self.calls.append(
            {
                "callback": callback,
                "interval": interval,
                "first": first,
                "name": name,
            }
        )


class _DummyApplication:
    def __init__(self) -> None:
        self.handlers: list[MessageHandler] = []
        self.job_queue = _DummyJobQueue()
        self.bot = type("_DummyBot", (), {"send_message": AsyncMock()})()

    def add_handler(self, handler: MessageHandler) -> None:
        self.handlers.append(handler)


class _DummyApplicationNoJobQueue:
    def __init__(self) -> None:
        self.handlers: list[MessageHandler] = []
        self.job_queue = None
        self.bot = type("_DummyBot", (), {"send_message": AsyncMock()})()

    def add_handler(self, handler: MessageHandler) -> None:
        self.handlers.append(handler)


class MumbleCommandTests(unittest.TestCase):
    def _config(self, *, monitor_interval_seconds: int = 10) -> BotConfig:
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
            mumble_monitor_interval_seconds=monitor_interval_seconds,
        )

    def test_register_adds_mumble_handler_and_monitor_job(self) -> None:
        app = _DummyApplication()
        mumble.register(app, self._config(monitor_interval_seconds=10))

        self.assertEqual(len(app.handlers), 1)
        self.assertIsInstance(app.handlers[0], MessageHandler)
        self.assertEqual(len(app.job_queue.calls), 1)
        self.assertEqual(app.job_queue.calls[0]["interval"], 10)
        self.assertEqual(app.job_queue.calls[0]["first"], 0)
        self.assertEqual(app.job_queue.calls[0]["name"], "mumble-monitor-snapshot")

    def test_register_uses_configured_monitor_interval(self) -> None:
        app = _DummyApplication()
        mumble.register(app, self._config(monitor_interval_seconds=7))

        self.assertEqual(len(app.job_queue.calls), 1)
        self.assertEqual(app.job_queue.calls[0]["interval"], 7)


class MumbleCommandAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_monitor_job_forwards_mumble_tele_messages(self) -> None:
        mumble._drain_persistent_tele_messages()
        app = _DummyApplication()
        config = MumbleCommandTests()._config(monitor_interval_seconds=7)
        config = replace(config, mumble_tele_chat_id=-1001234)
        mumble.register(app, config)

        monitor_callback = app.job_queue.calls[0]["callback"]
        self.assertTrue(callable(monitor_callback))

        with patch(
            "bot.commands.mumble._collect_monitored_snapshot",
            return_value={
                "server_address": "127.0.0.1:64738",
                "channels": [],
                "tele_messages": [("Alice", "hei telegram")],
            },
        ):
            await monitor_callback(None)

        app.bot.send_message.assert_awaited_once_with(
            chat_id=-1001234,
            text="🎧 Alice: hei telegram",
        )
        self.assertEqual(mumble._drain_persistent_tele_messages(), [])

    async def test_monitor_job_requeues_tele_messages_on_forward_failure(self) -> None:
        mumble._drain_persistent_tele_messages()
        app = _DummyApplication()
        app.bot = type(
            "_DummyBot",
            (),
            {"send_message": AsyncMock(side_effect=RuntimeError("telegram send failed"))},
        )()
        config = MumbleCommandTests()._config(monitor_interval_seconds=7)
        config = replace(config, mumble_tele_chat_id=-1001234)
        mumble.register(app, config)

        monitor_callback = app.job_queue.calls[0]["callback"]
        self.assertTrue(callable(monitor_callback))

        with patch(
            "bot.commands.mumble._collect_monitored_snapshot",
            return_value={
                "server_address": "127.0.0.1:64738",
                "channels": [],
                "tele_messages": [("Alice", "eka"), ("Bob", "toka")],
            },
        ):
            await monitor_callback(None)

        app.bot.send_message.assert_awaited_once_with(
            chat_id=-1001234,
            text="🎧 Alice: eka",
        )
        self.assertEqual(
            mumble._drain_persistent_tele_messages(),
            [("Alice", "eka"), ("Bob", "toka")],
        )


class MumbleFallbackMonitorAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await mumble.stop_background_monitor()

    async def test_start_background_monitor_uses_fallback_loop_without_job_queue(self) -> None:
        app = _DummyApplicationNoJobQueue()
        config = MumbleCommandTests()._config(monitor_interval_seconds=60)

        with patch(
            "bot.commands.mumble._refresh_monitored_snapshot",
            new=AsyncMock(),
        ) as refresh_mock:
            await mumble.start_background_monitor(app, config)
            await asyncio.sleep(0)

        refresh_mock.assert_awaited_once_with(app, config)

    async def test_start_background_monitor_skips_fallback_when_job_queue_exists(self) -> None:
        app = _DummyApplication()
        config = MumbleCommandTests()._config(monitor_interval_seconds=60)

        with patch("bot.commands.mumble.asyncio.create_task") as create_task_mock:
            await mumble.start_background_monitor(app, config)

        create_task_mock.assert_not_called()


class MumbleConnectionStateLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        mumble._reset_monitor_connection_state()

    def tearDown(self) -> None:
        mumble._reset_monitor_connection_state()

    def test_logs_connection_lifecycle_with_timestamps(self) -> None:
        with patch.object(mumble, "LOGGER") as logger_mock:
            mumble._update_monitor_connection_state(
                server_address="127.0.0.1:64738",
                connected=True,
                now_epoch=100.0,
            )
            mumble._update_monitor_connection_state(
                server_address="127.0.0.1:64738",
                connected=False,
                now_epoch=130.0,
            )
            mumble._update_monitor_connection_state(
                server_address="127.0.0.1:64738",
                connected=True,
                now_epoch=160.0,
            )

        self.assertEqual(logger_mock.info.call_count, 2)
        self.assertEqual(logger_mock.warning.call_count, 1)
        self.assertIn("connected", logger_mock.info.call_args_list[0].args[0])
        self.assertIn("lost connection", logger_mock.warning.call_args.args[0])
        self.assertIn("reconnected", logger_mock.info.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
