import unittest
from pathlib import Path
from unittest.mock import patch

from bot.commands.weather_logic import get_weather_cam_data, parse_weather_camera_location
from bot.config import BotConfig


class WeatherLogicTests(unittest.TestCase):
    def test_extracts_location_from_weather_command(self) -> None:
        self.assertEqual(
            parse_weather_camera_location("!sääkuva Helsinki"), (True, "Helsinki")
        )

    def test_extracts_location_from_ascii_alias(self) -> None:
        self.assertEqual(
            parse_weather_camera_location("  !saakuva  Oulu "), (True, "Oulu")
        )

    def test_matches_command_without_location(self) -> None:
        self.assertEqual(parse_weather_camera_location("!sääkuva"), (True, None))

    def test_ignores_non_command_text(self) -> None:
        self.assertEqual(parse_weather_camera_location("aih: test"), (False, None))

    def test_weather_image_fetch_sends_digitraffic_headers(self) -> None:
        location_data = {
            "features": [{"properties": {"name": "Helsinki", "presets": [{"id": "CAM123"}]}}]
        }
        config = BotConfig(
            telegram_bot_token="token",
            storage_dir=Path("."),
            ai_backend_url="http://example.invalid/query",
            ai_max_tokens=650,
            ai_backend_timeout_seconds=30,
            openweather_api_key="",
            weathercam_stations_url="https://stations.invalid",
            weathercam_image_base_url="https://images.invalid",
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

        with (
            patch("bot.commands.weather_logic._fetch_json", return_value=location_data) as fetch,
            patch("bot.commands.weather_logic._download_bytes", return_value=b"jpg") as download,
        ):
            image, filename = get_weather_cam_data("helsinki", config)

        self.assertEqual(image, b"jpg")
        self.assertEqual(filename, "CAM123.jpg")
        self.assertEqual(
            fetch.call_args.kwargs["headers"],
            {"Digitraffic-User": "telegram-bot-1.0", "If-None-Match": ""},
        )
        self.assertEqual(
            download.call_args.kwargs["headers"],
            {"Digitraffic-User": "telegram-bot-1.0", "If-None-Match": ""},
        )


if __name__ == "__main__":
    unittest.main()
