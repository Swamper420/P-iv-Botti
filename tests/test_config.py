import os
import unittest
from unittest.mock import patch

from bot.config import BotConfig


class ConfigTests(unittest.TestCase):
    def test_loads_ai_settings_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "AI_BACKEND_URL": "http://example.local/query",
                "AI_MAX_TOKENS": "321",
                "AI_BACKEND_TIMEOUT_SECONDS": "12",
                "OPENWEATHER_API_KEY": "ow-key",
                "WEATHER_API_TIMEOUT_SECONDS": "25",
                "MAX_REPLY_LENGTH": "777",
                "STEAM_CS2_RSS_URL": "https://steam.example/rss",
                "STEAM_RSS_POLL_INTERVAL_SECONDS": "123",
                "STEAM_RSS_REQUEST_TIMEOUT_SECONDS": "9",
                "STT_BACKEND_URL": "http://127.0.0.1:9000/transcribe",
                "STT_TIMEOUT_SECONDS": "21",
                "STT_MAX_AUDIO_SECONDS": "599",
                "MUMBLE_HOST": "127.0.0.2",
                "MUMBLE_PORT": "64739",
                "MUMBLE_USERNAME": "status-user",
                "MUMBLE_PASSWORD": "status-pass",
                "MUMBLE_CONNECT_TIMEOUT_SECONDS": "11",
                "MUMBLE_CONNECT_RETRIES": "3",
                "MUMBLE_STATUS_WAIT_SECONDS": "2",
                "MUMBLE_MONITOR_INTERVAL_SECONDS": "10",
                "MUMBLE_TELE_CHAT_ID": "-100123456789",
                "NAAMA_MODEL_NAME": "custom-naama-seg.pt",
                "NAAMA_CONFIDENCE_THRESHOLD": "0.25",
                "NAAMA_MASK_THRESHOLD": "0.45",
                "NAAMA_MAX_IMAGE_BYTES": "7654321",
            },
            clear=False,
        ):
            config = BotConfig.from_environment()

        self.assertEqual(config.ai_backend_url, "http://example.local/query")
        self.assertEqual(config.ai_max_tokens, 321)
        self.assertEqual(config.ai_backend_timeout_seconds, 12)
        self.assertEqual(config.openweather_api_key, "ow-key")
        self.assertEqual(config.weather_api_timeout_seconds, 25)
        self.assertEqual(config.max_reply_length, 777)
        self.assertEqual(config.steam_cs2_rss_url, "https://steam.example/rss")
        self.assertEqual(config.steam_rss_poll_interval_seconds, 123)
        self.assertEqual(config.steam_rss_request_timeout_seconds, 9)
        self.assertEqual(config.stt_backend_url, "http://127.0.0.1:9000/transcribe")
        self.assertEqual(config.stt_timeout_seconds, 21)
        self.assertEqual(config.stt_max_audio_seconds, 599)
        self.assertEqual(config.mumble_host, "127.0.0.2")
        self.assertEqual(config.mumble_port, 64739)
        self.assertEqual(config.mumble_username, "status-user")
        self.assertEqual(config.mumble_password, "status-pass")
        self.assertEqual(config.mumble_connect_timeout_seconds, 11)
        self.assertEqual(config.mumble_connect_retries, 3)
        self.assertEqual(config.mumble_status_wait_seconds, 2)
        self.assertEqual(config.mumble_monitor_interval_seconds, 10)
        self.assertEqual(config.mumble_tele_chat_id, -100123456789)
        self.assertEqual(config.naama_model_name, "custom-naama-seg.pt")
        self.assertEqual(config.naama_confidence_threshold, 0.25)
        self.assertEqual(config.naama_mask_threshold, 0.45)
        self.assertEqual(config.naama_max_image_bytes, 7654321)

    def test_rejects_invalid_mumble_monitor_interval(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "MUMBLE_MONITOR_INTERVAL_SECONDS": "0",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(
                ValueError, "MUMBLE_MONITOR_INTERVAL_SECONDS must be >= 1"
            ):
                BotConfig.from_environment()

    def test_rejects_invalid_mumble_connect_retries(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "MUMBLE_CONNECT_RETRIES": "0",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "MUMBLE_CONNECT_RETRIES must be >= 1"):
                BotConfig.from_environment()


if __name__ == "__main__":
    unittest.main()
