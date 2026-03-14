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
                "LINK_VIDEO_DOWNLOAD_ENABLED": "true",
                "LINK_VIDEO_DOWNLOAD_TIMEOUT_SECONDS": "45",
                "LINK_VIDEO_DOWNLOAD_MAX_FILESIZE_MB": "42",
                "LINK_VIDEO_DOWNLOAD_MAX_HEIGHT": "360",
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
        self.assertTrue(config.link_video_download_enabled)
        self.assertEqual(config.link_video_download_timeout_seconds, 45)
        self.assertEqual(config.link_video_download_max_filesize_mb, 42)
        self.assertEqual(config.link_video_download_max_height, 360)


if __name__ == "__main__":
    unittest.main()
