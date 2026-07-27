import os
import unittest
from unittest.mock import patch

from bot.config import BotConfig


class ConfigTests(unittest.TestCase):
    def test_loads_settings_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "OPENWEATHER_API_KEY": "ow-key",
                "WEATHER_API_TIMEOUT_SECONDS": "25",
                "MAX_REPLY_LENGTH": "777",
                "STEAM_CS2_RSS_URL": "https://steam.example/rss",
                "STEAM_RSS_POLL_INTERVAL_SECONDS": "123",
                "STEAM_RSS_REQUEST_TIMEOUT_SECONDS": "9",
                "NAAMA_MODEL_NAME": "custom-naama-seg.pt",
                "NAAMA_CONFIDENCE_THRESHOLD": "0.25",
                "NAAMA_MASK_THRESHOLD": "0.45",
                "NAAMA_MAX_IMAGE_BYTES": "7654321",
            },
            clear=False,
        ):
            config = BotConfig.from_environment()

        self.assertEqual(config.openweather_api_key, "ow-key")
        self.assertEqual(config.weather_api_timeout_seconds, 25)
        self.assertEqual(config.max_reply_length, 777)
        self.assertEqual(config.steam_cs2_rss_url, "https://steam.example/rss")
        self.assertEqual(config.steam_rss_poll_interval_seconds, 123)
        self.assertEqual(config.steam_rss_request_timeout_seconds, 9)
        self.assertEqual(config.naama_model_name, "custom-naama-seg.pt")
        self.assertEqual(config.naama_confidence_threshold, 0.25)
        self.assertEqual(config.naama_mask_threshold, 0.45)
        self.assertEqual(config.naama_max_image_bytes, 7654321)


if __name__ == "__main__":
    unittest.main()
