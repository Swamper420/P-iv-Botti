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
                "MAX_REPLY_LENGTH": "777",
            },
            clear=False,
        ):
            config = BotConfig.from_environment()

        self.assertEqual(config.ai_backend_url, "http://example.local/query")
        self.assertEqual(config.ai_max_tokens, 321)
        self.assertEqual(config.ai_backend_timeout_seconds, 12)
        self.assertEqual(config.max_reply_length, 777)


if __name__ == "__main__":
    unittest.main()
