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
                "OLLAMA_BASE_URL": "http://ollama.example:11434",
                "OLLAMA_MODEL": "qwen2.5",
                "OLLAMA_DEFAULT_NUM_PREDICT": "150",
                "OLLAMA_MAX_NUM_PREDICT": "1500",
                "OLLAMA_TIMEOUT_SECONDS": "60",
                "OLLAMA_STRIP_THINKING": "true",
                "TTS_BASE_URL": "http://tts.example:8080",
                "TTS_TIMEOUT_SECONDS": "45",
                "TTS_FORMAT": "ogg",
                "TTS_ERROR_MESSAGE": "Kustomoitu virhe",
                "TTS_LANGUAGE": "en",
                "TTS_SPEED": "1.2",
                "TTS_NUM_STEP": "24",
                "TTS_GUIDANCE_SCALE": "2.5",
                "TELKKARI_EPG_URL": "https://example.com/epg.xml",
                "TELKKARI_DEFAULT_CHANNELS": "1,2,3",
                "TELKKARI_CACHE_TIMEOUT_SECONDS": "600",
                "TELKKARI_TIMEOUT_SECONDS": "15",
                "STT_BASE_URL": "http://stt.example:8001",
                "STT_TIMEOUT_SECONDS": "90",
                "STT_BEAM_SIZE": "8",
                "STT_VAD_FILTER": "true",
                "STT_WORD_TIMESTAMPS": "true",
                "STT_INITIAL_PROMPT": "Syötä tekstiä",
            },
            clear=False,
        ):
            config = BotConfig.from_environment()

        # Domain config checks
        self.assertEqual(config.weather.openweather_api_key, "ow-key")
        self.assertEqual(config.weather.timeout_seconds, 25)
        self.assertEqual(config.cs2_rss.url, "https://steam.example/rss")
        self.assertEqual(config.cs2_rss.poll_interval_seconds, 123)
        self.assertEqual(config.cs2_rss.request_timeout_seconds, 9)
        self.assertEqual(config.naama.model_name, "custom-naama-seg.pt")
        self.assertEqual(config.naama.confidence_threshold, 0.25)
        self.assertEqual(config.naama.mask_threshold, 0.45)
        self.assertEqual(config.naama.max_image_bytes, 7654321)
        self.assertEqual(config.ollama.base_url, "http://ollama.example:11434")
        self.assertEqual(config.ollama.model, "qwen2.5")
        self.assertEqual(config.ollama.default_num_predict, 150)
        self.assertEqual(config.ollama.max_num_predict, 1500)
        self.assertEqual(config.ollama.timeout_seconds, 60)
        self.assertTrue(config.ollama.strip_thinking)
        self.assertEqual(config.tts.base_url, "http://tts.example:8080")
        self.assertEqual(config.tts.timeout_seconds, 45)
        self.assertEqual(config.tts.format, "ogg")
        self.assertEqual(config.tts.error_message, "Kustomoitu virhe")
        self.assertEqual(config.tts.language, "en")
        self.assertEqual(config.tts.speed, 1.2)
        self.assertEqual(config.tts.num_step, 24)
        self.assertEqual(config.tts.guidance_scale, 2.5)
        self.assertEqual(config.telkkari.epg_url, "https://example.com/epg.xml")
        self.assertEqual(config.telkkari.default_channels, (1, 2, 3))
        self.assertEqual(config.telkkari.cache_timeout_seconds, 600)
        self.assertEqual(config.telkkari.timeout_seconds, 15)
        self.assertEqual(config.stt.base_url, "http://stt.example:8001")
        self.assertEqual(config.stt.timeout_seconds, 90)
        self.assertEqual(config.stt.beam_size, 8)
        self.assertTrue(config.stt.vad_filter)
        self.assertTrue(config.stt.word_timestamps)
        self.assertEqual(config.stt.initial_prompt, "Syötä tekstiä")


        # Backward compatibility property checks
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
        self.assertEqual(config.ollama_base_url, "http://ollama.example:11434")
        self.assertEqual(config.ollama_model, "qwen2.5")
        self.assertEqual(config.ollama_default_num_predict, 150)
        self.assertEqual(config.ollama_max_num_predict, 1500)
        self.assertEqual(config.ollama_timeout_seconds, 60)
        self.assertTrue(config.ollama_strip_thinking)
        self.assertEqual(config.tts_base_url, "http://tts.example:8080")
        self.assertEqual(config.tts_timeout_seconds, 45)
        self.assertEqual(config.tts_format, "ogg")
        self.assertEqual(config.tts_error_message, "Kustomoitu virhe")
        self.assertEqual(config.tts_language, "en")
        self.assertEqual(config.tts_speed, 1.2)
        self.assertEqual(config.tts_num_step, 24)
        self.assertEqual(config.tts_guidance_scale, 2.5)
        self.assertEqual(config.stt_base_url, "http://stt.example:8001")
        self.assertEqual(config.stt_timeout_seconds, 90)
        self.assertEqual(config.stt_beam_size, 8)
        self.assertTrue(config.stt_vad_filter)
        self.assertTrue(config.stt_word_timestamps)
        self.assertEqual(config.stt_initial_prompt, "Syötä tekstiä")

    def test_invalid_stt_config_raises_error(self) -> None:
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "token", "STT_BEAM_SIZE": "0"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                BotConfig.from_environment()

        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "token", "STT_TIMEOUT_SECONDS": "0"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                BotConfig.from_environment()


if __name__ == "__main__":
    unittest.main()


