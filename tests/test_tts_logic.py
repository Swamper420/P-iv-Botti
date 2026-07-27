import unittest
from unittest.mock import AsyncMock, patch

import httpx

from bot.commands.tts_logic import parse_tts_command, synthesize_speech


class TtsLogicTests(unittest.IsolatedAsyncioTestCase):

    def test_parse_tts_command_default_voice(self) -> None:
        is_match, voice, text = parse_tts_command("puhuu: Terve maailma!")
        self.assertTrue(is_match)
        self.assertIsNone(voice)
        self.assertEqual(text, "Terve maailma!")

    def test_parse_tts_command_with_voice(self) -> None:
        is_match, voice, text = parse_tts_command("Matti puhuu: Terve maailma!")
        self.assertTrue(is_match)
        self.assertEqual(voice, "Matti")
        self.assertEqual(text, "Terve maailma!")

    def test_parse_tts_command_with_multi_word_voice(self) -> None:
        is_match, voice, text = parse_tts_command("Aku Ankka puhuu: Terve kaikille")
        self.assertTrue(is_match)
        self.assertEqual(voice, "Aku Ankka")
        self.assertEqual(text, "Terve kaikille")

    def test_parse_tts_command_with_exclamation_prefix(self) -> None:
        is_match, voice, text = parse_tts_command("!Pekka puhuu: Hei")
        self.assertTrue(is_match)
        self.assertEqual(voice, "Pekka")
        self.assertEqual(text, "Hei")

    def test_parse_tts_command_empty_text(self) -> None:
        is_match, voice, text = parse_tts_command("puhuu:")
        self.assertTrue(is_match)
        self.assertIsNone(voice)
        self.assertEqual(text, "")

    def test_parse_tts_command_no_match(self) -> None:
        is_match, voice, text = parse_tts_command("Tämä on tavallinen teksti")
        self.assertFalse(is_match)
        self.assertIsNone(voice)
        self.assertEqual(text, "")

    async def test_synthesize_speech_success(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.content = b"fake-audio-bytes"
        mock_response.raise_for_status = unittest.mock.MagicMock()

        mock_client.post.return_value = mock_response

        audio = await synthesize_speech(
            base_url="http://localhost:8080",
            text="Terve",
            voice="Matti",
            fmt="ogg",
            timeout_seconds=30,
            client=mock_client,
        )

        self.assertEqual(audio, b"fake-audio-bytes")
        mock_client.post.assert_called_once_with(
            "http://localhost:8080/api/tts",
            json={"text": "Terve", "format": "ogg", "voice": "Matti"},
        )

    async def test_handle_tts_error_response_on_exception(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock
        from bot.commands.tts import _build_handler
        from bot.config import BotConfig, TtsConfig

        config = BotConfig(
            telegram_bot_token="token",
            storage_dir=Path("."),
            max_reply_length=5000,
            weather=MagicMock(),
            cs2_rss=MagicMock(),
            naama=MagicMock(),
            tts=TtsConfig(error_message="Testivirhe viesti"),
        )
        handler = _build_handler(config)

        update = MagicMock()
        update.effective_message.text = "puhuu: Testi"
        update.effective_chat.id = 1234
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        with patch("bot.commands.common.track_active_chat"), \
             patch("bot.commands.tts.synthesize_speech", side_effect=RuntimeError("TTS failed")), \
             patch("bot.commands.tts.reply_in_chunks") as mock_reply:
            await handler(update, context)
            mock_reply.assert_called_once_with(update, "Testivirhe viesti", 5000)

    async def test_handle_tts_error_response_on_empty_audio(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock
        from bot.commands.tts import _build_handler
        from bot.config import BotConfig, TtsConfig

        config = BotConfig(
            telegram_bot_token="token",
            storage_dir=Path("."),
            max_reply_length=5000,
            weather=MagicMock(),
            cs2_rss=MagicMock(),
            naama=MagicMock(),
            tts=TtsConfig(error_message="Testivirhe viesti"),
        )
        handler = _build_handler(config)

        update = MagicMock()
        update.effective_message.text = "puhuu: Testi"
        update.effective_chat.id = 1234
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        with patch("bot.commands.common.track_active_chat"), \
             patch("bot.commands.tts.synthesize_speech", return_value=b""), \
             patch("bot.commands.tts.reply_in_chunks") as mock_reply:
            await handler(update, context)
            mock_reply.assert_called_once_with(update, "Testivirhe viesti", 5000)


if __name__ == "__main__":
    unittest.main()

