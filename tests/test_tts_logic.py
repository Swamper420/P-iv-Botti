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


if __name__ == "__main__":
    unittest.main()
