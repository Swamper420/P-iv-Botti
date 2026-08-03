import unittest
from unittest.mock import AsyncMock, patch

import httpx

from bot.commands.tts_logic import (
    chunk_text,
    combine_audio_chunks,
    parse_tts_command,
    synthesize_speech,
)


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

    def test_parse_tts_command_selkeasti_default_voice(self) -> None:
        is_match, voice, text = parse_tts_command("puhuu selkeästi: Terve maailma!")
        self.assertTrue(is_match)
        self.assertIsNone(voice)
        self.assertEqual(text, "Terve... maailma!...")

    def test_parse_tts_command_selkeasti_with_voice(self) -> None:
        is_match, voice, text = parse_tts_command("Matti puhuu selkeästi: Terve maailma!")
        self.assertTrue(is_match)
        self.assertEqual(voice, "Matti")
        self.assertEqual(text, "Terve... maailma!...")

    def test_parse_tts_command_selkeasti_exclamation_prefix(self) -> None:
        is_match, voice, text = parse_tts_command("!Pekka puhuu selkeästi: Hei kaveri")
        self.assertTrue(is_match)
        self.assertEqual(voice, "Pekka")
        self.assertEqual(text, "Hei... kaveri...")

    def test_parse_tts_command_selkeasti_existing_dots(self) -> None:
        is_match, voice, text = parse_tts_command("puhuu selkeästi: Terve... maailma")
        self.assertTrue(is_match)
        self.assertIsNone(voice)
        self.assertEqual(text, "Terve... maailma...")


    def test_chunk_text_padding_short_text(self) -> None:
        chunks = chunk_text("Hei!", max_chunk_size=50, min_chunk_len=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hei!......")
        self.assertGreaterEqual(len(chunks[0]), 10)

    def test_chunk_text_splits_on_punctuation(self) -> None:
        text = (
            "Ensimmäinen lause on tässä, ja se on aika pitkä. "
            "Toinen lause tulee tässä! Kolmas lause vastaa kysymykseen?"
        )
        chunks = chunk_text(text, max_chunk_size=40, min_chunk_len=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 50)
            self.assertGreaterEqual(len(chunk), 10)

    def test_chunk_text_empty(self) -> None:
        self.assertEqual(chunk_text(""), [])

    async def test_combine_audio_chunks_single(self) -> None:
        result = await combine_audio_chunks([b"audio1"])
        self.assertEqual(result, b"audio1")

    async def test_combine_audio_chunks_empty(self) -> None:
        result = await combine_audio_chunks([])
        self.assertEqual(result, b"")

    def test_extract_voice_ids(self) -> None:
        from bot.commands.tts_logic import extract_voice_ids

        data = {
            "count": 2,
            "voices": [
                {"voice_id": "voice_fi"},
                {"voice_id": "voice_en"},
            ],
        }
        self.assertEqual(extract_voice_ids(data), ["voice_fi", "voice_en"])

    def test_resolve_voice_exact_match(self) -> None:
        from bot.commands.tts_logic import resolve_voice

        available = ["voice_fi", "voice_en", "matti_speaker"]
        self.assertEqual(resolve_voice("voice_fi", available), "voice_fi")
        self.assertEqual(resolve_voice("VOICE_FI", available), "voice_fi")

    def test_resolve_voice_fuzzy_match(self) -> None:
        from bot.commands.tts_logic import resolve_voice

        available = ["voice_fi", "voice_en", "matti_speaker"]
        self.assertEqual(resolve_voice("matti", available), "matti_speaker")

    def test_resolve_voice_none_or_empty(self) -> None:
        from bot.commands.tts_logic import resolve_voice

        self.assertIsNone(resolve_voice(None, ["voice_fi"]))
        self.assertEqual(resolve_voice("matti", []), "matti")


    async def test_synthesize_speech_multiple_chunks(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_voices_resp = unittest.mock.MagicMock(spec=httpx.Response)
        mock_voices_resp.json.return_value = {"count": 1, "voices": [{"voice_id": "Matti"}]}
        mock_voices_resp.raise_for_status = unittest.mock.MagicMock()
        mock_client.get.return_value = mock_voices_resp

        mock_response1 = unittest.mock.MagicMock(spec=httpx.Response)
        mock_response1.content = b"chunk1-bytes"
        mock_response1.raise_for_status = unittest.mock.MagicMock()

        mock_response2 = unittest.mock.MagicMock(spec=httpx.Response)
        mock_response2.content = b"chunk2-bytes"
        mock_response2.raise_for_status = unittest.mock.MagicMock()

        mock_client.post.side_effect = [mock_response1, mock_response2]

        text = "Ensimmäinen lause, joka on pitkä. Toinen lause, joka on myös pitkä."
        with patch(
            "bot.commands.tts_logic.combine_audio_chunks",
            new_callable=AsyncMock,
            return_value=b"combined-audio",
        ) as mock_combine:
            audio = await synthesize_speech(
                base_url="http://localhost:8080",
                text=text,
                voice="Matti",
                fmt="ogg",
                language="fi",
                speed=1.0,
                num_step=32,
                guidance_scale=2.0,
                seed=42,
                timeout_seconds=30,
                max_chunk_size=40,
                min_chunk_len=10,
                client=mock_client,
            )

        self.assertEqual(audio, b"combined-audio")
        self.assertEqual(mock_client.post.call_count, 2)
        call_args = mock_client.post.call_args_list[0]
        self.assertEqual(call_args.args[0], "http://localhost:8080/api/v1/tts")
        self.assertEqual(
            call_args.kwargs["json"],
            {
                "text": "Ensimmäinen lause, joka on pitkä.",
                "voice": "Matti",
                "response_format": "ogg",
                "language": "fi",
                "speed": 1.0,
                "num_step": 32,
                "guidance_scale": 2.0,
                "seed": 42,
            },
        )
        mock_combine.assert_called_once_with(
            [b"chunk1-bytes", b"chunk2-bytes"], fmt="ogg"
        )

    async def test_synthesize_speech_fuzzy_voice_resolution(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_voices_resp = unittest.mock.MagicMock(spec=httpx.Response)
        mock_voices_resp.json.return_value = {
            "count": 2,
            "voices": [{"voice_id": "voice_fi"}, {"voice_id": "matti_speaker"}],
        }
        mock_voices_resp.raise_for_status = unittest.mock.MagicMock()
        mock_client.get.return_value = mock_voices_resp

        mock_tts_resp = unittest.mock.MagicMock(spec=httpx.Response)
        mock_tts_resp.content = b"audio-bytes"
        mock_tts_resp.raise_for_status = unittest.mock.MagicMock()
        mock_client.post.return_value = mock_tts_resp

        audio = await synthesize_speech(
            base_url="http://localhost:8080",
            text="Terve maailma!",
            voice="matti",
            client=mock_client,
        )

        self.assertEqual(audio, b"audio-bytes")
        call_args = mock_client.post.call_args
        self.assertEqual(call_args.kwargs["json"]["voice"], "matti_speaker")


    async def test_synthesize_speech_openai(self) -> None:
        from bot.commands.tts_logic import synthesize_speech_openai

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.content = b"mp3-bytes"
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_client.post.return_value = mock_response

        audio = await synthesize_speech_openai(
            base_url="http://localhost:8080",
            input_text="Hello world",
            voice="voice_fi",
            model="omnivoice",
            response_format="mp3",
            speed=1.0,
            client=mock_client,
        )

        self.assertEqual(audio, b"mp3-bytes")
        mock_client.post.assert_called_once_with(
            "http://localhost:8080/v1/audio/speech",
            json={
                "model": "omnivoice",
                "input": "Hello world",
                "voice": "voice_fi",
                "response_format": "mp3",
                "speed": 1.0,
            },
        )

    async def test_list_voices(self) -> None:
        from bot.commands.tts_logic import list_voices

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"count": 1, "voices": []}
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_client.get.return_value = mock_response

        result = await list_voices("http://localhost:8080", client=mock_client)
        self.assertEqual(result, {"count": 1, "voices": []})
        mock_client.get.assert_called_once_with("http://localhost:8080/api/v1/voices")

    async def test_reload_voices(self) -> None:
        from bot.commands.tts_logic import reload_voices

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.content = b'{"status": "reloaded"}'
        mock_response.json.return_value = {"status": "reloaded"}
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_client.post.return_value = mock_response

        result = await reload_voices("http://localhost:8080", client=mock_client)
        self.assertEqual(result, {"status": "reloaded"})
        mock_client.post.assert_called_once_with(
            "http://localhost:8080/api/v1/voices/reload"
        )

    async def test_check_health(self) -> None:
        from bot.commands.tts_logic import check_health

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"status": "ok", "model_loaded": True}
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_client.get.return_value = mock_response

        result = await check_health("http://localhost:8080", client=mock_client)
        self.assertEqual(result, {"status": "ok", "model_loaded": True})
        mock_client.get.assert_called_once_with("http://localhost:8080/health")


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
