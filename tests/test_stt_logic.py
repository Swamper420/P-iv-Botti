from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

import httpx

from bot.commands.stt_logic import (
    check_stt_health,
    format_stt_response,
    transcribe_audio,
)
from bot.config import SttConfig


class SttLogicTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = SttConfig(
            base_url="http://localhost:8001",
            timeout_seconds=60,
            beam_size=5,
            vad_filter=True,
            word_timestamps=False,
            initial_prompt="Context prompt",
        )

    async def test_transcribe_audio_success(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "status": "success",
            "filename": "puhe.wav",
            "text": "Tämä on esimerkki puheentunnistuksesta suomeksi.",
            "language": "fi",
            "duration": 3.45,
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        text = await transcribe_audio(
            file_bytes=b"fake_audio_bytes",
            filename="puhe.wav",
            config=self.config,
            httpx_client=mock_client,
        )

        self.assertEqual(text, "Tämä on esimerkki puheentunnistuksesta suomeksi.")
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args.kwargs
        self.assertEqual(call_kwargs["data"]["beam_size"], "5")
        self.assertEqual(call_kwargs["data"]["vad_filter"], "true")
        self.assertEqual(call_kwargs["data"]["word_timestamps"], "false")
        self.assertEqual(call_kwargs["data"]["initial_prompt"], "Context prompt")

    async def test_transcribe_audio_empty_file_bytes(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        text = await transcribe_audio(
            file_bytes=b"",
            filename="empty.wav",
            config=self.config,
            httpx_client=mock_client,
        )
        self.assertIsNone(text)
        mock_client.post.assert_not_called()

    async def test_transcribe_audio_http_error(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        text = await transcribe_audio(
            file_bytes=b"fake_bytes",
            filename="audio.wav",
            config=self.config,
            httpx_client=mock_client,
        )
        self.assertIsNone(text)

    async def test_transcribe_audio_empty_text_in_response(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "success", "text": "   "}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        text = await transcribe_audio(
            file_bytes=b"fake_bytes",
            filename="audio.wav",
            config=self.config,
            httpx_client=mock_client,
        )
        self.assertIsNone(text)

    async def test_check_stt_health_success(self) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "status": "healthy",
            "model": "RASMUS/whisper-large-v3-turbo-finnish-ct2",
            "language": "fi",
            "device": "cuda",
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        health = await check_stt_health(self.config, httpx_client=mock_client)
        self.assertIsNotNone(health)
        self.assertEqual(health.get("status"), "healthy")
        mock_client.get.assert_called_once_with(
            "http://localhost:8001/health",
            timeout=60,
        )

    async def test_check_stt_health_failure(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = RuntimeError("Network error")

        health = await check_stt_health(self.config, httpx_client=mock_client)
        self.assertIsNone(health)

    def test_format_stt_response(self) -> None:
        self.assertEqual(
            format_stt_response("Hei maailma"),
            "💬 Hei maailma",
        )
        self.assertEqual(format_stt_response(""), "")
        self.assertEqual(format_stt_response("   "), "")


if __name__ == "__main__":
    unittest.main()
