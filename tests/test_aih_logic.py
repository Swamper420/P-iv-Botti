import unittest
from unittest.mock import AsyncMock, MagicMock

from bot.commands.aih_logic import (
    filter_thinking_tags,
    parse_aih_command,
    stream_ollama_completion,
)


class AihLogicTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_default_command(self) -> None:
        matched, tokens, strip_thinking, prompt = parse_aih_command("!aih Kerro vitsi")
        self.assertTrue(matched)
        self.assertEqual(tokens, 100)
        self.assertTrue(strip_thinking)
        self.assertEqual(prompt, "Kerro vitsi")

    def test_parse_colon_syntax(self) -> None:
        matched, tokens, strip_thinking, prompt = parse_aih_command("!aih: Kerro tarina")
        self.assertTrue(matched)
        self.assertEqual(tokens, 100)
        self.assertTrue(strip_thinking)
        self.assertEqual(prompt, "Kerro tarina")

    def test_parse_tokens_attached(self) -> None:
        matched, tokens, strip_thinking, prompt = parse_aih_command("!aih400: Kerro pitkä tarina")
        self.assertTrue(matched)
        self.assertEqual(tokens, 400)
        self.assertTrue(strip_thinking)
        self.assertEqual(prompt, "Kerro pitkä tarina")

    def test_parse_tokens_without_exclamation(self) -> None:
        matched, tokens, strip_thinking, prompt = parse_aih_command("aih400: Kerro tarina")
        self.assertTrue(matched)
        self.assertEqual(tokens, 400)
        self.assertTrue(strip_thinking)
        self.assertEqual(prompt, "Kerro tarina")

    def test_parse_nothink_syntax(self) -> None:
        matched, tokens, strip_thinking, prompt = parse_aih_command("!aihnothink400: Kerro tarina")
        self.assertTrue(matched)
        self.assertEqual(tokens, 400)
        self.assertTrue(strip_thinking)
        self.assertEqual(prompt, "Kerro tarina")

    def test_parse_tokens_spaced(self) -> None:
        matched, tokens, strip_thinking, prompt = parse_aih_command("!aih 250: Kerro runo")
        self.assertTrue(matched)
        self.assertEqual(tokens, 250)
        self.assertTrue(strip_thinking)
        self.assertEqual(prompt, "Kerro runo")

    def test_parse_bare_command(self) -> None:
        matched, tokens, strip_thinking, prompt = parse_aih_command("!aih")
        self.assertTrue(matched)
        self.assertEqual(tokens, 100)
        self.assertTrue(strip_thinking)
        self.assertEqual(prompt, "")

    def test_parse_token_capping(self) -> None:
        matched, tokens, strip_thinking, prompt = parse_aih_command(
            "!aih5000: prompt", default_tokens=100, max_tokens=2000
        )
        self.assertTrue(matched)
        self.assertEqual(tokens, 2000)
        self.assertTrue(strip_thinking)

    def test_parse_non_matching(self) -> None:
        matched, tokens, strip_thinking, prompt = parse_aih_command("!weather Helsinki")
        self.assertFalse(matched)

    async def test_filter_thinking_tags(self) -> None:
        async def sample_stream():
            yield "<think>\nReasoning steps...\n</think>"
            yield "Vastaus "
            yield "on 42."

        filtered = []
        async for chunk in filter_thinking_tags(sample_stream()):
            filtered.append(chunk)

        self.assertEqual("".join(filtered), "Vastaus on 42.")

    async def test_stream_ollama_completion_strips_thinking(self) -> None:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def fake_aiter_lines():
            yield '{"response": "<think>thinking...</think>Hei ", "done": false}\n'
            yield '{"response": "maailma!", "done": true}\n'

        mock_response.aiter_lines = fake_aiter_lines

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_response

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_context

        chunks = []
        async for chunk in stream_ollama_completion(
            base_url="http://localhost:11434",
            model="gemma4",
            prompt="Hei",
            num_predict=100,
            strip_thinking=True,
            client=mock_client,
        ):
            chunks.append(chunk)

        self.assertEqual("".join(chunks), "Hei maailma!")
        mock_client.stream.assert_called_once_with(
            "POST",
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma4",
                "prompt": "Hei",
                "stream": True,
                "options": {"num_predict": 100, "think": False},
                "system": (
                    "Do not include internal thinking processes, reasoning steps, or <think> tags. "
                    "Answer directly and concisely."
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
