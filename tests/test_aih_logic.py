import unittest
from unittest.mock import AsyncMock, MagicMock

from bot.commands.aih_logic import (
    parse_aih_command,
    stream_ollama_completion,
    strip_think_blocks,
)


class AihLogicTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_default_command(self) -> None:
        matched, tokens, prompt = parse_aih_command("!aih Kerro vitsi")
        self.assertTrue(matched)
        self.assertEqual(tokens, 100)
        self.assertEqual(prompt, "Kerro vitsi")

    def test_parse_colon_syntax(self) -> None:
        matched, tokens, prompt = parse_aih_command("!aih: Kerro tarina")
        self.assertTrue(matched)
        self.assertEqual(tokens, 100)
        self.assertEqual(prompt, "Kerro tarina")

    def test_parse_tokens_attached(self) -> None:
        matched, tokens, prompt = parse_aih_command("!aih400: Kerro pitkä tarina")
        self.assertTrue(matched)
        self.assertEqual(tokens, 400)
        self.assertEqual(prompt, "Kerro pitkä tarina")

    def test_parse_tokens_without_exclamation(self) -> None:
        matched, tokens, prompt = parse_aih_command("aih400: Kerro tarina")
        self.assertTrue(matched)
        self.assertEqual(tokens, 400)
        self.assertEqual(prompt, "Kerro tarina")

    def test_parse_tokens_spaced(self) -> None:
        matched, tokens, prompt = parse_aih_command("!aih 250: Kerro runo")
        self.assertTrue(matched)
        self.assertEqual(tokens, 250)
        self.assertEqual(prompt, "Kerro runo")

    def test_parse_bare_command(self) -> None:
        matched, tokens, prompt = parse_aih_command("!aih")
        self.assertTrue(matched)
        self.assertEqual(tokens, 100)
        self.assertEqual(prompt, "")

    def test_parse_token_capping(self) -> None:
        matched, tokens, prompt = parse_aih_command(
            "!aih5000: prompt", default_tokens=100, max_tokens=2000
        )
        self.assertTrue(matched)
        self.assertEqual(tokens, 2000)

    def test_parse_non_matching(self) -> None:
        matched, tokens, prompt = parse_aih_command("!weather Helsinki")
        self.assertFalse(matched)

    async def test_stream_ollama_completion(self) -> None:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def fake_aiter_lines():
            yield '{"response": "Hei ", "done": false}\n'
            yield '{"response": "maailma!", "done": true}\n'

        mock_response.aiter_lines = fake_aiter_lines

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_response

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_context

        chunks = []
        async for chunk in stream_ollama_completion(
            base_url="http://localhost:11434",
            model="gemma3",
            prompt="Hei",
            num_predict=100,
            client=mock_client,
        ):
            chunks.append(chunk)

        self.assertEqual("".join(chunks), "Hei maailma!")
        mock_client.stream.assert_called_once_with(
            "POST",
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3",
                "prompt": "Hei",
                "stream": True,
                "think": False,
                "options": {"num_predict": 100, "num_ctx": 2048},
                "system": "Vastaa aina suomeksi suoraan ja tiiviisti.",
            },
        )

    async def test_stream_ollama_ignores_thinking_field(self) -> None:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def fake_aiter_lines():
            yield '{"thinking": "secret reasoning", "response": "", "done": false}\n'
            yield '{"response": "Valmis vastaus.", "done": true}\n'

        mock_response.aiter_lines = fake_aiter_lines

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_response

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_context

        chunks = []
        async for chunk in stream_ollama_completion(
            base_url="http://localhost:11434",
            model="lfm3",
            prompt="Hei",
            num_predict=100,
            client=mock_client,
        ):
            chunks.append(chunk)

        self.assertEqual("".join(chunks), "Valmis vastaus.")

    async def test_stream_ollama_strips_inline_think_block(self) -> None:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def fake_aiter_lines():
            yield '{"response": "<think>salainen pohdinta</think>Vastaus.", "done": true}\n'

        mock_response.aiter_lines = fake_aiter_lines

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_response

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_context

        chunks = []
        async for chunk in stream_ollama_completion(
            base_url="http://localhost:11434",
            model="lfm3",
            prompt="Hei",
            num_predict=100,
            client=mock_client,
        ):
            chunks.append(chunk)

        self.assertEqual("".join(chunks), "Vastaus.")

    async def test_stream_ollama_strips_split_think_tags(self) -> None:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def fake_aiter_lines():
            yield '{"response": "<thi", "done": false}\n'
            yield '{"response": "nk>salainen pohdinta</thi", "done": false}\n'
            yield '{"response": "nk>Vastaus.", "done": true}\n'

        mock_response.aiter_lines = fake_aiter_lines

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_response

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_context

        chunks = []
        async for chunk in stream_ollama_completion(
            base_url="http://localhost:11434",
            model="lfm3",
            prompt="Hei",
            num_predict=100,
            client=mock_client,
        ):
            chunks.append(chunk)

        self.assertEqual("".join(chunks), "Vastaus.")

    async def test_stream_ollama_strips_prefilled_think(self) -> None:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def fake_aiter_lines():
            yield '{"response": "pohdintaa pohdintaa</think>Vastaus.", "done": true}\n'

        mock_response.aiter_lines = fake_aiter_lines

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_response

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_context

        chunks = []
        async for chunk in stream_ollama_completion(
            base_url="http://localhost:11434",
            model="lfm3",
            prompt="Hei",
            num_predict=100,
            client=mock_client,
        ):
            chunks.append(chunk)

        self.assertEqual("".join(chunks), "Vastaus.")

    async def test_stream_ollama_drops_truncated_thinking(self) -> None:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def fake_aiter_lines():
            yield '{"response": "<think>kesken jaanyt pohdinta", "done": true}\n'

        mock_response.aiter_lines = fake_aiter_lines

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_response

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_context

        chunks = []
        async for chunk in stream_ollama_completion(
            base_url="http://localhost:11434",
            model="lfm3",
            prompt="Hei",
            num_predict=100,
            client=mock_client,
        ):
            chunks.append(chunk)

        self.assertEqual("".join(chunks), "")

    async def test_stream_ollama_think_none_omits_key(self) -> None:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def fake_aiter_lines():
            yield '{"response": "ok", "done": true}\n'

        mock_response.aiter_lines = fake_aiter_lines

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_response

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_context

        async for _ in stream_ollama_completion(
            base_url="http://localhost:11434",
            model="lfm3",
            prompt="Hei",
            num_predict=100,
            client=mock_client,
            think=None,
        ):
            pass

        sent_json = mock_client.stream.call_args.kwargs["json"]
        self.assertNotIn("think", sent_json)


class StripThinkBlocksTests(unittest.TestCase):
    def test_plain_text_unchanged(self) -> None:
        self.assertEqual(
            strip_think_blocks("Vastaus suomeksi."), "Vastaus suomeksi."
        )

    def test_inline_block_removed(self) -> None:
        self.assertEqual(
            strip_think_blocks("<think>pohdinta</think>Vastaus."), "Vastaus."
        )

    def test_prefilled_close_removed(self) -> None:
        self.assertEqual(
            strip_think_blocks("pohdintaa pohdintaa</think>Vastaus."),
            "Vastaus.",
        )

    def test_truncated_thinking_dropped(self) -> None:
        self.assertEqual(strip_think_blocks("<think>kesken..."), "")

    def test_word_think_kept(self) -> None:
        self.assertEqual(
            strip_think_blocks("I think this is good."),
            "I think this is good.",
        )

    def test_trailing_stray_tag_kept_answer(self) -> None:
        self.assertEqual(strip_think_blocks("Vastaus.</think>"), "Vastaus.")


if __name__ == "__main__":
    unittest.main()
