import unittest
from unittest.mock import AsyncMock, MagicMock

from bot.commands.tiivista_logic import (
    create_3_word_caption,
    extract_text_from_html,
    extract_urls,
    fetch_webpage_text,
    parse_tiivista_command,
    summarize_text_with_ollama,
)


class TiivistaLogicTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_simple_url(self) -> None:
        matched, voice, content = parse_tiivista_command("!tiivistä https://example.com/article")
        self.assertTrue(matched)
        self.assertIsNone(voice)
        self.assertEqual(content, "https://example.com/article")

    def test_parse_voice_and_url(self) -> None:
        matched, voice, content = parse_tiivista_command("!tiivistä Matti https://example.com/article")
        self.assertTrue(matched)
        self.assertEqual(voice, "Matti")
        self.assertEqual(content, "https://example.com/article")

    def test_parse_tiivista_no_dots(self) -> None:
        matched, voice, content = parse_tiivista_command("!tiivista https://example.com")
        self.assertTrue(matched)
        self.assertIsNone(voice)
        self.assertEqual(content, "https://example.com")

    def test_parse_raw_text(self) -> None:
        matched, voice, content = parse_tiivista_command("!tiivistä Tämä on suoraa tekstiä tiivistettäväksi")
        self.assertTrue(matched)
        self.assertEqual(voice, "Tämä")
        self.assertEqual(content, "on suoraa tekstiä tiivistettäväksi")

    def test_parse_bare_command(self) -> None:
        matched, voice, content = parse_tiivista_command("!tiivistä")
        self.assertTrue(matched)
        self.assertIsNone(voice)
        self.assertEqual(content, "")

    def test_parse_non_matching(self) -> None:
        matched, voice, content = parse_tiivista_command("!weather Helsinki")
        self.assertFalse(matched)
        self.assertIsNone(voice)
        self.assertEqual(content, "")

    def test_extract_urls(self) -> None:
        text = "Katso tämä uutinen https://yle.fi/a/74-20000000 ja myös http://example.com!"
        urls = extract_urls(text)
        self.assertEqual(urls, ["https://yle.fi/a/74-20000000", "http://example.com"])

    def test_extract_text_from_html(self) -> None:
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Testi</title><script>var x = 1;</script></head>
        <body>
            <header><nav><a href="#">Linkki</a></nav></header>
            <style>body { color: red; }</style>
            <h1>Otsikko</h1>
            <p>Ensimmäinen kappale tekstiä.</p>
            <div>Toinen kappale.</div>
            <footer>Copyright 2026</footer>
        </body>
        </html>
        """
        extracted = extract_text_from_html(html)
        self.assertIn("Otsikko", extracted)
        self.assertIn("Ensimmäinen kappale tekstiä.", extracted)
        self.assertIn("Toinen kappale.", extracted)
        self.assertNotIn("var x = 1;", extracted)
        self.assertNotIn("color: red;", extracted)
        self.assertNotIn("Copyright 2026", extracted)

    async def test_fetch_webpage_text(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.content = b"<html><body><h1>Testisivu</h1><p>Sisaltoa</p></body></html>"
        mock_response.text = "<html><body><h1>Testisivu</h1><p>Sisaltoa</p></body></html>"
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        text = await fetch_webpage_text(
            url="https://test.fi",
            client=mock_client,
        )
        self.assertIn("Testisivu", text)
        self.assertIn("Sisaltoa", text)

    async def test_summarize_text_with_ollama(self) -> None:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def fake_aiter_lines():
            yield '{"response": "Tämä on ", "done": false}\n'
            yield '{"response": "tiivistelmä.", "done": true}\n'

        mock_response.aiter_lines = fake_aiter_lines

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_response

        mock_client = MagicMock()
        mock_client.stream.return_value = stream_context

        summary = await summarize_text_with_ollama(
            base_url="http://localhost:11434",
            model="gemma3",
            text="Pitkä uutinen tähän",
            client=mock_client,
        )
        self.assertEqual(summary, "Tämä on tiivistelmä.")

    def test_create_3_word_caption(self) -> None:
        summary = "Tämä on erittäin mielenkiintoinen tiivistelmä uutisesta."
        caption = create_3_word_caption(summary, max_words=3)
        self.assertEqual(caption, "Tämä on erittäin")


if __name__ == "__main__":
    unittest.main()
