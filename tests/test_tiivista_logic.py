from io import BytesIO
import unittest
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from PIL import Image

from bot.commands.tiivista_logic import (
    create_3_word_caption,
    extract_text_from_html,
    extract_urls,
    fetch_webpage_text,
    parse_tiivista_command,
    recognize_objects_with_yolo,
    summarize_text_with_ollama,
)


def _create_dummy_image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (50, 50), color="blue").save(output, format="PNG")
    return output.getvalue()


class _DummyBoxes:
    def __init__(self, classes: list[float]) -> None:
        self._classes_array = np.array(classes, dtype=np.float32)
        cls_mock = MagicMock()
        cls_mock.cpu().numpy.return_value = self._classes_array
        self.cls = cls_mock

    def __len__(self) -> int:
        return len(self._classes_array)


class _DummyYoloResult:
    def __init__(self, classes: list[float], names: dict[int, str]) -> None:
        self.boxes = _DummyBoxes(classes) if classes else None
        self.names = names


class _DummyYoloModel:
    def __init__(self, classes: list[float], names: dict[int, str]) -> None:
        self._classes = classes
        self._names = names

    def predict(self, **kwargs: object) -> list[_DummyYoloResult]:
        return [_DummyYoloResult(self._classes, self._names)]


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

    def test_recognize_objects_with_yolo_success(self) -> None:
        img_bytes = _create_dummy_image_bytes()
        dummy_model = _DummyYoloModel(
            classes=[0.0, 0.0, 16.0, 2.0],
            names={0: "person", 16: "dog", 2: "car"},
        )
        result = recognize_objects_with_yolo(
            img_bytes,
            model_name="yolo26n.pt",
            model_loader=lambda name: dummy_model,
        )
        self.assertIn("Kuvasta tunnistettiin:", result)
        self.assertIn("2 henkilöä", result)
        self.assertIn("1 koira", result)
        self.assertIn("1 auto", result)

    def test_recognize_objects_with_yolo_no_detections(self) -> None:
        img_bytes = _create_dummy_image_bytes()
        dummy_model = _DummyYoloModel(classes=[], names={})
        result = recognize_objects_with_yolo(
            img_bytes,
            model_name="yolo26n.pt",
            model_loader=lambda name: dummy_model,
        )
        self.assertEqual(result, "Kuvassa ei tunnistettu kohteita.")

    def test_recognize_objects_with_yolo_invalid_bytes(self) -> None:
        result = recognize_objects_with_yolo(b"invalid image bytes")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
