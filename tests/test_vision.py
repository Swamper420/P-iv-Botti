from __future__ import annotations

import base64
import unittest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from PIL import Image

from bot.vision import describe_image_with_eyes, prepare_image_data_url


def _png_bytes(width: int = 100, height: int = 60, color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color=color).save(output, format="PNG")
    return output.getvalue()


class PrepareImageDataUrlTests(unittest.TestCase):
    def test_returns_jpeg_data_url(self) -> None:
        url = prepare_image_data_url(_png_bytes())
        assert url is not None
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        raw = base64.b64decode(url.split(",", 1)[1])
        self.assertTrue(raw.startswith(b"\xff\xd8"))

    def test_downscales_longest_side(self) -> None:
        url = prepare_image_data_url(_png_bytes(width=2000, height=1000), max_dimension=1024)
        assert url is not None
        raw = base64.b64decode(url.split(",", 1)[1])
        with Image.open(BytesIO(raw)) as img:
            self.assertEqual(max(img.size), 1024)

    def test_invalid_bytes_return_none(self) -> None:
        self.assertIsNone(prepare_image_data_url(b""))
        self.assertIsNone(prepare_image_data_url(b"not-an-image"))


class DescribeImageWithEyesTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_extracts_content(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Kuvassa on kissa pöydällä."}}]
        }
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        result = await describe_image_with_eyes(
            _png_bytes(),
            "Kuvaile kuva",
            base_url="https://eyes.tsih.fi/v1",
            model="eyes-gemma-4-12b-q4",
            client=mock_client,
        )
        self.assertEqual(result, "Kuvassa on kissa pöydällä.")
        mock_client.post.assert_awaited_once()
        args, kwargs = mock_client.post.call_args
        self.assertTrue(args[0].endswith("/chat/completions"))
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "eyes-gemma-4-12b-q4")
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(
            content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )

    async def test_list_content_blocks_joined(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Kissa"},
                            {"type": "text", "text": "pöydällä"},
                        ]
                    }
                }
            ]
        }
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        result = await describe_image_with_eyes(
            _png_bytes(),
            "Kuvaile",
            base_url="https://eyes.tsih.fi/v1",
            model="m",
            client=mock_client,
        )
        self.assertIn("Kissa", result)
        self.assertIn("pöydällä", result)

    async def test_http_error_returns_empty(self) -> None:
        mock_client = AsyncMock()
        mock_client.post.side_effect = RuntimeError("boom")
        result = await describe_image_with_eyes(
            _png_bytes(),
            "Kuvaile",
            base_url="https://eyes.tsih.fi/v1",
            model="m",
            client=mock_client,
        )
        self.assertEqual(result, "")

    async def test_connect_error_returns_empty_without_traceback(self) -> None:
        """DNS/connect failures (as seen on hosts without Eyes route) must
        degrade to "" with a one-line warning, not an exception traceback."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError(
            "[Errno -2] Name or service not known"
        )
        with patch("bot.vision.LOGGER") as mock_logger:
            result = await describe_image_with_eyes(
                _png_bytes(),
                "Kuvaile",
                base_url="https://eyes.tsih.fi/v1",
                model="m",
                client=mock_client,
            )
        self.assertEqual(result, "")
        mock_logger.warning.assert_called_once()
        mock_logger.exception.assert_not_called()

    async def test_empty_image_or_prompt_returns_empty(self) -> None:
        mock_client = AsyncMock()
        self.assertEqual(
            await describe_image_with_eyes(
                b"", "Kuvaile", base_url="https://x", model="m", client=mock_client
            ),
            "",
        )
        self.assertEqual(
            await describe_image_with_eyes(
                _png_bytes(),
                "  ",
                base_url="https://x",
                model="m",
                client=mock_client,
            ),
            "",
        )
        mock_client.post.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
