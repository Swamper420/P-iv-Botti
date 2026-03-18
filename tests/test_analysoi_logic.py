import base64
import unittest

from bot.commands.analysoi_logic import build_analysoi_prompt, encode_image_base64


class BuildAnalysoiPromptTests(unittest.TestCase):
    def test_returns_non_empty_string(self) -> None:
        prompt = build_analysoi_prompt()
        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 0)

    def test_prompt_is_in_finnish(self) -> None:
        prompt = build_analysoi_prompt()
        self.assertIn("suome", prompt.lower())

    def test_prompt_mentions_analysis(self) -> None:
        prompt = build_analysoi_prompt()
        lower = prompt.lower()
        self.assertTrue(
            "analysoi" in lower or "analyysi" in lower,
            "Prompt should mention analysis",
        )


class EncodeImageBase64Tests(unittest.TestCase):
    def test_encodes_bytes_to_base64(self) -> None:
        raw = b"\x89PNG\r\n\x1a\nfake-image-data"
        encoded = encode_image_base64(raw)
        self.assertEqual(base64.b64decode(encoded), raw)

    def test_returns_ascii_string(self) -> None:
        encoded = encode_image_base64(b"hello")
        self.assertIsInstance(encoded, str)
        encoded.encode("ascii")  # should not raise

    def test_empty_bytes(self) -> None:
        encoded = encode_image_base64(b"")
        self.assertEqual(encoded, "")


if __name__ == "__main__":
    unittest.main()
