from __future__ import annotations

import os
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from bot.commands.meme_logic import (
    NO_OBJECTS_RECOGNIZED,
    build_meme_prompt,
    fallback_caption,
    parse_meme_caption,
    parse_meme_command,
    render_meme_on_photo,
    sanitize_caption_line,
)
from bot.config import BotConfig


def _png_bytes(width: int = 640, height: int = 480, color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color=color).save(output, format="PNG")
    return output.getvalue()


class ParseMemeCommandTests(unittest.TestCase):
    def test_bare_command_matches_with_empty_hint(self) -> None:
        self.assertEqual(parse_meme_command("!meme"), (True, ""))

    def test_command_with_hint(self) -> None:
        self.assertEqual(parse_meme_command("!meme kissa"), (True, "kissa"))

    def test_command_is_case_insensitive_and_strips_hint(self) -> None:
        self.assertEqual(parse_meme_command("!MEME   vihje tähän  "), (True, "vihje tähän"))

    def test_multiline_hint_preserved(self) -> None:
        self.assertEqual(parse_meme_command("!meme eka\ntoka"), (True, "eka\ntoka"))

    def test_non_command_rejected(self) -> None:
        self.assertEqual(parse_meme_command("hello"), (False, ""))
        self.assertEqual(parse_meme_command(""), (False, ""))
        self.assertEqual(parse_meme_command("!memehelp"), (False, ""))


class BuildMemePromptTests(unittest.TestCase):
    def test_prompt_contains_all_sections(self) -> None:
        prompt = build_meme_prompt(
            image_description="Kuvasta tunnistettiin: 1 kissa.",
            ocr_text="MAITO 2€",
            user_hint="ruokakauppa",
        )
        self.assertIn("1 kissa", prompt)
        self.assertIn("MAITO 2€", prompt)
        self.assertIn("ruokakauppa", prompt)
        self.assertIn("YLÄ", prompt)

    def test_prompt_empty_when_nothing_to_describe(self) -> None:
        self.assertEqual(build_meme_prompt(), "")
        self.assertEqual(
            build_meme_prompt(image_description=NO_OBJECTS_RECOGNIZED), ""
        )

    def test_no_objects_sentinel_ignored_but_hint_kept(self) -> None:
        prompt = build_meme_prompt(
            image_description=NO_OBJECTS_RECOGNIZED, user_hint="vihje"
        )
        self.assertNotIn(NO_OBJECTS_RECOGNIZED, prompt)
        self.assertIn("vihje", prompt)


class ParseMemeCaptionTests(unittest.TestCase):
    def test_finnish_labels(self) -> None:
        self.assertEqual(
            parse_meme_caption("YLÄ: Kissa katsoo\nALA: Omistaja nukkuu"),
            ("Kissa katsoo", "Omistaja nukkuu"),
        )

    def test_english_labels_case_insensitive(self) -> None:
        self.assertEqual(
            parse_meme_caption("top: hello\nbottom: world"), ("hello", "world")
        )

    def test_numbered_labels(self) -> None:
        self.assertEqual(
            parse_meme_caption("1. YLÄ: foo\n2. ALA: bar"), ("foo", "bar")
        )

    def test_two_unlabeled_lines(self) -> None:
        self.assertEqual(parse_meme_caption("eka rivi\ntoka rivi"), ("eka rivi", "toka rivi"))

    def test_pipe_separated_single_line(self) -> None:
        self.assertEqual(parse_meme_caption("ylhäällä | alhaalla"), ("ylhäällä", "alhaalla"))

    def test_single_line_returns_empty_bottom(self) -> None:
        self.assertEqual(parse_meme_caption("vain yksi"), ("vain yksi", ""))

    def test_empty_input(self) -> None:
        self.assertEqual(parse_meme_caption(""), ("", ""))
        self.assertEqual(parse_meme_caption("   \n  "), ("", ""))

    def test_single_line_with_both_labels_split(self) -> None:
        # Regression: model emits both labels on one line; the embedded
        # second label must not leak into the top field.
        self.assertEqual(
            parse_meme_caption(
                "YLÄ: TYÖNHAKU PÄÄLLÄ? / ALA: HYVÄ, ETTÄ MUISTAT HAKEMUKSEN!"
            ),
            ("TYÖNHAKU PÄÄLLÄ?", "HYVÄ, ETTÄ MUISTAT HAKEMUKSEN!"),
        )

    def test_single_line_pipe_separated_labels(self) -> None:
        self.assertEqual(
            parse_meme_caption("TOP: foo | BOTTOM: bar"), ("foo", "bar")
        )

    def test_only_bottom_label_uses_preamble_as_top(self) -> None:
        self.assertEqual(
            parse_meme_caption("TYÖNHAKU PÄÄLLÄ? / ALA: HAUSKA!"),
            ("TYÖNHAKU PÄÄLLÄ?", "HAUSKA!"),
        )

    def test_label_inside_word_not_matched(self) -> None:
        self.assertEqual(
            parse_meme_caption("Salainen: juttu"), ("Salainen: juttu", "")
        )
        self.assertEqual(
            parse_meme_caption("Kissa on pöydän alla"),
            ("Kissa on pöydän alla", ""),
        )

    def test_reported_case_renders_both_fields(self) -> None:
        top, bottom = parse_meme_caption(
            "YLÄ: TYÖNHAKU PÄÄLLÄ? / ALA: HYVÄ, ETTÄ MUISTAT HAKEMUKSEN!"
        )
        top = sanitize_caption_line(top, 60)
        bottom = sanitize_caption_line(bottom, 60)
        self.assertTrue(top)
        self.assertTrue(bottom)
        self.assertNotIn("ALA:", top)


class SanitizeCaptionLineTests(unittest.TestCase):
    def test_uppercase_and_collapse(self) -> None:
        self.assertEqual(sanitize_caption_line("  kissa   katsoo  ", 60), "KISSA KATSOO")

    def test_strips_quotes(self) -> None:
        self.assertEqual(sanitize_caption_line('"hei maailma"', 60), "HEI MAAILMA")

    def test_truncates_with_ellipsis(self) -> None:
        result = sanitize_caption_line("abcdefghij", 5)
        self.assertEqual(result, "ABCD…")
        self.assertLessEqual(len(result), 5)

    def test_empty(self) -> None:
        self.assertEqual(sanitize_caption_line("   ", 60), "")


class FallbackCaptionTests(unittest.TestCase):
    def test_hint_and_description(self) -> None:
        top, bottom = fallback_caption(
            image_description="Kuvasta tunnistettiin: 1 kissa.",
            user_hint="maanantai",
        )
        self.assertEqual(top, "MAANANTAI")
        self.assertIn("KISSA", bottom)

    def test_ocr_only(self) -> None:
        top, bottom = fallback_caption(ocr_text="Ale -50%")
        self.assertEqual(top, "ALE -50%")
        self.assertEqual(bottom, "MEEMI")

    def test_description_only(self) -> None:
        top, bottom = fallback_caption(image_description="Kuvasta tunnistettiin: 2 koiraa.")
        self.assertEqual(top, "TÄMÄ KUVA")
        self.assertIn("KOIRAA", bottom)

    def test_all_empty(self) -> None:
        self.assertEqual(fallback_caption(), ("TÄMÄ KUVA", "MEEMI"))

    def test_sentinel_treated_as_empty(self) -> None:
        self.assertEqual(
            fallback_caption(image_description=NO_OBJECTS_RECOGNIZED),
            ("TÄMÄ KUVA", "MEEMI"),
        )

    def test_duplicate_top_bottom_avoided(self) -> None:
        top, bottom = fallback_caption(ocr_text="meemi", user_hint="meemi")
        self.assertEqual(top, "MEEMI")
        self.assertNotEqual(top, bottom)


class RenderMemeOnPhotoTests(unittest.TestCase):
    def test_renders_jpeg_with_captions(self) -> None:
        result = render_meme_on_photo(_png_bytes(), "YLÄTEKSTI", "ALATEKSTI")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.startswith(b"\xff\xd8"))
        with Image.open(BytesIO(result)) as img:
            self.assertEqual(img.size, (640, 480))

    def test_downscales_wide_image(self) -> None:
        result = render_meme_on_photo(
            _png_bytes(width=2000, height=1000), "A", "B", output_max_width=1080
        )
        self.assertIsNotNone(result)
        assert result is not None
        with Image.open(BytesIO(result)) as img:
            self.assertEqual(img.size, (1080, 540))

    def test_does_not_upscale_small_image(self) -> None:
        result = render_meme_on_photo(
            _png_bytes(width=320, height=200), "A", "B", output_max_width=1080
        )
        self.assertIsNotNone(result)
        assert result is not None
        with Image.open(BytesIO(result)) as img:
            self.assertEqual(img.size, (320, 200))

    def test_empty_captions_still_render_photo(self) -> None:
        result = render_meme_on_photo(_png_bytes(), "", "")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.startswith(b"\xff\xd8"))

    def test_invalid_bytes_return_none(self) -> None:
        self.assertIsNone(render_meme_on_photo(b"", "A", "B"))
        self.assertIsNone(render_meme_on_photo(b"not-an-image", "A", "B"))


class MemeConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        env = {"TELEGRAM_BOT_TOKEN": "token"}
        with patch("bot.config._load_env_file"):
            with patch.dict(os.environ, env, clear=True):
                config = BotConfig.from_environment()
        self.assertEqual(config.meme.yolo_model, "yolo26n.pt")
        self.assertEqual(config.meme.yolo_confidence_threshold, 0.25)
        self.assertTrue(config.meme.ocr_enabled)
        self.assertEqual(config.meme.caption_num_predict, 120)
        self.assertIn("YLÄ", config.meme.system_prompt)
        self.assertEqual(config.meme.output_max_width, 1080)
        self.assertEqual(config.meme.jpeg_quality, 90)

    def test_custom_values_from_environment(self) -> None:
        env = {
            "TELEGRAM_BOT_TOKEN": "token",
            "MEME_YOLO_MODEL": "yolo26s.pt",
            "MEME_YOLO_CONFIDENCE_THRESHOLD": "0.4",
            "MEME_OCR_ENABLED": "false",
            "MEME_CAPTION_NUM_PREDICT": "200",
            "MEME_MAX_TOP_CHARS": "42",
            "MEME_OUTPUT_MAX_WIDTH": "800",
            "MEME_JPEG_QUALITY": "80",
        }
        with patch("bot.config._load_env_file"):
            with patch.dict(os.environ, env, clear=True):
                config = BotConfig.from_environment()
        self.assertEqual(config.meme.yolo_model, "yolo26s.pt")
        self.assertEqual(config.meme.yolo_confidence_threshold, 0.4)
        self.assertFalse(config.meme.ocr_enabled)
        self.assertEqual(config.meme.caption_num_predict, 200)
        self.assertEqual(config.meme.max_top_chars, 42)
        self.assertEqual(config.meme.output_max_width, 800)
        self.assertEqual(config.meme.jpeg_quality, 80)

    def test_invalid_values_raise(self) -> None:
        invalid = {
            "MEME_MAX_IMAGE_BYTES": "0",
            "MEME_YOLO_CONFIDENCE_THRESHOLD": "1.5",
            "MEME_OCR_TIMEOUT_SECONDS": "0",
            "MEME_CAPTION_NUM_PREDICT": "0",
            "MEME_SYSTEM_PROMPT": "   ",
            "MEME_MAX_TOP_CHARS": "0",
            "MEME_MAX_BOTTOM_CHARS": "0",
            "MEME_OUTPUT_MAX_WIDTH": "50",
            "MEME_JPEG_QUALITY": "101",
        }
        for key, value in invalid.items():
            with self.subTest(key=key):
                with patch.dict(
                    os.environ,
                    {"TELEGRAM_BOT_TOKEN": "token", key: value},
                    clear=False,
                ):
                    with self.assertRaises(ValueError):
                        BotConfig.from_environment()


if __name__ == "__main__":
    unittest.main()
