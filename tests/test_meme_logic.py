from __future__ import annotations

import unittest
from io import BytesIO

from PIL import Image

from bot.commands.meme_logic import generate_meme_image, generate_custom_meme, list_templates


class MemeLogicTests(unittest.TestCase):
    def test_list_templates(self) -> None:
        templates = list_templates()
        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)
        for t in templates:
            self.assertIn(":", t)

    def test_generate_meme_image_drake(self) -> None:
        img_bytes = generate_meme_image("drake", "TOP", "BOTTOM", style="classic")
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 100)
        with Image.open(BytesIO(img_bytes)) as img:
            self.assertEqual(img.size, (800, 800))

    def test_generate_meme_image_random_template(self) -> None:
        img_bytes = generate_meme_image("random", "TOP", "BOTTOM", style="classic")
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 100)

    def test_generate_meme_image_dev_style(self) -> None:
        img_bytes = generate_meme_image(style="dev")
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 100)

    def test_generate_meme_image_motivational_style(self) -> None:
        img_bytes = generate_meme_image(style="motivational")
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 100)

    def test_generate_meme_image_expanding_brain(self) -> None:
        img_bytes = generate_meme_image("expanding_brain", style="classic")
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 100)
        with Image.open(BytesIO(img_bytes)) as img:
            self.assertEqual(img.size, (800, 800))

    def test_generate_meme_image_distracted_boyfriend(self) -> None:
        img_bytes = generate_meme_image("distracted_boyfriend", "Me\n\n\nNew thing\n\n\nOld thing", style="classic")
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 100)
        with Image.open(BytesIO(img_bytes)) as img:
            self.assertEqual(img.size, (800, 533))

    def test_generate_meme_image_two_buttons(self) -> None:
        img_bytes = generate_meme_image("two_buttons", "Button A\nButton B", "Sweating", style="classic")
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 100)
        with Image.open(BytesIO(img_bytes)) as img:
            self.assertEqual(img.size, (800, 800))

    def test_generate_custom_meme(self) -> None:
        img_bytes = generate_custom_meme("HELLO WORLD")
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 100)
        with Image.open(BytesIO(img_bytes)) as img:
            self.assertEqual(img.size, (800, 600))

    def test_generate_custom_meme_multiline(self) -> None:
        img_bytes = generate_custom_meme("LINE 1\nLINE 2\nLINE 3")
        self.assertIsInstance(img_bytes, bytes)
        with Image.open(BytesIO(img_bytes)) as img:
            self.assertEqual(img.size, (800, 600))

    def test_generate_custom_meme_rainbow_style(self) -> None:
        img_bytes = generate_custom_meme("RAINBOW", style="rainbow")
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 100)

    def test_generate_custom_meme_glitch_style(self) -> None:
        img_bytes = generate_custom_meme("GLITCH", style="glitch")
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 100)


if __name__ == "__main__":
    unittest.main()