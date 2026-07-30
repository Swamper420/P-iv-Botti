from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from bot.commands.paranna_logic import upscale_image


def _png_bytes_from_rgb(array: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(output, format="PNG")
    return output.getvalue()


class ParannaLogicTests(unittest.TestCase):
    def test_upscale_image_invalid_bytes(self) -> None:
        result = upscale_image(b"invalid image bytes")
        self.assertIsNone(result)

    def test_upscale_image_success(self) -> None:
        source_array = np.full((32, 32, 3), 120, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source_array)

        output_bytes = upscale_image(
            source_bytes,
            tile_size=64,
            tile_pad=4,
            max_input_dimension=1000,
            max_output_dimension=2000,
            jpeg_quality=90,
        )

        self.assertIsNotNone(output_bytes)
        with Image.open(BytesIO(output_bytes or b"")) as img:
            self.assertEqual(img.format, "JPEG")
            # 32x32 upscaled 4x -> 128x128
            self.assertEqual(img.size, (128, 128))

    def test_upscale_image_max_input_dimension_limit(self) -> None:
        # Create 100x50 image, set max_input_dimension to 50 -> input resized to 50x25 -> output 4x -> 200x100
        source_array = np.full((50, 100, 3), 150, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source_array)

        output_bytes = upscale_image(
            source_bytes,
            max_input_dimension=50,
            max_output_dimension=2000,
        )

        self.assertIsNotNone(output_bytes)
        with Image.open(BytesIO(output_bytes or b"")) as img:
            self.assertEqual(img.size, (200, 100))

    def test_upscale_image_max_output_dimension_limit(self) -> None:
        # Create 100x100 image -> upscaled 4x to 400x400 -> constrained to max_output_dimension=200
        source_array = np.full((100, 100, 3), 200, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source_array)

        output_bytes = upscale_image(
            source_bytes,
            max_input_dimension=1000,
            max_output_dimension=200,
        )

        self.assertIsNotNone(output_bytes)
        with Image.open(BytesIO(output_bytes or b"")) as img:
            self.assertEqual(img.size, (200, 200))


if __name__ == "__main__":
    unittest.main()
