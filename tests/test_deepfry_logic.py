from __future__ import annotations

from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from bot.commands.deepfry_logic import apply_segment_hue_overlay, segment_and_recolor_image


def _png_bytes_from_rgb(array: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _rgb_from_png(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


class _MaskWrapper:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def cpu(self) -> "_MaskWrapper":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _DummyResult:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self.masks = type("_DummyMasks", (), {"data": [_MaskWrapper(mask) for mask in masks]})()


class _DummyModel:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self._masks = masks

    def predict(self, **_: object) -> list[_DummyResult]:
        return [_DummyResult(self._masks)]


class DeepfryLogicTests(unittest.TestCase):
    def test_overlay_changes_only_segmented_pixels(self) -> None:
        image = np.array(
            [
                [[10, 10, 10], [20, 20, 20]],
                [[30, 30, 30], [40, 40, 40]],
            ],
            dtype=np.uint8,
        )
        mask = np.array([[True, False], [False, False]], dtype=np.bool_)

        output = apply_segment_hue_overlay(image, [mask], alpha=1.0, random_seed=123)

        self.assertFalse(np.array_equal(output[0, 0], image[0, 0]))
        self.assertTrue(np.array_equal(output[0, 1], image[0, 1]))
        self.assertTrue(np.array_equal(output[1, 0], image[1, 0]))

    def test_segment_and_recolor_uses_model_masks(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image_bytes = _png_bytes_from_rgb(image)
        mask = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)

        output = segment_and_recolor_image(
            image_bytes,
            model_name="yolo26n-seg.pt",
            alpha=1.0,
            model_loader=lambda _: _DummyModel([mask]),
            random_seed=1,
        )

        self.assertIsNotNone(output)
        output_rgb = _rgb_from_png(output or b"")
        self.assertFalse(np.array_equal(output_rgb[0, 0], image[0, 0]))
        self.assertTrue(np.array_equal(output_rgb[1, 1], image[1, 1]))

    def test_segment_and_recolor_returns_none_for_invalid_image(self) -> None:
        output = segment_and_recolor_image(
            b"not-a-real-image",
            model_name="yolo26n-seg.pt",
            alpha=0.45,
            model_loader=lambda _: _DummyModel([]),
        )
        self.assertIsNone(output)


if __name__ == "__main__":
    unittest.main()
