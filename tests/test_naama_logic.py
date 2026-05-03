from __future__ import annotations

from io import BytesIO
import unittest
from typing import Any

import numpy as np
from PIL import Image

from bot.commands.naama_logic import compose_naama_image


def _png_bytes_from_rgb(array: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(output, format="PNG")
    return output.getvalue()


class _MaskWrapper:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def cpu(self) -> "_MaskWrapper":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _DummyResult:
    def __init__(
        self,
        masks: list[np.ndarray],
        classes: list[float],
        keypoints: np.ndarray | None = None,
    ) -> None:
        if masks:
            self.masks = type("_DummyMasks", (), {"data": type("_DummyMasksData", (), {"cpu": lambda self: type("_DummyMasksCPU", (), {"numpy": lambda self: np.array([m.numpy() for m in self_outer._masks_wrapper])})()})()})()
            self._masks_wrapper = [_MaskWrapper(mask) for mask in masks]
            self_outer = self
        else:
            self.masks = None
            
        self.boxes = type(
            "_DummyBoxes",
            (),
            {
                "cls": type("_DummyCls", (), {"cpu": lambda self: type("_DummyClsCPU", (), {"numpy": lambda self: np.array(classes, dtype=np.float32)})()})()
            },
        )()
        
        if keypoints is not None:
            self.keypoints = type(
                "_DummyKeypoints", (), {"data": type("_DummyKptsData", (), {"cpu": lambda self: type("_DummyKptsCPU", (), {"numpy": lambda self: np.asarray(keypoints, dtype=np.float32)})()})()}
            )()
        else:
            self.keypoints = None


class _DummyModel:
    def __init__(
        self,
        masks: list[np.ndarray] | None = None,
        classes: list[float] | None = None,
        keypoints: np.ndarray | None = None,
    ) -> None:
        self._masks = masks
        self._classes = classes
        self._keypoints = keypoints

    def predict(self, **kwargs: object) -> list[_DummyResult]:
        task = str(kwargs.get("task", "segment"))
        if task == "pose":
            return [_DummyResult([], [], keypoints=self._keypoints)]
        return [_DummyResult(self._masks or [], self._classes or [])]


class NaamaLogicTests(unittest.TestCase):
    def test_compose_naama_image_mirrors_person(self) -> None:
        source = np.full((80, 80, 3), 30, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source)
        
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[10:70, 20:60] = 1.0  # person mask
        
        segment_model = _DummyModel([mask], [0.0])
        pose_model = _DummyModel(keypoints=None)

        def model_loader(model_name: str) -> _DummyModel:
            if "pose" in model_name:
                return pose_model
            return segment_model

        output = compose_naama_image(
            source_bytes,
            model_name="yolo26n-seg.pt",
            model_loader=model_loader,
            action="mirror",
            side="auto",
        )

        self.assertIsNotNone(output)
        with Image.open(BytesIO(output or b"")) as image:
            self.assertEqual(image.size, (80, 80))
            output_rgb = np.asarray(image.convert("RGB"))
            self.assertFalse(np.array_equal(output_rgb, source))

    def test_compose_naama_image_sticker(self) -> None:
        source = np.full((100, 100, 3), 30, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source)
        
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 1.0  # person mask
        
        segment_model = _DummyModel([mask], [0.0])
        pose_model = _DummyModel(keypoints=None)

        def model_loader(model_name: str) -> _DummyModel:
            return pose_model if "pose" in model_name else segment_model

        output = compose_naama_image(
            source_bytes,
            model_name="yolo26n-seg.pt",
            model_loader=model_loader,
            action="sticker",
            side="auto",
        )

        self.assertIsNotNone(output)
        with Image.open(BytesIO(output or b"")) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertLessEqual(image.width, 512)
            self.assertLessEqual(image.height, 512)

    def test_compose_naama_image_side_left(self) -> None:
        source = np.full((80, 80, 3), 30, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source)
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[10:70, 20:60] = 1.0
        segment_model = _DummyModel([mask], [0.0])
        pose_model = _DummyModel(keypoints=None)

        def model_loader(model_name: str) -> _DummyModel:
            return pose_model if "pose" in model_name else segment_model

        output = compose_naama_image(
            source_bytes,
            model_name="yolo26n-seg.pt",
            model_loader=model_loader,
            action="mirror",
            side="left",
        )

        self.assertIsNotNone(output)

    def test_compose_naama_image_side_up(self) -> None:
        source = np.full((80, 80, 3), 30, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source)
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[10:70, 20:60] = 1.0
        segment_model = _DummyModel([mask], [0.0])
        pose_model = _DummyModel(keypoints=None)

        def model_loader(model_name: str) -> _DummyModel:
            return pose_model if "pose" in model_name else segment_model

        output = compose_naama_image(
            source_bytes,
            model_name="yolo26n-seg.pt",
            model_loader=model_loader,
            action="mirror",
            side="up",
        )

        self.assertIsNotNone(output)

    def test_compose_naama_image_side_down(self) -> None:
        source = np.full((80, 80, 3), 30, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source)
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[10:70, 20:60] = 1.0
        segment_model = _DummyModel([mask], [0.0])
        pose_model = _DummyModel(keypoints=None)

        def model_loader(model_name: str) -> _DummyModel:
            return pose_model if "pose" in model_name else segment_model

        output = compose_naama_image(
            source_bytes,
            model_name="yolo26n-seg.pt",
            model_loader=model_loader,
            action="mirror",
            side="down",
        )

        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
