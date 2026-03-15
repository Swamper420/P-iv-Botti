from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from typing import Any

import numpy as np
from PIL import Image

from bot.commands.naama_logic import compose_naama_image


def _png_bytes_from_rgb(array: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _create_asset(path: Path, color: tuple[int, int, int, int], size: tuple[int, int] = (60, 60)) -> None:
    image = Image.new("RGBA", size, color)
    image.save(path, format="PNG")


class _MaskWrapper:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def cpu(self) -> "_MaskWrapper":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _DummyResult:
    def __init__(
        self, masks: list[np.ndarray], classes: list[float], keypoints: np.ndarray | None = None
    ) -> None:
        self.masks = type("_DummyMasks", (), {"data": [_MaskWrapper(mask) for mask in masks]})()
        self.boxes = type("_DummyBoxes", (), {"cls": np.array(classes, dtype=np.float32)})()
        if keypoints is not None:
            self.keypoints = type(
                "_DummyKeypoints", (), {"data": _MaskWrapper(np.asarray(keypoints, dtype=np.float32))}
            )()


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
        self.last_predict_kwargs: dict[str, Any] = {}
        self.predict_calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: object) -> list[_DummyResult]:
        self.last_predict_kwargs = dict(kwargs)
        self.predict_calls.append(dict(kwargs))
        task = str(kwargs.get("task", "segment"))
        if task == "pose":
            if self._keypoints is None:
                return []
            return [_DummyResult([], [], keypoints=self._keypoints)]
        return [_DummyResult(self._masks or [], self._classes or [])]


class NaamaLogicTests(unittest.TestCase):
    def test_compose_naama_image_builds_profile_picture(self) -> None:
        source = np.full((80, 80, 3), 30, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source)
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[10:70, 20:60] = 1.0
        segment_model = _DummyModel([mask], [0.0])
        pose_keypoints = np.zeros((1, 17, 3), dtype=np.float32)
        pose_keypoints[0, 0] = [42.0, 28.0, 0.9]  # nose
        pose_keypoints[0, 5] = [28.0, 40.0, 0.9]  # left shoulder
        pose_keypoints[0, 6] = [53.0, 44.0, 0.9]  # right shoulder
        pose_keypoints[0, 9] = [24.0, 58.0, 0.9]  # left wrist
        pose_keypoints[0, 10] = [58.0, 62.0, 0.9]  # right wrist
        pose_model = _DummyModel(keypoints=pose_keypoints)
        loaded_model_names: list[str] = []

        def model_loader(model_name: str) -> _DummyModel:
            loaded_model_names.append(model_name)
            if model_name.endswith("-pose.pt"):
                return pose_model
            return segment_model

        with TemporaryDirectory() as tmp_dir:
            assets_dir = Path(tmp_dir)
            _create_asset(assets_dir / "background1.png", (20, 40, 90, 255), size=(80, 80))
            _create_asset(assets_dir / "hat1.png", (255, 0, 0, 180), size=(40, 20))
            _create_asset(assets_dir / "suit1.png", (0, 255, 0, 180), size=(60, 45))
            _create_asset(assets_dir / "gloves1.png", (0, 0, 255, 180), size=(60, 25))
            _create_asset(assets_dir / "cigar1.png", (255, 255, 0, 200), size=(18, 8))
            _create_asset(assets_dir / "sun1.png", (255, 140, 0, 220), size=(25, 25))

            output = compose_naama_image(
                source_bytes,
                assets_dir=assets_dir,
                model_name="yolo26n-seg.pt",
                model_loader=model_loader,
                random_seed=5,
            )

        self.assertIsNotNone(output)
        with Image.open(BytesIO(output or b"")) as image:
            output_rgb = np.asarray(image.convert("RGB"))

        self.assertEqual(output_rgb.shape, source.shape)
        self.assertFalse(np.array_equal(output_rgb, source))
        self.assertEqual(segment_model.last_predict_kwargs.get("conf"), 0.15)
        self.assertIn("yolo26n-seg.pt", loaded_model_names)
        self.assertIn("yolo26n-pose.pt", loaded_model_names)
        self.assertEqual(segment_model.predict_calls[0]["task"], "segment")
        self.assertEqual(pose_model.predict_calls[0]["task"], "pose")

    def test_compose_naama_image_uses_mask_fallback_when_pose_is_missing(self) -> None:
        source = np.full((80, 80, 3), 30, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source)
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[10:70, 20:60] = 1.0
        segment_model = _DummyModel([mask], [0.0])
        pose_model = _DummyModel(keypoints=None)

        def model_loader(model_name: str) -> _DummyModel:
            if model_name.endswith("-pose.pt"):
                return pose_model
            return segment_model

        with TemporaryDirectory() as tmp_dir:
            assets_dir = Path(tmp_dir)
            _create_asset(assets_dir / "background1.png", (20, 40, 90, 255), size=(80, 80))
            _create_asset(assets_dir / "hat1.png", (255, 0, 0, 180), size=(40, 20))
            _create_asset(assets_dir / "suit1.png", (0, 255, 0, 180), size=(60, 45))
            _create_asset(assets_dir / "gloves1.png", (0, 0, 255, 180), size=(60, 25))
            _create_asset(assets_dir / "cigar1.png", (255, 255, 0, 200), size=(18, 8))
            _create_asset(assets_dir / "sun1.png", (255, 140, 0, 220), size=(25, 25))

            output = compose_naama_image(
                source_bytes,
                assets_dir=assets_dir,
                model_name="yolo26n-seg.pt",
                model_loader=model_loader,
                random_seed=5,
            )

        self.assertIsNotNone(output)
        self.assertEqual(segment_model.predict_calls[0]["task"], "segment")
        self.assertEqual(pose_model.predict_calls[0]["task"], "pose")

    def test_compose_naama_image_returns_none_without_required_assets(self) -> None:
        source = np.full((60, 60, 3), 50, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source)
        mask = np.ones((60, 60), dtype=np.float32)
        model = _DummyModel([mask], [0.0])

        with TemporaryDirectory() as tmp_dir:
            assets_dir = Path(tmp_dir)
            _create_asset(assets_dir / "background1.png", (10, 10, 10, 255), size=(60, 60))
            _create_asset(assets_dir / "hat1.png", (255, 0, 0, 180))

            output = compose_naama_image(
                source_bytes,
                assets_dir=assets_dir,
                model_name="yolo26n-seg.pt",
                model_loader=lambda _: model,
            )

        self.assertIsNone(output)

    def test_compose_naama_image_returns_none_when_person_not_detected(self) -> None:
        source = np.full((40, 40, 3), 70, dtype=np.uint8)
        source_bytes = _png_bytes_from_rgb(source)
        mask = np.ones((40, 40), dtype=np.float32)
        model = _DummyModel([mask], [16.0])

        with TemporaryDirectory() as tmp_dir:
            assets_dir = Path(tmp_dir)
            _create_asset(assets_dir / "background1.png", (20, 40, 90, 255), size=(40, 40))
            _create_asset(assets_dir / "hat1.png", (255, 0, 0, 180), size=(20, 10))
            _create_asset(assets_dir / "suit1.png", (0, 255, 0, 180), size=(25, 20))
            _create_asset(assets_dir / "gloves1.png", (0, 0, 255, 180), size=(25, 14))
            _create_asset(assets_dir / "cigar1.png", (255, 255, 0, 200), size=(10, 4))
            _create_asset(assets_dir / "sun1.png", (255, 140, 0, 220), size=(14, 14))

            output = compose_naama_image(
                source_bytes,
                assets_dir=assets_dir,
                model_name="yolo26n-seg.pt",
                model_loader=lambda _: model,
            )

        self.assertIsNone(output)


if __name__ == "__main__":
    unittest.main()
