from __future__ import annotations

import os
import random
import tempfile
import unittest
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from bot.commands.niilo_logic import (
    DETECTION_REPLY_TEMPLATES,
    NiiloMatchResult,
    clear_caches,
    detect_person_in_image,
    detect_person_in_video,
    format_niilo_detection_reply,
    load_reference_embeddings,
    match_faces_against_references,
)
from bot.config import NiiloConfig


@dataclass
class _FakeFace:
    embedding: np.ndarray
    det_score: float = 0.95
    bbox: tuple[float, float, float, float] = (10.0, 10.0, 110.0, 110.0)


def _make_jpeg_bytes(color: tuple[int, int, int] = (120, 80, 40), size: tuple[int, int] = (32, 32)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


class _FakeAnalyzer:
    """Deterministic fake FaceAnalysis keyed by the image's top-left red channel value."""

    def __init__(self, faces_by_red: dict[int, list[_FakeFace]]) -> None:
        self.faces_by_red = faces_by_red
        self.calls: int = 0

    def get(self, bgr_image: np.ndarray) -> list[_FakeFace]:
        self.calls += 1
        # BGR -> red channel is index 2
        red_value = int(bgr_image[0, 0, 2])
        # JPEG compression can shift uniform colors by +/- 3
        best_key = min(self.faces_by_red.keys(), key=lambda k: abs(k - red_value))
        if abs(best_key - red_value) <= 10:
            return self.faces_by_red[best_key]
        return []


class NiiloLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_caches()

    def tearDown(self) -> None:
        clear_caches()

    def _config(self, ref_dir: Path, **overrides: object) -> NiiloConfig:
        base = {
            "reference_dir": str(ref_dir),
            "det_threshold": 0.5,
            "similarity_threshold": 0.45,
        }
        base.update(overrides)
        return NiiloConfig(**base)  # type: ignore[arg-type]

    def test_load_reference_embeddings_empty_or_missing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist"
            cfg = self._config(missing)
            self.assertEqual(load_reference_embeddings(cfg), [])

            empty = Path(tmp) / "empty"
            empty.mkdir()
            (empty / "README.md").write_text("docs", encoding="utf-8")
            (empty / ".hidden.jpg").write_bytes(_make_jpeg_bytes())
            cfg_empty = self._config(empty)
            self.assertEqual(load_reference_embeddings(cfg_empty), [])

    def test_load_reference_embeddings_caches_and_reloads_on_change(self) -> None:
        niilo_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        niilo_vec2 = np.array([0.9, 0.43589, 0.0, 0.0], dtype=np.float32)
        small_face = _FakeFace(
            embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            bbox=(0.0, 0.0, 20.0, 20.0),
        )
        large_face = _FakeFace(
            embedding=niilo_vec * 5.0,  # unnormalized on purpose
            bbox=(0.0, 0.0, 200.0, 200.0),
        )
        analyzer = _FakeAnalyzer(
            {
                200: [small_face, large_face],
                100: [_FakeFace(embedding=niilo_vec2)],
                50: [],  # no face detected
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            (ref_dir / "ref1.jpg").write_bytes(_make_jpeg_bytes((200, 0, 0)))
            (ref_dir / "noface.jpg").write_bytes(_make_jpeg_bytes((50, 0, 0)))
            (ref_dir / "corrupt.jpg").write_bytes(b"not-an-image")

            cfg = self._config(ref_dir)
            embs1 = load_reference_embeddings(cfg, analyzer_loader=lambda _: analyzer)
            self.assertEqual(len(embs1), 1)
            # Largest face was selected and L2-normalized
            np.testing.assert_allclose(embs1[0], niilo_vec, atol=1e-5)
            calls_after_first = analyzer.calls

            # Second call with unchanged directory uses cache
            embs2 = load_reference_embeddings(cfg, analyzer_loader=lambda _: analyzer)
            self.assertEqual(analyzer.calls, calls_after_first)
            self.assertEqual(len(embs2), 1)

            # Adding a new reference photo invalidates cache automatically
            (ref_dir / "ref2.jpg").write_bytes(_make_jpeg_bytes((100, 0, 0)))
            embs3 = load_reference_embeddings(cfg, analyzer_loader=lambda _: analyzer)
            self.assertEqual(len(embs3), 2)
            self.assertGreater(analyzer.calls, calls_after_first)

    def test_match_faces_against_references_thresholds_and_multiface(self) -> None:
        ref = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
        stranger = _FakeFace(embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32), det_score=0.99)
        low_det_niilo = _FakeFace(embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32), det_score=0.2)
        real_niilo = _FakeFace(embedding=np.array([0.8, 0.6, 0.0], dtype=np.float32), det_score=0.9)

        found_low, sim_low = match_faces_against_references(
            [stranger, low_det_niilo], ref, det_threshold=0.5, similarity_threshold=0.45
        )
        self.assertFalse(found_low)
        self.assertAlmostEqual(sim_low, 0.0, places=4)

        found_ok, sim_ok = match_faces_against_references(
            [stranger, real_niilo], ref, det_threshold=0.5, similarity_threshold=0.45
        )
        self.assertTrue(found_ok)
        self.assertAlmostEqual(sim_ok, 0.8, places=4)

        found_strict, _ = match_faces_against_references(
            [real_niilo], ref, det_threshold=0.5, similarity_threshold=0.85
        )
        self.assertFalse(found_strict)

    def test_detect_person_in_image_match_and_mismatch(self) -> None:
        niilo_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        other_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        analyzer = _FakeAnalyzer(
            {
                210: [_FakeFace(embedding=niilo_vec)],
                30: [_FakeFace(embedding=other_vec)],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            (ref_dir / "niilo.jpg").write_bytes(_make_jpeg_bytes((210, 0, 0)))
            cfg = self._config(ref_dir)

            res_hit = detect_person_in_image(
                _make_jpeg_bytes((210, 0, 0)), cfg, analyzer_loader=lambda _: analyzer
            )
            self.assertTrue(res_hit.found)
            self.assertGreaterEqual(res_hit.similarity, 0.99)

            res_miss = detect_person_in_image(
                _make_jpeg_bytes((30, 0, 0)), cfg, analyzer_loader=lambda _: analyzer
            )
            self.assertFalse(res_miss.found)

            res_bad = detect_person_in_image(
                b"broken-bytes", cfg, analyzer_loader=lambda _: analyzer
            )
            self.assertFalse(res_bad.found)

    def test_detect_person_in_video_exits_early_on_first_match(self) -> None:
        niilo_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        other_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        analyzer = _FakeAnalyzer(
            {
                210: [_FakeFace(embedding=niilo_vec)],
                30: [_FakeFace(embedding=other_vec)],
            }
        )

        def _solid_bgr(red: int) -> np.ndarray:
            arr = np.zeros((16, 16, 3), dtype=np.uint8)
            arr[:, :, 2] = red
            return arr

        yielded_frames: list[int] = []

        def fake_frame_extractor(
            video_bytes: bytes, interval: float, max_frames: int, suffix: str
        ):
            frames = [
                (0, 0.0, _solid_bgr(30)),
                (25, 1.0, _solid_bgr(210)),
                (50, 2.0, _solid_bgr(30)),
                (75, 3.0, _solid_bgr(30)),
            ]
            for item in frames:
                yielded_frames.append(item[0])
                yield item

        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            (ref_dir / "niilo.jpg").write_bytes(_make_jpeg_bytes((210, 0, 0)))
            cfg = self._config(ref_dir)

            res = detect_person_in_video(
                b"fake-video-bytes",
                cfg,
                analyzer_loader=lambda _: analyzer,
                frame_extractor=fake_frame_extractor,
            )

        self.assertTrue(res.found)
        self.assertEqual(res.frame_index, 25)
        self.assertEqual(res.timestamp_seconds, 1.0)
        # Early exit: frame 50 and 75 were never requested
        self.assertEqual(yielded_frames, [0, 25])

    def test_detect_person_in_video_returns_false_when_not_found(self) -> None:
        niilo_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        other_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        analyzer = _FakeAnalyzer(
            {
                210: [_FakeFace(embedding=niilo_vec)],
                30: [_FakeFace(embedding=other_vec)],
            }
        )

        def _solid_bgr(red: int) -> np.ndarray:
            arr = np.zeros((16, 16, 3), dtype=np.uint8)
            arr[:, :, 2] = red
            return arr

        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            (ref_dir / "niilo.jpg").write_bytes(_make_jpeg_bytes((210, 0, 0)))
            cfg = self._config(ref_dir)

            res = detect_person_in_video(
                b"fake-video-bytes",
                cfg,
                analyzer_loader=lambda _: analyzer,
                frame_extractor=lambda *_: [(0, 0.0, _solid_bgr(30)), (25, 1.0, _solid_bgr(30))],
            )

        self.assertFalse(res.found)
        self.assertIsNone(res.frame_index)

    def test_format_niilo_detection_reply_variations(self) -> None:
        rng = random.Random(42)
        replies = {
            format_niilo_detection_reply("Niilo", NiiloMatchResult(found=True, similarity=0.7), rng=rng)
            for _ in range(50)
        }
        self.assertGreater(len(replies), 1)
        for reply in replies:
            self.assertIn("Niilo", reply)

        custom = format_niilo_detection_reply("Matti", rng=random.Random(0))
        self.assertIn("Matti", custom)
        self.assertGreaterEqual(len(DETECTION_REPLY_TEMPLATES), 5)


class NiiloMediaExtractionTests(unittest.TestCase):
    def test_extract_media_target_handles_photos_videos_and_documents(self) -> None:
        try:
            from bot.commands.niilo import _extract_media_target
        except ModuleNotFoundError:
            self.skipTest("python-telegram-bot not installed in this environment")

        photo_msg = SimpleNamespace(photo=["small", "large"], video=None, video_note=None, animation=None, document=None)
        self.assertEqual(_extract_media_target(photo_msg), ("image", "large", ".jpg"))

        video_obj = SimpleNamespace(file_name="clip.mov")
        video_msg = SimpleNamespace(photo=None, video=video_obj, video_note=None, animation=None, document=None)
        self.assertEqual(_extract_media_target(video_msg), ("video", video_obj, ".mov"))

        note_obj = SimpleNamespace()
        note_msg = SimpleNamespace(photo=None, video=None, video_note=note_obj, animation=None, document=None)
        self.assertEqual(_extract_media_target(note_msg), ("video", note_obj, ".mp4"))

        anim_obj = SimpleNamespace(file_name="funny.mp4")
        anim_msg = SimpleNamespace(photo=None, video=None, video_note=None, animation=anim_obj, document=None)
        self.assertEqual(_extract_media_target(anim_msg), ("video", anim_obj, ".mp4"))

        doc_img = SimpleNamespace(mime_type="image/png", file_name="pic.png")
        doc_img_msg = SimpleNamespace(photo=None, video=None, video_note=None, animation=None, document=doc_img)
        self.assertEqual(_extract_media_target(doc_img_msg), ("image", doc_img, ".png"))

        doc_gif = SimpleNamespace(mime_type="image/gif", file_name="loop.gif")
        doc_gif_msg = SimpleNamespace(photo=None, video=None, video_note=None, animation=None, document=doc_gif)
        self.assertEqual(_extract_media_target(doc_gif_msg), ("video", doc_gif, ".gif"))

        doc_vid = SimpleNamespace(mime_type="video/webm", file_name="vid.webm")
        doc_vid_msg = SimpleNamespace(photo=None, video=None, video_note=None, animation=None, document=doc_vid)
        self.assertEqual(_extract_media_target(doc_vid_msg), ("video", doc_vid, ".webm"))

        text_msg = SimpleNamespace(photo=None, video=None, video_note=None, animation=None, document=None)
        self.assertIsNone(_extract_media_target(text_msg))


if __name__ == "__main__":
    unittest.main()
