from __future__ import annotations

import logging
import math
import os
import random
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from bot.config import NiiloConfig

LOGGER = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
)

DETECTION_REPLY_TEMPLATES: tuple[str, ...] = (
    "{name} löydetty!",
    "{name} bongattu!",
    "Kas, {name}!",
    "{name} havaittu!",
    "{name} näkyvissä!",
    "Bongaus: {name}!",
    "Tuttu naama — {name} löydetty!",
    "Hei, tuossahan on {name}!",
    "Spotattu: {name}!",
    "Sieltähän se {name} löytyi!",
)

_ANALYZER_CACHE: dict[tuple[str, str, int], Any] = {}
_ANALYZER_LOCK = threading.Lock()

_REFERENCE_CACHE: dict[
    tuple[str, str, str, int],
    tuple[tuple[tuple[str, int, int], ...], list[np.ndarray]],
] = {}
_REFERENCE_LOCK = threading.Lock()


@dataclass(frozen=True)
class NiiloMatchResult:
    found: bool
    similarity: float = 0.0
    frame_index: int | None = None
    timestamp_seconds: float | None = None


def clear_caches() -> None:
    """Clear cached analyzers and reference embeddings (useful for tests)."""
    with _ANALYZER_LOCK:
        _ANALYZER_CACHE.clear()
    with _REFERENCE_LOCK:
        _REFERENCE_CACHE.clear()


def _get_analyzer(
    config: NiiloConfig,
    analyzer_loader: Callable[[NiiloConfig], Any] | None = None,
) -> Any:
    if analyzer_loader is not None:
        return analyzer_loader(config)

    model_root = Path(config.model_root).resolve()
    cache_key = (config.model_name, str(model_root), config.det_size)
    with _ANALYZER_LOCK:
        if cache_key not in _ANALYZER_CACHE:
            from insightface.app import FaceAnalysis

            model_root.mkdir(parents=True, exist_ok=True)
            analyzer = FaceAnalysis(
                name=config.model_name,
                root=str(model_root),
                providers=["CPUExecutionProvider"],
            )
            analyzer.prepare(
                ctx_id=-1,
                det_size=(config.det_size, config.det_size),
            )
            _ANALYZER_CACHE[cache_key] = analyzer
        return _ANALYZER_CACHE[cache_key]


def _decode_image_to_bgr(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            rgb_img = ImageOps.exif_transpose(img).convert("RGB")
            rgb_arr = np.asarray(rgb_img, dtype=np.uint8)
    except OSError:
        return None

    if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
        return None
    # InsightFace expects BGR uint8 numpy arrays (OpenCV convention)
    return np.ascontiguousarray(rgb_arr[:, :, ::-1])


def _extract_normalized_embedding(face: Any) -> np.ndarray | None:
    raw = getattr(face, "normed_embedding", None)
    if raw is None:
        raw = getattr(face, "embedding", None)
    if raw is None:
        return None

    vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return None
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return vec / norm


def _face_det_score(face: Any) -> float:
    score = getattr(face, "det_score", None)
    if score is None:
        return 1.0
    return float(score)


def _face_bbox_area(face: Any) -> float:
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return 0.0
    try:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except (IndexError, TypeError, ValueError):
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _list_reference_files(reference_dir: Path) -> list[Path]:
    if not reference_dir.exists() or not reference_dir.is_dir():
        return []
    files: list[Path] = []
    for entry in sorted(reference_dir.iterdir(), key=lambda p: p.name.casefold()):
        if not entry.is_file() or entry.name.startswith("."):
            continue
        if entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            files.append(entry)
    return files


def _compute_dir_signature(files: list[Path]) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        signature.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _select_reference_face(faces: Iterable[Any], det_threshold: float) -> Any | None:
    valid = [face for face in faces if _face_det_score(face) >= det_threshold]
    if not valid:
        return None
    # Prefer the largest face in the reference image, breaking ties by detection score
    return max(valid, key=lambda f: (_face_bbox_area(f), _face_det_score(f)))


def load_reference_embeddings(
    config: NiiloConfig,
    *,
    analyzer_loader: Callable[[NiiloConfig], Any] | None = None,
) -> list[np.ndarray]:
    """Load and cache L2-normalized reference face embeddings from config.reference_dir.

    Automatically reloads when files are added, modified, or removed in the directory.
    """
    reference_dir = Path(config.reference_dir).resolve()
    files = _list_reference_files(reference_dir)
    signature = _compute_dir_signature(files)

    cache_key = (
        str(reference_dir),
        config.model_name,
        str(Path(config.model_root).resolve()),
        config.det_size,
    )

    with _REFERENCE_LOCK:
        cached = _REFERENCE_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

    if not files:
        LOGGER.info(
            "No reference images found for %s in %s; skipping detection.",
            config.person_name,
            reference_dir,
        )
        with _REFERENCE_LOCK:
            _REFERENCE_CACHE[cache_key] = (signature, [])
        return []

    analyzer = _get_analyzer(config, analyzer_loader=analyzer_loader)
    embeddings: list[np.ndarray] = []

    for image_path in files:
        try:
            raw_bytes = image_path.read_bytes()
        except OSError:
            LOGGER.warning("Failed to read reference image: %s", image_path)
            continue

        bgr_image = _decode_image_to_bgr(raw_bytes)
        if bgr_image is None:
            LOGGER.warning("Could not decode reference image: %s", image_path)
            continue

        faces = analyzer.get(bgr_image)
        selected = _select_reference_face(faces, config.det_threshold)
        if selected is None:
            LOGGER.warning("No face detected in reference image: %s", image_path)
            continue

        embedding = _extract_normalized_embedding(selected)
        if embedding is None:
            LOGGER.warning("No valid face embedding in reference image: %s", image_path)
            continue

        embeddings.append(embedding)

    LOGGER.info(
        "Loaded %d reference face embedding(s) for %s from %s.",
        len(embeddings),
        config.person_name,
        reference_dir,
    )

    with _REFERENCE_LOCK:
        _REFERENCE_CACHE[cache_key] = (signature, embeddings)

    return embeddings


def match_faces_against_references(
    faces: Iterable[Any],
    reference_embeddings: list[np.ndarray],
    *,
    det_threshold: float,
    similarity_threshold: float,
) -> tuple[bool, float]:
    """Compare detected faces against reference embeddings using cosine similarity."""
    if not reference_embeddings:
        return False, 0.0

    best_similarity = 0.0
    for face in faces:
        if _face_det_score(face) < det_threshold:
            continue
        embedding = _extract_normalized_embedding(face)
        if embedding is None:
            continue
        for ref in reference_embeddings:
            if ref.shape != embedding.shape:
                continue
            sim = float(np.dot(embedding, ref))
            if sim > best_similarity:
                best_similarity = sim

    return best_similarity >= similarity_threshold, best_similarity


def detect_person_in_image(
    image_bytes: bytes,
    config: NiiloConfig,
    *,
    analyzer_loader: Callable[[NiiloConfig], Any] | None = None,
) -> NiiloMatchResult:
    """Detect whether the configured person appears in the given image bytes."""
    reference_embeddings = load_reference_embeddings(
        config, analyzer_loader=analyzer_loader
    )
    if not reference_embeddings:
        return NiiloMatchResult(found=False, similarity=0.0)

    bgr_image = _decode_image_to_bgr(image_bytes)
    if bgr_image is None:
        return NiiloMatchResult(found=False, similarity=0.0)

    analyzer = _get_analyzer(config, analyzer_loader=analyzer_loader)
    faces = analyzer.get(bgr_image)
    found, similarity = match_faces_against_references(
        faces,
        reference_embeddings,
        det_threshold=config.det_threshold,
        similarity_threshold=config.similarity_threshold,
    )
    return NiiloMatchResult(found=found, similarity=similarity)


def _iter_video_frames(
    video_bytes: bytes,
    *,
    sample_interval_seconds: float,
    max_frames: int,
    suffix: str = ".mp4",
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield sampled (frame_index, timestamp_seconds, bgr_frame) tuples from video bytes."""
    import cv2

    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    tmp_path: str | None = None
    cap: Any = None
    try:
        with tempfile.NamedTemporaryFile(suffix=safe_suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            LOGGER.warning("Could not open video stream for face detection.")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0.0:
            fps = 25.0

        step = max(1, int(round(fps * sample_interval_seconds)))
        frame_idx = 0
        yielded = 0

        while yielded < max_frames:
            if frame_idx % step == 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                timestamp = frame_idx / fps
                yield frame_idx, timestamp, frame
                yielded += 1
            else:
                if not cap.grab():
                    break
            frame_idx += 1
    finally:
        if cap is not None:
            cap.release()
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def detect_person_in_video(
    video_bytes: bytes,
    config: NiiloConfig,
    *,
    suffix: str = ".mp4",
    analyzer_loader: Callable[[NiiloConfig], Any] | None = None,
    frame_extractor: Callable[
        [bytes, float, int, str], Iterable[tuple[int, float, np.ndarray]]
    ]
    | None = None,
) -> NiiloMatchResult:
    """Detect whether the configured person appears in sampled frames of a video.

    Exits early on the first frame that matches the person.
    """
    if not video_bytes:
        return NiiloMatchResult(found=False, similarity=0.0)

    reference_embeddings = load_reference_embeddings(
        config, analyzer_loader=analyzer_loader
    )
    if not reference_embeddings:
        return NiiloMatchResult(found=False, similarity=0.0)

    analyzer = _get_analyzer(config, analyzer_loader=analyzer_loader)
    if frame_extractor is not None:
        frames = frame_extractor(
            video_bytes,
            config.video_sample_interval_seconds,
            config.max_video_frames,
            suffix,
        )
    else:
        frames = _iter_video_frames(
            video_bytes,
            sample_interval_seconds=config.video_sample_interval_seconds,
            max_frames=config.max_video_frames,
            suffix=suffix,
        )

    best_similarity = 0.0
    for frame_idx, timestamp, frame_bgr in frames:
        faces = analyzer.get(frame_bgr)
        found, similarity = match_faces_against_references(
            faces,
            reference_embeddings,
            det_threshold=config.det_threshold,
            similarity_threshold=config.similarity_threshold,
        )
        if similarity > best_similarity:
            best_similarity = similarity
        if found:
            return NiiloMatchResult(
                found=True,
                similarity=similarity,
                frame_index=frame_idx,
                timestamp_seconds=timestamp,
            )

    return NiiloMatchResult(found=False, similarity=best_similarity)


def format_niilo_detection_reply(
    person_name: str,
    result: NiiloMatchResult | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Format a playful Finnish reply when the target person is detected."""
    clean_name = person_name.strip() or "Niilo"
    chooser = rng if rng is not None else random
    template = chooser.choice(DETECTION_REPLY_TEMPLATES)
    return template.format(name=clean_name)
