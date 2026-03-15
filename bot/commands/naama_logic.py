from __future__ import annotations

import logging
import random
import threading
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)
_PERSON_CLASS_ID = 0
_ACCESSORY_NAMES = ("hat", "suit", "gloves", "cigar", "sun")
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_LOAD_LOCKS: dict[str, threading.Lock] = {}


def _default_model_loader(model_name: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_name)


def _get_model(model_name: str, model_loader: Callable[[str], Any] | None) -> Any:
    if model_loader is not None:
        return model_loader(model_name)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached
        load_lock = _MODEL_LOAD_LOCKS.setdefault(model_name, threading.Lock())

    with load_lock:
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(model_name)
            if cached is not None:
                return cached

        model = _default_model_loader(model_name)

        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[model_name] = model
            return model


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _mask_to_bool(mask: object, *, height: int, width: int, threshold: float) -> np.ndarray | None:
    value: Any = mask
    cpu_fn = getattr(value, "cpu", None)
    if callable(cpu_fn):
        value = cpu_fn()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()

    try:
        mask_array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if mask_array.ndim != 2:
        return None

    if mask_array.shape != (height, width):
        resized = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        )
        mask_array = np.asarray(resized, dtype=np.float32) / 255.0

    return mask_array > threshold


def _extract_person_mask(
    image_rgb: np.ndarray,
    *,
    model_name: str,
    confidence_threshold: float,
    mask_threshold: float,
    model_loader: Callable[[str], Any] | None,
) -> np.ndarray | None:
    try:
        model = _get_model(model_name, model_loader)
        results = model.predict(
            source=image_rgb,
            task="segment",
            device="cpu",
            conf=max(0.0, min(confidence_threshold, 1.0)),
            verbose=False,
        )
    except Exception:
        LOGGER.exception("Naama image segmentation failed")
        return None

    if not isinstance(results, (list, tuple)) or len(results) == 0:
        return None

    first_result = results[0]
    masks_data = getattr(getattr(first_result, "masks", None), "data", None)
    class_values = getattr(getattr(first_result, "boxes", None), "cls", None)
    if masks_data is None or class_values is None:
        return None

    try:
        class_ids = np.asarray(class_values, dtype=np.float32).astype(np.int32).tolist()
    except (TypeError, ValueError):
        return None

    safe_mask_threshold = max(0.0, min(mask_threshold, 1.0))
    image_height, image_width = image_rgb.shape[:2]
    person_mask = np.zeros((image_height, image_width), dtype=np.bool_)
    for index, raw_mask in enumerate(masks_data):
        if index >= len(class_ids) or class_ids[index] != _PERSON_CLASS_ID:
            continue
        mask = _mask_to_bool(
            raw_mask, height=image_height, width=image_width, threshold=safe_mask_threshold
        )
        if mask is not None:
            person_mask |= mask

    return person_mask if person_mask.any() else None


def _load_assets_by_prefix(assets_dir: Path, prefix: str) -> list[Path]:
    if not assets_dir.exists():
        return []
    return sorted(
        [
            path
            for path in assets_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in _ALLOWED_EXTENSIONS
            and path.stem.lower().startswith(prefix)
        ]
    )


def _paste_scaled_overlay(
    canvas: Image.Image, overlay: Image.Image, *, center_x: int, top_y: int, width: int
) -> None:
    safe_width = max(1, width)
    aspect_ratio = overlay.height / max(overlay.width, 1)
    safe_height = max(1, int(round(safe_width * aspect_ratio)))
    resized_overlay = overlay.resize((safe_width, safe_height), Image.Resampling.LANCZOS)
    x = center_x - (safe_width // 2)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(resized_overlay, (x, top_y), resized_overlay)
    composited = Image.alpha_composite(canvas, layer)
    canvas.paste(composited)


def compose_naama_image(
    image_bytes: bytes,
    *,
    assets_dir: Path,
    model_name: str,
    confidence_threshold: float = 0.15,
    mask_threshold: float = 0.35,
    model_loader: Callable[[str], Any] | None = None,
    random_seed: int | None = None,
) -> bytes | None:
    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            source_rgba = source_image.convert("RGBA")
    except OSError:
        return None

    image_rgb = np.asarray(source_rgba.convert("RGB"))
    person_mask = _extract_person_mask(
        image_rgb,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        mask_threshold=mask_threshold,
        model_loader=model_loader,
    )
    if person_mask is None:
        return None

    background_candidates = _load_assets_by_prefix(assets_dir, "background")
    if not background_candidates:
        return None

    accessory_candidates: dict[str, list[Path]] = {
        name: _load_assets_by_prefix(assets_dir, name) for name in _ACCESSORY_NAMES
    }
    if any(not paths for paths in accessory_candidates.values()):
        return None

    rng = random.Random(random_seed)
    with Image.open(rng.choice(background_candidates)) as background_image:
        composed = background_image.convert("RGBA").resize(source_rgba.size, Image.Resampling.LANCZOS)

    alpha_channel = (person_mask.astype(np.uint8) * 255)
    person_pixels = np.asarray(source_rgba, dtype=np.uint8).copy()
    person_pixels[..., 3] = alpha_channel
    person_layer = Image.fromarray(person_pixels, mode="RGBA")
    composed = Image.alpha_composite(composed, person_layer)

    y_coords, x_coords = np.where(person_mask)
    left, right = int(x_coords.min()), int(x_coords.max())
    top, bottom = int(y_coords.min()), int(y_coords.max())
    person_width = max(1, right - left + 1)
    person_height = max(1, bottom - top + 1)
    center_x = left + (person_width // 2)

    with Image.open(rng.choice(accessory_candidates["hat"])) as hat_image:
        _paste_scaled_overlay(
            composed,
            hat_image.convert("RGBA"),
            center_x=center_x,
            top_y=top - int(person_height * 0.35),
            width=int(person_width * 0.95),
        )
    with Image.open(rng.choice(accessory_candidates["suit"])) as suit_image:
        _paste_scaled_overlay(
            composed,
            suit_image.convert("RGBA"),
            center_x=center_x,
            top_y=bottom - int(person_height * 0.2),
            width=int(person_width * 1.15),
        )
    with Image.open(rng.choice(accessory_candidates["gloves"])) as gloves_image:
        _paste_scaled_overlay(
            composed,
            gloves_image.convert("RGBA"),
            center_x=center_x,
            top_y=bottom - int(person_height * 0.45),
            width=int(person_width * 1.1),
        )
    with Image.open(rng.choice(accessory_candidates["cigar"])) as cigar_image:
        _paste_scaled_overlay(
            composed,
            cigar_image.convert("RGBA"),
            center_x=right - int(person_width * 0.08),
            top_y=top + int(person_height * 0.58),
            width=max(1, int(person_width * 0.28)),
        )
    with Image.open(rng.choice(accessory_candidates["sun"])) as sun_image:
        _paste_scaled_overlay(
            composed,
            sun_image.convert("RGBA"),
            center_x=source_rgba.width - int(source_rgba.width * 0.14),
            top_y=int(source_rgba.height * 0.03),
            width=max(1, int(source_rgba.width * 0.22)),
        )

    return _encode_png(composed.convert("RGB"))
