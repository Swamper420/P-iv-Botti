from __future__ import annotations

import colorsys
import logging
import random
import threading
from collections.abc import Callable, Sequence
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)
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


def _encode_png(image_rgb: np.ndarray) -> bytes:
    output_buffer = BytesIO()
    Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").save(output_buffer, format="PNG")
    return output_buffer.getvalue()


def _mask_to_bool(mask: object, *, height: int, width: int) -> np.ndarray | None:
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

    return mask_array > 0.5


def apply_segment_hue_overlay(
    image_rgb: np.ndarray,
    masks: Sequence[np.ndarray],
    *,
    alpha: float,
    random_seed: int | None = None,
) -> np.ndarray:
    if alpha <= 0 or not masks:
        return image_rgb.copy()

    overlay_alpha = min(alpha, 1.0)
    rng = random.Random(random_seed)
    output = image_rgb.astype(np.float32).copy()

    for mask in masks:
        if mask.dtype != np.bool_:
            continue
        if mask.shape != image_rgb.shape[:2]:
            continue
        hue = rng.random()
        red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color = np.array((red, green, blue), dtype=np.float32) * 255.0
        output[mask] = output[mask] * (1.0 - overlay_alpha) + color * overlay_alpha

    return np.clip(output, 0, 255).astype(np.uint8)


def segment_and_recolor_image(
    image_bytes: bytes,
    *,
    model_name: str,
    alpha: float,
    model_loader: Callable[[str], Any] | None = None,
    random_seed: int | None = None,
) -> bytes | None:
    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            image_rgb = np.asarray(source_image.convert("RGB"))
    except OSError:
        return None

    try:
        model = _get_model(model_name, model_loader)
        results = model.predict(
            source=image_rgb,
            task="segment",
            device="cpu",
            verbose=False,
        )
    except Exception:
        LOGGER.exception("Image segmentation failed")
        return None

    if not results:
        return _encode_png(image_rgb)

    masks_data = getattr(getattr(results[0], "masks", None), "data", None)
    if masks_data is None:
        return _encode_png(image_rgb)

    image_height, image_width = image_rgb.shape[:2]
    masks: list[np.ndarray] = []
    for raw_mask in masks_data:
        mask = _mask_to_bool(raw_mask, height=image_height, width=image_width)
        if mask is not None:
            masks.append(mask)

    recolored = apply_segment_hue_overlay(
        image_rgb, masks, alpha=alpha, random_seed=random_seed
    )
    return _encode_png(recolored)
