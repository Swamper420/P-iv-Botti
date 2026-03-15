from __future__ import annotations

import logging
import math
import random
import threading
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)
_PERSON_CLASS_ID = 0
_ACCESSORY_NAMES = ("hat", "cigar", "sun")
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_COCO_NOSE_INDEX = 0
_COCO_LEFT_EYE_INDEX = 1
_COCO_RIGHT_EYE_INDEX = 2
_COCO_LEFT_SHOULDER_INDEX = 5
_COCO_RIGHT_SHOULDER_INDEX = 6
_KEYPOINT_MIN_CONFIDENCE = 0.2
_HEAD_WIDTH_MAX_RATIO = 0.9
_HEAD_WIDTH_MIN_RATIO = 0.28
_HEAD_WIDTH_SHOULDER_RATIO = 0.72
_DEFAULT_MOUTH_Y_RATIO = 0.45
_MOUTH_WIDTH_MIN_RATIO = 0.14
_MOUTH_WIDTH_SHOULDER_RATIO = 0.26
_MOUTH_Y_OFFSET_MIN = 2.0
_MOUTH_Y_OFFSET_HEAD_RATIO = 0.38
_DEPTH_SCALE_RATIO = 0.86
_DEPTH_BOTTOM_MARGIN_RATIO = 0.02
_HAT_ROTATION_SCALE = 0.45
_CIGAR_ROTATION_SCALE = 0.75
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_LOAD_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class _NaamaAnchors:
    top_head: tuple[int, int]
    mouth: tuple[int, int]
    head_angle_degrees: float
    head_width: int
    mouth_width: int


@dataclass(frozen=True)
class _PersonDetection:
    mask: np.ndarray
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(1, self.right - self.left + 1)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top + 1)


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


def _extract_person_detection(
    image_rgb: np.ndarray,
    *,
    model_name: str,
    confidence_threshold: float,
    mask_threshold: float,
    model_loader: Callable[[str], Any] | None,
) -> _PersonDetection | None:
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
    candidate_detections: list[_PersonDetection] = []
    boxes_xyxy = getattr(getattr(first_result, "boxes", None), "xyxy", None)
    boxes_value: Any = boxes_xyxy
    cpu_fn = getattr(boxes_value, "cpu", None)
    if callable(cpu_fn):
        boxes_value = cpu_fn()
    numpy_fn = getattr(boxes_value, "numpy", None)
    if callable(numpy_fn):
        boxes_value = numpy_fn()
    try:
        boxes_array = np.asarray(boxes_value, dtype=np.float32)
    except (TypeError, ValueError):
        boxes_array = np.empty((0, 4), dtype=np.float32)

    for index, raw_mask in enumerate(masks_data):
        if index >= len(class_ids) or class_ids[index] != _PERSON_CLASS_ID:
            continue
        mask = _mask_to_bool(
            raw_mask, height=image_height, width=image_width, threshold=safe_mask_threshold
        )
        if mask is None or not mask.any():
            continue

        y_coords, x_coords = np.where(mask)
        left = int(x_coords.min())
        right = int(x_coords.max())
        top = int(y_coords.min())
        bottom = int(y_coords.max())
        if boxes_array.ndim == 2 and boxes_array.shape[1] >= 4 and index < boxes_array.shape[0]:
            raw_box = boxes_array[index]
            box_left = max(0, min(image_width - 1, int(round(float(raw_box[0])))))
            box_top = max(0, min(image_height - 1, int(round(float(raw_box[1])))))
            box_right = max(0, min(image_width - 1, int(round(float(raw_box[2])))))
            box_bottom = max(0, min(image_height - 1, int(round(float(raw_box[3])))))
            if box_right > box_left and box_bottom > box_top:
                left, top, right, bottom = box_left, box_top, box_right, box_bottom
        candidate_detections.append(
            _PersonDetection(mask=mask, left=left, top=top, right=right, bottom=bottom)
        )

    if not candidate_detections:
        return None

    return max(candidate_detections, key=lambda detection: int(np.count_nonzero(detection.mask)))


def _derive_pose_model_name(segmentation_model_name: str) -> str:
    lower_name = segmentation_model_name.lower()
    if "-seg" in lower_name:
        start_index = lower_name.index("-seg")
        return segmentation_model_name[:start_index] + "-pose" + segmentation_model_name[start_index + 4 :]
    return segmentation_model_name


def _extract_pose_keypoints(
    image_rgb: np.ndarray,
    *,
    model_name: str,
    confidence_threshold: float,
    model_loader: Callable[[str], Any] | None,
) -> np.ndarray | None:
    pose_model_name = _derive_pose_model_name(model_name)
    try:
        model = _get_model(pose_model_name, model_loader)
        results = model.predict(
            source=image_rgb,
            task="pose",
            device="cpu",
            conf=max(0.0, min(confidence_threshold, 1.0)),
            verbose=False,
        )
    except Exception:
        LOGGER.debug("Naama pose detection failed", exc_info=True)
        return None

    if not isinstance(results, (list, tuple)) or len(results) == 0:
        return None

    keypoints_data = getattr(getattr(results[0], "keypoints", None), "data", None)
    if keypoints_data is None:
        return None

    value: Any = keypoints_data
    cpu_fn = getattr(value, "cpu", None)
    if callable(cpu_fn):
        value = cpu_fn()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()

    try:
        keypoints_array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if keypoints_array.ndim == 2:
        keypoints_array = keypoints_array[np.newaxis, ...]
    if keypoints_array.ndim != 3 or keypoints_array.shape[1] <= _COCO_RIGHT_EYE_INDEX:
        return None
    if keypoints_array.shape[2] < 2:
        return None

    return keypoints_array


def _mask_point(mask: np.ndarray, x_ratio: float, y_ratio: float) -> tuple[int, int]:
    y_coords, x_coords = np.where(mask)
    left, right = int(x_coords.min()), int(x_coords.max())
    top, bottom = int(y_coords.min()), int(y_coords.max())
    width = max(1, right - left + 1)
    height = max(1, bottom - top + 1)
    point_x = left + int(round(width * x_ratio))
    point_y = top + int(round(height * y_ratio))
    return point_x, point_y


def _clamp_point(
    point: tuple[int, int], *, left: int, top: int, right: int, bottom: int
) -> tuple[int, int]:
    return (
        min(max(point[0], left), right),
        min(max(point[1], top), bottom),
    )


def _extract_anchor_from_keypoints(
    keypoints: np.ndarray | None, keypoint_index: int
) -> tuple[int, int] | None:
    if keypoints is None or keypoints.shape[0] == 0:
        return None

    person_points = keypoints[0]
    if person_points.shape[0] <= keypoint_index:
        return None

    point = person_points[keypoint_index]
    confidence = float(point[2]) if point.shape[0] > 2 else 1.0
    if confidence < _KEYPOINT_MIN_CONFIDENCE:
        return None

    return int(round(float(point[0]))), int(round(float(point[1])))


def _build_naama_anchors(
    person_mask: np.ndarray, person_detection: _PersonDetection, pose_keypoints: np.ndarray | None
) -> _NaamaAnchors:
    left, top, right, bottom = (
        person_detection.left,
        person_detection.top,
        person_detection.right,
        person_detection.bottom,
    )
    person_width = person_detection.width
    person_height = person_detection.height
    shoulder_span = max(1.0, float(person_width) * 0.35)

    y_coords, x_coords = np.where(person_mask)
    top_band = y_coords <= (top + max(1, int(person_height * 0.08)))
    top_x_values = x_coords[top_band] if np.any(top_band) else x_coords
    top_head_x = int(np.median(top_x_values))
    top_head = (top_head_x, top)

    nose = _extract_anchor_from_keypoints(pose_keypoints, _COCO_NOSE_INDEX)
    left_eye = _extract_anchor_from_keypoints(pose_keypoints, _COCO_LEFT_EYE_INDEX)
    right_eye = _extract_anchor_from_keypoints(pose_keypoints, _COCO_RIGHT_EYE_INDEX)
    left_shoulder = _extract_anchor_from_keypoints(pose_keypoints, _COCO_LEFT_SHOULDER_INDEX)
    right_shoulder = _extract_anchor_from_keypoints(pose_keypoints, _COCO_RIGHT_SHOULDER_INDEX)

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_span = max(
            shoulder_span, abs(float(right_shoulder[0]) - float(left_shoulder[0]))
        )

    head_width = max(
        1,
        int(
            round(
                min(
                    person_width * _HEAD_WIDTH_MAX_RATIO,
                    max(
                        person_width * _HEAD_WIDTH_MIN_RATIO,
                        shoulder_span * _HEAD_WIDTH_SHOULDER_RATIO,
                    ),
                )
            )
        ),
    )
    mouth_width = max(
        1,
        int(
            round(
                max(
                    person_width * _MOUTH_WIDTH_MIN_RATIO,
                    shoulder_span * _MOUTH_WIDTH_SHOULDER_RATIO,
                )
            )
        ),
    )

    default_mouth = _mask_point(person_mask, x_ratio=0.5, y_ratio=_DEFAULT_MOUTH_Y_RATIO)
    if nose is not None:
        mouth = (
            nose[0],
            int(round(nose[1] + max(_MOUTH_Y_OFFSET_MIN, head_width * _MOUTH_Y_OFFSET_HEAD_RATIO))),
        )
    else:
        mouth = default_mouth
    mouth = _clamp_point(mouth, left=left, top=top, right=right, bottom=bottom)

    if left_eye is not None and right_eye is not None:
        head_angle = math.degrees(
            math.atan2(
                float(right_eye[1] - left_eye[1]),
                float(right_eye[0] - left_eye[0]),
            )
        )
    elif left_shoulder is not None and right_shoulder is not None:
        head_angle = math.degrees(
            math.atan2(
                float(right_shoulder[1] - left_shoulder[1]),
                float(right_shoulder[0] - left_shoulder[0]),
            )
        )
    else:
        head_angle = 0.0

    return _NaamaAnchors(
        top_head=top_head,
        mouth=mouth,
        head_angle_degrees=head_angle,
        head_width=head_width,
        mouth_width=mouth_width,
    )


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
    canvas: Image.Image,
    overlay: Image.Image,
    *,
    center_x: int,
    center_y: int,
    width: int,
    angle_degrees: float = 0.0,
) -> None:
    prepared_overlay = ImageOps.exif_transpose(overlay)
    safe_width = max(1, width)
    aspect_ratio = prepared_overlay.height / max(prepared_overlay.width, 1)
    safe_height = max(1, int(round(safe_width * aspect_ratio)))
    resized_overlay = prepared_overlay.resize((safe_width, safe_height), Image.Resampling.LANCZOS).rotate(
        angle_degrees, resample=Image.Resampling.BICUBIC, expand=True
    )
    x = center_x - (resized_overlay.width // 2)
    y = center_y - (resized_overlay.height // 2)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(resized_overlay, (x, y), resized_overlay)
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
            source_rgba = ImageOps.exif_transpose(source_image).convert("RGBA")
    except OSError:
        return None

    image_rgb = np.asarray(source_rgba.convert("RGB"))
    person_detection = _extract_person_detection(
        image_rgb,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        mask_threshold=mask_threshold,
        model_loader=model_loader,
    )
    if person_detection is None:
        return None
    person_mask = person_detection.mask
    pose_keypoints = _extract_pose_keypoints(
        image_rgb,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        model_loader=model_loader,
    )

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
    depth_width = max(1, int(round(source_rgba.width * _DEPTH_SCALE_RATIO)))
    depth_height = max(1, int(round(source_rgba.height * _DEPTH_SCALE_RATIO)))
    depth_x = (source_rgba.width - depth_width) // 2
    bottom_margin = int(round(source_rgba.height * _DEPTH_BOTTOM_MARGIN_RATIO))
    depth_y = source_rgba.height - depth_height - bottom_margin
    depth_person = person_layer.resize((depth_width, depth_height), Image.Resampling.LANCZOS)
    depth_layer = Image.new("RGBA", source_rgba.size, (0, 0, 0, 0))
    depth_layer.paste(depth_person, (depth_x, depth_y), depth_person)
    composed = Image.alpha_composite(composed, depth_layer)

    person_height = person_detection.height
    anchors = _build_naama_anchors(person_mask, person_detection, pose_keypoints)
    scaled_person_height = max(1, int(round(person_height * _DEPTH_SCALE_RATIO)))
    scaled_top_head = (
        int(round(anchors.top_head[0] * _DEPTH_SCALE_RATIO)) + depth_x,
        int(round(anchors.top_head[1] * _DEPTH_SCALE_RATIO)) + depth_y,
    )
    scaled_mouth = (
        int(round(anchors.mouth[0] * _DEPTH_SCALE_RATIO)) + depth_x,
        int(round(anchors.mouth[1] * _DEPTH_SCALE_RATIO)) + depth_y,
    )

    with Image.open(rng.choice(accessory_candidates["hat"])) as hat_image:
        _paste_scaled_overlay(
            composed,
            hat_image.convert("RGBA"),
            center_x=scaled_top_head[0],
            center_y=scaled_top_head[1] - max(1, int(round(anchors.head_width * _DEPTH_SCALE_RATIO * 0.45))),
            width=max(1, int(round(anchors.head_width * _DEPTH_SCALE_RATIO * 1.35))),
            angle_degrees=anchors.head_angle_degrees * _HAT_ROTATION_SCALE,
        )
    with Image.open(rng.choice(accessory_candidates["cigar"])) as cigar_image:
        _paste_scaled_overlay(
            composed,
            cigar_image.convert("RGBA"),
            center_x=scaled_mouth[0] + int(round(anchors.mouth_width * _DEPTH_SCALE_RATIO * 0.65)),
            center_y=scaled_mouth[1] + int(scaled_person_height * 0.02),
            width=max(1, int(round(anchors.mouth_width * _DEPTH_SCALE_RATIO * 1.45))),
            angle_degrees=anchors.head_angle_degrees * _CIGAR_ROTATION_SCALE,
        )
    with Image.open(rng.choice(accessory_candidates["sun"])) as sun_image:
        _paste_scaled_overlay(
            composed,
            sun_image.convert("RGBA"),
            center_x=source_rgba.width - int(source_rgba.width * 0.14),
            center_y=int(source_rgba.height * 0.14),
            width=max(1, int(source_rgba.width * 0.22)),
        )

    return _encode_png(composed.convert("RGB"))
