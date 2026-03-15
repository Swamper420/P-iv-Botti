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
_ACCESSORY_NAMES = ("hat", "suit", "gloves", "cigar", "sun")
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_COCO_NOSE_INDEX = 0
_COCO_LEFT_SHOULDER_INDEX = 5
_COCO_RIGHT_SHOULDER_INDEX = 6
_COCO_LEFT_WRIST_INDEX = 9
_COCO_RIGHT_WRIST_INDEX = 10
_KEYPOINT_MIN_CONFIDENCE = 0.2
_BODY_CENTER_BELOW_SHOULDERS_RATIO = 0.33
_BODY_CENTER_FALLBACK_OFFSET_RATIO = 0.16
_HAT_ROTATION_SCALE = 0.45
_SUIT_ROTATION_SCALE = 0.25
_GLOVES_ROTATION_SCALE = 0.8
_CIGAR_ROTATION_SCALE = 0.75
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_LOAD_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class _NaamaAnchors:
    top_head: tuple[int, int]
    mouth: tuple[int, int]
    left_hand: tuple[int, int]
    right_hand: tuple[int, int]
    body_center: tuple[int, int]
    shoulder_angle_degrees: float
    hands_angle_degrees: float
    head_width: int
    mouth_width: int
    hands_width: int
    body_width: int


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
    if keypoints_array.ndim != 3 or keypoints_array.shape[1] <= _COCO_RIGHT_WRIST_INDEX:
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
    center_x = left + (person_width // 2)
    center_y = top + (person_height // 2)
    shoulder_span = max(1.0, float(person_width) * 0.35)

    y_coords, x_coords = np.where(person_mask)
    top_band = y_coords <= (top + max(1, int(person_height * 0.08)))
    top_x_values = x_coords[top_band] if np.any(top_band) else x_coords
    top_head_x = int(np.median(top_x_values))
    top_head = (top_head_x, top)

    nose = _extract_anchor_from_keypoints(pose_keypoints, _COCO_NOSE_INDEX)
    left_shoulder = _extract_anchor_from_keypoints(pose_keypoints, _COCO_LEFT_SHOULDER_INDEX)
    right_shoulder = _extract_anchor_from_keypoints(pose_keypoints, _COCO_RIGHT_SHOULDER_INDEX)
    left_wrist = _extract_anchor_from_keypoints(pose_keypoints, _COCO_LEFT_WRIST_INDEX)
    right_wrist = _extract_anchor_from_keypoints(pose_keypoints, _COCO_RIGHT_WRIST_INDEX)

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_span = max(
            shoulder_span, abs(float(right_shoulder[0]) - float(left_shoulder[0]))
        )

    head_width = max(1, int(round(min(person_width * 0.9, max(person_width * 0.28, shoulder_span * 0.72)))))
    mouth_width = max(1, int(round(max(person_width * 0.14, shoulder_span * 0.26))))
    hand_box_width = max(1, int(round(max(person_width * 0.14, shoulder_span * 0.22))))
    body_width = max(1, int(round(max(person_width * 0.65, shoulder_span * 1.15))))

    default_mouth = _mask_point(person_mask, x_ratio=0.5, y_ratio=0.45)
    if nose is not None:
        mouth = (nose[0], int(round(nose[1] + max(2.0, head_width * 0.38))))
    else:
        mouth = default_mouth
    mouth = _clamp_point(mouth, left=left, top=top, right=right, bottom=bottom)

    left_hand = left_wrist if left_wrist is not None else _mask_point(person_mask, x_ratio=0.2, y_ratio=0.63)
    right_hand = (
        right_wrist if right_wrist is not None else _mask_point(person_mask, x_ratio=0.8, y_ratio=0.63)
    )
    left_hand = _clamp_point(left_hand, left=left, top=top, right=right, bottom=bottom)
    right_hand = _clamp_point(right_hand, left=left, top=top, right=right, bottom=bottom)
    hands_angle = math.degrees(
        math.atan2(float(right_hand[1] - left_hand[1]), float(right_hand[0] - left_hand[0]))
    )

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_angle = math.degrees(
            math.atan2(
                float(right_shoulder[1] - left_shoulder[1]),
                float(right_shoulder[0] - left_shoulder[0]),
            )
        )
        body_center = (
            int(round((left_shoulder[0] + right_shoulder[0]) / 2.0)),
            int(
                round(
                    (left_shoulder[1] + right_shoulder[1]) / 2.0
                    + person_height * _BODY_CENTER_BELOW_SHOULDERS_RATIO
                )
            ),
        )
    else:
        shoulder_angle = hands_angle
        body_center = (center_x, center_y + int(round(person_height * _BODY_CENTER_FALLBACK_OFFSET_RATIO)))
    body_center = _clamp_point(body_center, left=left, top=top, right=right, bottom=bottom)

    return _NaamaAnchors(
        top_head=top_head,
        mouth=mouth,
        left_hand=left_hand,
        right_hand=right_hand,
        body_center=body_center,
        shoulder_angle_degrees=shoulder_angle,
        hands_angle_degrees=hands_angle,
        head_width=head_width,
        mouth_width=mouth_width,
        hands_width=max(hand_box_width * 2, abs(right_hand[0] - left_hand[0]) + hand_box_width),
        body_width=body_width,
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
    composed = Image.alpha_composite(composed, person_layer)

    person_height = person_detection.height
    anchors = _build_naama_anchors(person_mask, person_detection, pose_keypoints)
    hand_midpoint_x = int(round((anchors.left_hand[0] + anchors.right_hand[0]) / 2.0))
    hand_midpoint_y = int(round((anchors.left_hand[1] + anchors.right_hand[1]) / 2.0))

    with Image.open(rng.choice(accessory_candidates["hat"])) as hat_image:
        _paste_scaled_overlay(
            composed,
            hat_image.convert("RGBA"),
            center_x=anchors.top_head[0],
            center_y=anchors.top_head[1] - max(1, int(round(anchors.head_width * 0.45))),
            width=max(1, int(round(anchors.head_width * 1.35))),
            angle_degrees=anchors.shoulder_angle_degrees * _HAT_ROTATION_SCALE,
        )
    with Image.open(rng.choice(accessory_candidates["suit"])) as suit_image:
        _paste_scaled_overlay(
            composed,
            suit_image.convert("RGBA"),
            center_x=anchors.body_center[0],
            center_y=anchors.body_center[1] + int(person_height * 0.2),
            width=max(1, int(round(anchors.body_width * 1.2))),
            angle_degrees=anchors.shoulder_angle_degrees * _SUIT_ROTATION_SCALE,
        )
    with Image.open(rng.choice(accessory_candidates["gloves"])) as gloves_image:
        _paste_scaled_overlay(
            composed,
            gloves_image.convert("RGBA"),
            center_x=hand_midpoint_x,
            center_y=hand_midpoint_y,
            width=max(1, int(round(anchors.hands_width * 1.1))),
            angle_degrees=anchors.hands_angle_degrees * _GLOVES_ROTATION_SCALE,
        )
    with Image.open(rng.choice(accessory_candidates["cigar"])) as cigar_image:
        _paste_scaled_overlay(
            composed,
            cigar_image.convert("RGBA"),
            center_x=anchors.mouth[0] + int(round(anchors.mouth_width * 0.65)),
            center_y=anchors.mouth[1] + int(person_height * 0.02),
            width=max(1, int(round(anchors.mouth_width * 1.45))),
            angle_degrees=anchors.shoulder_angle_degrees * _CIGAR_ROTATION_SCALE,
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
