from __future__ import annotations

import logging
import math
import random
import threading
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0
_ACCESSORY_NAMES = ("hat", "cigar", "sun")
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_KEYPOINT_MIN_CONFIDENCE = 0.4

# Depth settings
_DEPTH_SCALE_RATIO = 0.86
_DEPTH_BOTTOM_MARGIN_RATIO = 0.02

# YOLO caching to prevent reloading models in memory
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_model(model_name: str) -> Any:
    with _MODEL_CACHE_LOCK:
        if model_name not in _MODEL_CACHE:
            from ultralytics import YOLO
            _MODEL_CACHE[model_name] = YOLO(model_name)
        return _MODEL_CACHE[model_name]


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _get_keypoint(keypoints: np.ndarray, index: int) -> tuple[int, int] | None:
    if keypoints is None or keypoints.shape[0] == 0:
        return None
    pt = keypoints[0][index]
    confidence = float(pt[2]) if pt.shape[0] > 2 else 1.0
    if confidence < _KEYPOINT_MIN_CONFIDENCE:
        return None
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def _paste_accessory(
    canvas: Image.Image,
    overlay: Image.Image,
    x: int,
    y: int,
    scale: float,
    angle_degrees: float,
    anchor: str = "center",
) -> None:
    """Pastes an overlay onto the canvas rotating it around a specific anchor point."""
    w = max(1, int(overlay.width * scale))
    h = max(1, int(overlay.height * scale))
    resized = overlay.resize((w, h), Image.Resampling.LANCZOS)

    # PIL rotates counter-clockwise.
    rotated = resized.rotate(angle_degrees, resample=Image.Resampling.BICUBIC, expand=True)

    # Calculate offset to keep anchor steady
    cx, cy = w / 2, h / 2
    if anchor == "center":
        ax, ay = cx, cy
    elif anchor == "bottom_center":
        ax, ay = cx, h
    elif anchor == "left_center":
        ax, ay = 0, cy
    else:
        ax, ay = cx, cy

    # Rotate the anchor offset vector
    rad = math.radians(-angle_degrees)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    vx, vy = ax - cx, ay - cy
    rot_vx = vx * cos_a - vy * sin_a
    rot_vy = vx * sin_a + vy * cos_a

    # Calculate final paste coordinates
    paste_x = int(x - (rotated.width / 2) - rot_vx)
    paste_y = int(y - (rotated.height / 2) - rot_vy)

    canvas.alpha_composite(rotated, (paste_x, paste_y))


def _load_assets(assets_dir: Path, prefix: str) -> list[Path]:
    if not assets_dir.exists():
        return []
    return sorted([
        p for p in assets_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _ALLOWED_EXTENSIONS and p.stem.lower().startswith(prefix)
    ])


def compose_naama_image(
    image_bytes: bytes,
    *,
    assets_dir: Path,
    model_name: str,
    confidence_threshold: float = 0.25,
    mask_threshold: float = 0.35,
    **kwargs,  # Catch extra args safely
) -> bytes | None:
    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            source_rgba = ImageOps.exif_transpose(source_image).convert("RGBA")
    except OSError:
        return None

    image_rgb = np.asarray(source_rgba.convert("RGB"))
    seg_model = _get_model(model_name)
    pose_model_name = model_name.replace("-seg", "-pose") if "-seg" in model_name else model_name
    pose_model = _get_model(pose_model_name)

    # 1. Run inference (FORCED TO CPU)
    seg_results = seg_model.predict(source=image_rgb, task="segment", conf=confidence_threshold, device="cpu", verbose=False)
    pose_results = pose_model.predict(source=image_rgb, task="pose", conf=confidence_threshold, device="cpu", verbose=False)

    if not seg_results or seg_results[0].masks is None:
        return None

    # 2. Extract best mask (largest person)
    masks = seg_results[0].masks.data.cpu().numpy()
    classes = seg_results[0].boxes.cls.cpu().numpy()
    best_mask, max_area = None, 0
    for idx, cls in enumerate(classes):
        if int(cls) == _PERSON_CLASS_ID:
            area = np.sum(masks[idx] > mask_threshold)
            if area > max_area:
                max_area = area
                best_mask = masks[idx] > mask_threshold

    if best_mask is None:
        return None

    # Fix mask shape if model output differs from image
    if best_mask.shape != (source_rgba.height, source_rgba.width):
        best_mask = np.asarray(Image.fromarray(best_mask).resize((source_rgba.width, source_rgba.height), Image.Resampling.NEAREST))

    # 3. Build Person Layer
    alpha_channel = (best_mask.astype(np.uint8) * 255)
    person_pixels = np.asarray(source_rgba).copy()
    person_pixels[..., 3] = alpha_channel
    person_layer = Image.fromarray(person_pixels, mode="RGBA")

    # 4. Create Composition Canvas
    backgrounds = _load_assets(assets_dir, "background")
    if not backgrounds:
        return None

    rng = random.Random()
    with Image.open(rng.choice(backgrounds)) as bg_image:
        composed = bg_image.convert("RGBA").resize(source_rgba.size, Image.Resampling.LANCZOS)

    # 5. Apply Depth Scale & Paste Person
    depth_w = int(source_rgba.width * _DEPTH_SCALE_RATIO)
    depth_h = int(source_rgba.height * _DEPTH_SCALE_RATIO)
    depth_x = (source_rgba.width - depth_w) // 2
    depth_y = source_rgba.height - depth_h - int(source_rgba.height * _DEPTH_BOTTOM_MARGIN_RATIO)

    person_scaled = person_layer.resize((depth_w, depth_h), Image.Resampling.LANCZOS)
    composed.alpha_composite(person_scaled, (depth_x, depth_y))

    # 6. Extract and Scale Keypoints
    kpts = pose_results[0].keypoints.data.cpu().numpy() if pose_results and pose_results[0].keypoints is not None else None

    def get_scaled_kpt(idx):
        pt = _get_keypoint(kpts, idx)
        if pt is None: return None
        return (int(pt[0] * _DEPTH_SCALE_RATIO + depth_x), int(pt[1] * _DEPTH_SCALE_RATIO + depth_y))

    nose = get_scaled_kpt(0)
    l_eye = get_scaled_kpt(1) # Person's left eye (right side of image)
    r_eye = get_scaled_kpt(2) # Person's right eye (left side of image)

    # 7. Math for Anchors
    pil_angle = 0.0
    eye_dist = depth_w * 0.15 # Fallback

    if l_eye and r_eye:
        dx = l_eye[0] - r_eye[0]
        dy = l_eye[1] - r_eye[1]

        head_angle = math.degrees(math.atan2(dy, dx))
        pil_angle = -head_angle
        eye_dist = math.hypot(dx, dy)
        eye_center = ((l_eye[0] + r_eye[0]) / 2, (l_eye[1] + r_eye[1]) / 2)
    else:
        eye_center = (depth_x + depth_w / 2, depth_y + depth_h * 0.2)

    # Calculate Top of Head & Mouth based on facial vectors
    if nose and l_eye and r_eye:
        nx = nose[0] - eye_center[0]
        ny = nose[1] - eye_center[1]
        # Pulled the mouth slightly closer to the nose (0.75 instead of 0.9) for better baseline accuracy
        mouth = (nose[0] + nx * 0.75, nose[1] + ny * 0.75)
        top_head = (eye_center[0] - nx * 1.8, eye_center[1] - ny * 1.8)
    else:
        mouth = (depth_x + depth_w / 2, depth_y + depth_h * 0.35)
        top_head = (depth_x + depth_w / 2, depth_y + depth_h * 0.1)

    # 8. Add Accessories
    hats = _load_assets(assets_dir, "hat")
    cigars = _load_assets(assets_dir, "cigar")
    suns = _load_assets(assets_dir, "sun")

    if hats:
        with Image.open(rng.choice(hats)).convert("RGBA") as hat_img:
            hat_scale = (eye_dist * 3.2) / hat_img.width
            _paste_accessory(
                composed, hat_img,
                x=top_head[0], y=top_head[1],
                scale=hat_scale, angle_degrees=pil_angle, anchor="bottom_center"
            )

    if cigars:
        with Image.open(rng.choice(cigars)).convert("RGBA") as cigar_img:
            cigar_scale = (eye_dist * 2.5) / cigar_img.width
            cigar_angle = pil_angle - 15

            # Shift the cigar inward along the angle of the head so it overlaps the face
            # This visually "embeds" it in the lips
            rad = math.radians(-pil_angle)
            overlap_shift = eye_dist * 0.35
            embedded_mouth_x = mouth[0] - (math.cos(rad) * overlap_shift)
            embedded_mouth_y = mouth[1] - (math.sin(rad) * overlap_shift)

            _paste_accessory(
                composed, cigar_img,
                x=embedded_mouth_x, y=embedded_mouth_y,
                scale=cigar_scale, angle_degrees=cigar_angle, anchor="left_center"
            )

    if suns:
        with Image.open(rng.choice(suns)).convert("RGBA") as sun_img:
            sun_target_w = int(source_rgba.width * 0.25)
            sun_scale = sun_target_w / sun_img.width
            margin = int(source_rgba.width * 0.05)
            sun_x = source_rgba.width - (sun_target_w // 2) - margin
            sun_y = (int(sun_img.height * sun_scale) // 2) + margin
            _paste_accessory(
                composed, sun_img,
                x=sun_x, y=sun_y,
                scale=sun_scale, angle_degrees=0, anchor="center"
            )

    return _encode_png(composed.convert("RGB"))
