from __future__ import annotations

import logging
import threading
from io import BytesIO
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)

# COCO Classes: 0: person, 15: cat, 16: dog, 17: horse, 18: sheep,
# 19: cow, 20: elephant, 21: bear, 22: zebra, 23: giraffe
_TARGET_CLASSES = {0, 15, 16, 17, 18, 19, 20, 21, 22, 23}
_KEYPOINT_MIN_CONFIDENCE = 0.4

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_model(model_name: str, model_loader: Any = None) -> Any:
    if model_loader is not None:
        return model_loader(model_name)
    with _MODEL_CACHE_LOCK:
        if model_name not in _MODEL_CACHE:
            from ultralytics import YOLO
            _MODEL_CACHE[model_name] = YOLO(model_name)
        return _MODEL_CACHE[model_name]


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def compose_naama_image(
    image_bytes: bytes,
    *,
    model_name: str,
    confidence_threshold: float = 0.25,
    mask_threshold: float = 0.35,
    action: Literal["mirror", "sticker"] = "mirror",
    side: Literal["auto", "left", "right", "up", "down"] = "auto",
    model_loader: Any = None,
    **kwargs: object,
) -> bytes | None:
    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            source_rgba = ImageOps.exif_transpose(source_image).convert("RGBA")
    except OSError:
        return None

    image_rgb = np.asarray(source_rgba.convert("RGB"))
    seg_model = _get_model(model_name, model_loader)
    pose_model_name = model_name.replace("-seg", "-pose") if "-seg" in model_name else model_name
    pose_model = _get_model(pose_model_name, model_loader)

    # 1. Run inference
    seg_results = seg_model.predict(
        source=image_rgb, task="segment", conf=confidence_threshold, device="cpu", verbose=False
    )
    pose_results = pose_model.predict(
        source=image_rgb, task="pose", conf=confidence_threshold, device="cpu", verbose=False
    )

    if not seg_results or seg_results[0].masks is None:
        return None

    masks = seg_results[0].masks.data.cpu().numpy()
    classes = seg_results[0].boxes.cls.cpu().numpy()

    # 2. Collect all valid instances
    instances = []
    for idx, cls in enumerate(classes):
        if int(cls) in _TARGET_CLASSES:
            area = np.sum(masks[idx] > mask_threshold)
            if area > 0:
                instances.append({
                    "idx": idx,
                    "cls": int(cls),
                    "area": area,
                    "mask": masks[idx] > mask_threshold
                })

    if not instances:
        return None

    # Sort by area ascending so smaller/background targets are drawn first for mirroring
    instances.sort(key=lambda x: x["area"])

    if action == "sticker":
        output_image = Image.new("RGBA", source_rgba.size, (0, 0, 0, 0))
    else:
        output_image = source_rgba.copy()

    # Base array for extracting original pixels
    source_np = np.asarray(source_rgba)

    # 3. Process each target individually
    all_target_bounds = []
    for inst in instances:
        idx = inst["idx"]
        cls_id = inst["cls"]
        best_mask = inst["mask"]

        if best_mask.shape != (source_rgba.height, source_rgba.width):
            best_mask = np.asarray(
                Image.fromarray(best_mask).resize(
                    (source_rgba.width, source_rgba.height), Image.Resampling.NEAREST
                )
            )

        rows = np.any(best_mask, axis=1)
        cols = np.any(best_mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            continue

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Expand slightly to avoid harsh clipping
        rmin, rmax = max(0, rmin - 2), min(source_rgba.height - 1, rmax + 2)
        cmin, cmax = max(0, cmin - 2), min(source_rgba.width - 1, cmax + 2)

        # Build isolated target layer (pixels + mask alpha)
        alpha_channel = (best_mask.astype(np.uint8) * 255)
        target_pixels = source_np.copy()
        target_pixels[..., 3] = alpha_channel
        target_layer = Image.fromarray(target_pixels, mode="RGBA")

        # Crop tightly to the target
        cropped_target = target_layer.crop((cmin, rmin, cmax + 1, rmax + 1))
        width, height = cropped_target.size
        
        if side != "auto":
            mirror_x = cmin + (cmax - cmin) // 2
            mirror_y = rmin + (rmax - rmin) // 2
            
            if cls_id == 0 and pose_results and pose_results[0].keypoints is not None:
                kpts = pose_results[0].keypoints.data.cpu().numpy()
                if len(kpts) > idx:
                    person_kpts = kpts[idx]
                    if len(person_kpts) > 0:
                        nose_pt = person_kpts[0]
                        if float(nose_pt[2]) >= _KEYPOINT_MIN_CONFIDENCE:
                            mirror_x = int(round(float(nose_pt[0])))
                            mirror_y = int(round(float(nose_pt[1])))
            
            mirror_x = max(cmin + 1, min(mirror_x, cmax - 1))
            mirror_y = max(rmin + 1, min(mirror_y, rmax - 1))
            
            local_split_x = mirror_x - cmin
            local_split_y = mirror_y - rmin
            
            if side in ("up", "down"):
                if side == "up":
                    half_h = local_split_y
                    symmetrical = Image.new("RGBA", (width, half_h * 2))
                    top_half = cropped_target.crop((0, 0, width, half_h))
                    symmetrical.paste(top_half, (0, 0))
                    symmetrical.paste(ImageOps.flip(top_half), (0, half_h))
                else:
                    half_h = height - local_split_y
                    symmetrical = Image.new("RGBA", (width, half_h * 2))
                    bottom_half = cropped_target.crop((0, local_split_y, width, height))
                    symmetrical.paste(ImageOps.flip(bottom_half), (0, 0))
                    symmetrical.paste(bottom_half, (0, half_h))
                    
                paste_x = cmin
                paste_y = mirror_y - (symmetrical.height // 2)
            else: # left, right
                if side == "left":
                    half_w = local_split_x
                    symmetrical = Image.new("RGBA", (half_w * 2, height))
                    left_half = cropped_target.crop((0, 0, half_w, height))
                    symmetrical.paste(left_half, (0, 0))
                    symmetrical.paste(ImageOps.mirror(left_half), (half_w, 0))
                else:
                    half_w = width - local_split_x
                    symmetrical = Image.new("RGBA", (half_w * 2, height))
                    right_half = cropped_target.crop((local_split_x, 0, width, height))
                    symmetrical.paste(ImageOps.mirror(right_half), (0, 0))
                    symmetrical.paste(right_half, (half_w, 0))
                    
                paste_x = mirror_x - (symmetrical.width // 2)
                paste_y = rmin

            if action == "sticker":
                output_image.alpha_composite(symmetrical, (paste_x, paste_y))
                all_target_bounds.append((paste_x, paste_y, paste_x + symmetrical.width, paste_y + symmetrical.height))
                continue

        elif action == "sticker":
            output_image.alpha_composite(cropped_target, (cmin, rmin))
            all_target_bounds.append((cmin, rmin, cmax + 1, rmax + 1))
            continue

        if action == "mirror" and side == "auto":
            mirror_x = cmin + (cmax - cmin) // 2
            if cls_id == 0 and pose_results and pose_results[0].keypoints is not None:
                kpts = pose_results[0].keypoints.data.cpu().numpy()
                if len(kpts) > idx:
                    person_kpts = kpts[idx]
                    if len(person_kpts) > 0:
                        nose_pt = person_kpts[0]
                        confidence = float(nose_pt[2]) if len(nose_pt) > 2 else 1.0
                        if confidence >= _KEYPOINT_MIN_CONFIDENCE:
                            mirror_x = int(round(float(nose_pt[0])))

            mirror_x = max(cmin + 1, min(mirror_x, cmax - 1))
            local_split_x = mirror_x - cmin

            target_alpha = np.array(cropped_target)[..., 3]
            left_area = np.sum(target_alpha[:, :local_split_x] > 0)
            right_area = np.sum(target_alpha[:, local_split_x:] > 0)
            use_left = left_area >= right_area

            if use_left:
                half_w = local_split_x
                symmetrical = Image.new("RGBA", (half_w * 2, height))
                left_half = cropped_target.crop((0, 0, half_w, height))
                symmetrical.paste(left_half, (0, 0))
                symmetrical.paste(ImageOps.mirror(left_half), (half_w, 0))
            else:
                half_w = width - local_split_x
                symmetrical = Image.new("RGBA", (half_w * 2, height))
                right_half = cropped_target.crop((local_split_x, 0, width, height))
                symmetrical.paste(ImageOps.mirror(right_half), (0, 0))
                symmetrical.paste(right_half, (half_w, 0))

            paste_x = mirror_x - (symmetrical.width // 2)
            paste_y = rmin

        scale_factor = 1.15
        scaled_w = max(1, int(symmetrical.width * scale_factor))
        scaled_h = max(1, int(symmetrical.height * scale_factor))
        symmetrical = symmetrical.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        orig_center_y = rmin + (rmax - rmin) // 2
        paste_y = orig_center_y - scaled_h // 2
        paste_x = paste_x - (scaled_w - symmetrical.width) // 2
        
        # Recalculate paste_x using mirror_x directly
        if side in ("up", "down"):
            paste_x = cmin - (scaled_w - width) // 2
        else:
            paste_x = mirror_x - (scaled_w // 2)

        output_image.alpha_composite(symmetrical, (paste_x, paste_y))

    if action == "sticker":
        # Crop the transparent canvas to the bounding box of all targets
        if all_target_bounds:
            b_cmin = min(b[0] for b in all_target_bounds)
            b_rmin = min(b[1] for b in all_target_bounds)
            b_cmax = max(b[2] for b in all_target_bounds)
            b_rmax = max(b[3] for b in all_target_bounds)
            
            b_cmin = max(0, b_cmin)
            b_rmin = max(0, b_rmin)
            b_cmax = min(output_image.width, b_cmax)
            b_rmax = min(output_image.height, b_rmax)
            
            if b_cmax > b_cmin and b_rmax > b_rmin:
                output_image = output_image.crop((b_cmin, b_rmin, b_cmax, b_rmax))
        
        # Scale to max 512px
        max_dim = max(output_image.width, output_image.height)
        if max_dim > 512:
            scale = 512 / max_dim
            new_w = int(output_image.width * scale)
            new_h = int(output_image.height * scale)
            output_image = output_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return _encode_png(output_image)

    return _encode_png(output_image.convert("RGB"))
