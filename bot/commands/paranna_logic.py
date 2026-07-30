from __future__ import annotations

import logging
import math
import os
import threading
from io import BytesIO
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32, bias: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        nb: int = 6,
        gc: int = 32,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if scale == 4:
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk
        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        if self.scale == 4:
            fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


def _ensure_model_file(model_path: str, model_url: str) -> Path:
    path = Path(model_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Downloading Real-ESRGAN weights from %s to %s", model_url, path)
        urllib.request.urlretrieve(model_url, str(path))
    return path


def _get_model(
    model_path: str = "storage/models/RealESRGAN_x4plus_anime_6B.pth",
    model_url: str = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    model_loader: Any = None,
) -> Any:
    if model_loader is not None:
        return model_loader(model_path)

    with _MODEL_CACHE_LOCK:
        if model_path not in _MODEL_CACHE:
            local_path = _ensure_model_file(model_path, model_url)
            model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=6, gc=32, scale=4)
            checkpoint = torch.load(str(local_path), map_location="cpu")
            if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
                state_dict = checkpoint["params_ema"]
            elif isinstance(checkpoint, dict) and "params" in checkpoint:
                state_dict = checkpoint["params"]
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            _MODEL_CACHE[model_path] = model
        return _MODEL_CACHE[model_path]


def _tiled_upscale(
    model: Any,
    img_tensor: torch.Tensor,
    scale: int = 4,
    tile_size: int = 256,
    tile_pad: int = 10,
) -> torch.Tensor:
    """Tiled inference to handle large images efficiently on CPU."""
    batch, channel, height, width = img_tensor.shape
    output_height = height * scale
    output_width = width * scale
    output_tensor = torch.zeros(
        (batch, channel, output_height, output_width), dtype=img_tensor.dtype
    )

    num_tiles_x = math.ceil(width / tile_size)
    num_tiles_y = math.ceil(height / tile_size)

    num_threads = min(os.cpu_count() or 4, 8)
    torch.set_num_threads(num_threads)

    with torch.no_grad():
        for y_idx in range(num_tiles_y):
            for x_idx in range(num_tiles_x):
                # Extract tile input coordinates
                x_start = x_idx * tile_size
                y_start = y_idx * tile_size
                x_end = min(x_start + tile_size, width)
                y_end = min(y_start + tile_size, height)

                # Pad boundaries
                x_start_pad = max(x_start - tile_pad, 0)
                y_start_pad = max(y_start - tile_pad, 0)
                x_end_pad = min(x_end + tile_pad, width)
                y_end_pad = min(y_end + tile_pad, height)

                # Crop tile with padding
                tile = img_tensor[:, :, y_start_pad:y_end_pad, x_start_pad:x_end_pad]
                tile_output = model(tile)

                # Calculate corresponding coordinates in output
                out_x_start = x_start * scale
                out_y_start = y_start * scale
                out_x_end = x_end * scale
                out_y_end = y_end * scale

                out_tile_x_start = (x_start - x_start_pad) * scale
                out_tile_y_start = (y_start - y_start_pad) * scale
                out_tile_x_end = out_tile_x_start + (x_end - x_start) * scale
                out_tile_y_end = out_tile_y_start + (y_end - y_start) * scale

                output_tensor[
                    :, :, out_y_start:out_y_end, out_x_start:out_x_end
                ] = tile_output[
                    :, :, out_tile_y_start:out_tile_y_end, out_tile_x_start:out_tile_x_end
                ]

    return torch.clamp(output_tensor, 0.0, 1.0)


def upscale_image(
    image_bytes: bytes,
    *,
    model_path: str = "storage/models/RealESRGAN_x4plus_anime_6B.pth",
    model_url: str = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    tile_size: int = 256,
    tile_pad: int = 10,
    max_input_dimension: int = 2560,
    max_output_dimension: int = 4096,
    jpeg_quality: int = 95,
    model_loader: Any = None,
    **kwargs: object,
) -> bytes | None:
    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            rgb_image = ImageOps.exif_transpose(source_image).convert("RGB")
    except Exception as exc:
        LOGGER.error("Failed to open image for upscaling: %s", exc)
        return None

    # Pre-scale if image exceeds max_input_dimension
    width, height = rgb_image.size
    max_dim = max(width, height)
    if max_dim > max_input_dimension:
        ratio = max_input_dimension / max_dim
        new_width = max(1, int(width * ratio))
        new_height = max(1, int(height * ratio))
        rgb_image = rgb_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        width, height = new_width, new_height

    try:
        model = _get_model(model_path=model_path, model_url=model_url, model_loader=model_loader)
        img_np = np.array(rgb_image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

        out_tensor = _tiled_upscale(
            model=model,
            img_tensor=img_tensor,
            scale=4,
            tile_size=tile_size,
            tile_pad=tile_pad,
        )

        out_np = (out_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        upscaled_image = Image.fromarray(out_np, mode="RGB")
    except Exception as exc:
        LOGGER.warning("Neural upscaling failed (%s). Falling back to high-quality Lanczos upscale.", exc)
        upscaled_image = rgb_image.resize((width * 4, height * 4), Image.Resampling.LANCZOS)

    # Post-scale if output exceeds max_output_dimension for Telegram preview limits
    out_w, out_h = upscaled_image.size
    out_max_dim = max(out_w, out_h)
    if out_max_dim > max_output_dimension:
        scale_ratio = max_output_dimension / out_max_dim
        final_w = max(1, int(out_w * scale_ratio))
        final_h = max(1, int(out_h * scale_ratio))
        upscaled_image = upscaled_image.resize((final_w, final_h), Image.Resampling.LANCZOS)

    # Encode as JPEG quality 95
    output = BytesIO()
    upscaled_image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    return output.getvalue()
