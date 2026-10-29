from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any

import httpx
from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)


def prepare_image_data_url(
    image_bytes: bytes,
    *,
    max_dimension: int = 1024,
    jpeg_quality: int = 85,
) -> str | None:
    """Downscale + JPEG-encode image bytes into a data: URL for vision APIs.

    Returns None when input is not a readable image.
    """
    if not image_bytes:
        return None
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
    except Exception:
        LOGGER.warning("vision: input is not a readable image")
        return None

    if max_dimension > 0:
        longest = max(photo.width, photo.height)
        if longest > max_dimension:
            scale = max_dimension / float(longest)
            new_size = (max(1, round(photo.width * scale)), max(1, round(photo.height * scale)))
            photo = photo.resize(new_size, Image.LANCZOS)

    output = BytesIO()
    try:
        photo.save(output, format="JPEG", quality=jpeg_quality)
    except Exception:
        LOGGER.warning("vision: JPEG encoding failed")
        return None
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_content_text(payload: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible chat completion payload."""
    try:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            return "\n".join(parts).strip()
        return ""
    except Exception:
        return ""


async def describe_image_with_eyes(
    image_bytes: bytes,
    prompt: str,
    *,
    base_url: str,
    model: str,
    max_tokens: int = 500,
    timeout_seconds: int = 60,
    max_dimension: int = 1024,
    jpeg_quality: int = 85,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Describe an image via the external Eyes OpenAI-compatible API.

    Returns "" on any failure (unreadable image, network error, bad payload).
    """
    if not image_bytes or not (prompt or "").strip():
        return ""

    data_url = await _prepare_data_url_threaded(image_bytes, max_dimension, jpeg_quality)
    if not data_url:
        return ""

    target_url = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.strip()},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds)
        close_client = True
    try:
        response = await client.post(target_url, json=payload)
        response.raise_for_status()
        try:
            data = response.json()
        except Exception:
            LOGGER.warning("vision: non-JSON response from Eyes API")
            return ""
        if not isinstance(data, dict):
            return ""
        return _extract_content_text(data)
    except Exception:
        LOGGER.exception("vision: Eyes image description failed")
        return ""
    finally:
        if close_client:
            await client.aclose()


async def _prepare_data_url_threaded(
    image_bytes: bytes, max_dimension: int, jpeg_quality: int
) -> str | None:
    import asyncio

    return await asyncio.to_thread(
        prepare_image_data_url,
        image_bytes,
        max_dimension=max_dimension,
        jpeg_quality=jpeg_quality,
    )
