from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from bot.config import SttConfig

LOGGER = logging.getLogger(__name__)


async def transcribe_audio(
    file_bytes: bytes,
    filename: str,
    config: SttConfig,
    httpx_client: httpx.AsyncClient | None = None,
) -> str | None:
    """Send audio/video file to STT API for transcription.

    API Endpoint: POST /api/v1/transcribe (multipart/form-data)
    """
    if not file_bytes:
        LOGGER.warning("Empty file bytes provided for STT transcription.")
        return None

    url = f"{config.base_url.rstrip('/')}/api/v1/transcribe"
    files = {"file": (filename, file_bytes)}
    data: dict[str, str] = {
        "beam_size": str(config.beam_size),
        "vad_filter": "true" if config.vad_filter else "false",
        "word_timestamps": "true" if config.word_timestamps else "false",
    }
    if config.initial_prompt:
        data["initial_prompt"] = config.initial_prompt

    close_client = False
    client = httpx_client
    if client is None:
        client = httpx.AsyncClient()
        close_client = True

    try:
        response = await client.post(
            url,
            files=files,
            data=data,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None
    except httpx.HTTPStatusError as exc:
        LOGGER.error("STT API HTTP error status %s: %s", exc.response.status_code, exc)
        return None
    except Exception as exc:
        LOGGER.error("Error communicating with STT API: %s", exc)
        return None
    finally:
        if close_client:
            await client.aclose()


async def check_stt_health(
    config: SttConfig,
    httpx_client: httpx.AsyncClient | None = None,
) -> dict[str, Any] | None:
    """Perform health check against STT server: GET /health."""
    url = f"{config.base_url.rstrip('/')}/health"
    close_client = False
    client = httpx_client
    if client is None:
        client = httpx.AsyncClient()
        close_client = True

    try:
        response = await client.get(url, timeout=config.timeout_seconds)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        LOGGER.error("STT health check failed: %s", exc)
        return None
    finally:
        if close_client:
            await client.aclose()


def format_stt_response(text: str) -> str:
    """Format transcribed text into reply format."""
    clean_text = text.strip() if text else ""
    if not clean_text:
        return ""
    return f"💬 {clean_text}"
