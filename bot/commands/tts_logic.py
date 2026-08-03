from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)



def parse_tts_command(text: str) -> tuple[bool, str | None, str]:
    """Parse a TTS command string into (is_match, voice, text).

    Syntax: [voice] puhuu [selkeästi]: <text>
    If voice is omitted, returns voice=None.
    If 'selkeästi' is specified, adds '...' after each word to make the speech slower.
    """
    if not text:
        return False, None, ""

    match = re.match(
        r"(?i)^\s*!?(?:([^\n:]+)\s+)?puhuu(\s+selkeästi)?:\s*(.*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return False, None, ""

    voice_raw = match.group(1)
    is_slow = bool(match.group(2))
    prompt_text = (match.group(3) or "").strip()

    voice: str | None = None
    if voice_raw:
        voice_clean = voice_raw.strip().lstrip("!")
        if voice_clean:
            voice = voice_clean

    if is_slow and prompt_text:
        words = prompt_text.split()
        prompt_text = " ".join(
            w if w.endswith("...") else f"{w}..." for w in words
        )

    return True, voice, prompt_text



def chunk_text(
    text: str,
    max_chunk_size: int = 50,
    min_chunk_len: int = 10,
) -> list[str]:
    """Split text into chunks up to max_chunk_size characters.

    Prefers splitting at punctuation marks (., !, ?, ,, :, ;, \n).
    Pads any chunk shorter than min_chunk_len with '...' (and '.' if needed).
    """
    text = text.strip()
    if not text:
        return []

    if max_chunk_size < min_chunk_len:
        max_chunk_size = min_chunk_len

    # Split into initial segments by punctuation marks
    raw_segments = re.split(r"(?<=[.!?,\n;:])\s+", text)
    segments: list[str] = []

    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) > max_chunk_size:
            words = seg.split()
            current_word_chunk = ""
            for word in words:
                candidate = (
                    f"{current_word_chunk} {word}".strip()
                    if current_word_chunk
                    else word
                )
                if len(candidate) <= max_chunk_size:
                    current_word_chunk = candidate
                else:
                    if current_word_chunk:
                        segments.append(current_word_chunk)
                    while len(word) > max_chunk_size:
                        segments.append(word[:max_chunk_size])
                        word = word[max_chunk_size:]
                    current_word_chunk = word
            if current_word_chunk:
                segments.append(current_word_chunk)
        else:
            segments.append(seg)

    chunks: list[str] = []
    current_chunk = ""

    for seg in segments:
        if not current_chunk:
            current_chunk = seg
        elif len(current_chunk) + 1 + len(seg) <= max_chunk_size:
            current_chunk = f"{current_chunk} {seg}"
        else:
            chunks.append(current_chunk)
            current_chunk = seg

    if current_chunk:
        chunks.append(current_chunk)

    final_chunks: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) < min_chunk_len:
            chunk += "..."
            if len(chunk) < min_chunk_len:
                chunk += "." * (min_chunk_len - len(chunk))
        final_chunks.append(chunk)

    return final_chunks


async def combine_audio_chunks(
    audio_chunks: list[bytes],
    fmt: str = "ogg",
) -> bytes:
    """Combine multiple audio byte chunks into a single audio file via ffmpeg."""
    if not audio_chunks:
        return b""
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_files: list[Path] = []

        for idx, chunk in enumerate(audio_chunks):
            file_path = tmp_path / f"chunk_{idx}.{fmt}"
            file_path.write_bytes(chunk)
            input_files.append(file_path)

        list_file = tmp_path / "files.txt"
        list_contents = "\n".join(
            f"file '{f.resolve().as_posix()}'" for f in input_files
        )
        list_file.write_text(list_contents, encoding="utf-8")

        output_file = tmp_path / f"output.{fmt}"

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file.resolve()),
            "-c",
            "copy",
            str(output_file.resolve()),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await proc.communicate()

        if proc.returncode != 0 or not output_file.exists():
            cmd_reencode = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file.resolve()),
                str(output_file.resolve()),
            ]
            proc2 = await asyncio.create_subprocess_exec(
                *cmd_reencode,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout2, stderr2 = await proc2.communicate()
            if proc2.returncode != 0 or not output_file.exists():
                raise RuntimeError(
                    f"ffmpeg concat failed with code {proc2.returncode}: {stderr2.decode(errors='ignore')}"
                )

        return output_file.read_bytes()


def extract_voice_ids(data: dict[str, Any]) -> list[str]:
    """Extract list of voice_id strings from GET /api/v1/voices response."""
    voices_raw = data.get("voices", [])
    if not isinstance(voices_raw, list):
        return []
    voice_ids: list[str] = []
    for item in voices_raw:
        if isinstance(item, dict) and "voice_id" in item:
            voice_ids.append(str(item["voice_id"]))
        elif isinstance(item, str):
            voice_ids.append(item)
    return voice_ids


def resolve_voice(
    requested_voice: str | None,
    available_voices: list[str],
) -> str | None:
    """Find nearest matching voice ID from available_voices using rapidfuzz.

    If requested_voice is None or available_voices is empty, returns requested_voice.
    If exact case-insensitive match exists, returns that exact voice ID.
    Otherwise, uses rapidfuzz to find the closest matching voice ID.
    """
    if not requested_voice or not available_voices:
        return requested_voice

    req_clean = requested_voice.strip()
    if not req_clean:
        return requested_voice

    req_lower = req_clean.lower()
    for v in available_voices:
        if v.lower() == req_lower:
            return v

    try:
        from rapidfuzz import fuzz, process

        match = process.extractOne(req_clean, available_voices, scorer=fuzz.WRatio)
        if match:
            best_voice = match[0]
            return str(best_voice)
    except Exception as err:
        LOGGER.warning(
            "RapidFuzz matching failed for voice '%s': %s", requested_voice, err
        )

    return requested_voice


async def synthesize_speech(
    base_url: str,
    text: str,
    voice: str | None = None,
    fmt: str = "ogg",
    language: str | None = None,
    speed: float | None = None,
    num_step: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    model: str | None = None,
    timeout_seconds: int = 60,
    max_chunk_size: int = 50,
    min_chunk_len: int = 10,
    client: httpx.AsyncClient | None = None,
    available_voices: list[str] | None = None,
) -> bytes:
    """Synthesize speech using custom TTS API (/api/v1/tts) with chunking and rapidfuzz voice resolution."""
    chunks = chunk_text(text, max_chunk_size=max_chunk_size, min_chunk_len=min_chunk_len)
    if not chunks:
        return b""

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds)
        close_client = True

    try:
        resolved_voice = voice
        if voice:
            if available_voices is None:
                try:
                    voices_data = await list_voices(
                        base_url, timeout_seconds=timeout_seconds, client=client
                    )
                    available_voices = extract_voice_ids(voices_data)
                except Exception as err:
                    LOGGER.warning("Could not fetch voice catalog for fuzzy matching: %s", err)
                    available_voices = []

            resolved_voice = resolve_voice(voice, available_voices)

        audio_chunks: list[bytes] = []
        target_url = f"{base_url.rstrip('/')}/api/v1/tts"
        for chunk in chunks:
            payload: dict[str, Any] = {
                "text": chunk,
                "response_format": fmt,
            }
            if resolved_voice:
                payload["voice"] = resolved_voice
            if language:
                payload["language"] = language
            if speed is not None:
                payload["speed"] = speed
            if num_step is not None:
                payload["num_step"] = num_step
            if guidance_scale is not None:
                payload["guidance_scale"] = guidance_scale
            if seed is not None:
                payload["seed"] = seed

            response = await client.post(target_url, json=payload)
            response.raise_for_status()
            audio_chunks.append(response.content)

        return await combine_audio_chunks(audio_chunks, fmt=fmt)
    finally:
        if close_client:
            await client.aclose()


async def synthesize_speech_openai(
    base_url: str,
    input_text: str,
    voice: str = "voice_fi",
    model: str = "omnivoice",
    response_format: str = "mp3",
    speed: float | None = None,
    timeout_seconds: int = 60,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    """Synthesize speech using OpenAI compatible endpoint (/v1/audio/speech)."""
    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds)
        close_client = True

    try:
        target_url = f"{base_url.rstrip('/')}/v1/audio/speech"
        payload: dict[str, Any] = {
            "model": model,
            "input": input_text,
            "voice": voice,
            "response_format": response_format,
        }
        if speed is not None:
            payload["speed"] = speed

        response = await client.post(target_url, json=payload)
        response.raise_for_status()
        return response.content
    finally:
        if close_client:
            await client.aclose()


async def list_voices(
    base_url: str,
    timeout_seconds: int = 60,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch voice catalog from TTS API (/api/v1/voices)."""
    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds)
        close_client = True

    try:
        target_url = f"{base_url.rstrip('/')}/api/v1/voices"
        response = await client.get(target_url)
        response.raise_for_status()
        return response.json()
    finally:
        if close_client:
            await client.aclose()


async def reload_voices(
    base_url: str,
    timeout_seconds: int = 60,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Reload voice catalog on TTS API (/api/v1/voices/reload)."""
    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds)
        close_client = True

    try:
        target_url = f"{base_url.rstrip('/')}/api/v1/voices/reload"
        response = await client.post(target_url)
        response.raise_for_status()
        return response.json() if response.content else {}
    finally:
        if close_client:
            await client.aclose()


async def check_health(
    base_url: str,
    timeout_seconds: int = 60,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Check health status of TTS API (/health)."""
    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds)
        close_client = True

    try:
        target_url = f"{base_url.rstrip('/')}/health"
        response = await client.get(target_url)
        response.raise_for_status()
        return response.json()
    finally:
        if close_client:
            await client.aclose()

