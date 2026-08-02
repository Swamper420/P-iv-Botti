from __future__ import annotations

import asyncio
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx


def parse_tts_command(text: str) -> tuple[bool, str | None, str]:
    """Parse a TTS command string into (is_match, voice, text).

    Syntax: [voice] puhuu: <text>
    If voice is omitted, returns voice=None.
    """
    if not text:
        return False, None, ""

    match = re.match(
        r"(?i)^\s*!?(?:([^\n:]+)\s+)?puhuu:\s*(.*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return False, None, ""

    voice_raw = match.group(1)
    prompt_text = (match.group(2) or "").strip()

    voice: str | None = None
    if voice_raw:
        voice_clean = voice_raw.strip().lstrip("!")
        if voice_clean:
            voice = voice_clean

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


async def synthesize_speech(
    base_url: str,
    text: str,
    voice: str | None = None,
    fmt: str = "ogg",
    model: str | None = None,
    timeout_seconds: int = 60,
    max_chunk_size: int = 50,
    min_chunk_len: int = 10,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    """Synthesize speech using Chatterbox TTS API with chunking and audio concatenation."""
    chunks = chunk_text(text, max_chunk_size=max_chunk_size, min_chunk_len=min_chunk_len)
    if not chunks:
        return b""

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds)
        close_client = True

    try:
        audio_chunks: list[bytes] = []
        target_url = f"{base_url.rstrip('/')}/api/tts"
        for chunk in chunks:
            payload: dict[str, Any] = {
                "text": chunk,
                "format": fmt,
            }
            if voice:
                payload["voice"] = voice
            if model:
                payload["model"] = model

            response = await client.post(target_url, json=payload)
            response.raise_for_status()
            audio_chunks.append(response.content)

        return await combine_audio_chunks(audio_chunks, fmt=fmt)
    finally:
        if close_client:
            await client.aclose()
