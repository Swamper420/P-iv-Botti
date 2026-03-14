from __future__ import annotations

import re

_URL_PATTERN = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".,!?;:)]}"


def extract_urls(text: str | None) -> list[str]:
    if not text:
        return []

    return [match.rstrip(_TRAILING_PUNCTUATION) for match in _URL_PATTERN.findall(text)]


def get_url_regex() -> str:
    return _URL_PATTERN.pattern


def build_yt_dlp_format_selector(max_height: int) -> str:
    return (
        f"bv*[height<={max_height}][vcodec^=vp09]+ba/"
        f"bv*[height<={max_height}][vcodec^=avc]+ba/"
        f"b[height<={max_height}]"
    )
