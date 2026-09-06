from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

LOGGER = logging.getLogger(__name__)

# Matches the "!meme" command prefix; the remainder is an optional user hint.
MEME_COMMAND_PATTERN = r"(?i)^\s*!meme\b(.*)$"

# Sentinel returned by tiivista_logic.recognize_objects_with_yolo when nothing is found.
NO_OBJECTS_RECOGNIZED = "Kuvassa ei tunnistettu kohteita."

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
_SANS_BOLD_CANDIDATES = [
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Bold.ttf",
]

_TOP_LABELS = ("YLÄ", "YLA", "TOP", "UPPER")
_BOTTOM_LABELS = ("ALA", "BOTTOM", "LOWER")


def parse_meme_command(text: str) -> tuple[bool, str]:
    """Parse "!meme [vihje]" into (is_match, hint).

    The hint is free-form text after the command and may be empty.
    """
    if not text:
        return False, ""
    match = re.match(MEME_COMMAND_PATTERN, text, re.DOTALL)
    if not match:
        return False, ""
    return True, (match.group(1) or "").strip()


def build_meme_prompt(
    image_description: str = "",
    ocr_text: str = "",
    user_hint: str = "",
) -> str:
    """Build the Ollama user prompt from local image perception + user hint.

    Returns "" when there is nothing to base a caption on.
    """
    sections: list[str] = []

    desc = (image_description or "").strip()
    if desc and desc != NO_OBJECTS_RECOGNIZED:
        sections.append(f"Kuvahavainnot: {desc[:500]}")

    ocr = (ocr_text or "").strip()
    if ocr:
        sections.append(f"Kuvan teksti: {ocr[:800]}")

    hint = (user_hint or "").strip()
    if hint:
        sections.append(f"Käyttäjän vihje: {hint[:300]}")

    if not sections:
        return ""

    sections.append(
        "Keksi hauska meemiteksti. Vastaa täsmälleen kahdella rivillä: "
        "YLÄ: ... / ALA: ..."
    )
    return "\n".join(sections)


def _clean_line(line: str) -> str:
    """Strip numbering, bullets, surrounding quotes and extra whitespace."""
    cleaned = line.strip()
    cleaned = re.sub(r"^[\d]+[.)\-:]\s*", "", cleaned)
    cleaned = re.sub(r"^[-*•–—]\s*", "", cleaned)
    cleaned = cleaned.strip().strip("\"'””‘’«»").strip()
    return " ".join(cleaned.split())


def _truncate(text: str, max_chars: int) -> str:
    if max_chars < 1:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def sanitize_caption_line(line: str, max_chars: int) -> str:
    """Normalize one caption line to classic uppercase meme style."""
    cleaned = _clean_line(line)
    if not cleaned:
        return ""
    return _truncate(cleaned.upper(), max_chars)


def parse_meme_caption(raw: str) -> tuple[str, str]:
    """Parse raw LLM output into (top_text, bottom_text).

    Understands labeled ("YLÄ: ...", "ALA: ...", "TOP: ...", "BOTTOM: ..."),
    multi-line, and "top | bottom" formats. Returns raw (unsanitized) lines.
    """
    if not raw or not raw.strip():
        return "", ""

    lines = [_clean_line(line) for line in raw.strip().splitlines()]
    lines = [line for line in lines if line]

    label_pattern = re.compile(
        r"^\s*(YLÄ|YLA|TOP|UPPER|ALA|BOTTOM|LOWER)\s*[:\-–—|]\s*(.+)$",
        re.IGNORECASE,
    )
    top: str | None = None
    bottom: str | None = None
    unlabeled: list[str] = []
    for line in lines:
        label_match = label_pattern.match(line)
        if not label_match:
            unlabeled.append(line)
            continue
        label = label_match.group(1).upper()
        value = _clean_line(label_match.group(2))
        if label in _TOP_LABELS and top is None:
            top = value
        elif label in _BOTTOM_LABELS and bottom is None:
            bottom = value
        else:
            unlabeled.append(value)

    if top is not None or bottom is not None:
        if top is None:
            top = " ".join(unlabeled).strip()
        if bottom is None:
            bottom = ""
        return top.strip(), bottom.strip()

    if len(unlabeled) >= 2:
        return unlabeled[0], " ".join(unlabeled[1:]).strip()

    single = unlabeled[0] if unlabeled else ""
    if "|" in single:
        first, rest = single.split("|", 1)
        return first.strip(), rest.strip()
    return single, ""


def _shorten_description(image_description: str) -> str:
    """Turn "Kuvasta tunnistettiin: 3 kissaa, 1 koira." into "3 kissaa, 1 koira"."""
    desc = (image_description or "").strip()
    if not desc or desc.rstrip(".").strip() == NO_OBJECTS_RECOGNIZED.rstrip("."):
        return ""
    desc = desc.rstrip(".")
    prefix = "Kuvasta tunnistettiin:"
    if desc.lower().startswith(prefix.lower()):
        desc = desc[len(prefix):].strip()
    return desc


def fallback_caption(
    image_description: str = "",
    ocr_text: str = "",
    user_hint: str = "",
    max_top_chars: int = 60,
    max_bottom_chars: int = 60,
) -> tuple[str, str]:
    """Deterministic caption used when Ollama is unavailable or returns nothing.

    Built only from local perception (YOLO/OCR) and the user hint, so the
    meme still reflects the actual photo.
    """
    ocr_first = ""
    for line in (ocr_text or "").splitlines():
        cleaned = _clean_line(line)
        if cleaned:
            ocr_first = cleaned
            break

    hint = _clean_line(user_hint or "")
    desc_short = _clean_line(_shorten_description(image_description))

    if hint:
        top_raw = hint
        bottom_raw = desc_short or ocr_first or "MEEMI"
    elif ocr_first:
        top_raw = ocr_first
        bottom_raw = desc_short or "MEEMI"
    elif desc_short:
        top_raw = "TÄMÄ KUVA"
        bottom_raw = desc_short
    else:
        top_raw = "TÄMÄ KUVA"
        bottom_raw = "MEEMI"

    top_clean = _clean_line(top_raw).upper()
    if not _clean_line(bottom_raw) or _clean_line(bottom_raw).upper() == top_clean:
        for candidate in ("MEEMI", "TÄMÄ KUVA", "HAUSKA KUVA"):
            if candidate != top_clean:
                bottom_raw = candidate
                break

    return (
        sanitize_caption_line(top_raw, max_top_chars),
        sanitize_caption_line(bottom_raw, max_bottom_chars),
    )


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in _SANS_BOLD_CANDIDATES:
        if Path(path).is_file():
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _wrap_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = line.split(" ")
    wrapped: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                wrapped.append(current)
            current = word
    if current:
        wrapped.append(current)
    return wrapped or [""]


def _draw_centered_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    width: int,
    start_y: int,
    outline_width: int,
) -> int:
    y = start_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (width - text_w) // 2
        for ox in range(-outline_width, outline_width + 1):
            for oy in range(-outline_width, outline_width + 1):
                if ox != 0 or oy != 0:
                    draw.text((x + ox, y + oy), line, fill=(0, 0, 0), font=font)
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += text_h + 4
    return y


def render_meme_on_photo(
    image_bytes: bytes,
    top_text: str = "",
    bottom_text: str = "",
    *,
    output_max_width: int = 1080,
    jpeg_quality: int = 90,
) -> bytes | None:
    """Render classic top/bottom meme captions over the given photo.

    Returns JPEG bytes, or None when the input is not a readable image.
    """
    if not image_bytes:
        return None
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
    except Exception:
        LOGGER.warning("render_meme_on_photo: input is not a readable image")
        return None

    if output_max_width > 0 and photo.width > output_max_width:
        new_height = max(1, round(photo.height * (output_max_width / photo.width)))
        photo = photo.resize((output_max_width, new_height), Image.LANCZOS)

    width, height = photo.size
    font_size = min(72, max(24, width // 12))
    outline_width = max(2, font_size // 16)
    margin = max(10, width // 40)
    max_text_width = max(50, width - margin * 2)

    font = _load_bold_font(font_size)
    draw = ImageDraw.Draw(photo)

    top_lines: list[str] = []
    for raw_line in (top_text or "").splitlines():
        if raw_line.strip():
            top_lines.extend(_wrap_line(draw, raw_line.strip(), font, max_text_width))

    bottom_lines: list[str] = []
    for raw_line in (bottom_text or "").splitlines():
        if raw_line.strip():
            bottom_lines.extend(
                _wrap_line(draw, raw_line.strip(), font, max_text_width)
            )

    if top_lines:
        _draw_centered_block(draw, top_lines, font, width, margin, outline_width)

    if bottom_lines:
        probe_draw = ImageDraw.Draw(photo)
        block_height = 0
        for line in bottom_lines:
            bbox = probe_draw.textbbox((0, 0), line, font=font)
            block_height += (bbox[3] - bbox[1]) + 4
        start_y = max(margin, height - margin - block_height)
        _draw_centered_block(draw, bottom_lines, font, width, start_y, outline_width)

    output = BytesIO()
    photo.save(output, format="JPEG", quality=jpeg_quality)
    return output.getvalue()
