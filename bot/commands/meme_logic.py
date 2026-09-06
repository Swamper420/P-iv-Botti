from __future__ import annotations

import logging
import random
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageFilter

LOGGER = logging.getLogger(__name__)

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
_SANS_CANDIDATES = [
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Bold.ttf",
]
_MONO_CANDIDATES = [
    "/usr/share/fonts/TTF/Hack-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
]

_TEMPLATES: dict[str, dict] = {
    "drake": {
        "name": "Drake Hotline Bling",
        "top_text": "Ei tämä",
        "bottom_text": "Vaikka tämä",
        "aspect": (1, 1),
    },
    "change_mind": {
        "name": "Change My Mind",
        "top_text": "Pineapple on pizza",
        "bottom_text": "Change my mind",
        "aspect": (3, 2),
    },
    "expanding_brain": {
        "name": "Expanding Brain",
        "levels": 4,
        "top_text": "Level 1\nLevel 2\nLevel 3\nLevel 4 (GALAXY BRAIN)",
        "bottom_text": "",
        "aspect": (1, 1),
    },
    "distracted_boyfriend": {
        "name": "Distracted Boyfriend",
        "top_text": "Minä\n\n\n\nUusi juttu\n\n\n\nVanha juttu",
        "bottom_text": "",
        "aspect": (3, 2),
    },
    "two_buttons": {
        "name": "Two Buttons",
        "top_text": "Painaa nappia A\nPainaa nappia B",
        "bottom_text": "Hidastuu",
        "aspect": (1, 1),
    },
    "this_is_fine": {
        "name": "This Is Fine",
        "top_text": "Tämä on hienoa",
        "bottom_text": "Kaikki palaa",
        "aspect": (1, 1),
    },
}

_FUNNY_QUOTES = [
    ("Koodi toimii", "Mutta en tiedä miksi"),
    ("Ensimmäinen kerta", "Toimii heti"),
    ("Toinen kerta", "Rikki"),
    ("Git push --force", "Mitä voi mennä pieleen?"),
    ("Kahvi", "Kääntää koodin toimivaksi"),
    ("Stack Overflow", "Kopioi, liitä, rugaile"),
    ("Variable naming", "a, b, c, data, temp, final_final"),
    ("Debugging", "90% etsii, 10% korjaa"),
    ("Deadline", "Panika mode: ON"),
    ("Code review", "LGTM (en luettanut)"),
    ("Rubber duck", "Kertoo bugista, korjaa itsensä"),
    ("Legacy code", "Älä koske, se toimii"),
    ("Tests", "Kirjoitan myöhemmin (enkos)"),
    ("Documentation", "Mikä dokumentaatio?"),
    ("Refactor", "Rikkaa kaikki, korjaa viikonloppuisin"),
]

_MOTIVATIONAL = [
    ("ÄLÄ LOPETA", "VAAN ETTÄ SAAAT PERÄÄ"),
    ("OLET VOIMAKAS", "KÄY KOTOON JA NUKKU"),
    ("ONNISTUMINEN", "ON 1% INSPIRAATIO, 99% KOFFEIINIA"),
    ("KÄVELY", "ON PARAS DEBUGGAUS"),
    ("SEURAA AIVOJESI", "EI KÄSKYJÄ"),
]


def _load_font(font_type: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (font_type, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    candidates = _SANS_CANDIDATES if font_type == "sans" else _MONO_CANDIDATES
    for path in candidates:
        if Path(path).is_file():
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[key] = font
                return font
            except Exception:
                pass

    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        if not raw_line:
            lines.append("")
            continue
        words = raw_line.split(" ")
        current = ""
        for word in words:
            test = f"{current} {word}".strip() if current else word
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def _draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    x: int,
    y: int,
    fill: tuple[int, int, int] = (255, 255, 255),
    outline: tuple[int, int, int] = (0, 0, 0),
    outline_width: int = 3,
    align: str = "center",
    max_width: int | None = None,
) -> int:
    lines = [text] if max_width is None else _wrap_text(draw, text, font, max_width)
    total_h = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if align == "center":
            tx = x - w // 2
        elif align == "right":
            tx = x - w
        else:
            tx = x
        for ox in range(-outline_width, outline_width + 1):
            for oy in range(-outline_width, outline_width + 1):
                if ox != 0 or oy != 0:
                    draw.text((tx + ox, y + oy), line, fill=outline, font=font)
        draw.text((tx, y), line, fill=fill, font=font)
        y += h + 4
        total_h += h + 4
    return total_h


def generate_meme_image(
    template: str = "random",
    top_text: str | None = None,
    bottom_text: str | None = None,
    custom_text: str | None = None,
    width: int = 800,
    style: Literal["classic", "motivational", "dev", "random"] = "random",
) -> bytes:
    if template == "random":
        template = random.choice(list(_TEMPLATES.keys()))

    tpl = _TEMPLATES.get(template, _TEMPLATES["drake"])

    aspect_w, aspect_h = tpl["aspect"]
    height = int(width * aspect_h / aspect_w)

    img = Image.new("RGB", (width, height), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    if style == "motivational" or (style == "random" and random.random() < 0.2):
        quote = random.choice(_MOTIVATIONAL)
        font_big = _load_font("sans", max(48, width // 16))
        font_small = _load_font("sans", max(24, width // 32))
        _draw_text_with_outline(draw, quote[0], font_big, width // 2, height // 3, fill=(255, 255, 255))
        _draw_text_with_outline(draw, quote[1], font_small, width // 2, height * 2 // 3, fill=(200, 200, 200))
        return _encode_png(img)

    if style == "dev" or (style == "random" and random.random() < 0.4):
        quote = random.choice(_FUNNY_QUOTES)
        font_main = _load_font("sans", max(36, width // 20))
        font_sub = _load_font("mono", max(22, width // 36))
        _draw_text_with_outline(draw, quote[0], font_main, width // 2, height // 3, fill=(100, 200, 255))
        _draw_text_with_outline(draw, quote[1], font_sub, width // 2, height * 2 // 3, fill=(180, 180, 180))
        return _encode_png(img)

    font = _load_font("sans", max(40, width // 18))

    top = top_text or tpl.get("top_text", "TOP TEXT")
    bottom = bottom_text or tpl.get("bottom_text", "BOTTOM TEXT")

    if template == "expanding_brain":
        levels = tpl.get("levels", 4)
        step = height // (levels + 1)
        for i, line in enumerate(top.split("\n")):
            level_font = _load_font("sans", max(20, width // 30) + i * max(8, width // 100))
            y_pos = step * (i + 1)
            _draw_text_with_outline(draw, line, level_font, width // 2, y_pos, fill=(255, 255, 100 + i * 40))
        return _encode_png(img)

    if template == "distracted_boyfriend":
        parts = top.split("\n\n\n")
        if len(parts) >= 3:
            font_label = _load_font("sans", max(28, width // 25))
            y_start = height // 6
            spacing = height // 4
            for i, part in enumerate(parts[:3]):
                _draw_text_with_outline(draw, part.strip(), font_label, width // 2, y_start + i * spacing, fill=(255, 255, 255))
        return _encode_png(img)

    if template == "two_buttons":
        buttons = top.split("\n")
        btn_font = _load_font("sans", max(28, width // 25))
        btn_h = height // 3
        for i, btn in enumerate(buttons[:2]):
            y_pos = btn_h * i + btn_h // 2
            _draw_text_with_outline(draw, btn.strip(), btn_font, width // 2, y_pos, fill=(255, 255, 255))
        _draw_text_with_outline(draw, bottom, _load_font("sans", max(24, width // 30)), width // 2, height * 5 // 6, fill=(255, 100, 100))
        return _encode_png(img)

    _draw_text_with_outline(draw, top, font, width // 2, height // 6, fill=(255, 255, 255), max_width=width - 80)
    _draw_text_with_outline(draw, bottom, font, width // 2, height * 5 // 6, fill=(255, 255, 255), max_width=width - 80)

    return _encode_png(img)


def generate_custom_meme(
    text: str,
    width: int = 800,
    height: int = 600,
    style: Literal["impact", "clean", "glitch", "rainbow"] = "impact",
) -> bytes:
    img = Image.new("RGB", (width, height), color=(15, 15, 25))
    draw = ImageDraw.Draw(img)

    if style == "rainbow":
        for y in range(height):
            r = int(128 + 127 * (y / height))
            g = int(128 + 127 * ((y + height // 3) % height / height))
            b = int(128 + 127 * ((y + 2 * height // 3) % height / height))
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        draw = ImageDraw.Draw(img)

    font_size = max(32, min(width // 12, height // 8))
    font = _load_font("sans", font_size)

    lines = _wrap_text(draw, text, font, width - 100)
    total_h = len(lines) * (font_size + 10)
    start_y = (height - total_h) // 2

    colors = [
        (255, 100, 100), (100, 255, 100), (100, 200, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
    ]

    for i, line in enumerate(lines):
        y = start_y + i * (font_size + 10)
        color = colors[i % len(colors)] if style == "rainbow" else (255, 255, 255)
        _draw_text_with_outline(draw, line, font, width // 2, y, fill=color, outline=(0, 0, 0), outline_width=4)

    if style == "glitch":
        for _ in range(10):
            y = random.randint(0, height - 20)
            h = random.randint(5, 20)
            x_shift = random.randint(-20, 20)
            region = img.crop((0, y, width, y + h))
            img.paste(region, (x_shift, y))

    return _encode_png(img)


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def list_templates() -> list[str]:
    return [f"{k}: {v['name']}" for k, v in _TEMPLATES.items()]