from __future__ import annotations

import functools
from io import BytesIO
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from bot.rendering.models import (
    DARK_THEME,
    Badge,
    Card,
    CardElement,
    CodeBlockElement,
    DividerElement,
    KeyValuesElement,
    ProgressBarElement,
    TableElement,
    TextElement,
    Theme,
)

# Common TTF search candidates on Linux and other platforms
SANS_REGULAR_CANDIDATES = [
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
]

SANS_BOLD_CANDIDATES = [
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Bold.ttf",
]

MONO_CANDIDATES = [
    "/usr/share/fonts/TTF/Hack-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
]


@functools.lru_cache(maxsize=32)
def _find_font_path(candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


@functools.lru_cache(maxsize=64)
def _load_font(font_type: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path: str | None = None
    if font_type == "bold":
        path = _find_font_path(tuple(SANS_BOLD_CANDIDATES))
    elif font_type == "mono":
        path = _find_font_path(tuple(MONO_CANDIDATES))
    else:
        path = _find_font_path(tuple(SANS_REGULAR_CANDIDATES))

    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass

    return ImageFont.load_default()


def _get_text_bbox(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font)


def _get_text_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> int:
    bbox = _get_text_bbox(draw, text, font)
    return bbox[2] - bbox[0]


def _get_text_height(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> int:
    bbox = _get_text_bbox(draw, text, font)
    return bbox[3] - bbox[1]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            lines.append("")
            continue

        words = raw_line.split(" ")
        current_line = ""

        for word in words:
            candidate = f"{current_line} {word}".strip() if current_line else word
            if _get_text_width(draw, candidate, font) <= max_width:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word exceeds max width, split characters
                    chunk = ""
                    for char in word:
                        if _get_text_width(draw, chunk + char, font) <= max_width:
                            chunk += char
                        else:
                            lines.append(chunk)
                            chunk = char
                    current_line = chunk

        if current_line:
            lines.append(current_line)

    return lines


def _measure_badge(
    draw: ImageDraw.ImageDraw,
    badge: Badge,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    text = badge.text.strip().upper()
    pad_h = 8
    pad_v = 3
    text_w = _get_text_width(draw, text, font)
    text_h = _get_text_height(draw, text, font)
    return text_w + pad_h * 2, text_h + pad_v * 2


def _draw_sharp_badge(
    draw: ImageDraw.ImageDraw,
    badge: Badge,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    """Draws a badge with sharp rectangular corners and returns its (width, height)."""
    style = badge.get_style()
    text = badge.text.strip().upper()
    pad_h = 8
    pad_v = 3

    text_w = _get_text_width(draw, text, font)
    text_h = _get_text_height(draw, text, font)

    badge_w = text_w + pad_h * 2
    badge_h = text_h + pad_v * 2

    # Draw sharp rectangle background and border (NO rounded corners)
    draw.rectangle(
        [(x, y), (x + badge_w, y + badge_h)],
        fill=style.bg_color,
        outline=style.border_color,
        width=1,
    )
    draw.text((x + pad_h, y + pad_v), text, fill=style.fg_color, font=font)

    return badge_w, badge_h


class CardRenderer:
    def __init__(self, card: Card, theme: Theme = DARK_THEME) -> None:
        self.card = card
        self.theme = theme

        # Load fonts
        self.font_title = _load_font("bold", 24)
        self.font_subtitle = _load_font("regular", 14)
        self.font_body = _load_font("regular", 15)
        self.font_body_bold = _load_font("bold", 15)
        self.font_mono = _load_font("mono", 14)
        self.font_small = _load_font("regular", 12)
        self.font_small_bold = _load_font("bold", 12)

        # Measurements
        self.card_width = theme.card_width
        self.padding = theme.padding
        self.content_width = self.card_width - (self.padding * 2)

    def render(self) -> bytes:
        # Pass 1: Measure required height
        dummy_img = Image.new("RGB", (self.card_width, 100), color=self.theme.canvas_bg)
        measure_draw = ImageDraw.Draw(dummy_img)
        total_height = self._measure_and_layout(measure_draw)

        # Pass 2: Draw onto target canvas
        img = Image.new("RGB", (self.card_width, total_height), color=self.theme.canvas_bg)
        draw = ImageDraw.Draw(img)
        self._draw_all(draw, total_height)

        output = BytesIO()
        img.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def _measure_and_layout(self, draw: ImageDraw.ImageDraw) -> int:
        y = self.padding

        # Accent top bar (sharp rectangle)
        y += 4 + 14

        # Header: Title, Subtitle, Badge
        title_lines = _wrap_text(draw, self.card.title, self.font_title, self.content_width - 120)
        title_h = sum(_get_text_height(draw, line, self.font_title) + 4 for line in title_lines)

        subtitle_h = 0
        if self.card.subtitle:
            sub_lines = _wrap_text(draw, self.card.subtitle, self.font_subtitle, self.content_width - 120)
            subtitle_h = sum(_get_text_height(draw, line, self.font_subtitle) + 3 for line in sub_lines) + 4

        header_h = max(title_h + subtitle_h, 32)
        y += header_h + 16

        # Header divider
        y += 1 + 16

        # Elements
        for el in self.card.elements:
            y += self._measure_element(draw, el) + self.theme.element_spacing

        # Footer
        if self.card.footer:
            footer_lines = _wrap_text(draw, self.card.footer, self.font_small, self.content_width)
            footer_h = sum(_get_text_height(draw, line, self.font_small) + 3 for line in footer_lines)
            y += 8 + footer_h

        y += self.padding
        return max(y, 160)

    def _measure_element(self, draw: ImageDraw.ImageDraw, element: CardElement) -> int:
        if isinstance(element, TextElement):
            font = self.font_mono if element.is_code else (self.font_body_bold if element.bold else self.font_body)
            lines = _wrap_text(draw, element.text, font, self.content_width)
            line_h = max(_get_text_height(draw, "Ag", font) + 5, 20)
            return max(len(lines) * line_h, line_h)

        elif isinstance(element, KeyValuesElement):
            cols = max(1, min(element.columns, 4))
            rows_count = (len(element.items) + cols - 1) // cols
            row_h = 32
            return rows_count * row_h

        elif isinstance(element, TableElement):
            header_h = 34
            row_h = 30
            divider_h = 1
            return header_h + divider_h + (len(element.rows) * row_h)

        elif isinstance(element, DividerElement):
            return element.margin_top + (1 if element.line else 0) + element.margin_bottom

        elif isinstance(element, ProgressBarElement):
            return 46

        elif isinstance(element, CodeBlockElement):
            lines = element.code.splitlines() or [""]
            line_h = _get_text_height(draw, "Ag", self.font_mono) + 4
            return (len(lines) * line_h) + 20

        return 20

    def _draw_all(self, draw: ImageDraw.ImageDraw, total_height: int) -> None:
        # Draw outer card box with sharp corners (NO rounded corners)
        draw.rectangle(
            [(0, 0), (self.card_width - 1, total_height - 1)],
            fill=self.theme.card_bg,
            outline=self.theme.border_color,
            width=1,
        )

        accent = self.card.accent_color or self.theme.accent_color

        # Accent top bar: sharp rectangle
        draw.rectangle(
            [(0, 0), (self.card_width - 1, 4)],
            fill=accent,
        )

        y = self.padding + 6
        x = self.padding

        # Header: Title & optional Subtitle
        title_lines = _wrap_text(draw, self.card.title, self.font_title, self.content_width - 130)
        curr_y = y
        for line in title_lines:
            draw.text((x, curr_y), line, fill=self.theme.text_primary, font=self.font_title)
            curr_y += _get_text_height(draw, line, self.font_title) + 4

        if self.card.subtitle:
            sub_lines = _wrap_text(draw, self.card.subtitle, self.font_subtitle, self.content_width - 130)
            for line in sub_lines:
                draw.text((x, curr_y), line, fill=self.theme.text_secondary, font=self.font_subtitle)
                curr_y += _get_text_height(draw, line, self.font_subtitle) + 3

        # Header Badge: top right with sharp rectangle
        if self.card.badge:
            badge_w, badge_h = _measure_badge(draw, self.card.badge, self.font_small_bold)
            badge_x = self.card_width - self.padding - badge_w
            _draw_sharp_badge(draw, self.card.badge, x=badge_x, y=y + 2, font=self.font_small_bold)

        header_h = max(curr_y - y, 32)
        y += header_h + 14

        # Header Divider: sharp 1px rectangle
        draw.rectangle([(x, y), (x + self.content_width, y)], fill=self.theme.border_color)
        y += 1 + 16

        # Draw Elements
        for el in self.card.elements:
            y = self._draw_element(draw, el, x, y)
            y += self.theme.element_spacing

        # Draw Footer
        if self.card.footer:
            y += 4
            footer_lines = _wrap_text(draw, self.card.footer, self.font_small, self.content_width)
            for line in footer_lines:
                draw.text((x, y), line, fill=self.theme.text_muted, font=self.font_small)
                y += _get_text_height(draw, line, self.font_small) + 3

    def _draw_element(self, draw: ImageDraw.ImageDraw, element: CardElement, x: int, y: int) -> int:
        if isinstance(element, TextElement):
            font = self.font_mono if element.is_code else (self.font_body_bold if element.bold else self.font_body)
            color = element.color or (self.theme.text_muted if element.muted else self.theme.text_primary)
            lines = _wrap_text(draw, element.text, font, self.content_width)
            line_h = max(_get_text_height(draw, "Ag", font) + 5, 20)
            for line in lines:
                draw.text((x, y), line, fill=color, font=font)
                y += line_h
            return y

        elif isinstance(element, KeyValuesElement):
            cols = max(1, min(element.columns, 4))
            col_w = self.content_width // cols
            row_h = 32

            for idx, (k, v) in enumerate(element.items):
                col_idx = idx % cols
                row_idx = idx // cols
                item_x = x + (col_idx * col_w)
                item_y = y + (row_idx * row_h)

                # Key in secondary color
                key_text = f"{k}:"
                draw.text((item_x, item_y + 4), key_text, fill=self.theme.text_secondary, font=self.font_body)
                kw = _get_text_width(draw, key_text, self.font_body) + 8

                # Value
                if isinstance(v, Badge):
                    _draw_sharp_badge(draw, v, item_x + kw, item_y + 3, self.font_small_bold)
                else:
                    draw.text((item_x + kw, item_y + 4), str(v), fill=self.theme.text_primary, font=self.font_body_bold)

            rows_count = (len(element.items) + cols - 1) // cols
            return y + (rows_count * row_h)

        elif isinstance(element, TableElement):
            cols = len(element.headers)
            if cols == 0:
                return y

            header_h = 34
            row_h = 30
            alignments = element.alignments or ["left"] * cols

            # Calculate column widths evenly or proportionally
            col_w = self.content_width // cols

            # Header background: sharp rectangle (panel_bg)
            draw.rectangle([(x, y), (x + self.content_width, y + header_h)], fill=self.theme.panel_bg)

            # Draw header labels
            for i, header in enumerate(element.headers):
                cell_x = x + (i * col_w)
                align = alignments[i] if i < len(alignments) else "left"
                hw = _get_text_width(draw, header, self.font_body_bold)
                if align == "right":
                    tx = cell_x + col_w - hw - 8
                elif align == "center":
                    tx = cell_x + (col_w - hw) // 2
                else:
                    tx = cell_x + 8
                draw.text((tx, y + 8), header, fill=self.theme.text_primary, font=self.font_body_bold)

            y += header_h

            # Header divider: sharp rectangle
            draw.rectangle([(x, y), (x + self.content_width, y)], fill=self.theme.border_color)
            y += 1

            # Draw rows
            for row_idx, row in enumerate(element.rows):
                row_bg = self.theme.card_bg if row_idx % 2 == 0 else self.theme.panel_bg
                draw.rectangle([(x, y), (x + self.content_width, y + row_h)], fill=row_bg)

                for i in range(cols):
                    val = str(row[i]) if i < len(row) else ""
                    cell_x = x + (i * col_w)
                    align = alignments[i] if i < len(alignments) else "left"
                    vw = _get_text_width(draw, val, self.font_body)
                    if align == "right":
                        tx = cell_x + col_w - vw - 8
                    elif align == "center":
                        tx = cell_x + (col_w - vw) // 2
                    else:
                        tx = cell_x + 8
                    draw.text((tx, y + 6), val, fill=self.theme.text_secondary, font=self.font_body)

                y += row_h

            return y

        elif isinstance(element, DividerElement):
            y += element.margin_top
            if element.line:
                draw.rectangle([(x, y), (x + self.content_width, y)], fill=self.theme.border_color)
                y += 1
            y += element.margin_bottom
            return y

        elif isinstance(element, ProgressBarElement):
            label_text = element.label
            pct = 0.0
            if element.max_value > 0:
                pct = max(0.0, min(1.0, element.value / element.max_value))
            val_text = f"{element.value:g}{element.unit} / {element.max_value:g}{element.unit} ({pct * 100:.1f}%)"

            # Labels row
            draw.text((x, y), label_text, fill=self.theme.text_primary, font=self.font_body_bold)
            vw = _get_text_width(draw, val_text, self.font_small)
            draw.text((x + self.content_width - vw, y + 2), val_text, fill=self.theme.text_secondary, font=self.font_small)
            y += 24

            # Sharp track rectangle (panel_bg)
            bar_h = 10
            draw.rectangle(
                [(x, y), (x + self.content_width, y + bar_h)],
                fill=self.theme.panel_bg,
                outline=self.theme.border_color,
                width=1,
            )

            # Sharp fill rectangle (accent_color or element color)
            fill_w = int(self.content_width * pct)
            if fill_w > 0:
                bar_color = element.color or self.theme.accent_color
                draw.rectangle([(x, y), (x + fill_w, y + bar_h)], fill=bar_color)

            y += bar_h + 12
            return y

        elif isinstance(element, CodeBlockElement):
            lines = element.code.splitlines() or [""]
            line_h = _get_text_height(draw, "Ag", self.font_mono) + 4
            block_h = (len(lines) * line_h) + 20

            # Background and border: sharp rectangle (NO rounded corners)
            draw.rectangle(
                [(x, y), (x + self.content_width, y + block_h)],
                fill=self.theme.code_bg,
                outline=self.theme.border_color,
                width=1,
            )

            curr_y = y + 10
            for line in lines:
                draw.text((x + 12, curr_y), line, fill=self.theme.text_primary, font=self.font_mono)
                curr_y += line_h

            return y + block_h

        return y


def render_card(card: Card, theme: Theme = DARK_THEME) -> bytes:
    """Renders a Card into PNG bytes with sharp corners."""
    renderer = CardRenderer(card, theme)
    return renderer.render()


def render_text_card(
    title: str,
    text: str,
    subtitle: str | None = None,
    badge: str | None = None,
    badge_color: str = "blue",
    footer: str | None = None,
    theme: Theme = DARK_THEME,
) -> bytes:
    """Convenience function: renders a formatted text card to PNG bytes with sharp corners."""
    card = Card(title=title, subtitle=subtitle, footer=footer)
    if badge:
        card.set_badge(badge, color=badge_color)

    # Process text paragraphs / code blocks
    lines = text.splitlines()
    in_code_block = False
    code_lines: list[str] = []
    normal_lines: list[str] = []

    def flush_normal() -> None:
        if normal_lines:
            card.add_text("\n".join(normal_lines))
            normal_lines.clear()

    def flush_code() -> None:
        if code_lines:
            card.add_code_block("\n".join(code_lines))
            code_lines.clear()

    for line in lines:
        if line.strip().startswith("```"):
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                flush_normal()
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
        else:
            normal_lines.append(line)

    if in_code_block:
        flush_code()
    else:
        flush_normal()

    return render_card(card, theme)


def render_table_card(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    subtitle: str | None = None,
    badge: str | None = None,
    badge_color: str = "blue",
    footer: str | None = None,
    theme: Theme = DARK_THEME,
) -> bytes:
    """Convenience function: renders a table card into PNG bytes with sharp corners."""
    card = Card(title=title, subtitle=subtitle, footer=footer)
    if badge:
        card.set_badge(badge, color=badge_color)

    card.add_table(headers=headers, rows=rows)
    return render_card(card, theme)
