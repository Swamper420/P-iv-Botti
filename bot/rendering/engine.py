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
    ImageElement,
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


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    overflow: str = "ellipsis",
) -> str:
    """Ensure text fits within max_width using specified overflow strategy ('ellipsis', 'clip', 'none')."""
    if max_width <= 0:
        return ""
    if _get_text_width(draw, text, font) <= max_width:
        return text

    if overflow == "none":
        return text

    if overflow == "clip":
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _get_text_width(draw, text[:mid], font) <= max_width:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]

    # Default: "ellipsis"
    ellipsis = "..."
    ell_w = _get_text_width(draw, ellipsis, font)
    if ell_w > max_width:
        ellipsis = "…"
        ell_w = _get_text_width(draw, ellipsis, font)
        if ell_w > max_width:
            return ""

    avail_w = max_width - ell_w
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _get_text_width(draw, text[:mid], font) <= avail_w:
            lo = mid
        else:
            hi = mid - 1

    return text[:lo].rstrip() + ellipsis


def _calculate_col_widths(
    content_width: int,
    cols: int,
    col_widths: Sequence[int | float] | None = None,
) -> list[int]:
    """Calculate pixel column widths from proportional or explicit weights."""
    if cols <= 0:
        return []
    if not col_widths or len(col_widths) != cols:
        base_w = content_width // cols
        remainder = content_width - (base_w * cols)
        widths = [base_w] * cols
        for i in range(remainder):
            widths[i] += 1
        return widths

    total_spec = sum(col_widths)
    if total_spec <= 0:
        return _calculate_col_widths(content_width, cols, None)

    widths = [int(content_width * (w / total_spec)) for w in col_widths]
    remainder = content_width - sum(widths)
    for i in range(remainder):
        widths[i % cols] += 1
    return widths


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


def _probe_image_size(image_bytes: bytes) -> tuple[int, int] | None:
    """Return (width, height) of image bytes without fully loading, or None on failure."""
    if not image_bytes:
        return None
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                return None
            return w, h
    except Exception:
        return None


def _compute_image_display_size(
    orig_w: int,
    orig_h: int,
    content_width: int,
    max_height: int | None,
) -> tuple[int, int]:
    """Compute aspect-fit display size for content width with optional max height.

    Fits to full content width unless that would exceed max_height, in which
    case scales down to max_height and lets width shrink (centered on draw).
    All corners stay sharp (no rounding applied anywhere).
    """
    if orig_w <= 0 or orig_h <= 0 or content_width <= 0:
        return content_width, 200
    scaled_h = int(round(orig_h * (content_width / orig_w)))
    if max_height is not None and max_height > 0 and scaled_h > max_height:
        disp_h = max_height
        disp_w = max(1, int(round(orig_w * (disp_h / orig_h))))
        disp_w = min(disp_w, content_width)
        return disp_w, disp_h
    return content_width, max(1, scaled_h)


class CardRenderer:
    def __init__(self, card: Card, theme: Theme = DARK_THEME) -> None:
        self.card = card
        self.theme = theme

        # Load fonts - optimized hierarchy for clarity & density
        self.font_title = _load_font("bold", 22)
        self.font_subtitle = _load_font("regular", 13)
        self.font_body = _load_font("regular", 14)
        self.font_body_bold = _load_font("bold", 14)
        self.font_primary_item = _load_font("bold", 16)  # Prioritized & bigger for player names & key items
        self.font_table_header = _load_font("bold", 12)  # Concise header font
        self.font_label = _load_font("regular", 13)       # Subtle labels
        self.font_mono = _load_font("mono", 13)
        self.font_small = _load_font("regular", 11)
        self.font_small_bold = _load_font("bold", 11)

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
        self._draw_all(draw, img, total_height)

        output = BytesIO()
        img.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def _measure_and_layout(self, draw: ImageDraw.ImageDraw) -> int:
        y = self.padding

        # Accent top bar (sharp rectangle)
        y += 3 + 8

        # Header: Title, Subtitle, Badge
        title_lines = _wrap_text(draw, self.card.title, self.font_title, self.content_width - 120)
        title_h = sum(_get_text_height(draw, line, self.font_title) + 3 for line in title_lines)

        subtitle_h = 0
        if self.card.subtitle:
            sub_lines = _wrap_text(draw, self.card.subtitle, self.font_subtitle, self.content_width - 120)
            subtitle_h = sum(_get_text_height(draw, line, self.font_subtitle) + 2 for line in sub_lines) + 2

        header_h = max(title_h + subtitle_h, 28)
        y += header_h + 8

        # Header divider
        y += 1 + 8

        # Elements
        for el in self.card.elements:
            y += self._measure_element(draw, el) + self.theme.element_spacing

        # Footer
        if self.card.footer:
            footer_lines = _wrap_text(draw, self.card.footer, self.font_small, self.content_width)
            footer_h = sum(_get_text_height(draw, line, self.font_small) + 2 for line in footer_lines)
            y += 4 + footer_h

        y += self.padding
        if self.card.max_height is not None:
            return min(max(y, 80), self.card.max_height)
        return max(y, 80)

    def _measure_element(self, draw: ImageDraw.ImageDraw, element: CardElement) -> int:
        if isinstance(element, TextElement):
            font = self.font_mono if element.is_code else (self.font_body_bold if element.bold else self.font_body)
            lines = _wrap_text(draw, element.text, font, self.content_width)
            if element.max_lines is not None and len(lines) > element.max_lines:
                lines = lines[: element.max_lines]
            line_h = max(_get_text_height(draw, "Ag", font) + 4, 18)
            return max(len(lines) * line_h, line_h)

        elif isinstance(element, KeyValuesElement):
            cols = max(1, min(element.columns, 4))
            rows_count = (len(element.items) + cols - 1) // cols
            row_h = 28
            return rows_count * row_h

        elif isinstance(element, TableElement):
            cols = len(element.headers)
            if cols == 0:
                return 0
            header_h = 28
            row_h = 28
            divider_h = 1
            has_overflow_row = bool(element.max_rows is not None and len(element.rows) > element.max_rows)
            rendered_rows = min(len(element.rows), element.max_rows) if element.max_rows is not None else len(element.rows)
            total_rows = rendered_rows + (1 if has_overflow_row else 0)
            return header_h + divider_h + (total_rows * row_h)

        elif isinstance(element, DividerElement):
            return element.margin_top + (1 if element.line else 0) + element.margin_bottom

        elif isinstance(element, ProgressBarElement):
            return 32

        elif isinstance(element, CodeBlockElement):
            lines = element.code.splitlines() or [""]
            hidden_lines = 0
            if element.max_lines is not None and len(lines) > element.max_lines:
                hidden_lines = len(lines) - element.max_lines
                lines = lines[: element.max_lines]
            total_lines_count = len(lines) + (1 if hidden_lines > 0 else 0)
            line_h = _get_text_height(draw, "Ag", self.font_mono) + 3
            return (total_lines_count * line_h) + 14

        elif isinstance(element, ImageElement):
            return self._measure_image_element(draw, element)

        return 18

    def _measure_image_element(
        self, draw: ImageDraw.ImageDraw, element: ImageElement
    ) -> int:
        size = _probe_image_size(element.image_bytes)
        if size is None:
            base_h = 48
        else:
            _disp_w, disp_h = _compute_image_display_size(
                size[0], size[1], self.content_width, element.max_height
            )
            base_h = disp_h
        total = base_h
        if element.caption:
            cap_lines = _wrap_text(
                draw, element.caption, self.font_small, self.content_width
            )
            if cap_lines:
                cap_h = sum(
                    _get_text_height(draw, line, self.font_small) + 2
                    for line in cap_lines
                )
                total += 6 + cap_h
        return total

    def _draw_all(
        self, draw: ImageDraw.ImageDraw, img: Image.Image, total_height: int
    ) -> None:
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
            [(0, 0), (self.card_width - 1, 3)],
            fill=accent,
        )

        y = self.padding + 2
        x = self.padding

        # Header: Title & optional Subtitle
        title_lines = _wrap_text(draw, self.card.title, self.font_title, self.content_width - 130)
        curr_y = y
        for line in title_lines:
            draw.text((x, curr_y), line, fill=self.theme.text_primary, font=self.font_title)
            curr_y += _get_text_height(draw, line, self.font_title) + 3

        if self.card.subtitle:
            sub_lines = _wrap_text(draw, self.card.subtitle, self.font_subtitle, self.content_width - 130)
            for line in sub_lines:
                draw.text((x, curr_y), line, fill=self.theme.text_secondary, font=self.font_subtitle)
                curr_y += _get_text_height(draw, line, self.font_subtitle) + 2

        # Header Badge: top right with sharp rectangle
        if self.card.badge:
            badge_w, badge_h = _measure_badge(draw, self.card.badge, self.font_small_bold)
            badge_x = self.card_width - self.padding - badge_w
            _draw_sharp_badge(draw, self.card.badge, x=badge_x, y=y + 2, font=self.font_small_bold)

        header_h = max(curr_y - y, 28)
        y += header_h + 8

        # Header Divider: sharp 1px rectangle
        draw.rectangle([(x, y), (x + self.content_width, y)], fill=self.theme.border_color)
        y += 1 + 8

        # Draw Elements
        for el in self.card.elements:
            y = self._draw_element(draw, img, el, x, y)
            y += self.theme.element_spacing

        # Draw Footer
        if self.card.footer:
            y += 4
            footer_lines = _wrap_text(draw, self.card.footer, self.font_small, self.content_width)
            for line in footer_lines:
                draw.text((x, y), line, fill=self.theme.text_muted, font=self.font_small)
                y += _get_text_height(draw, line, self.font_small) + 2

    def _draw_element(
        self,
        draw: ImageDraw.ImageDraw,
        img: Image.Image,
        element: CardElement,
        x: int,
        y: int,
    ) -> int:
        if isinstance(element, TextElement):
            font = self.font_mono if element.is_code else (self.font_body_bold if element.bold else self.font_body)
            color = element.color or (self.theme.text_muted if element.muted else self.theme.text_primary)
            lines = _wrap_text(draw, element.text, font, self.content_width)
            if element.max_lines is not None and len(lines) > element.max_lines:
                lines = lines[: element.max_lines]
                if lines:
                    lines[-1] = _fit_text(draw, f"{lines[-1]}", font, self.content_width, overflow="ellipsis")
            line_h = max(_get_text_height(draw, "Ag", font) + 4, 18)
            for line in lines:
                draw.text((x, y), line, fill=color, font=font)
                y += line_h
            return y

        elif isinstance(element, KeyValuesElement):
            cols = max(1, min(element.columns, 4))
            col_w = self.content_width // cols
            row_h = 28

            for idx, (k, v) in enumerate(element.items):
                col_idx = idx % cols
                row_idx = idx // cols
                item_x = x + (col_idx * col_w)
                item_y = y + (row_idx * row_h)

                avail_w = col_w - 12
                # Key in secondary color
                key_text = f"{k}:"
                fitted_key = _fit_text(draw, key_text, self.font_label, max(20, avail_w - 20), overflow=element.overflow)
                draw.text((item_x, item_y + 4), fitted_key, fill=self.theme.text_secondary, font=self.font_label)
                kw = _get_text_width(draw, fitted_key, self.font_label) + 6

                max_val_w = max(10, avail_w - kw)

                # Value: prominent bold in text_primary
                if isinstance(v, Badge):
                    _draw_sharp_badge(draw, v, item_x + kw, item_y + 2, self.font_small_bold)
                else:
                    fitted_val = _fit_text(draw, str(v), self.font_body_bold, max_val_w, overflow=element.overflow)
                    draw.text((item_x + kw, item_y + 3), fitted_val, fill=self.theme.text_primary, font=self.font_body_bold)

            rows_count = (len(element.items) + cols - 1) // cols
            return y + (rows_count * row_h)

        elif isinstance(element, TableElement):
            cols = len(element.headers)
            if cols == 0:
                return y

            header_h = 28
            row_h = 28
            alignments = element.alignments or ["left"] * cols
            col_widths = _calculate_col_widths(self.content_width, cols, element.col_widths)

            # Header background: sharp rectangle (panel_bg)
            draw.rectangle([(x, y), (x + self.content_width, y + header_h)], fill=self.theme.panel_bg)

            # Draw header labels (concise, subtle uppercase headers)
            curr_x = x
            for i, header in enumerate(element.headers):
                w_i = col_widths[i]
                align = alignments[i] if i < len(alignments) else "left"
                fitted_header = _fit_text(draw, header.upper(), self.font_table_header, w_i - 14, overflow=element.overflow)
                hw = _get_text_width(draw, fitted_header, self.font_table_header)
                if align == "right":
                    tx = curr_x + w_i - hw - 8
                elif align == "center":
                    tx = curr_x + (w_i - hw) // 2
                else:
                    tx = curr_x + 8
                draw.text((tx, y + 6), fitted_header, fill=self.theme.text_secondary, font=self.font_table_header)
                curr_x += w_i

            y += header_h

            # Header divider: sharp rectangle
            draw.rectangle([(x, y), (x + self.content_width, y)], fill=self.theme.border_color)
            y += 1

            # Determine rows to render & check for row overflow
            rows_to_render = element.rows
            has_overflow_row = False
            hidden_rows_count = 0
            if element.max_rows is not None and len(element.rows) > element.max_rows:
                rows_to_render = element.rows[: element.max_rows]
                has_overflow_row = True
                hidden_rows_count = len(element.rows) - element.max_rows

            # Draw rows
            for row_idx, row in enumerate(rows_to_render):
                row_bg = self.theme.card_bg if row_idx % 2 == 0 else self.theme.panel_bg
                draw.rectangle([(x, y), (x + self.content_width, y + row_h)], fill=row_bg)

                curr_x = x
                for i in range(cols):
                    w_i = col_widths[i]
                    val = str(row[i]) if i < len(row) else ""
                    align = alignments[i] if i < len(alignments) else "left"

                    is_primary = (element.primary_col is not None and i == element.primary_col)
                    is_bold = is_primary or (element.bold_cols is not None and i in element.bold_cols)

                    if is_primary:
                        cell_font = self.font_primary_item
                        cell_color = self.theme.text_primary
                        y_offset = 4
                    elif is_bold:
                        cell_font = self.font_body_bold
                        cell_color = self.theme.text_primary
                        y_offset = 5
                    else:
                        cell_font = self.font_body
                        cell_color = self.theme.text_secondary
                        y_offset = 5

                    fitted_val = _fit_text(draw, val, cell_font, w_i - 14, overflow=element.overflow)
                    vw = _get_text_width(draw, fitted_val, cell_font)
                    if align == "right":
                        tx = curr_x + w_i - vw - 8
                    elif align == "center":
                        tx = curr_x + (w_i - vw) // 2
                    else:
                        tx = curr_x + 8
                    draw.text((tx, y + y_offset), fitted_val, fill=cell_color, font=cell_font)
                    curr_x += w_i

                y += row_h

            if has_overflow_row:
                overflow_bg = self.theme.card_bg if len(rows_to_render) % 2 == 0 else self.theme.panel_bg
                draw.rectangle([(x, y), (x + self.content_width, y + row_h)], fill=overflow_bg)
                overflow_text = f"… (+{hidden_rows_count} muuta riviä)"
                draw.text((x + 8, y + 6), overflow_text, fill=self.theme.text_muted, font=self.font_small)
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

            # Labels row: label in secondary and value in bold primary
            draw.text((x, y), label_text, fill=self.theme.text_secondary, font=self.font_label)
            vw = _get_text_width(draw, val_text, self.font_body_bold)
            draw.text((x + self.content_width - vw, y), val_text, fill=self.theme.text_primary, font=self.font_body_bold)
            y += 18

            # Sharp track rectangle (panel_bg)
            bar_h = 8
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

            y += bar_h + 6
            return y

        elif isinstance(element, CodeBlockElement):
            lines = element.code.splitlines() or [""]
            hidden_lines = 0
            if element.max_lines is not None and len(lines) > element.max_lines:
                hidden_lines = len(lines) - element.max_lines
                lines = lines[: element.max_lines]

            line_h = _get_text_height(draw, "Ag", self.font_mono) + 3
            total_lines_count = len(lines) + (1 if hidden_lines > 0 else 0)
            block_h = (total_lines_count * line_h) + 14
            max_code_w = self.content_width - 20

            # Background and border: sharp rectangle (NO rounded corners)
            draw.rectangle(
                [(x, y), (x + self.content_width, y + block_h)],
                fill=self.theme.code_bg,
                outline=self.theme.border_color,
                width=1,
            )

            curr_y = y + 7
            for line in lines:
                fitted_line = _fit_text(draw, line, self.font_mono, max_code_w, overflow=element.overflow)
                draw.text((x + 10, curr_y), fitted_line, fill=self.theme.text_primary, font=self.font_mono)
                curr_y += line_h

            if hidden_lines > 0:
                overflow_text = f"… (+{hidden_lines} riviä)"
                draw.text((x + 10, curr_y), overflow_text, fill=self.theme.text_muted, font=self.font_mono)
                curr_y += line_h

            return y + block_h

        elif isinstance(element, ImageElement):
            return self._draw_image_element(draw, img, element, x, y)

        return y

    def _draw_image_element(
        self,
        draw: ImageDraw.ImageDraw,
        img: Image.Image,
        element: ImageElement,
        x: int,
        y: int,
    ) -> int:
        """Draw photo with sharp 1px border and optional centered caption."""
        size = _probe_image_size(element.image_bytes)
        if size is None:
            box_h = 48
            draw.rectangle(
                [(x, y), (x + self.content_width, y + box_h)],
                fill=self.theme.panel_bg,
                outline=self.theme.border_color,
                width=1,
            )
            draw.text(
                (x + 8, y + 14),
                "Kuvaa ei voitu näyttää",
                fill=self.theme.text_muted,
                font=self.font_small,
            )
            y += box_h
        else:
            disp_w, disp_h = _compute_image_display_size(
                size[0], size[1], self.content_width, element.max_height
            )
            x_offset = x + (self.content_width - disp_w) // 2 if disp_w < self.content_width else x
            try:
                with Image.open(BytesIO(element.image_bytes)) as pil_img:
                    try:
                        from PIL import ImageOps

                        pil_img = ImageOps.exif_transpose(pil_img)
                    except Exception:
                        pass
                    if pil_img.mode in ("RGBA", "LA", "PA"):
                        canvas = Image.new("RGB", pil_img.size, self.theme.card_bg)
                        try:
                            alpha = pil_img.split()[-1]
                            canvas.paste(pil_img.convert("RGB"), mask=alpha)
                            pil_img = canvas
                        except Exception:
                            pil_img = pil_img.convert("RGB")
                    else:
                        pil_img = pil_img.convert("RGB")
                    if (pil_img.width, pil_img.height) != (disp_w, disp_h):
                        pil_img = pil_img.resize((disp_w, disp_h), Image.LANCZOS)
                    img.paste(pil_img, (x_offset, y))
            except Exception:
                draw.rectangle(
                    [(x, y), (x + self.content_width, y + 48)],
                    fill=self.theme.panel_bg,
                    outline=self.theme.border_color,
                    width=1,
                )
                draw.text(
                    (x + 8, y + 14),
                    "Kuvaa ei voitu näyttää",
                    fill=self.theme.text_muted,
                    font=self.font_small,
                )
                y += 48
            else:
                # Sharp border around photo (NO rounded corners)
                draw.rectangle(
                    [(x_offset, y), (x_offset + disp_w, y + disp_h)],
                    outline=self.theme.border_color,
                    width=1,
                )
                y += disp_h

        if element.caption:
            cap_lines = _wrap_text(draw, element.caption, self.font_small, self.content_width)
            y += 6
            for line in cap_lines:
                lw = _get_text_width(draw, line, self.font_small)
                tx = x + (self.content_width - lw) // 2 if lw < self.content_width else x
                draw.text((tx, y), line, fill=self.theme.text_muted, font=self.font_small)
                y += _get_text_height(draw, line, self.font_small) + 2
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
    col_widths: list[int | float] | None = None,
    max_rows: int | None = None,
    overflow: str = "ellipsis",
    primary_col: int | None = 0,
    bold_cols: list[int] | None = None,
    theme: Theme = DARK_THEME,
) -> bytes:
    """Convenience function: renders a table card into PNG bytes with sharp corners."""
    card = Card(title=title, subtitle=subtitle, footer=footer)
    if badge:
        card.set_badge(badge, color=badge_color)

    card.add_table(
        headers=headers,
        rows=rows,
        col_widths=col_widths,
        max_rows=max_rows,
        overflow=overflow,
        primary_col=primary_col,
        bold_cols=bold_cols,
    )
    return render_card(card, theme)
