from __future__ import annotations

import functools
from io import BytesIO
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from bot.rendering.icons import draw_icon, sanitize_for_pil
from bot.rendering.models import (
    DARK_THEME,
    Badge,
    BarListElement,
    CalloutElement,
    Card,
    CardElement,
    CodeBlockElement,
    DividerElement,
    HeroElement,
    ImageElement,
    ImageGridElement,
    KeyValuesElement,
    ProgressBarElement,
    RangeBarElement,
    SectionHeaderElement,
    StatGridElement,
    TableElement,
    TextElement,
    Theme,
    TimelineElement,
)

# Common TTF/OTF search candidates on Linux and other platforms
DISPLAY_CANDIDATES = (
    "/usr/share/fonts/gsfonts/NimbusSansNarrow-Bold.otf",
    "/usr/share/fonts/gsfonts/NimbusSans-Bold.otf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
)

SANS_REGULAR_CANDIDATES = (
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
)

SANS_BOLD_CANDIDATES = (
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
)

MONO_CANDIDATES = (
    "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/TTF/Hack-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
)

MONO_BOLD_CANDIDATES = (
    "/usr/share/fonts/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/TTF/Hack-Bold.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = str(hex_color or "").strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(ch * 2 for ch in cleaned)
    if len(cleaned) != 6:
        return (56, 189, 248)
    try:
        return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))
    except ValueError:
        return (56, 189, 248)


def _blend_hex(base_hex: str, overlay_hex: str, alpha: float) -> str:
    a = max(0.0, min(1.0, alpha))
    b = _hex_to_rgb(base_hex)
    o = _hex_to_rgb(overlay_hex)
    r = int(b[0] * (1.0 - a) + o[0] * a)
    g = int(b[1] * (1.0 - a) + o[1] * a)
    bl = int(b[2] * (1.0 - a) + o[2] * a)
    return f"#{r:02x}{g:02x}{bl:02x}"


@functools.lru_cache(maxsize=32)
def _find_font_path(candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


@functools.lru_cache(maxsize=96)
def _load_font(font_type: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates_map = {
        "display": DISPLAY_CANDIDATES,
        "bold": SANS_BOLD_CANDIDATES,
        "mono": MONO_CANDIDATES,
        "mono_bold": MONO_BOLD_CANDIDATES,
        "regular": SANS_REGULAR_CANDIDATES,
    }
    path = _find_font_path(candidates_map.get(font_type, SANS_REGULAR_CANDIDATES))
    if path:
        try:
            f = ImageFont.truetype(path, size)
            if font_type in ("display", "bold") and "AdwaitaSans-Regular" in path:
                try:
                    f.set_variation_by_name("Black" if font_type == "display" else "Bold")
                except Exception:
                    pass
            return f
        except Exception:
            pass
    return ImageFont.load_default()


def _get_text_bbox(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text or " ", font=font)


def _get_text_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> int:
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _get_text_height(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> int:
    bbox = _get_text_bbox(draw, text, font)
    return bbox[3] - bbox[1]


def _line_advance(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    extra: int = 5,
) -> int:
    bbox = draw.textbbox((0, 0), "Äg", font=font)
    return max((bbox[3] - bbox[1]) + extra, 14)


def _draw_text_ink(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> tuple[int, int]:
    """Draw text with top-left ink edge anchored exactly at (x, y)."""
    if not text:
        return 0, 0
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - bbox[0], y - bbox[1]), text, fill=fill, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_text_vcenter(
    draw: ImageDraw.ImageDraw,
    x: int,
    box_y: int,
    box_h: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> tuple[int, int]:
    """Draw text vertically centered by ink bounding box inside [box_y, box_y + box_h]."""
    if not text:
        return 0, 0
    bbox = draw.textbbox((0, 0), text, font=font)
    ink_h = bbox[3] - bbox[1]
    y = box_y + (box_h - ink_h) // 2
    draw.text((x - bbox[0], y - bbox[1]), text, fill=fill, font=font)
    return bbox[2] - bbox[0], ink_h


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    clean = sanitize_for_pil(text)
    if max_width <= 8:
        return clean.splitlines() or [""]
    lines: list[str] = []
    for raw_line in clean.splitlines():
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
                    chunk = ""
                    for char in word:
                        if _get_text_width(draw, chunk + char, font) <= max_width:
                            chunk += char
                        else:
                            if chunk:
                                lines.append(chunk)
                            chunk = char
                    current_line = chunk

        if current_line:
            lines.append(current_line)

    return lines or [""]


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    overflow: str = "ellipsis",
) -> str:
    """Ensure text fits within max_width using specified overflow strategy ('ellipsis', 'clip', 'none')."""
    clean = sanitize_for_pil(text)
    if max_width <= 0:
        return ""
    if _get_text_width(draw, clean, font) <= max_width:
        return clean

    if overflow == "none":
        return clean

    if overflow == "clip":
        lo, hi = 0, len(clean)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _get_text_width(draw, clean[:mid], font) <= max_width:
                lo = mid
            else:
                hi = mid - 1
        return clean[:lo]

    ellipsis = "…"
    ell_w = _get_text_width(draw, ellipsis, font)
    if ell_w > max_width:
        ellipsis = "."
        ell_w = _get_text_width(draw, ellipsis, font)
        if ell_w > max_width:
            return ""

    avail_w = max_width - ell_w
    lo, hi = 0, len(clean)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _get_text_width(draw, clean[:mid], font) <= avail_w:
            lo = mid
        else:
            hi = mid - 1

    return clean[:lo].rstrip() + ellipsis


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

    widths = [max(24, int(content_width * (w / total_spec))) for w in col_widths]
    diff = content_width - sum(widths)
    if diff != 0:
        widths[-1] = max(24, widths[-1] + diff)
    return widths


def _measure_badge(
    draw: ImageDraw.ImageDraw,
    badge: Badge,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    text = sanitize_for_pil(badge.text).strip().upper()
    pad_h = 9
    pad_v = 5
    text_w = _get_text_width(draw, text, font)
    ref_h = _get_text_height(draw, "ÄG", font)
    return text_w + pad_h * 2, ref_h + pad_v * 2


def _draw_sharp_badge(
    draw: ImageDraw.ImageDraw,
    badge: Badge,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    """Draws a badge with sharp rectangular corners and returns its (width, height)."""
    style = badge.get_style()
    text = sanitize_for_pil(badge.text).strip().upper()
    pad_h = 9
    badge_w, badge_h = _measure_badge(draw, badge, font)

    # Draw sharp rectangle background and border (NO rounded corners)
    draw.rectangle(
        [(x, y), (x + badge_w, y + badge_h)],
        fill=style.bg_color,
        outline=style.border_color,
        width=1,
    )
    _draw_text_vcenter(draw, x + pad_h, y, badge_h, text, font, style.fg_color)
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
    """Compute aspect-fit display size for content width with optional max height."""
    if orig_w <= 0 or orig_h <= 0 or content_width <= 0:
        return content_width, 200
    scaled_h = int(round(orig_h * (content_width / orig_w)))
    if max_height is not None and max_height > 0 and scaled_h > max_height:
        disp_h = max_height
        disp_w = max(1, int(round(orig_w * (disp_h / orig_h))))
        disp_w = min(disp_w, content_width)
        return disp_w, disp_h
    return content_width, max(1, scaled_h)


def _compute_grid_cell_display_size(
    orig_w: int,
    orig_h: int,
    cell_width: int,
    max_cell_height: int | None,
) -> tuple[int, int]:
    """Aspect-fit a single grid cell, capping height to keep 2x2 compact."""
    if orig_w <= 0 or orig_h <= 0 or cell_width <= 0:
        return max(1, cell_width), 120
    scaled_h = int(round(orig_h * (cell_width / orig_w)))
    if (
        max_cell_height is not None
        and max_cell_height > 0
        and scaled_h > max_cell_height
    ):
        disp_h = max_cell_height
        disp_w = max(1, int(round(orig_w * (disp_h / orig_h))))
        return min(disp_w, cell_width), disp_h
    return cell_width, max(1, scaled_h)


def _compute_image_grid_layout(
    sizes: list[tuple[int, int] | None],
    content_width: int,
    gap: int,
    max_cell_height: int | None,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Compute 2-column grid layout in content-relative coordinates."""
    n = len(sizes)
    if n == 0 or content_width <= 0:
        return [], 0
    cols = 1 if n == 1 else 2
    rows = (n + cols - 1) // cols
    gap = max(0, gap)
    cell_width = (
        (content_width - gap * (cols - 1)) // cols if cols > 1 else content_width
    )
    cell_width = max(1, cell_width)

    disps: list[tuple[int, int]] = []
    for size in sizes:
        if size is None:
            disps.append((cell_width, 120))
        else:
            disps.append(
                _compute_grid_cell_display_size(
                    size[0], size[1], cell_width, max_cell_height
                )
            )

    row_heights: list[int] = []
    for r in range(rows):
        heights = [
            disps[r * cols + c][1]
            for c in range(cols)
            if r * cols + c < n
        ]
        row_heights.append(max(heights) if heights else 0)

    total_h = sum(row_heights) + gap * (rows - 1) if rows else 0

    cells: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * n
    y_cursor = 0
    for r in range(rows):
        row_h = row_heights[r]
        for c in range(cols):
            idx = r * cols + c
            if idx >= n:
                break
            disp_w, disp_h = disps[idx]
            x_cell = c * (cell_width + gap) if cols > 1 else 0
            x_rel = x_cell + (cell_width - disp_w) // 2
            y_rel = y_cursor + (row_h - disp_h) // 2
            cells[idx] = (disp_w, disp_h, x_rel, y_rel)
        y_cursor += row_h + gap

    return cells, total_h


def _tone_color(tone: str, theme: Theme) -> str:
    t = (tone or "info").lower()
    if t == "live" or t == "success":
        return "#22c55e"
    if t == "warning":
        return "#eab308"
    if t == "error":
        return "#ef4444"
    return theme.accent_color


class CardRenderer:
    def __init__(self, card: Card, theme: Theme = DARK_THEME) -> None:
        self.card = card
        self.theme = theme

        # Font hierarchy
        self.font_hero = _load_font("display", 38)
        self.font_title = _load_font("bold", 22)
        self.font_subtitle = _load_font("regular", 13)
        self.font_section = _load_font("bold", 11)
        self.font_body = _load_font("regular", 14)
        self.font_body_bold = _load_font("bold", 14)
        self.font_primary_item = _load_font("bold", 15)
        self.font_stat_value = _load_font("bold", 20)
        self.font_table_header = _load_font("bold", 11)
        self.font_label = _load_font("regular", 13)
        self.font_mono = _load_font("mono", 13)
        self.font_mono_bold = _load_font("mono_bold", 13)
        self.font_small = _load_font("regular", 11)
        self.font_small_bold = _load_font("bold", 11)

        # Layout metrics
        self.card_width = theme.card_width
        self.padding = theme.padding
        self.content_width = self.card_width - (self.padding * 2)

    def render(self) -> bytes:
        # Pass 1: Measure exact height
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

    def _header_title_width(self, draw: ImageDraw.ImageDraw) -> int:
        if self.card.badge:
            badge_w, _ = _measure_badge(draw, self.card.badge, self.font_small_bold)
            return max(120, self.content_width - badge_w - 16)
        return self.content_width

    def _measure_header(self, draw: ImageDraw.ImageDraw) -> tuple[int, int]:
        """Return (header_band_height, header_content_height)."""
        avail_w = self._header_title_width(draw)
        title_lines = _wrap_text(draw, self.card.title, self.font_title, avail_w)
        title_adv = _line_advance(draw, self.font_title, extra=4)
        title_h = len(title_lines) * title_adv

        subtitle_h = 0
        if self.card.subtitle:
            sub_lines = _wrap_text(draw, self.card.subtitle, self.font_subtitle, avail_w)
            sub_adv = _line_advance(draw, self.font_subtitle, extra=3)
            subtitle_h = 4 + len(sub_lines) * sub_adv

        badge_h = 0
        if self.card.badge:
            _, badge_h = _measure_badge(draw, self.card.badge, self.font_small_bold)

        content_h = max(title_h + subtitle_h, badge_h, 28)
        band_h = 4 + 16 + content_h + 14  # top accent bar + top pad + content + bottom pad
        return band_h, content_h

    def _measure_footer(self, draw: ImageDraw.ImageDraw) -> int:
        if not self.card.footer:
            return 0
        footer_lines = _wrap_text(draw, self.card.footer, self.font_small, self.content_width)
        adv = _line_advance(draw, self.font_small, extra=3)
        return 10 + (len(footer_lines) * adv) + 10

    def _measure_and_layout(self, draw: ImageDraw.ImageDraw) -> int:
        band_h, _ = self._measure_header(draw)
        y = band_h + 14  # spacing after header divider

        for idx, el in enumerate(self.card.elements):
            el_h = self._measure_element(draw, el)
            y += el_h
            if idx < len(self.card.elements) - 1:
                y += self.theme.element_spacing

        footer_h = self._measure_footer(draw)
        if footer_h > 0:
            y += 16 + footer_h
        else:
            y += self.padding

        if self.card.max_height is not None:
            return min(max(y, 80), self.card.max_height)
        return max(y, 80)

    def _measure_element(self, draw: ImageDraw.ImageDraw, element: CardElement) -> int:
        if isinstance(element, TextElement):
            font = (
                self.font_mono
                if element.is_code
                else (self.font_body_bold if element.bold else self.font_body)
            )
            lines = _wrap_text(draw, element.text, font, self.content_width)
            if element.max_lines is not None and len(lines) > element.max_lines:
                lines = lines[: element.max_lines]
            line_h = _line_advance(draw, font, extra=5)
            return max(len(lines) * line_h, line_h)

        if isinstance(element, KeyValuesElement):
            cols = max(1, min(element.columns, 4))
            rows_count = max(1, (len(element.items) + cols - 1) // cols)
            return 16 + (rows_count * 30)

        if isinstance(element, TableElement):
            cols = len(element.headers)
            if cols == 0:
                return 0
            header_h = 30
            row_h = 32
            has_overflow_row = bool(
                element.max_rows is not None and len(element.rows) > element.max_rows
            )
            rendered_rows = (
                min(len(element.rows), element.max_rows)
                if element.max_rows is not None
                else len(element.rows)
            )
            total_rows = rendered_rows + (1 if has_overflow_row else 0)
            return header_h + 1 + (total_rows * row_h) + 2

        if isinstance(element, DividerElement):
            return element.margin_top + (1 if element.line else 0) + element.margin_bottom

        if isinstance(element, ProgressBarElement):
            return 38

        if isinstance(element, CodeBlockElement):
            clean_code = sanitize_for_pil(element.code)
            lines = clean_code.splitlines() or [""]
            hidden_lines = 0
            if element.max_lines is not None and len(lines) > element.max_lines:
                hidden_lines = len(lines) - element.max_lines
                lines = lines[: element.max_lines]
            total_lines_count = len(lines) + (1 if hidden_lines > 0 else 0)
            line_h = _line_advance(draw, self.font_mono, extra=4)
            return (total_lines_count * line_h) + 18

        if isinstance(element, ImageElement):
            return self._measure_image_element(draw, element)

        if isinstance(element, ImageGridElement):
            return self._measure_image_grid_element(draw, element)

        if isinstance(element, SectionHeaderElement):
            return 22

        if isinstance(element, HeroElement):
            return self._measure_hero_element(draw, element)

        if isinstance(element, StatGridElement):
            return self._measure_stat_grid_element(draw, element)

        if isinstance(element, TimelineElement):
            return self._measure_timeline_element(draw, element)

        if isinstance(element, BarListElement):
            return self._measure_bar_list_element(draw, element)

        if isinstance(element, RangeBarElement):
            return 56

        if isinstance(element, CalloutElement):
            return self._measure_callout_element(draw, element)

        return 18

    def _measure_hero_element(self, draw: ImageDraw.ImageDraw, element: HeroElement) -> int:
        val_h = _get_text_height(draw, sanitize_for_pil(element.value) or "0", self.font_hero)
        lbl_h = _line_advance(draw, self.font_body_bold, extra=4)
        sub_h = _line_advance(draw, self.font_small, extra=3) if element.subtext else 0
        text_stack_h = max(val_h + 4, lbl_h + sub_h + 6)
        icon_h = 56 if element.icon else 0
        return max(text_stack_h, icon_h, 60) + 28

    def _measure_stat_grid_element(self, draw: ImageDraw.ImageDraw, element: StatGridElement) -> int:
        if not element.items:
            return 0
        cols = max(1, min(element.columns, 4))
        rows = (len(element.items) + cols - 1) // cols
        has_sub = any(t.subtext for t in element.items)
        tile_h = 74 if has_sub else 58
        gap = 8
        return (rows * tile_h) + max(0, rows - 1) * gap

    def _measure_timeline_element(self, draw: ImageDraw.ImageDraw, element: TimelineElement) -> int:
        if not element.items:
            return 0
        axis_h = 22
        row_h = 46
        return axis_h + (len(element.items) * row_h) + 8

    def _measure_bar_list_element(self, draw: ImageDraw.ImageDraw, element: BarListElement) -> int:
        if not element.items:
            return 0
        row_h = 48
        return 12 + (len(element.items) * row_h)

    def _measure_callout_element(self, draw: ImageDraw.ImageDraw, element: CalloutElement) -> int:
        inner_w = self.content_width - 28
        h = 12
        if element.title:
            h += _line_advance(draw, self.font_body_bold, extra=4)
        lines = _wrap_text(draw, element.text, self.font_body, inner_w)
        h += len(lines) * _line_advance(draw, self.font_body, extra=4)
        return h + 10

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
                adv = _line_advance(draw, self.font_small, extra=3)
                total += 6 + (len(cap_lines) * adv)
        return total

    def _measure_image_grid_element(
        self, draw: ImageDraw.ImageDraw, element: ImageGridElement
    ) -> int:
        images = list(element.images or [])[:4]
        if not images:
            base_h = 0
        else:
            sizes = [_probe_image_size(b) for b in images]
            _cells, total_h = _compute_image_grid_layout(
                sizes, self.content_width, element.gap, element.max_cell_height
            )
            base_h = total_h
        total = base_h
        if element.caption:
            cap_lines = _wrap_text(
                draw, element.caption, self.font_small, self.content_width
            )
            if cap_lines:
                adv = _line_advance(draw, self.font_small, extra=3)
                total += (6 + len(cap_lines) * adv) if images else (len(cap_lines) * adv)
        return total

    def _draw_all(
        self, draw: ImageDraw.ImageDraw, img: Image.Image, total_height: int
    ) -> None:
        accent = self.card.accent_color or self.theme.accent_color

        # Card canvas background
        draw.rectangle(
            [(0, 0), (self.card_width - 1, total_height - 1)],
            fill=self.theme.card_bg,
        )

        # Header band background (subtle tint toward accent)
        band_h, _ = self._measure_header(draw)
        header_bg = _blend_hex(self.theme.panel_bg, accent, 0.08)
        draw.rectangle(
            [(0, 4), (self.card_width - 1, band_h - 1)],
            fill=header_bg,
        )

        # Sharp top accent bar (4px)
        draw.rectangle([(0, 0), (self.card_width - 1, 3)], fill=accent)

        # Header divider line
        draw.rectangle(
            [(0, band_h), (self.card_width - 1, band_h)],
            fill=self.theme.border_color,
        )

        x = self.padding
        y = 4 + 16
        avail_w = self._header_title_width(draw)

        title_lines = _wrap_text(draw, self.card.title, self.font_title, avail_w)
        title_adv = _line_advance(draw, self.font_title, extra=4)
        curr_y = y
        for line in title_lines:
            _draw_text_ink(draw, x, curr_y, line, self.font_title, self.theme.text_primary)
            curr_y += title_adv

        if self.card.subtitle:
            curr_y += 4
            sub_lines = _wrap_text(draw, self.card.subtitle, self.font_subtitle, avail_w)
            sub_adv = _line_advance(draw, self.font_subtitle, extra=3)
            for line in sub_lines:
                _draw_text_ink(draw, x, curr_y, line, self.font_subtitle, self.theme.text_secondary)
                curr_y += sub_adv

        if self.card.badge:
            badge_w, _ = _measure_badge(draw, self.card.badge, self.font_small_bold)
            badge_x = self.card_width - self.padding - badge_w
            _draw_sharp_badge(draw, self.card.badge, x=badge_x, y=y + 2, font=self.font_small_bold)

        # Elements region
        y = band_h + 14
        for idx, el in enumerate(self.card.elements):
            y = self._draw_element(draw, img, el, x, y)
            if idx < len(self.card.elements) - 1:
                y += self.theme.element_spacing

        # Footer strip
        footer_h = self._measure_footer(draw)
        if self.card.footer and footer_h > 0:
            fy = total_height - footer_h
            draw.rectangle(
                [(0, fy), (self.card_width - 1, total_height - 1)],
                fill=self.theme.panel_recessed,
            )
            draw.rectangle(
                [(0, fy), (self.card_width - 1, fy)],
                fill=self.theme.border_color,
            )
            footer_lines = _wrap_text(draw, self.card.footer, self.font_small, self.content_width)
            adv = _line_advance(draw, self.font_small, extra=3)
            line_y = fy + 10
            for line in footer_lines:
                _draw_text_ink(draw, x, line_y, line, self.font_small, self.theme.text_muted)
                line_y += adv

        # Outer sharp 1px border (drawn last so max_height clipping keeps crisp border)
        draw.rectangle(
            [(0, 0), (self.card_width - 1, total_height - 1)],
            outline=self.theme.border_color,
            width=1,
        )
        draw.rectangle([(0, 0), (self.card_width - 1, 3)], fill=accent)

    def _draw_element(
        self,
        draw: ImageDraw.ImageDraw,
        img: Image.Image,
        element: CardElement,
        x: int,
        y: int,
    ) -> int:
        if isinstance(element, TextElement):
            font = (
                self.font_mono
                if element.is_code
                else (self.font_body_bold if element.bold else self.font_body)
            )
            color = element.color or (
                self.theme.text_muted if element.muted else self.theme.text_primary
            )
            lines = _wrap_text(draw, element.text, font, self.content_width)
            if element.max_lines is not None and len(lines) > element.max_lines:
                lines = lines[: element.max_lines]
                if lines:
                    lines[-1] = _fit_text(
                        draw, lines[-1], font, self.content_width, overflow="ellipsis"
                    )
            line_h = _line_advance(draw, font, extra=5)
            for line in lines:
                _draw_text_ink(draw, x, y, line, font, color)
                y += line_h
            return y

        if isinstance(element, KeyValuesElement):
            cols = max(1, min(element.columns, 4))
            rows_count = max(1, (len(element.items) + cols - 1) // cols)
            row_h = 30
            box_h = 16 + (rows_count * row_h)

            draw.rectangle(
                [(x, y), (x + self.content_width, y + box_h)],
                fill=self.theme.panel_recessed,
                outline=self.theme.border_color,
                width=1,
            )

            inner_x = x + 12
            inner_w = self.content_width - 24
            col_w = inner_w // cols

            for c in range(1, cols):
                sep_x = inner_x + c * col_w - 6
                draw.line(
                    [(sep_x, y + 8), (sep_x, y + box_h - 8)],
                    fill=self.theme.border_color,
                    width=1,
                )

            for idx, (k, v) in enumerate(element.items):
                col_idx = idx % cols
                row_idx = idx // cols
                item_x = inner_x + (col_idx * col_w)
                item_y = y + 8 + (row_idx * row_h)
                avail_w = col_w - 14

                key_text = f"{sanitize_for_pil(k)}:"
                fitted_key = _fit_text(
                    draw, key_text, self.font_label, max(20, avail_w - 20), overflow=element.overflow
                )
                kw, _ = _draw_text_vcenter(
                    draw, item_x, item_y, row_h, fitted_key, self.font_label, self.theme.text_secondary
                )
                kw += 8
                max_val_w = max(10, avail_w - kw)

                if isinstance(v, Badge):
                    _, bh = _measure_badge(draw, v, self.font_small_bold)
                    by = item_y + (row_h - bh) // 2
                    _draw_sharp_badge(draw, v, item_x + kw, by, self.font_small_bold)
                else:
                    fitted_val = _fit_text(
                        draw, str(v), self.font_body_bold, max_val_w, overflow=element.overflow
                    )
                    _draw_text_vcenter(
                        draw, item_x + kw, item_y, row_h, fitted_val, self.font_body_bold, self.theme.text_primary
                    )

            return y + box_h

        if isinstance(element, TableElement):
            return self._draw_table_element(draw, element, x, y)

        if isinstance(element, DividerElement):
            y += element.margin_top
            if element.line:
                draw.rectangle(
                    [(x, y), (x + self.content_width, y)],
                    fill=self.theme.border_color,
                )
                y += 1
            y += element.margin_bottom
            return y

        if isinstance(element, ProgressBarElement):
            return self._draw_progress_bar_element(draw, element, x, y)

        if isinstance(element, CodeBlockElement):
            return self._draw_code_block_element(draw, element, x, y)

        if isinstance(element, ImageElement):
            return self._draw_image_element(draw, img, element, x, y)

        if isinstance(element, ImageGridElement):
            return self._draw_image_grid_element(draw, img, element, x, y)

        if isinstance(element, SectionHeaderElement):
            return self._draw_section_header(draw, element, x, y)

        if isinstance(element, HeroElement):
            return self._draw_hero_element(draw, element, x, y)

        if isinstance(element, StatGridElement):
            return self._draw_stat_grid_element(draw, element, x, y)

        if isinstance(element, TimelineElement):
            return self._draw_timeline_element(draw, element, x, y)

        if isinstance(element, BarListElement):
            return self._draw_bar_list_element(draw, element, x, y)

        if isinstance(element, RangeBarElement):
            return self._draw_range_bar_element(draw, element, x, y)

        if isinstance(element, CalloutElement):
            return self._draw_callout_element(draw, element, x, y)

        return y

    def _draw_table_element(
        self, draw: ImageDraw.ImageDraw, element: TableElement, x: int, y: int
    ) -> int:
        cols = len(element.headers)
        if cols == 0:
            return y

        header_h = 30
        row_h = 32
        alignments = element.alignments or ["left"] * cols
        col_widths = _calculate_col_widths(self.content_width, cols, element.col_widths)
        mono_cols = set(element.mono_cols or [])
        bold_cols = set(element.bold_cols or [])

        rows_to_render = element.rows
        has_overflow_row = False
        hidden_rows_count = 0
        if element.max_rows is not None and len(element.rows) > element.max_rows:
            rows_to_render = element.rows[: element.max_rows]
            has_overflow_row = True
            hidden_rows_count = len(element.rows) - element.max_rows

        total_rows = len(rows_to_render) + (1 if has_overflow_row else 0)
        total_h = header_h + 1 + (total_rows * row_h) + 2
        table_top = y

        # Header bar
        draw.rectangle(
            [(x, y), (x + self.content_width, y + header_h)],
            fill=self.theme.panel_bg,
        )
        curr_x = x
        for i, header in enumerate(element.headers):
            w_i = col_widths[i]
            align = alignments[i] if i < len(alignments) else "left"
            fitted_header = _fit_text(
                draw, header.upper(), self.font_table_header, w_i - 16, overflow=element.overflow
            )
            hw = _get_text_width(draw, fitted_header, self.font_table_header)
            if align == "right":
                tx = curr_x + w_i - hw - 10
            elif align == "center":
                tx = curr_x + (w_i - hw) // 2
            else:
                tx = curr_x + 10
            _draw_text_vcenter(
                draw, tx, y, header_h, fitted_header, self.font_table_header, self.theme.text_muted
            )
            curr_x += w_i

        y += header_h
        draw.rectangle([(x, y), (x + self.content_width, y)], fill=self.theme.border_bright)
        y += 1

        for row_idx, row in enumerate(rows_to_render):
            row_bg = self.theme.card_bg if row_idx % 2 == 0 else self.theme.panel_recessed
            draw.rectangle([(x, y), (x + self.content_width, y + row_h)], fill=row_bg)

            curr_x = x
            for i in range(cols):
                w_i = col_widths[i]
                cell_val = row[i] if i < len(row) else ""
                align = alignments[i] if i < len(alignments) else "left"

                if isinstance(cell_val, Badge):
                    bw, bh = _measure_badge(draw, cell_val, self.font_small_bold)
                    if align == "right":
                        bx = curr_x + w_i - bw - 10
                    elif align == "center":
                        bx = curr_x + (w_i - bw) // 2
                    else:
                        bx = curr_x + 10
                    by = y + (row_h - bh) // 2
                    _draw_sharp_badge(draw, cell_val, bx, by, self.font_small_bold)
                else:
                    is_primary = element.primary_col is not None and i == element.primary_col
                    is_bold = is_primary or (i in bold_cols)
                    is_mono = i in mono_cols

                    if is_mono and is_bold:
                        cell_font = self.font_mono_bold
                        cell_color = self.theme.text_primary
                    elif is_mono:
                        cell_font = self.font_mono
                        cell_color = self.theme.text_secondary
                    elif is_primary:
                        cell_font = self.font_primary_item
                        cell_color = self.theme.text_primary
                    elif is_bold:
                        cell_font = self.font_body_bold
                        cell_color = self.theme.text_primary
                    else:
                        cell_font = self.font_body
                        cell_color = self.theme.text_secondary

                    fitted_val = _fit_text(
                        draw, str(cell_val), cell_font, w_i - 16, overflow=element.overflow
                    )
                    vw = _get_text_width(draw, fitted_val, cell_font)
                    if align == "right":
                        tx = curr_x + w_i - vw - 10
                    elif align == "center":
                        tx = curr_x + (w_i - vw) // 2
                    else:
                        tx = curr_x + 10
                    _draw_text_vcenter(draw, tx, y, row_h, fitted_val, cell_font, cell_color)
                curr_x += w_i
            y += row_h

        if has_overflow_row:
            overflow_bg = (
                self.theme.card_bg if len(rows_to_render) % 2 == 0 else self.theme.panel_recessed
            )
            draw.rectangle([(x, y), (x + self.content_width, y + row_h)], fill=overflow_bg)
            overflow_text = f"… (+{hidden_rows_count} muuta riviä)"
            _draw_text_vcenter(
                draw, x + 10, y, row_h, overflow_text, self.font_small, self.theme.text_muted
            )
            y += row_h

        # Sharp outer border around the table
        draw.rectangle(
            [(x, table_top), (x + self.content_width, table_top + total_h - 1)],
            outline=self.theme.border_color,
            width=1,
        )
        return table_top + total_h

    def _draw_progress_bar_element(
        self, draw: ImageDraw.ImageDraw, element: ProgressBarElement, x: int, y: int
    ) -> int:
        label_text = sanitize_for_pil(element.label)
        pct = 0.0
        if element.max_value > 0:
            pct = max(0.0, min(1.0, element.value / element.max_value))

        unit = element.unit or ""
        if unit.strip() == "%" and abs(element.max_value - 100.0) < 1e-6:
            pct_str = f"{element.value:g}%"
            val_text = f"{element.sublabel} • {pct_str}" if element.sublabel else pct_str
        else:
            unit_part = f" {unit}" if unit else ""
            base = f"{element.value:g} / {element.max_value:g}{unit_part}"
            if element.show_percentage:
                base = f"{base} ({pct * 100:.0f}%)"
            val_text = f"{element.sublabel} • {base}" if element.sublabel else base
        val_text = sanitize_for_pil(val_text)

        _draw_text_ink(draw, x, y + 1, label_text, self.font_body_bold, self.theme.text_secondary)
        vw = _get_text_width(draw, val_text, self.font_mono_bold)
        _draw_text_ink(
            draw, x + self.content_width - vw, y + 1, val_text, self.font_mono_bold, self.theme.text_primary
        )
        y += 22

        bar_h = 10
        draw.rectangle(
            [(x, y), (x + self.content_width, y + bar_h)],
            fill=self.theme.panel_recessed,
            outline=self.theme.border_color,
            width=1,
        )

        # Subtle quartile tick marks on the track
        for frac in (0.25, 0.5, 0.75):
            tx = x + int(self.content_width * frac)
            draw.line([(tx, y + 1), (tx, y + bar_h - 1)], fill=self.theme.border_color, width=1)

        bar_color = element.color or self.theme.accent_color
        fill_w = int(self.content_width * pct)
        if fill_w > 0:
            draw.rectangle([(x + 1, y + 1), (x + fill_w - 1, y + bar_h - 1)], fill=bar_color)
            if fill_w >= 4:
                tip_color = _blend_hex(bar_color, "#ffffff", 0.35)
                draw.rectangle([(x + fill_w - 2, y + 1), (x + fill_w, y + bar_h - 1)], fill=tip_color)

        return y + bar_h + 6

    def _draw_code_block_element(
        self, draw: ImageDraw.ImageDraw, element: CodeBlockElement, x: int, y: int
    ) -> int:
        clean_code = sanitize_for_pil(element.code)
        lines = clean_code.splitlines() or [""]
        hidden_lines = 0
        if element.max_lines is not None and len(lines) > element.max_lines:
            hidden_lines = len(lines) - element.max_lines
            lines = lines[: element.max_lines]

        line_h = _line_advance(draw, self.font_mono, extra=4)
        total_lines_count = len(lines) + (1 if hidden_lines > 0 else 0)
        block_h = (total_lines_count * line_h) + 18
        max_code_w = self.content_width - 28

        draw.rectangle(
            [(x, y), (x + self.content_width, y + block_h)],
            fill=self.theme.code_bg,
            outline=self.theme.border_color,
            width=1,
        )
        # Left 3px accent stripe
        draw.rectangle(
            [(x, y), (x + 2, y + block_h)],
            fill=self.card.accent_color or self.theme.accent_color,
        )

        if element.language:
            lang = sanitize_for_pil(element.language).upper()
            lw = _get_text_width(draw, lang, self.font_small_bold)
            _draw_text_ink(
                draw, x + self.content_width - lw - 8, y + 5, lang, self.font_small_bold, self.theme.text_muted
            )

        curr_y = y + 9
        for line in lines:
            fitted_line = _fit_text(
                draw, line, self.font_mono, max_code_w, overflow=element.overflow
            )
            _draw_text_ink(draw, x + 14, curr_y, fitted_line, self.font_mono, self.theme.text_primary)
            curr_y += line_h

        if hidden_lines > 0:
            overflow_text = f"… (+{hidden_lines} riviä)"
            _draw_text_ink(draw, x + 14, curr_y, overflow_text, self.font_mono, self.theme.text_muted)

        return y + block_h

    def _draw_section_header(
        self, draw: ImageDraw.ImageDraw, element: SectionHeaderElement, x: int, y: int
    ) -> int:
        accent = element.accent_color or self.card.accent_color or self.theme.accent_color
        title = sanitize_for_pil(element.title).upper()
        # Small sharp square accent pip
        draw.rectangle([(x, y + 4), (x + 4, y + 16)], fill=accent)
        tw, _ = _draw_text_vcenter(
            draw, x + 11, y, 20, title, self.font_section, self.theme.text_secondary
        )
        line_start = x + 11 + tw + 10
        line_end = x + self.content_width
        if element.right_text:
            rtext = sanitize_for_pil(element.right_text)
            rw = _get_text_width(draw, rtext, self.font_small_bold)
            _draw_text_vcenter(
                draw, x + self.content_width - rw, y, 20, rtext, self.font_small_bold, self.theme.text_muted
            )
            line_end -= rw + 10
        if line_end > line_start + 12:
            draw.rectangle([(line_start, y + 10), (line_end, y + 10)], fill=self.theme.border_color)
        return y + 22

    def _draw_hero_element(
        self, draw: ImageDraw.ImageDraw, element: HeroElement, x: int, y: int
    ) -> int:
        h = self._measure_hero_element(draw, element)
        accent = element.accent_color or self.card.accent_color or self.theme.accent_color
        panel_fill = _blend_hex(self.theme.panel_recessed, accent, 0.06)

        draw.rectangle(
            [(x, y), (x + self.content_width, y + h)],
            fill=panel_fill,
            outline=self.theme.border_color,
            width=1,
        )
        # Left 3px accent stripe
        draw.rectangle([(x, y), (x + 3, y + h)], fill=accent)

        cursor_x = x + 18
        if element.icon:
            icon_cx = cursor_x + 26
            icon_cy = y + h // 2
            draw_icon(
                draw,
                element.icon,
                icon_cx,
                icon_cy,
                size=46,
                color=accent,
                secondary_color=self.theme.text_primary,
            )
            cursor_x += 68

        val_text = sanitize_for_pil(element.value)
        vw, _ = _draw_text_vcenter(
            draw, cursor_x, y, h, val_text, self.font_hero, self.theme.text_primary
        )
        cursor_x += vw + 18

        right_reserved = 0
        if element.right_value or element.right_label:
            r_val = sanitize_for_pil(element.right_value or "")
            r_lbl = sanitize_for_pil(element.right_label or "")
            rw = max(
                _get_text_width(draw, r_val, self.font_mono_bold),
                _get_text_width(draw, r_lbl, self.font_small),
            )
            right_reserved = rw + 28
            rx = x + self.content_width - 16
            if r_lbl and r_val:
                lw = _get_text_width(draw, r_lbl, self.font_small)
                vw_r = _get_text_width(draw, r_val, self.font_mono_bold)
                _draw_text_ink(draw, rx - lw, y + 18, r_lbl, self.font_small, self.theme.text_muted)
                _draw_text_ink(
                    draw, rx - vw_r, y + 36, r_val, self.font_mono_bold, self.theme.text_primary
                )
            elif r_val:
                vw_r = _get_text_width(draw, r_val, self.font_mono_bold)
                _draw_text_vcenter(
                    draw, rx - vw_r, y, h, r_val, self.font_mono_bold, self.theme.text_primary
                )

        avail_lbl_w = max(60, (x + self.content_width - right_reserved) - cursor_x - 10)
        lbl = _fit_text(draw, element.label, self.font_body_bold, avail_lbl_w)
        if element.subtext:
            sub = _fit_text(draw, element.subtext, self.font_label, avail_lbl_w)
            lbl_adv = _line_advance(draw, self.font_body_bold, extra=4)
            sub_h = _get_text_height(draw, sub or "Ag", self.font_label)
            stack_h = lbl_adv + sub_h
            top_y = y + (h - stack_h) // 2
            _draw_text_ink(draw, cursor_x, top_y, lbl, self.font_body_bold, self.theme.text_primary)
            _draw_text_ink(
                draw, cursor_x, top_y + lbl_adv, sub, self.font_label, self.theme.text_secondary
            )
        else:
            _draw_text_vcenter(draw, cursor_x, y, h, lbl, self.font_body_bold, self.theme.text_primary)

        return y + h

    def _draw_stat_grid_element(
        self, draw: ImageDraw.ImageDraw, element: StatGridElement, x: int, y: int
    ) -> int:
        if not element.items:
            return y
        cols = max(1, min(element.columns, 4))
        gap = 8
        has_sub = any(t.subtext for t in element.items)
        tile_h = 74 if has_sub else 58
        tile_w = (self.content_width - gap * (cols - 1)) // cols

        for idx, tile in enumerate(element.items):
            r = idx // cols
            c = idx % cols
            # Last tile stretches to fill any remainder pixels on the right edge
            is_last_col = c == cols - 1
            tx = x + c * (tile_w + gap)
            tw = (self.content_width - (tx - x)) if is_last_col else tile_w
            ty = y + r * (tile_h + gap)
            accent = tile.accent_color or self.card.accent_color or self.theme.accent_color

            draw.rectangle(
                [(tx, ty), (tx + tw, ty + tile_h)],
                fill=self.theme.panel_recessed,
                outline=self.theme.border_color,
                width=1,
            )
            # Top 2px accent bar
            draw.rectangle([(tx, ty), (tx + tw, ty + 2)], fill=accent)

            inner_w = max(20, tw - 20)
            lbl = _fit_text(draw, tile.label.upper(), self.font_table_header, inner_w)
            _draw_text_ink(draw, tx + 10, ty + 10, lbl, self.font_table_header, self.theme.text_muted)

            val = _fit_text(draw, tile.value, self.font_stat_value, inner_w)
            _draw_text_ink(draw, tx + 10, ty + 27, val, self.font_stat_value, self.theme.text_primary)

            if tile.subtext:
                sub = _fit_text(draw, tile.subtext, self.font_small, inner_w)
                _draw_text_ink(draw, tx + 10, ty + 53, sub, self.font_small, self.theme.text_secondary)

        rows = (len(element.items) + cols - 1) // cols
        return y + (rows * tile_h) + max(0, rows - 1) * gap

    def _draw_timeline_element(
        self, draw: ImageDraw.ImageDraw, element: TimelineElement, x: int, y: int
    ) -> int:
        if not element.items:
            return y

        total_h = self._measure_timeline_element(draw, element)
        axis_h = 22
        row_h = 46
        label_col_w = min(132, max(96, int(self.content_width * 0.18)))
        track_x = x + label_col_w + 10
        track_w = self.content_width - label_col_w - 20

        # Outer sharp container
        draw.rectangle(
            [(x, y), (x + self.content_width, y + total_h)],
            fill=self.theme.panel_recessed,
            outline=self.theme.border_color,
            width=1,
        )

        # Axis header strip
        draw.rectangle(
            [(x, y), (x + self.content_width, y + axis_h)],
            fill=self.theme.panel_bg,
        )
        draw.rectangle([(x, y + axis_h), (x + self.content_width, y + axis_h)], fill=self.theme.border_color)

        _draw_text_vcenter(
            draw, x + 10, y, axis_h, "KANAVA", self.font_table_header, self.theme.text_muted
        )
        start_lbl = sanitize_for_pil(element.start_label)
        end_lbl = sanitize_for_pil(element.end_label)
        if start_lbl:
            _draw_text_vcenter(
                draw, track_x, y, axis_h, start_lbl, self.font_mono_bold, self.theme.text_secondary
            )
        if end_lbl:
            ew = _get_text_width(draw, end_lbl, self.font_mono_bold)
            _draw_text_vcenter(
                draw, track_x + track_w - ew, y, axis_h, end_lbl, self.font_mono_bold, self.theme.text_secondary
            )

        # Quarter guide lines across all rows
        rows_top = y + axis_h + 1
        rows_bot = y + total_h - 4
        for frac in (0.25, 0.5, 0.75):
            gx = track_x + int(track_w * frac)
            draw.line([(gx, rows_top), (gx, rows_bot)], fill=self.theme.border_color, width=1)

        # Draw row backgrounds first, then subtle NOW background line, then text & bars on top
        curr_y = rows_top + 2
        for idx, _item in enumerate(element.items):
            if idx % 2 == 1:
                draw.rectangle(
                    [(x + 1, curr_y), (x + self.content_width - 1, curr_y + row_h - 1)],
                    fill=_blend_hex(self.theme.panel_recessed, self.theme.panel_bg, 0.45),
                )
            curr_y += row_h

        now_color = "#22c55e"
        nx: int | None = None
        if element.now_fraction is not None:
            nf = max(0.0, min(1.0, element.now_fraction))
            nx = track_x + int(track_w * nf)
            # Subtle guide line behind text so titles remain 100% legible
            draw.line(
                [(nx, rows_top), (nx, rows_bot)],
                fill=_blend_hex(self.theme.panel_recessed, now_color, 0.30),
                width=1,
            )

        # Draw programme rows (text + bars)
        curr_y = rows_top + 2
        for idx, item in enumerate(element.items):
            # Channel label column
            ch_txt = _fit_text(draw, item.label, self.font_body_bold, label_col_w - 14)
            _draw_text_vcenter(
                draw, x + 10, curr_y, row_h, ch_txt, self.font_body_bold, self.theme.text_primary
            )

            # Top line above bar: Programme title (left) + time text (right)
            time_txt = sanitize_for_pil(item.time_text)
            time_w = _get_text_width(draw, time_txt, self.font_mono)
            title_avail = max(40, track_w - time_w - 14)
            title_txt = _fit_text(draw, item.title, self.font_body_bold, title_avail)
            title_color = self.theme.text_primary if item.is_live else self.theme.text_secondary
            _draw_text_ink(
                draw, track_x, curr_y + 6, title_txt, self.font_body_bold, title_color
            )
            _draw_text_ink(
                draw, track_x + track_w - time_w, curr_y + 7, time_txt, self.font_mono, self.theme.text_muted
            )

            # Timeline bar slot
            bar_y = curr_y + 27
            bar_h = 11
            draw.rectangle(
                [(track_x, bar_y), (track_x + track_w, bar_y + bar_h)],
                fill=self.theme.card_bg,
                outline=self.theme.border_color,
                width=1,
            )

            sf = max(0.0, min(1.0, item.start_frac))
            ef = max(sf, min(1.0, item.end_frac))
            bx1 = track_x + int(track_w * sf)
            bx2 = max(bx1 + 4, track_x + int(track_w * ef))
            bx2 = min(bx2, track_x + track_w)

            bar_color = item.color or ("#22c55e" if item.is_live else "#3b82f6")
            if item.is_live:
                draw.rectangle(
                    [(bx1, bar_y), (bx2, bar_y + bar_h)],
                    fill=_blend_hex(self.theme.card_bg, bar_color, 0.32),
                    outline=bar_color,
                    width=1,
                )
                if item.progress is not None:
                    prog = max(0.0, min(1.0, item.progress))
                    fill_end = bx1 + int((bx2 - bx1) * prog)
                    if fill_end > bx1 + 2:
                        draw.rectangle(
                            [(bx1 + 1, bar_y + 1), (fill_end, bar_y + bar_h - 1)],
                            fill=bar_color,
                        )
            else:
                draw.rectangle(
                    [(bx1, bar_y), (bx2, bar_y + bar_h)],
                    fill=_blend_hex(self.theme.card_bg, bar_color, 0.55),
                    outline=bar_color,
                    width=1,
                )

            # Crisp bright NOW tick directly across each bar track (never cuts through text)
            if nx is not None:
                draw.rectangle(
                    [(nx - 1, bar_y - 2), (nx + 1, bar_y + bar_h + 2)],
                    fill=now_color,
                )

            curr_y += row_h

        # Crisp NOW badge in axis header
        if nx is not None:
            now_lbl = sanitize_for_pil(element.now_label).upper()
            if now_lbl:
                nw = _get_text_width(draw, now_lbl, self.font_small_bold) + 8
                nh = 16
                tag_x = max(track_x, min(nx - nw // 2, track_x + track_w - nw))
                tag_y = y + (axis_h - nh) // 2
                draw.rectangle([(tag_x, tag_y), (tag_x + nw, tag_y + nh)], fill=now_color)
                _draw_text_vcenter(draw, tag_x + 4, tag_y, nh, now_lbl, self.font_small_bold, "#08120c")

        return y + total_h

    def _draw_bar_list_element(
        self, draw: ImageDraw.ImageDraw, element: BarListElement, x: int, y: int
    ) -> int:
        if not element.items:
            return y

        total_h = self._measure_bar_list_element(draw, element)
        max_val = element.max_value
        if max_val is None or max_val <= 0:
            max_val = max((max(0.0, it.value) for it in element.items), default=1.0) or 1.0

        draw.rectangle(
            [(x, y), (x + self.content_width, y + total_h)],
            fill=self.theme.panel_recessed,
            outline=self.theme.border_color,
            width=1,
        )

        row_h = 48
        inner_x = x + 12
        inner_w = self.content_width - 24
        curr_y = y + 6

        for idx, item in enumerate(element.items):
            bar_color = item.color or self.card.accent_color or self.theme.accent_color
            val_txt = sanitize_for_pil(item.value_text)
            val_w = _get_text_width(draw, val_txt, self.font_mono_bold)

            left_x = inner_x
            if element.show_rank:
                rank_txt = f"#{idx + 1}"
                rw, _ = _draw_text_ink(
                    draw, left_x, curr_y + 4, rank_txt, self.font_mono_bold, self.theme.text_muted
                )
                left_x += rw + 10

            badge_w = 0
            if item.badge:
                bw, _ = _measure_badge(draw, item.badge, self.font_small_bold)
                badge_w = bw + 8

            sub_txt = sanitize_for_pil(item.subtext or "")
            sub_w = _get_text_width(draw, f" • {sub_txt}", self.font_small) if sub_txt else 0
            lbl_max = max(40, (inner_x + inner_w) - left_x - val_w - badge_w - sub_w - 12)
            lbl_txt = _fit_text(draw, item.label, self.font_body_bold, lbl_max)
            lw, _ = _draw_text_ink(
                draw, left_x, curr_y + 4, lbl_txt, self.font_body_bold, self.theme.text_primary
            )

            badge_x = left_x + lw + 8
            if item.badge:
                bw, _ = _draw_sharp_badge(
                    draw, item.badge, badge_x, curr_y + 1, self.font_small_bold
                )
                badge_x += bw + 8

            if sub_txt:
                avail_sub = (inner_x + inner_w) - badge_x - val_w - 10
                if avail_sub > 30:
                    fitted_sub = _fit_text(draw, sub_txt, self.font_small, avail_sub)
                    _draw_text_ink(
                        draw, badge_x, curr_y + 6, fitted_sub, self.font_small, self.theme.text_muted
                    )

            _draw_text_ink(
                draw,
                inner_x + inner_w - val_w,
                curr_y + 4,
                val_txt,
                self.font_mono_bold,
                self.theme.text_primary,
            )

            # Horizontal comparison bar
            bar_y = curr_y + 28
            bar_h = 9
            draw.rectangle(
                [(inner_x, bar_y), (inner_x + inner_w, bar_y + bar_h)],
                fill=self.theme.card_bg,
                outline=self.theme.border_color,
                width=1,
            )
            frac = max(0.0, min(1.0, item.value / max_val))
            fw = int(inner_w * frac)
            if fw > 0:
                draw.rectangle([(inner_x + 1, bar_y + 1), (inner_x + max(3, fw), bar_y + bar_h - 1)], fill=bar_color)

            curr_y += row_h

        return y + total_h

    def _draw_range_bar_element(
        self, draw: ImageDraw.ImageDraw, element: RangeBarElement, x: int, y: int
    ) -> int:
        h = 56
        accent = element.color or self.card.accent_color or self.theme.accent_color
        draw.rectangle(
            [(x, y), (x + self.content_width, y + h)],
            fill=self.theme.panel_recessed,
            outline=self.theme.border_color,
            width=1,
        )

        inner_x = x + 12
        inner_w = self.content_width - 24
        _draw_text_ink(
            draw, inner_x, y + 8, sanitize_for_pil(element.label), self.font_body_bold, self.theme.text_secondary
        )
        if element.current_label:
            cur = sanitize_for_pil(element.current_label)
            cw = _get_text_width(draw, cur, self.font_mono_bold)
            _draw_text_ink(
                draw, inner_x + inner_w - cw, y + 8, cur, self.font_mono_bold, self.theme.text_primary
            )

        bar_y = y + 28
        bar_h = 8
        track_fill = element.track_color or self.theme.card_bg
        draw.rectangle(
            [(inner_x, bar_y), (inner_x + inner_w, bar_y + bar_h)],
            fill=track_fill,
            outline=self.theme.border_color,
            width=1,
        )
        frac = max(0.0, min(1.0, element.value_frac))
        fill_x = inner_x + int(inner_w * frac)
        if fill_x > inner_x + 1:
            draw.rectangle([(inner_x + 1, bar_y + 1), (fill_x, bar_y + bar_h - 1)], fill=accent)

        # Sharp vertical marker
        marker_x = max(inner_x + 2, min(inner_x + inner_w - 2, fill_x))
        draw.rectangle(
            [(marker_x - 2, bar_y - 3), (marker_x + 2, bar_y + bar_h + 3)],
            fill=self.theme.text_primary,
        )

        # Start and end labels below track
        s_lbl = sanitize_for_pil(element.start_label)
        e_lbl = sanitize_for_pil(element.end_label)
        _draw_text_ink(draw, inner_x, y + 40, s_lbl, self.font_small, self.theme.text_muted)
        ew = _get_text_width(draw, e_lbl, self.font_small)
        _draw_text_ink(
            draw, inner_x + inner_w - ew, y + 40, e_lbl, self.font_small, self.theme.text_muted
        )
        return y + h

    def _draw_callout_element(
        self, draw: ImageDraw.ImageDraw, element: CalloutElement, x: int, y: int
    ) -> int:
        h = self._measure_callout_element(draw, element)
        accent = element.color or _tone_color(element.tone, self.theme)
        bg = _blend_hex(self.theme.panel_recessed, accent, 0.09)

        draw.rectangle(
            [(x, y), (x + self.content_width, y + h)],
            fill=bg,
            outline=self.theme.border_color,
            width=1,
        )
        draw.rectangle([(x, y), (x + 3, y + h)], fill=accent)

        inner_x = x + 14
        inner_w = self.content_width - 28
        curr_y = y + 10
        if element.title:
            title_txt = _fit_text(draw, element.title, self.font_body_bold, inner_w)
            _draw_text_ink(draw, inner_x, curr_y, title_txt, self.font_body_bold, accent)
            curr_y += _line_advance(draw, self.font_body_bold, extra=4)

        lines = _wrap_text(draw, element.text, self.font_body, inner_w)
        adv = _line_advance(draw, self.font_body, extra=4)
        for line in lines:
            _draw_text_ink(draw, inner_x, curr_y, line, self.font_body, self.theme.text_primary)
            curr_y += adv

        return y + h

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
            _draw_text_vcenter(
                draw, x + 12, y, box_h, "Kuvaa ei voitu näyttää", self.font_small, self.theme.text_muted
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
                _draw_text_vcenter(
                    draw, x + 12, y, 48, "Kuvaa ei voitu näyttää", self.font_small, self.theme.text_muted
                )
                y += 48
            else:
                draw.rectangle(
                    [(x_offset, y), (x_offset + disp_w, y + disp_h)],
                    outline=self.theme.border_bright,
                    width=1,
                )
                y += disp_h

        if element.caption:
            cap_lines = _wrap_text(draw, element.caption, self.font_small, self.content_width)
            adv = _line_advance(draw, self.font_small, extra=3)
            y += 6
            for line in cap_lines:
                lw = _get_text_width(draw, line, self.font_small)
                tx = x + (self.content_width - lw) // 2 if lw < self.content_width else x
                _draw_text_ink(draw, tx, y, line, self.font_small, self.theme.text_muted)
                y += adv
        return y

    def _paste_cell_image(
        self, img: Image.Image, image_bytes: bytes, abs_x: int, abs_y: int, disp_w: int, disp_h: int
    ) -> bool:
        try:
            with Image.open(BytesIO(image_bytes)) as pil_img:
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
                img.paste(pil_img, (abs_x, abs_y))
                return True
        except Exception:
            return False

    def _draw_grid_label(
        self, draw: ImageDraw.ImageDraw, abs_x: int, abs_y: int, label: str
    ) -> None:
        text = sanitize_for_pil(str(label)).strip()
        if not text:
            return
        pad_h, pad_v = 6, 3
        text_w = _get_text_width(draw, text, self.font_small_bold)
        text_h = _get_text_height(draw, "ÄG", self.font_small_bold)
        box_w, box_h = text_w + pad_h * 2, text_h + pad_v * 2
        bx, by = abs_x + 6, abs_y + 6
        draw.rectangle(
            [(bx, by), (bx + box_w, by + box_h)],
            fill=self.theme.panel_recessed,
            outline=self.theme.border_bright,
            width=1,
        )
        _draw_text_vcenter(draw, bx + pad_h, by, box_h, text, self.font_small_bold, self.theme.text_primary)

    def _draw_image_grid_element(
        self,
        draw: ImageDraw.ImageDraw,
        img: Image.Image,
        element: ImageGridElement,
        x: int,
        y: int,
    ) -> int:
        images = list(element.images or [])[:4]
        if not images:
            if element.caption:
                cap_lines = _wrap_text(draw, element.caption, self.font_small, self.content_width)
                adv = _line_advance(draw, self.font_small, extra=3)
                for line in cap_lines:
                    lw = _get_text_width(draw, line, self.font_small)
                    tx = x + (self.content_width - lw) // 2 if lw < self.content_width else x
                    _draw_text_ink(draw, tx, y, line, self.font_small, self.theme.text_muted)
                    y += adv
            return y

        sizes = [_probe_image_size(b) for b in images]
        cells, total_h = _compute_image_grid_layout(
            sizes, self.content_width, element.gap, element.max_cell_height
        )
        labels = list(element.labels or [])

        for idx, (disp_w, disp_h, x_rel, y_rel) in enumerate(cells):
            abs_x, abs_y = x + x_rel, y + y_rel
            ok = self._paste_cell_image(img, images[idx], abs_x, abs_y, disp_w, disp_h)
            if not ok:
                draw.rectangle(
                    [(abs_x, abs_y), (abs_x + disp_w, abs_y + disp_h)],
                    fill=self.theme.panel_bg,
                    outline=self.theme.border_color,
                    width=1,
                )
                _draw_text_vcenter(
                    draw, abs_x + 10, abs_y, disp_h, "Kuvaa ei voitu näyttää", self.font_small, self.theme.text_muted
                )
            else:
                draw.rectangle(
                    [(abs_x, abs_y), (abs_x + disp_w, abs_y + disp_h)],
                    outline=self.theme.border_bright,
                    width=1,
                )
            if idx < len(labels) and labels[idx]:
                self._draw_grid_label(draw, abs_x, abs_y, labels[idx])

        y += total_h

        if element.caption:
            cap_lines = _wrap_text(draw, element.caption, self.font_small, self.content_width)
            adv = _line_advance(draw, self.font_small, extra=3)
            y += 6
            for line in cap_lines:
                lw = _get_text_width(draw, line, self.font_small)
                tx = x + (self.content_width - lw) // 2 if lw < self.content_width else x
                _draw_text_ink(draw, tx, y, line, self.font_small, self.theme.text_muted)
                y += adv
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
    rows: list[list[str | Badge]],
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
