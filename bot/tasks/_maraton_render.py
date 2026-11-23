from __future__ import annotations

import functools
from io import BytesIO
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from bot.config import MaratonConfig
from bot.tasks.maraton_logic import (
    RELAY_LEGS,
    WEEKDAY_NAMES_FI,
    MaratonBriefing,
    format_finnish_date,
    select_phase,
)

# Strictly sharp 90-degree dark tactical/athletic poster palette
BG_OBSIDIAN = (11, 12, 16)
PANEL_DARK = (19, 21, 28)
PANEL_ELEVATED = (26, 29, 39)
PANEL_RECESSED = (14, 16, 22)
BORDER_SUBTLE = (44, 48, 62)
BORDER_BRIGHT = (68, 74, 94)
TEXT_WHITE = (245, 246, 250)
TEXT_SOFT = (205, 210, 222)
TEXT_MUTED = (138, 145, 164)
TEXT_DIM = (96, 103, 120)

DISPLAY_FONT_CANDIDATES = (
    "/usr/share/fonts/gsfonts/NimbusSansNarrow-Bold.otf",
    "/usr/share/fonts/gsfonts/NimbusSans-Bold.otf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/gnu-free/FreeSansBold.otf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
)

SANS_BOLD_CANDIDATES = (
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/gsfonts/NimbusSans-Bold.otf",
    "/usr/share/fonts/gnu-free/FreeSansBold.otf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
)

SANS_REGULAR_CANDIDATES = (
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/gsfonts/NimbusSans-Regular.otf",
    "/usr/share/fonts/gnu-free/FreeSans.otf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
)

MONO_BOLD_CANDIDATES = (
    "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf",
    "/usr/share/fonts/TTF/Hack-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/gnu-free/FreeMonoBold.otf",
    "/usr/share/fonts/gsfonts/NimbusMonoPS-Bold.otf",
    "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
)

MONO_REGULAR_CANDIDATES = (
    "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
    "/usr/share/fonts/TTF/Hack-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/gnu-free/FreeMono.otf",
    "/usr/share/fonts/gsfonts/NimbusMonoPS-Regular.otf",
)

RELAY_DISTANCES_KM: tuple[float, ...] = (11.2, 7.0, 14.1, 9.9)
TOTAL_MARATHON_KM = 42.195


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = hex_color.strip().lstrip("#")
    if len(cleaned) != 6:
        return (239, 68, 68)
    return (
        int(cleaned[0:2], 16),
        int(cleaned[2:4], 16),
        int(cleaned[4:6], 16),
    )


def _blend(
    base: tuple[int, int, int],
    overlay: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    a = max(0.0, min(1.0, alpha))
    return (
        int(base[0] * (1.0 - a) + overlay[0] * a),
        int(base[1] * (1.0 - a) + overlay[1] * a),
        int(base[2] * (1.0 - a) + overlay[2] * a),
    )


@functools.lru_cache(maxsize=16)
def _find_font(candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


@functools.lru_cache(maxsize=128)
def _font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates_map = {
        "display": DISPLAY_FONT_CANDIDATES,
        "bold": SANS_BOLD_CANDIDATES,
        "regular": SANS_REGULAR_CANDIDATES,
        "mono_bold": MONO_BOLD_CANDIDATES,
        "mono": MONO_REGULAR_CANDIDATES,
    }
    path = _find_font(candidates_map.get(kind, SANS_REGULAR_CANDIDATES))
    if path:
        try:
            f = ImageFont.truetype(path, size)
            if kind in ("display", "bold") and "AdwaitaSans-Regular" in path:
                try:
                    f.set_variation_by_name("Black" if kind == "display" else "Bold")
                except Exception:
                    pass
            return f
        except Exception:
            pass
    return ImageFont.load_default()


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_text_top_left(
    draw: ImageDraw.ImageDraw,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> tuple[int, int]:
    """Draws text cleanly anchored by its actual top-left ink bbox to prevent overlap."""
    bbox = draw.textbbox((0, 0), text, font=font)
    x = pos[0] - bbox[0]
    y = pos[1] - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            w, _ = _text_size(draw, candidate, font)
            if w <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                    current = word
                else:
                    chunk = ""
                    for ch in word:
                        cw, _ = _text_size(draw, chunk + ch, font)
                        if cw <= max_width or not chunk:
                            chunk += ch
                        else:
                            lines.append(chunk)
                            chunk = ch
                    current = chunk
        if current:
            lines.append(current)
    return lines


def _measure_wrapped(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    line_gap: int,
) -> int:
    if not lines:
        return 0
    _, sample_h = _text_size(draw, "ÄgM0", font)
    return len(lines) * sample_h + max(0, len(lines) - 1) * line_gap


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    lines: list[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_gap: int,
) -> int:
    _, sample_h = _text_size(draw, "ÄgM0", font)
    cur_y = y
    for line in lines:
        if line:
            _draw_text_top_left(draw, (x, cur_y), line, font, fill)
        cur_y += sample_h + line_gap
    return cur_y - line_gap if lines else y


def _draw_corner_brackets(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    length: int = 12,
    thickness: int = 2,
) -> None:
    x0, y0, x1, y1 = box
    # Top-left
    draw.rectangle([(x0, y0), (x0 + length, y0 + thickness - 1)], fill=color)
    draw.rectangle([(x0, y0), (x0 + thickness - 1, y0 + length)], fill=color)
    # Top-right
    draw.rectangle([(x1 - length, y0), (x1, y0 + thickness - 1)], fill=color)
    draw.rectangle([(x1 - thickness + 1, y0), (x1, y0 + length)], fill=color)
    # Bottom-left
    draw.rectangle([(x0, y1 - thickness + 1), (x0 + length, y1)], fill=color)
    draw.rectangle([(x0, y1 - length), (x0 + thickness - 1, y1)], fill=color)
    # Bottom-right
    draw.rectangle([(x1 - length, y1 - thickness + 1), (x1, y1)], fill=color)
    draw.rectangle([(x1 - thickness + 1, y1 - length), (x1, y1)], fill=color)


def _draw_hazard_stripe_bar(
    img: Image.Image,
    y: int,
    height: int,
    accent: tuple[int, int, int],
    intense: bool,
) -> None:
    w = img.width
    draw = ImageDraw.Draw(img)
    if not intense:
        draw.rectangle([(0, y), (w, y + height - 1)], fill=accent)
        return

    # Sharp diagonal warning chevrons for high-alert phases (<= 60 days)
    dark_stripe = _blend(BG_OBSIDIAN, accent, 0.22)
    draw.rectangle([(0, y), (w, y + height - 1)], fill=dark_stripe)
    step = 26
    for sx in range(-height, w + step, step):
        poly = [
            (sx, y + height),
            (sx + 12, y + height),
            (sx + 12 + height, y),
            (sx + height, y),
        ]
        draw.polygon(poly, fill=accent)


def _create_atmospheric_canvas(
    width: int,
    height: int,
    accent: tuple[int, int, int],
    dread_percent: int,
) -> Image.Image:
    """Builds the obsidian base canvas with blurred atmospheric glow and subtle track lines."""
    base = Image.new("RGBA", (width, height), (*BG_OBSIDIAN, 255))

    # Subtle ambient glow behind hero countdown & right telemetry
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    intensity = 34 + int((dread_percent / 100.0) * 38)
    gdraw.ellipse(
        [(-120, -140), (int(width * 0.62), 340)],
        fill=(*accent, intensity),
    )
    gdraw.ellipse(
        [(int(width * 0.55), -80), (width + 160, 280)],
        fill=(*accent, max(16, intensity // 2)),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=85))
    base = Image.alpha_composite(base, glow)

    # Subtle architectural tactical grid + 4 running track lanes in the hero zone
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Fine vertical guide lines
    for gx in range(36, width, 46):
        odraw.line([(gx, 0), (gx, 320)], fill=(255, 255, 255, 7), width=1)
    # 4 faint horizontal running-track lane lines in hero background
    for lane_idx in range(4):
        ly = 88 + lane_idx * 42
        odraw.line([(24, ly), (width - 24, ly)], fill=(*accent, 22), width=1)

    base = Image.alpha_composite(base, overlay)
    return base.convert("RGB")


def _draw_ecg_waveform(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    accent: tuple[int, int, int],
    dread_percent: int,
) -> None:
    """Draws a crisp procedural heart-rate ECG trace scaling with threat intensity."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    mid_y = y0 + int(h * 0.58)

    # Faint oscilloscope grid
    grid_color = _blend(PANEL_RECESSED, BORDER_SUBTLE, 0.55)
    for gx in range(x0 + 24, x1, 24):
        draw.line([(gx, y0 + 2), (gx, y1 - 2)], fill=grid_color, width=1)
    draw.line([(x0 + 2, mid_y), (x1 - 2, mid_y)], fill=grid_color, width=1)

    # Number of beats across the box scales with intensity (2 beats calm -> 5 beats panic)
    beats = 2 if dread_percent < 35 else (3 if dread_percent < 70 else (4 if dread_percent < 95 else 5))
    amp = int(h * (0.32 + 0.14 * (dread_percent / 100.0)))

    points: list[tuple[int, int]] = []
    for px in range(x0 + 4, x1 - 4, 2):
        t = ((px - x0) / max(1, w)) * beats
        phase = t - math.floor(t)

        # Classic P-QRS-T cardiac waveform
        if 0.12 <= phase < 0.20:
            # P wave
            val = 0.18 * math.sin(((phase - 0.12) / 0.08) * math.pi)
        elif 0.26 <= phase < 0.30:
            # Q dip
            val = -0.22 * math.sin(((phase - 0.26) / 0.04) * math.pi)
        elif 0.30 <= phase < 0.36:
            # R spike (sharp peak!)
            val = 1.0 * math.sin(((phase - 0.30) / 0.06) * math.pi)
        elif 0.36 <= phase < 0.41:
            # S dip
            val = -0.38 * math.sin(((phase - 0.36) / 0.05) * math.pi)
        elif 0.52 <= phase < 0.66:
            # T wave
            val = 0.28 * math.sin(((phase - 0.52) / 0.14) * math.pi)
        else:
            val = 0.0

        py = int(mid_y - val * amp)
        py = max(y0 + 4, min(y1 - 4, py))
        points.append((px, py))

    if len(points) >= 2:
        # Soft glow trail under main trace
        dim_accent = _blend(PANEL_RECESSED, accent, 0.35)
        draw.line([(px, py + 2) for px, py in points], fill=dim_accent, width=3)
        draw.line(points, fill=accent, width=2)
        # Bright live pulse dot at right tip
        lx, ly = points[-1]
        draw.rectangle([(lx - 2, ly - 2), (lx + 2, ly + 2)], fill=TEXT_WHITE)


def _draw_segmented_meter(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    percent: float,
    accent: tuple[int, int, int],
    segments: int = 20,
    gap: int = 3,
) -> None:
    seg_w = max(2, (width - gap * (segments - 1)) // segments)
    filled_count = int(round((max(0.0, min(100.0, percent)) / 100.0) * segments))
    for i in range(segments):
        sx = x + i * (seg_w + gap)
        if i < filled_count:
            # Gradually brighten towards active tip
            ratio = (i + 1) / max(1, filled_count)
            fill_col = _blend(accent, TEXT_WHITE, 0.15 if ratio > 0.85 else 0.0)
        else:
            fill_col = (32, 36, 48)
        draw.rectangle([(sx, y), (sx + seg_w, y + height)], fill=fill_col)


def _draw_relay_track_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    accent: tuple[int, int, int],
    highlight_leg_number: int | None,
) -> list[tuple[int, int]]:
    """Draws the 4-leg proportional 42.195 km maratonviesti bar and returns (x0, x1) per leg."""
    gap = 6
    avail_w = width - gap * (len(RELAY_LEGS) - 1)
    bounds: list[tuple[int, int]] = []
    cur_x = x
    total_dist = sum(RELAY_DISTANCES_KM)

    f_num = _font("mono_bold", 12)
    f_km = _font("bold", 15)

    for idx, (leg, dist) in enumerate(zip(RELAY_LEGS, RELAY_DISTANCES_KM)):
        if idx == len(RELAY_LEGS) - 1:
            seg_w = (x + width) - cur_x
        else:
            seg_w = int(round(avail_w * (dist / total_dist)))

        is_hot = highlight_leg_number == leg.number
        if is_hot:
            bg = _blend(PANEL_DARK, accent, 0.28)
            border = accent
            top_strip = accent
            km_col = TEXT_WHITE
            num_col = _blend(accent, TEXT_WHITE, 0.5)
        else:
            bg = PANEL_ELEVATED
            border = BORDER_SUBTLE
            top_strip = BORDER_BRIGHT
            km_col = TEXT_SOFT
            num_col = TEXT_MUTED

        draw.rectangle([(cur_x, y), (cur_x + seg_w, y + height)], fill=bg, outline=border, width=1)
        draw.rectangle([(cur_x, y), (cur_x + seg_w, y + 3)], fill=top_strip)

        label_top = f"OSUUS {leg.number}"
        label_km = leg.distance_km
        tw1, _ = _text_size(draw, label_top, f_num)
        tw2, _ = _text_size(draw, label_km, f_km)
        _draw_text_top_left(draw, (cur_x + (seg_w - tw1) // 2, y + 10), label_top, f_num, num_col)
        _draw_text_top_left(draw, (cur_x + (seg_w - tw2) // 2, y + 27), label_km, f_km, km_col)

        bounds.append((cur_x, cur_x + seg_w))
        cur_x += seg_w + gap

    return bounds


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_maraton_briefing_image(briefing: MaratonBriefing) -> bytes:
    """Renders a bespoke high-impact dark-mode countdown poster for a MaratonBriefing."""
    width = 920
    pad = 32
    content_w = width - pad * 2
    accent = _hex_to_rgb(briefing.accent_color)
    high_alert = (not briefing.is_post_race) and briefing.days_remaining <= 60

    # Fonts
    f_eyebrow = _font("mono_bold", 13)
    f_badge = _font("mono_bold", 13)
    f_hero_num = _font("display", 118)
    f_hero_word = _font("display", 46)
    f_hero_unit = _font("mono_bold", 16)
    f_hero_meta = _font("regular", 15)
    f_stat_label = _font("mono", 12)
    f_stat_val = _font("mono_bold", 15)
    f_milestone = _font("bold", 16)
    f_headline = _font("bold", 23)
    f_body = _font("regular", 17)
    f_section = _font("mono_bold", 13)
    f_tip = _font("regular", 15)
    f_footer = _font("mono", 12)

    # Measure pass on dummy draw
    scratch = Image.new("RGB", (width, 100), BG_OBSIDIAN)
    mdraw = ImageDraw.Draw(scratch)

    milestone_lines = (
        _wrap_text(mdraw, briefing.milestone_banner, f_milestone, content_w - 36)
        if briefing.milestone_banner
        else []
    )
    headline_lines = _wrap_text(mdraw, briefing.headline, f_headline, content_w - 44)
    dispatch_lines = _wrap_text(mdraw, briefing.dispatch, f_body, content_w - 44)
    workout_lines = _wrap_text(mdraw, briefing.workout_detail, f_body, content_w - 40)
    tip_lines = _wrap_text(mdraw, f"VALMENTAJAN VINKKI: {briefing.coach_tip}", f_tip, content_w - 40)

    leg = briefing.relay_spotlight
    leg_callout_text = f"{leg.number}. OSUUS ({leg.distance_km} • {leg.role.upper()}): {leg.danger_note}. {leg.tactical_tip}"
    leg_lines = _wrap_text(mdraw, leg_callout_text, f_tip, content_w - 36)

    # Calculate exact dynamic height
    y = 0
    top_bar_h = 8 if high_alert else 6
    y += top_bar_h + 24
    # Eyebrow row
    eyebrow_h = 30
    y += eyebrow_h + 18
    # Hero block
    hero_h = 196
    y += hero_h + 20
    # Milestone banner (optional)
    milestone_h = 0
    if milestone_lines:
        milestone_h = _measure_wrapped(mdraw, milestone_lines, f_milestone, 5) + 26
        y += milestone_h + 18
    # Dispatch card
    dispatch_h = (
        22
        + _measure_wrapped(mdraw, headline_lines, f_headline, 6)
        + 12
        + _measure_wrapped(mdraw, dispatch_lines, f_body, 6)
        + 22
    )
    y += dispatch_h + 18
    # Training directive card
    training_body_h = _measure_wrapped(mdraw, workout_lines, f_body, 6)
    tip_h = _measure_wrapped(mdraw, tip_lines, f_tip, 5)
    training_h = 44 + 14 + training_body_h + 16 + (tip_h + 22) + 16
    y += training_h + 18
    # Relay 4-leg bar + spotlight (when not post-race)
    relay_section_h = 0
    if not briefing.is_post_race:
        leg_box_h = _measure_wrapped(mdraw, leg_lines, f_tip, 5) + 22
        relay_section_h = 26 + 52 + 10 + leg_box_h
        y += relay_section_h + 18
    # 365-day countdown timeline bar (when not post-race)
    timeline_h = 0
    if not briefing.is_post_race:
        timeline_h = 54
        y += timeline_h + 16
    # Footer
    footer_h = 34
    y += footer_h + 14

    total_height = y

    # Render final canvas
    img = _create_atmospheric_canvas(width, total_height, accent, briefing.dread_percent)
    _draw_hazard_stripe_bar(img, 0, top_bar_h, accent, intense=high_alert)
    draw = ImageDraw.Draw(img)

    # Crisp outer frame
    draw.rectangle([(0, 0), (width - 1, total_height - 1)], outline=BORDER_SUBTLE, width=1)

    cur_y = top_bar_h + 24

    # 1. Eyebrow header & threat badge
    date_str = format_finnish_date(briefing.event_date)
    eyebrow = f"TAMPERE MARATON 2027  //  MARATONVIESTI  //  {date_str}"
    _draw_text_top_left(draw, (pad, cur_y + 7), eyebrow, f_eyebrow, TEXT_MUTED)

    badge_txt = briefing.threat_label.upper()
    bw, bh = _text_size(draw, badge_txt, f_badge)
    badge_w = bw + 22
    badge_h = bh + 12
    bx0 = width - pad - badge_w
    by0 = cur_y
    badge_bg = _blend(BG_OBSIDIAN, accent, 0.24)
    draw.rectangle([(bx0, by0), (bx0 + badge_w, by0 + badge_h)], fill=badge_bg, outline=accent, width=1)
    _draw_text_top_left(draw, (bx0 + 11, by0 + 6), badge_txt, f_badge, _blend(accent, TEXT_WHITE, 0.45))

    cur_y += eyebrow_h + 18

    # 2. Hero Block (Left: Giant countdown numeral; Right: ECG + Telemetry Panel)
    hero_box = (pad, cur_y, width - pad, cur_y + hero_h)
    draw.rectangle(hero_box, fill=PANEL_DARK, outline=BORDER_SUBTLE, width=1)
    _draw_corner_brackets(draw, hero_box, accent, length=14, thickness=2)

    left_w = int(content_w * 0.53)
    split_x = pad + left_w
    draw.line([(split_x, cur_y + 16), (split_x, cur_y + hero_h - 16)], fill=BORDER_SUBTLE, width=1)

    # Left hero content
    hx = pad + 26
    if briefing.is_race_day:
        # Dramatic KISAPÄIVÄ hero
        _draw_text_top_left(draw, (hx + 2, cur_y + 28), "00", f_hero_num, _blend(PANEL_DARK, accent, 0.35))
        nw, nh = _draw_text_top_left(draw, (hx, cur_y + 24), "00", f_hero_num, TEXT_WHITE)
        _draw_text_top_left(draw, (hx + nw + 18, cur_y + 44), "KISAPÄIVÄ", f_hero_word, accent)
        _draw_text_top_left(draw, (hx + nw + 18, cur_y + 100), "RATINAN STADION", f_hero_unit, TEXT_SOFT)
        _draw_text_top_left(draw, (hx, cur_y + 158), "42,195 KM  •  4 OSUUTTA  •  NYT MENNÄÄN", f_hero_meta, TEXT_MUTED)
    elif briefing.is_post_race:
        days_since = abs(briefing.days_remaining)
        _draw_text_top_left(draw, (hx, cur_y + 26), "MAALI", f_hero_word, accent)
        _draw_text_top_left(draw, (hx, cur_y + 88), f"JUOSTU {days_since} PV SITTEN", f_hero_unit, TEXT_WHITE)
        _draw_text_top_left(draw, (hx, cur_y + 124), briefing.phase_name, f_hero_meta, TEXT_MUTED)
    else:
        num_str = str(briefing.days_remaining)
        # Subtle glow shadow behind giant digits
        _draw_text_top_left(draw, (hx + 3, cur_y + 21), num_str, f_hero_num, _blend(PANEL_DARK, accent, 0.42))
        nw, _ = _draw_text_top_left(draw, (hx, cur_y + 18), num_str, f_hero_num, TEXT_WHITE)

        label_x = hx + nw + 18
        unit_top = "PÄIVÄ" if briefing.days_remaining == 1 else "PÄIVÄÄ"
        _draw_text_top_left(draw, (label_x, cur_y + 34), unit_top, f_hero_unit, accent)
        _draw_text_top_left(draw, (label_x, cur_y + 58), "LÄHTÖÖN", f_hero_unit, TEXT_SOFT)
        _draw_text_top_left(draw, (label_x, cur_y + 92), f"{briefing.weeks_remaining} VIIKKOA", f_stat_val, TEXT_MUTED)

        # Phase bar at bottom of left hero
        phase_clean = briefing.phase_name.upper()
        _draw_text_top_left(draw, (hx, cur_y + 154), phase_clean, f_stat_label, TEXT_MUTED)

    # Right hero telemetry panel (ECG + simulated BPM + Dread meter)
    rx = split_x + 22
    rw = (width - pad) - rx - 22

    # Simulated target HR based on phase intensity
    if briefing.is_post_race:
        bpm_label = "54 BPM • PALAUTUMINEN"
    elif briefing.is_race_day:
        bpm_label = "MAX BPM • KISASYKE"
    else:
        sim_bpm = 128 + int((briefing.dread_percent / 100.0) * 64)
        bpm_label = f"{sim_bpm} BPM • PAINE {briefing.dread_percent}%"

    _draw_text_top_left(draw, (rx, cur_y + 18), "VALMIUSTELEMETRIA", f_stat_label, TEXT_MUTED)
    bw_r, _ = _text_size(draw, bpm_label, f_stat_label)
    _draw_text_top_left(draw, (rx + rw - bw_r, cur_y + 18), bpm_label, f_stat_label, accent)

    ecg_box = (rx, cur_y + 42, rx + rw, cur_y + 118)
    draw.rectangle(ecg_box, fill=PANEL_RECESSED, outline=BORDER_SUBTLE, width=1)
    _draw_ecg_waveform(draw, ecg_box, accent, briefing.dread_percent)

    _draw_text_top_left(draw, (rx, cur_y + 132), "INTENSITEETTI- JA PAINETASO", f_stat_label, TEXT_MUTED)
    pct_str = f"{briefing.dread_percent}%"
    pw_r, _ = _text_size(draw, pct_str, f_stat_val)
    _draw_text_top_left(draw, (rx + rw - pw_r, cur_y + 130), pct_str, f_stat_val, TEXT_WHITE)
    _draw_segmented_meter(draw, rx, cur_y + 154, rw, 16, float(briefing.dread_percent), accent, segments=20, gap=3)

    cur_y += hero_h + 20

    # 3. Milestone Banner (if present)
    if milestone_lines:
        m_box = (pad, cur_y, width - pad, cur_y + milestone_h)
        m_bg = _blend(BG_OBSIDIAN, accent, 0.22)
        draw.rectangle(m_box, fill=m_bg, outline=accent, width=1)
        draw.rectangle([(pad, cur_y), (pad + 6, cur_y + milestone_h)], fill=accent)
        _draw_wrapped(draw, pad + 20, cur_y + 13, milestone_lines, f_milestone, TEXT_WHITE, 5)
        cur_y += milestone_h + 18

    # 4. Scary & Motivating Dispatch Panel
    d_box = (pad, cur_y, width - pad, cur_y + dispatch_h)
    draw.rectangle(d_box, fill=PANEL_ELEVATED, outline=BORDER_SUBTLE, width=1)
    draw.rectangle([(pad, cur_y), (pad + 5, cur_y + dispatch_h)], fill=accent)

    dy = _draw_wrapped(draw, pad + 24, cur_y + 22, headline_lines, f_headline, TEXT_WHITE, 6)
    _draw_wrapped(draw, pad + 24, dy + 12, dispatch_lines, f_body, TEXT_SOFT, 6)

    cur_y += dispatch_h + 18

    # 5. Today's Training Directive Panel
    t_box = (pad, cur_y, width - pad, cur_y + training_h)
    draw.rectangle(t_box, fill=PANEL_DARK, outline=BORDER_SUBTLE, width=1)

    # Header bar inside training card
    t_hdr_h = 44
    draw.rectangle([(pad, cur_y), (width - pad, cur_y + t_hdr_h)], fill=PANEL_ELEVATED, outline=BORDER_SUBTLE, width=1)
    draw.rectangle([(pad, cur_y), (pad + 5, cur_y + t_hdr_h)], fill=accent)
    _draw_text_top_left(
        draw,
        (pad + 20, cur_y + 14),
        f"PÄIVÄN TREENIKÄSKY  //  {briefing.weekday_name.upper()}",
        f_section,
        accent,
    )
    tag_right = "KISAPROTOKOLLA" if briefing.is_race_day else "PAKOLLINEN SUORITUS"
    if briefing.is_post_race:
        tag_right = "PALAUTUMINEN"
    tw_r, _ = _text_size(draw, tag_right, f_stat_label)
    _draw_text_top_left(draw, (width - pad - 18 - tw_r, cur_y + 15), tag_right, f_stat_label, TEXT_DIM)

    wy = _draw_wrapped(draw, pad + 20, cur_y + t_hdr_h + 14, workout_lines, f_body, TEXT_WHITE, 6)

    # Coach tip recessed bar
    tip_box_y0 = wy + 16
    tip_box_y1 = tip_box_y0 + tip_h + 22
    draw.rectangle(
        [(pad + 14, tip_box_y0), (width - pad - 14, tip_box_y1)],
        fill=PANEL_RECESSED,
        outline=BORDER_SUBTLE,
        width=1,
    )
    _draw_wrapped(draw, pad + 24, tip_box_y0 + 11, tip_lines, f_tip, TEXT_MUTED, 5)

    cur_y += training_h + 18

    # 6. 4-Leg Relay Track Diagram + Spotlight
    if not briefing.is_post_race:
        _draw_text_top_left(
            draw,
            (pad, cur_y),
            "MARATONVIESTIN OSUUDET (42,195 KM)  //  PÄIVÄN OSUUSNOSTO",
            f_section,
            TEXT_MUTED,
        )
        bar_y = cur_y + 24
        bounds = _draw_relay_track_bar(draw, pad, bar_y, content_w, 52, accent, leg.number)

        callout_y0 = bar_y + 62
        callout_h = _measure_wrapped(mdraw, leg_lines, f_tip, 5) + 22
        draw.rectangle(
            [(pad, callout_y0), (width - pad, callout_y0 + callout_h)],
            fill=PANEL_RECESSED,
            outline=BORDER_SUBTLE,
            width=1,
        )
        # Sharp connector tick pointing from spotlight leg down into callout box
        lx0, lx1 = bounds[leg.number - 1]
        mid_leg_x = (lx0 + lx1) // 2
        draw.rectangle([(mid_leg_x - 2, bar_y + 52), (mid_leg_x + 2, callout_y0)], fill=accent)
        draw.rectangle([(pad, callout_y0), (width - pad, callout_y0 + 2)], fill=_blend(BORDER_SUBTLE, accent, 0.5))

        _draw_wrapped(draw, pad + 18, callout_y0 + 11, leg_lines, f_tip, TEXT_SOFT, 5)
        cur_y += relay_section_h + 18

    # 7. 365-Day Countdown Timeline Bar
    if not briefing.is_post_race:
        _draw_text_top_left(draw, (pad, cur_y), "LÄHTÖLAUKAUS LÄHESTYY (365 PV IKKUNA)", f_stat_label, TEXT_MUTED)
        prog_lbl = f"{briefing.progress_percent:.1f}% KULUNUT"
        plw, _ = _text_size(draw, prog_lbl, f_stat_label)
        _draw_text_top_left(draw, (width - pad - plw, cur_y), prog_lbl, f_stat_label, accent)

        track_y = cur_y + 22
        track_h = 10
        draw.rectangle([(pad, track_y), (width - pad, track_y + track_h)], fill=PANEL_ELEVATED, outline=BORDER_SUBTLE, width=1)
        fill_w = int(round((briefing.progress_percent / 100.0) * content_w))
        if fill_w > 0:
            draw.rectangle([(pad, track_y), (pad + fill_w, track_y + track_h)], fill=accent)
        # Sharp vertical position marker needle
        needle_x = max(pad + 2, min(width - pad - 2, pad + fill_w))
        draw.rectangle([(needle_x - 2, track_y - 4), (needle_x + 2, track_y + track_h + 4)], fill=TEXT_WHITE)

        # Scale ticks
        for tick_days, tick_lbl in ((365, "365 PV"), (270, "270 PV"), (180, "180 PV"), (90, "90 PV"), (0, "RATINA")):
            frac = (365 - tick_days) / 365.0
            tx = pad + int(round(frac * content_w))
            tw, _ = _text_size(draw, tick_lbl, f_stat_label)
            draw_x = max(pad, min(width - pad - tw, tx - tw // 2))
            _draw_text_top_left(draw, (draw_x, track_y + 18), tick_lbl, f_stat_label, TEXT_DIM)

        cur_y += timeline_h + 16

    # 8. Footer
    draw.line([(pad, cur_y), (width - pad, cur_y)], fill=BORDER_SUBTLE, width=1)
    footer_left = "TAMPERE MARATON 2027 • RATINAN STADION"
    footer_right = "!maraton  |  !maraton osuudet  |  !maraton treeni"
    _draw_text_top_left(draw, (pad, cur_y + 12), footer_left, f_footer, TEXT_DIM)
    frw, _ = _text_size(draw, footer_right, f_footer)
    _draw_text_top_left(draw, (width - pad - frw, cur_y + 12), footer_right, f_footer, TEXT_DIM)

    return _to_png_bytes(img)


def render_maraton_legs_image(briefing: MaratonBriefing) -> bytes:
    """Renders a bespoke 4-leg maratonviesti tactical breakdown poster."""
    width = 920
    pad = 32
    content_w = width - pad * 2
    accent = _hex_to_rgb(briefing.accent_color)

    f_eyebrow = _font("mono_bold", 13)
    f_title = _font("display", 42)
    f_badge = _font("mono_bold", 13)
    f_leg_num = _font("mono_bold", 13)
    f_leg_dist = _font("display", 38)
    f_leg_role = _font("bold", 17)
    f_body = _font("regular", 15)
    f_footer = _font("mono", 12)

    col_gap = 16
    card_w = (content_w - col_gap) // 2

    scratch = Image.new("RGB", (width, 100), BG_OBSIDIAN)
    mdraw = ImageDraw.Draw(scratch)

    leg_danger_wrapped: list[list[str]] = []
    leg_tip_wrapped: list[list[str]] = []
    for leg in RELAY_LEGS:
        leg_danger_wrapped.append(_wrap_text(mdraw, f"UHKA: {leg.danger_note}.", f_body, card_w - 36))
        leg_tip_wrapped.append(_wrap_text(mdraw, f"TAKTIIKKA: {leg.tactical_tip}", f_body, card_w - 36))

    row1_h = max(
        118 + _measure_wrapped(mdraw, leg_danger_wrapped[i], f_body, 5) + 10 + _measure_wrapped(mdraw, leg_tip_wrapped[i], f_body, 5) + 20
        for i in (0, 1)
    )
    row2_h = max(
        118 + _measure_wrapped(mdraw, leg_danger_wrapped[i], f_body, 5) + 10 + _measure_wrapped(mdraw, leg_tip_wrapped[i], f_body, 5) + 20
        for i in (2, 3)
    )

    total_height = 6 + 24 + 26 + 56 + 22 + 52 + 24 + row1_h + col_gap + row2_h + 24 + 42

    img = _create_atmospheric_canvas(width, total_height, accent, briefing.dread_percent)
    _draw_hazard_stripe_bar(img, 0, 6, accent, intense=briefing.days_remaining <= 60 and not briefing.is_post_race)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (width - 1, total_height - 1)], outline=BORDER_SUBTLE, width=1)

    cur_y = 30
    date_str = format_finnish_date(briefing.event_date)
    _draw_text_top_left(draw, (pad, cur_y), f"TAMPERE MARATON 2027  //  {date_str}  //  42,195 KM", f_eyebrow, TEXT_MUTED)

    badge_txt = briefing.badge_text
    bw, bh = _text_size(draw, badge_txt, f_badge)
    bx0 = width - pad - (bw + 22)
    draw.rectangle([(bx0, cur_y - 4), (width - pad, cur_y - 4 + bh + 12)], fill=_blend(BG_OBSIDIAN, accent, 0.25), outline=accent, width=1)
    _draw_text_top_left(draw, (bx0 + 11, cur_y + 2), badge_txt, f_badge, TEXT_WHITE)

    cur_y += 28
    _draw_text_top_left(draw, (pad, cur_y), "MARATONVIESTIN OSUUDET", f_title, TEXT_WHITE)
    cur_y += 58

    # Proportional 4-leg track overview (highlighting the 14.1 km monster leg!)
    _draw_relay_track_bar(draw, pad, cur_y, content_w, 52, accent, highlight_leg_number=3)
    cur_y += 52 + 24

    # 2x2 Tactical Leg Cards
    for idx, leg in enumerate(RELAY_LEGS):
        col = idx % 2
        row = idx // 2
        cx = pad + col * (card_w + col_gap)
        cy = cur_y + (0 if row == 0 else row1_h + col_gap)
        ch = row1_h if row == 0 else row2_h
        is_monster = leg.number == 3

        box = (cx, cy, cx + card_w, cy + ch)
        bg = _blend(PANEL_DARK, accent, 0.10) if is_monster else PANEL_DARK
        border = accent if is_monster else BORDER_SUBTLE
        draw.rectangle(box, fill=bg, outline=border, width=1)
        draw.rectangle([(cx, cy), (cx + card_w, cy + 4)], fill=accent if is_monster else BORDER_BRIGHT)
        if is_monster:
            _draw_corner_brackets(draw, box, accent, length=10, thickness=2)

        tag = f"OSUUS {leg.number}  •  {leg.loops.upper()}"
        _draw_text_top_left(draw, (cx + 18, cy + 16), tag, f_leg_num, accent if is_monster else TEXT_MUTED)
        if is_monster:
            mw, _ = _text_size(draw, "HIRVIÖOSUUS", f_leg_num)
            _draw_text_top_left(draw, (cx + card_w - 18 - mw, cy + 16), "HIRVIÖOSUUS", f_leg_num, accent)

        _draw_text_top_left(draw, (cx + 18, cy + 40), leg.distance_km, f_leg_dist, TEXT_WHITE)
        _draw_text_top_left(draw, (cx + 18, cy + 88), leg.role.upper(), f_leg_role, accent if is_monster else TEXT_SOFT)

        dy = _draw_wrapped(draw, cx + 18, cy + 118, leg_danger_wrapped[idx], f_body, TEXT_SOFT, 5)
        _draw_wrapped(draw, cx + 18, dy + 10, leg_tip_wrapped[idx], f_body, TEXT_MUTED, 5)

    cur_y += row1_h + col_gap + row2_h + 24
    draw.line([(pad, cur_y), (width - pad, cur_y)], fill=BORDER_SUBTLE, width=1)
    _draw_text_top_left(draw, (pad, cur_y + 12), "VALITKAA 14,1 KM KOLMOSOSUUDELLE JOUKKUEEN KOVIN KESTÄVYYSKONE.", f_footer, TEXT_DIM)

    return _to_png_bytes(img)


def render_maraton_week_image(config: MaratonConfig, briefing: MaratonBriefing) -> bytes:
    """Renders a bespoke 7-day weekly training plan poster for the current phase."""
    width = 920
    pad = 32
    content_w = width - pad * 2
    accent = _hex_to_rgb(briefing.accent_color)
    phase = select_phase(max(1, briefing.days_remaining))

    f_eyebrow = _font("mono_bold", 13)
    f_title = _font("display", 40)
    f_badge = _font("mono_bold", 13)
    f_day = _font("mono_bold", 14)
    f_body = _font("regular", 15)
    f_footer = _font("mono", 12)

    scratch = Image.new("RGB", (width, 100), BG_OBSIDIAN)
    mdraw = ImageDraw.Draw(scratch)

    day_col_w = 165
    text_col_w = content_w - day_col_w - 36
    row_lines: list[list[str]] = []
    row_heights: list[int] = []
    for day_name, workout in zip(WEEKDAY_NAMES_FI, phase.weekly_workouts):
        lines = _wrap_text(mdraw, workout, f_body, text_col_w)
        row_lines.append(lines)
        min_h = 56 if day_name == briefing.weekday_name else 46
        row_heights.append(max(min_h, _measure_wrapped(mdraw, lines, f_body, 5) + 24))

    tip_lines = _wrap_text(mdraw, f"VALMENTAJAN VINKKI: {briefing.coach_tip}", f_body, content_w - 36)
    tip_h = _measure_wrapped(mdraw, tip_lines, f_body, 5) + 24

    total_height = 6 + 24 + 28 + 56 + 16 + sum(row_heights) + 6 * 8 + 18 + tip_h + 22 + 42

    img = _create_atmospheric_canvas(width, total_height, accent, briefing.dread_percent)
    _draw_hazard_stripe_bar(img, 0, 6, accent, intense=briefing.days_remaining <= 60 and not briefing.is_post_race)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (width - 1, total_height - 1)], outline=BORDER_SUBTLE, width=1)

    cur_y = 30
    date_str = format_finnish_date(briefing.event_date)
    _draw_text_top_left(draw, (pad, cur_y), f"{phase.name.upper()}  //  {date_str}", f_eyebrow, TEXT_MUTED)

    badge_txt = briefing.badge_text
    bw, bh = _text_size(draw, badge_txt, f_badge)
    bx0 = width - pad - (bw + 22)
    draw.rectangle([(bx0, cur_y - 4), (width - pad, cur_y - 4 + bh + 12)], fill=_blend(BG_OBSIDIAN, accent, 0.25), outline=accent, width=1)
    _draw_text_top_left(draw, (bx0 + 11, cur_y + 2), badge_txt, f_badge, TEXT_WHITE)

    cur_y += 28
    _draw_text_top_left(draw, (pad, cur_y), "VIIKON TREENIOHJELMA", f_title, TEXT_WHITE)
    cur_y += 56 + 16

    for day_name, lines, rh in zip(WEEKDAY_NAMES_FI, row_lines, row_heights):
        is_today = day_name == briefing.weekday_name
        bg = _blend(PANEL_DARK, accent, 0.16) if is_today else PANEL_DARK
        border = accent if is_today else BORDER_SUBTLE
        draw.rectangle([(pad, cur_y), (width - pad, cur_y + rh)], fill=bg, outline=border, width=1)
        if is_today:
            draw.rectangle([(pad, cur_y), (pad + 5, cur_y + rh)], fill=accent)

        day_label = f"{day_name.upper()} *" if is_today else day_name.upper()
        _draw_text_top_left(draw, (pad + 18, cur_y + 14), day_label, f_day, accent if is_today else TEXT_MUTED)
        if is_today:
            _draw_text_top_left(draw, (pad + 18, cur_y + 32), "TÄNÄÄN", _font("mono", 11), TEXT_WHITE)

        _draw_wrapped(draw, pad + day_col_w, cur_y + 11, lines, f_body, TEXT_WHITE if is_today else TEXT_SOFT, 5)
        cur_y += rh + 8

    cur_y += 10
    draw.rectangle([(pad, cur_y), (width - pad, cur_y + tip_h)], fill=PANEL_RECESSED, outline=BORDER_SUBTLE, width=1)
    _draw_wrapped(draw, pad + 18, cur_y + 12, tip_lines, f_body, TEXT_MUTED, 5)

    cur_y += tip_h + 22
    draw.line([(pad, cur_y), (width - pad, cur_y)], fill=BORDER_SUBTLE, width=1)
    _draw_text_top_left(
        draw,
        (pad, cur_y + 12),
        f"* = TÄMÄN PÄIVÄN HARJOITUS  •  PÄIVITTÄINEN MUISTUTUS KLO {config.send_time} ({config.timezone})",
        f_footer,
        TEXT_DIM,
    )

    return _to_png_bytes(img)
