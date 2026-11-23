"""Crisp sharp-edged geometric vector icons and text sanitization for PIL cards."""

from __future__ import annotations

import math
import re

from PIL import ImageDraw


_MULTI_SPACE_RE = re.compile(r" {2,}")


def _is_unsupported_emoji_cp(cp: int) -> bool:
    """Return True for emoji/pictograph/arrow codepoints that render as tofu in PIL Sans fonts."""
    if cp in (0xFE0E, 0xFE0F, 0x200D, 0x20E3):
        return True
    # Arrows block (diagonal/cardinal arrows missing in LiberationSans)
    if 0x2190 <= cp <= 0x21FF:
        return True
    # Misc symbols & dingbats
    if 0x2600 <= cp <= 0x27BF:
        return True
    # Supplemental Symbols, Emoticons, Transport, Misc Pictographs, Flags, etc.
    if 0x1F000 <= cp <= 0x1FAFF:
        return True
    return False


def sanitize_for_pil(text: str) -> str:
    """Strip unsupported Unicode emoji/pictograph codepoints that render as empty boxes.

    Preserves Finnish letters (äöåÄÖÅ), degree sign (°), arrows (↑↓→←),
    en/em dashes (–—), bullets (•), ellipsis (…), and standard punctuation.
    """
    if not text:
        return ""
    cleaned_lines: list[str] = []
    for line in str(text).splitlines():
        kept = [ch for ch in line if not _is_unsupported_emoji_cp(ord(ch))]
        collapsed = _MULTI_SPACE_RE.sub(" ", "".join(kept)).strip()
        cleaned_lines.append(collapsed)
    return "\n".join(cleaned_lines)


def weather_icon_name(condition_id: int | None, icon_code: str | None = None) -> str:
    """Map an OpenWeather condition ID and icon code to a geometric vector icon name."""
    is_night = bool(icon_code and icon_code.endswith("n"))
    if condition_id is None:
        return "moon" if is_night else "sun"
    if 200 <= condition_id < 300:
        return "storm"
    if 300 <= condition_id < 600:
        return "rain"
    if 600 <= condition_id < 700:
        return "snow"
    if 700 <= condition_id < 800:
        return "fog"
    if condition_id == 800:
        return "moon" if is_night else "sun"
    if condition_id in (801, 802):
        return "partly_cloudy"
    return "cloud"


def _draw_cloud_blocks(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    size: int,
    fill: str,
) -> None:
    """Draw a crisp sharp-edged geometric cloud silhouette using rectangular blocks."""
    s = max(12, size)
    # Lower wide base
    base_left = cx - int(s * 0.44)
    base_right = cx + int(s * 0.44)
    base_top = cy - int(s * 0.04)
    base_bot = cy + int(s * 0.22)
    draw.rectangle([(base_left, base_top), (base_right, base_bot)], fill=fill)
    # Left shoulder
    draw.rectangle(
        [(cx - int(s * 0.30), cy - int(s * 0.22)), (cx + int(s * 0.04), base_top + 1)],
        fill=fill,
    )
    # Right lower dome
    draw.rectangle(
        [(cx + int(s * 0.02), cy - int(s * 0.14)), (cx + int(s * 0.32), base_top + 1)],
        fill=fill,
    )


def draw_icon(
    draw: ImageDraw.ImageDraw,
    icon_name: str,
    cx: int,
    cy: int,
    size: int,
    color: str,
    secondary_color: str | None = None,
) -> None:
    """Draw a crisp sharp-edged geometric vector icon centered at (cx, cy).

    Strictly avoids rounded_rectangle so all geometry remains sharp and crisp.
    Supports: sun, moon, cloud, partly_cloudy, rain, snow, storm, fog,
    live, server, voice, warning, info, wind, wind:<deg>.
    """
    name = (icon_name or "").strip().lower()
    s = max(12, int(size))
    sec = secondary_color or "#f4f6fb"
    stroke = max(2, s // 14)

    if name == "sun":
        core_r = max(4, int(s * 0.20))
        # Sharp octagonal sun core
        diag = int(core_r * 0.72)
        oct_pts = [
            (cx, cy - core_r),
            (cx + diag, cy - diag),
            (cx + core_r, cy),
            (cx + diag, cy + diag),
            (cx, cy + core_r),
            (cx - diag, cy + diag),
            (cx - core_r, cy),
            (cx - diag, cy - diag),
        ]
        draw.polygon(oct_pts, fill=color)
        inner_r = int(s * 0.28)
        outer_r = int(s * 0.44)
        for i in range(8):
            angle = math.radians(i * 45)
            x1 = cx + int(round(math.cos(angle) * inner_r))
            y1 = cy + int(round(math.sin(angle) * inner_r))
            x2 = cx + int(round(math.cos(angle) * outer_r))
            y2 = cy + int(round(math.sin(angle) * outer_r))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=stroke)
        return

    if name == "moon":
        # Geometric sharp crescent polygon
        pts = [
            (cx + int(s * 0.10), cy - int(s * 0.40)),
            (cx - int(s * 0.16), cy - int(s * 0.30)),
            (cx - int(s * 0.32), cy - int(s * 0.08)),
            (cx - int(s * 0.30), cy + int(s * 0.18)),
            (cx - int(s * 0.10), cy + int(s * 0.38)),
            (cx + int(s * 0.20), cy + int(s * 0.40)),
            (cx + int(s * 0.34), cy + int(s * 0.26)),
            (cx + int(s * 0.10), cy + int(s * 0.24)),
            (cx - int(s * 0.06), cy + int(s * 0.08)),
            (cx - int(s * 0.06), cy - int(s * 0.12)),
            (cx + int(s * 0.12), cy - int(s * 0.26)),
        ]
        draw.polygon(pts, fill=color)
        # Small geometric star accent
        sx, sy = cx + int(s * 0.26), cy - int(s * 0.18)
        sr = max(2, s // 10)
        draw.line([(sx - sr, sy), (sx + sr, sy)], fill=sec, width=max(1, stroke - 1))
        draw.line([(sx, sy - sr), (sx, sy + sr)], fill=sec, width=max(1, stroke - 1))
        return

    if name == "cloud":
        _draw_cloud_blocks(draw, cx, cy, s, color)
        return

    if name == "partly_cloudy":
        # Sun in top-right behind cloud
        sun_cx = cx + int(s * 0.18)
        sun_cy = cy - int(s * 0.18)
        sun_r = max(3, int(s * 0.13))
        draw.rectangle(
            [(sun_cx - sun_r, sun_cy - sun_r), (sun_cx + sun_r, sun_cy + sun_r)],
            fill=color,
        )
        for angle_deg in (-90, -45, 0, 45, -135):
            rad = math.radians(angle_deg)
            x1 = sun_cx + int(round(math.cos(rad) * (sun_r + 2)))
            y1 = sun_cy + int(round(math.sin(rad) * (sun_r + 2)))
            x2 = sun_cx + int(round(math.cos(rad) * (sun_r + max(4, s // 8))))
            y2 = sun_cy + int(round(math.sin(rad) * (sun_r + max(4, s // 8))))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=max(1, stroke - 1))
        _draw_cloud_blocks(draw, cx - int(s * 0.06), cy + int(s * 0.08), int(s * 0.88), sec)
        return

    if name == "rain":
        _draw_cloud_blocks(draw, cx, cy - int(s * 0.10), int(s * 0.90), sec)
        drop_top = cy + int(s * 0.18)
        drop_bot = cy + int(s * 0.40)
        for dx in (-int(s * 0.24), 0, int(s * 0.24)):
            draw.line(
                [(cx + dx, drop_top), (cx + dx - int(s * 0.08), drop_bot)],
                fill=color,
                width=stroke,
            )
        return

    if name == "snow":
        _draw_cloud_blocks(draw, cx, cy - int(s * 0.12), int(s * 0.88), sec)
        fl_r = max(2, s // 10)
        for dx, dy in (
            (-int(s * 0.24), int(s * 0.28)),
            (0, int(s * 0.34)),
            (int(s * 0.24), int(s * 0.28)),
        ):
            fx, fy = cx + dx, cy + dy
            draw.line([(fx - fl_r, fy), (fx + fl_r, fy)], fill=color, width=max(1, stroke - 1))
            draw.line([(fx, fy - fl_r), (fx, fy + fl_r)], fill=color, width=max(1, stroke - 1))
            d = max(1, int(fl_r * 0.7))
            draw.line([(fx - d, fy - d), (fx + d, fy + d)], fill=color, width=1)
            draw.line([(fx - d, fy + d), (fx + d, fy - d)], fill=color, width=1)
        return

    if name == "storm":
        _draw_cloud_blocks(draw, cx, cy - int(s * 0.12), int(s * 0.90), sec)
        bolt = [
            (cx + int(s * 0.06), cy + int(s * 0.10)),
            (cx - int(s * 0.12), cy + int(s * 0.26)),
            (cx + int(s * 0.02), cy + int(s * 0.26)),
            (cx - int(s * 0.08), cy + int(s * 0.46)),
            (cx + int(s * 0.16), cy + int(s * 0.22)),
            (cx + int(s * 0.02), cy + int(s * 0.22)),
        ]
        draw.polygon(bolt, fill=color)
        return

    if name == "fog":
        half_w = int(s * 0.40)
        for idx, dy in enumerate((-int(s * 0.22), -int(s * 0.06), int(s * 0.10), int(s * 0.26))):
            inset = int(s * 0.08) if idx % 2 else 0
            draw.rectangle(
                [(cx - half_w + inset, cy + dy), (cx + half_w - inset, cy + dy + max(2, stroke))],
                fill=color if idx % 2 == 0 else sec,
            )
        return

    if name == "live":
        r_outer = max(5, int(s * 0.34))
        r_inner = max(2, int(s * 0.15))
        draw.rectangle(
            [(cx - r_outer, cy - r_outer), (cx + r_outer, cy + r_outer)],
            outline=color,
            width=stroke,
        )
        draw.rectangle(
            [(cx - r_inner, cy - r_inner), (cx + r_inner, cy + r_inner)],
            fill=color,
        )
        return

    if name == "server":
        w_half = int(s * 0.38)
        unit_h = max(4, int(s * 0.18))
        for offset in (-int(s * 0.24), 0, int(s * 0.24)):
            top = cy + offset - unit_h // 2
            draw.rectangle(
                [(cx - w_half, top), (cx + w_half, top + unit_h)],
                outline=color,
                width=max(1, stroke - 1),
            )
            draw.rectangle(
                [(cx - w_half + 3, top + 2), (cx - w_half + 6, top + unit_h - 2)],
                fill=sec,
            )
        return

    if name == "voice":
        bar_w = max(2, s // 7)
        heights = (0.35, 0.72, 0.52, 0.85, 0.40)
        total_w = len(heights) * bar_w + (len(heights) - 1) * max(2, bar_w // 2)
        bx = cx - total_w // 2
        for h_frac in heights:
            bh = max(4, int(s * h_frac))
            draw.rectangle([(bx, cy - bh // 2), (bx + bar_w, cy + bh // 2)], fill=color)
            bx += bar_w + max(2, bar_w // 2)
        return

    if name == "warning":
        r = int(s * 0.38)
        tri = [(cx, cy - r), (cx + r, cy + r - 2), (cx - r, cy + r - 2)]
        draw.polygon(tri, outline=color, width=stroke)
        draw.rectangle([(cx - 1, cy - int(r * 0.35)), (cx + 1, cy + int(r * 0.32))], fill=color)
        draw.rectangle([(cx - 1, cy + int(r * 0.52)), (cx + 1, cy + int(r * 0.68))], fill=color)
        return

    if name.startswith("wind"):
        deg = 0.0
        if ":" in name:
            try:
                deg = float(name.split(":", 1)[1])
            except ValueError:
                deg = 0.0
        # Meteorological wind_deg is direction wind blows FROM; arrow points WHERE it blows (+180 deg)
        rad = math.radians(deg + 180.0 - 90.0)
        tip_r = int(s * 0.38)
        tail_r = int(s * 0.30)
        wing_r = int(s * 0.22)
        tip = (cx + int(round(math.cos(rad) * tip_r)), cy + int(round(math.sin(rad) * tip_r)))
        tail = (cx - int(round(math.cos(rad) * tail_r)), cy - int(round(math.sin(rad) * tail_r)))
        wing_l = (
            cx + int(round(math.cos(rad + 2.4) * wing_r)),
            cy + int(round(math.sin(rad + 2.4) * wing_r)),
        )
        wing_r_pt = (
            cx + int(round(math.cos(rad - 2.4) * wing_r)),
            cy + int(round(math.sin(rad - 2.4) * wing_r)),
        )
        draw.polygon([tip, wing_l, tail, wing_r_pt], fill=color)
        return

    # Default "info"
    r = max(5, int(s * 0.34))
    draw.rectangle([(cx - r, cy - r), (cx + r, cy + r)], outline=color, width=stroke)
    draw.rectangle([(cx - 1, cy - int(r * 0.45)), (cx + 1, cy - int(r * 0.25))], fill=color)
    draw.rectangle([(cx - 1, cy - int(r * 0.05)), (cx + 1, cy + int(r * 0.50))], fill=color)
