from __future__ import annotations

import gzip
import logging
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from bot.config import TelkkariConfig

logger = logging.getLogger(__name__)

# Finnish free-to-air channels mapping: channel_number -> (display_name, epg_channel_id)
FREE_CHANNELS: dict[int, tuple[str, str]] = {
    1: ("YLE TV1", "YLE.TV1.fi"),
    2: ("YLE TV2", "YLE.TV2.fi"),
    3: ("MTV3", "MTV3.fi"),
    4: ("Nelonen", "Nelonen.fi"),
    5: ("Yle Teema Fem", "Yle.Teema.Fem.fi"),
    6: ("MTV Sub", "MTV.Sub.fi"),
    7: ("TV5", "TV.5.fi"),
    8: ("Liv", "Liv.fi"),
    9: ("JIM", "JIM.fi"),
    10: ("Kutonen", "Kutonen.fi"),
    11: ("TLC", "TLC.fi"),
    12: ("Star Channel", "Star.Channel.fi"),
    13: ("MTV Ava", "MTV.Ava.fi"),
    14: ("Hero", "Hero.fi"),
    15: ("Frii", "Frii.fi"),
    16: ("National Geographic", "National.Geographic.fi"),
    17: ("Eveo", "Eveo.fi"),
}

HELSINKI_TZ = ZoneInfo("Europe/Helsinki")

# Global cache tuple: (timestamp, root_element)
_EPG_CACHE: tuple[float, ET.Element] | None = None


def clear_epg_cache() -> None:
    """Clear the in-memory EPG cache (useful for testing)."""
    global _EPG_CACHE
    _EPG_CACHE = None


def parse_xmltv_time(time_str: str) -> datetime:
    """Parse XMLTV datetime string into timezone-aware datetime.

    Format example: '20260802040000 +0000' or '20260802070000 +0300'
    """
    parts = time_str.strip().split()
    dt_str = parts[0]
    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")

    if len(parts) > 1:
        tz_str = parts[1]
        sign = -1 if tz_str.startswith("-") else 1
        hours = int(tz_str[1:3])
        minutes = int(tz_str[3:5])
        tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
    else:
        tz = timezone.utc

    return dt.replace(tzinfo=tz)


def fetch_epg_data(config: TelkkariConfig) -> ET.Element:
    """Fetch and parse EPG XML TV data with caching support."""
    global _EPG_CACHE

    now_ts = time.time()
    if (
        _EPG_CACHE is not None
        and config.cache_timeout_seconds > 0
        and (now_ts - _EPG_CACHE[0]) < config.cache_timeout_seconds
    ):
        return _EPG_CACHE[1]

    req = urllib.request.Request(
        config.epg_url,
        headers={"User-Agent": "telegram-bot-telkkari/1.0"},
    )
    with urllib.request.urlopen(req, timeout=config.timeout_seconds) as resp:
        content = resp.read()

    if config.epg_url.endswith(".gz") or content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)

    root = ET.fromstring(content)
    _EPG_CACHE = (now_ts, root)
    return root


def get_channel_day_schedule(
    channel_num: int,
    config: TelkkariConfig,
    now: datetime | None = None,
    xml_root: ET.Element | None = None,
) -> str:
    """Format full day's TV program schedule for a specific channel number."""
    if channel_num not in FREE_CHANNELS:
        available = "\n".join(
            f"{num}: {name}" for num, (name, _) in sorted(FREE_CHANNELS.items())
        )
        return (
            f"⚠️ Tuntematon kanavanumero: {channel_num}.\n\n"
            f"Vapaasti katsottavat kanavat:\n{available}"
        )

    ch_name, epg_id = FREE_CHANNELS[channel_num]

    if now is None:
        now = datetime.now(HELSINKI_TZ)
    else:
        now = now.astimezone(HELSINKI_TZ)

    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)

    try:
        root = xml_root if xml_root is not None else fetch_epg_data(config)
    except Exception as e:
        logger.error("Failed to fetch EPG data: %s", e)
        return "⚠️ TV-ohjelmatietojen haku epäonnistui. Yritä myöhemmin uudelleen."

    programmes = root.findall("programme")
    ch_progs = [p for p in programmes if p.get("channel") == epg_id]

    today_items: list[tuple[datetime, datetime, str]] = []
    for p in ch_progs:
        start_raw = p.get("start")
        stop_raw = p.get("stop")
        if not start_raw or not stop_raw:
            continue

        try:
            start_dt = parse_xmltv_time(start_raw).astimezone(HELSINKI_TZ)
            stop_dt = parse_xmltv_time(stop_raw).astimezone(HELSINKI_TZ)
        except Exception:
            continue

        if stop_dt > today_start and start_dt < today_end:
            title_elem = p.find("title")
            title = title_elem.text if title_elem is not None and title_elem.text else "Tuntematon ohjelma"
            today_items.append((start_dt, stop_dt, title))

    today_items.sort(key=lambda x: x[0])

    if not today_items:
        return f"Kanavan {ch_name} ohjelmatietoja ei löytynyt tälle päivälle."

    lines = [f"📺 {ch_name} (tänään):", ""]
    for start_dt, stop_dt, title in today_items:
        start_str = start_dt.strftime("%H:%M")
        stop_str = stop_dt.strftime("%H:%M")
        lines.append(f"{start_str} - {stop_str}: {title}")

    return "\n".join(lines)


def get_next_hour_schedule(
    config: TelkkariConfig,
    now: datetime | None = None,
    xml_root: ET.Element | None = None,
) -> str:
    """Format next hour TV program schedule for configured default channels."""
    if now is None:
        now = datetime.now(HELSINKI_TZ)
    else:
        now = now.astimezone(HELSINKI_TZ)

    next_hour_end = now + timedelta(hours=1)

    try:
        root = xml_root if xml_root is not None else fetch_epg_data(config)
    except Exception as e:
        logger.error("Failed to fetch EPG data: %s", e)
        return "⚠️ TV-ohjelmatietojen haku epäonnistui. Yritä myöhemmin uudelleen."

    programmes = root.findall("programme")

    lines = ["📺 TV-ohjelmat seuraavan tunnin aikana:", ""]
    has_any = False

    for ch_num in config.default_channels:
        if ch_num not in FREE_CHANNELS:
            continue

        ch_name, epg_id = FREE_CHANNELS[ch_num]
        ch_progs = [p for p in programmes if p.get("channel") == epg_id]

        ch_items: list[tuple[datetime, datetime, str]] = []
        for p in ch_progs:
            start_raw = p.get("start")
            stop_raw = p.get("stop")
            if not start_raw or not stop_raw:
                continue

            try:
                start_dt = parse_xmltv_time(start_raw).astimezone(HELSINKI_TZ)
                stop_dt = parse_xmltv_time(stop_raw).astimezone(HELSINKI_TZ)
            except Exception:
                continue

            if stop_dt > now and start_dt < next_hour_end:
                title_elem = p.find("title")
                title = title_elem.text if title_elem is not None and title_elem.text else "Tuntematon ohjelma"
                ch_items.append((start_dt, stop_dt, title))

        ch_items.sort(key=lambda x: x[0])

        if ch_items:
            has_any = True
            lines.append(f"{ch_name}:")
            for start_dt, stop_dt, title in ch_items:
                start_str = start_dt.strftime("%H:%M")
                stop_str = stop_dt.strftime("%H:%M")
                lines.append(f"  {start_str} - {stop_str}: {title}")
            lines.append("")

    if not has_any:
        return "Seuraavan tunnin aikana ei löytynyt ohjelmatietoja."

    return "\n".join(lines).strip()
