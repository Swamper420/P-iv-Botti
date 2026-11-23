from __future__ import annotations

import gzip
import logging
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from bot.config import TelkkariConfig
from bot.rendering import BadgeColor, Card, TimelineItem

logger = logging.getLogger(__name__)

TELKKARI_CARD_FOOTER = "TV-ohjelmat • P-iv-Botti"

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

        if stop_dt > now and start_dt < today_end:
            title_elem = p.find("title")
            title = title_elem.text if title_elem is not None and title_elem.text else "Tuntematon ohjelma"
            today_items.append((start_dt, stop_dt, title))

    today_items.sort(key=lambda x: x[0])

    if not today_items:
        return f"Kanavan {ch_name} ohjelmatietoja ei löytynyt loppupäivälle."


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


# ---------------------------------------------------------------------------
# Picture-card builders (pure, unit-testable — no network, no Telegram)
# ---------------------------------------------------------------------------


def build_telkkari_info_card(
    title: str,
    text: str,
    badge_text: str = "TIETO",
    badge_color: BadgeColor | str = BadgeColor.BLUE,
    subtitle: str | None = None,
) -> Card:
    """Build a simple informational Card for errors, notices and help."""
    return (
        Card(title=title, subtitle=subtitle, footer=TELKKARI_CARD_FOOTER)
        .set_badge(badge_text, badge_color)
        .add_text(text)
    )


def build_unknown_channel_card(channel_num: int) -> Card:
    """Card for unknown channel numbers, listing available channels."""
    rows: list[list[str]] = [[str(num), name] for num, (name, _) in sorted(FREE_CHANNELS.items())]
    return (
        Card(
            title="Tuntematon kanava",
            subtitle=f"Kanava {channel_num} ei löytynyt",
            footer=TELKKARI_CARD_FOOTER,
            accent_color="#eab308",
        )
        .set_badge("VIRHE", BadgeColor.YELLOW)
        .add_callout(f"Tuntematon kanavanumero: {channel_num}.", title="Valitse listalta", tone="warning")
        .add_section("SAATAVILLA OLEVAT KANAVAT", right_text=f"{len(rows)} kanavaa")
        .add_table(
            headers=["#", "Kanava"],
            rows=rows,
            col_widths=[1, 6],
            primary_col=1,
            bold_cols=[1],
            mono_cols=[0],
            overflow="ellipsis",
        )
    )


def build_telkkari_error_card(message: str, subtitle: str | None = None) -> Card:
    """Card for fetch failures and other error states."""
    return (
        Card(title="TV-ohjelmat", subtitle=subtitle, footer=TELKKARI_CARD_FOOTER, accent_color="#ef4444")
        .set_badge("VIRHE", BadgeColor.RED)
        .add_callout(message, title="Haku epäonnistui", tone="error")
    )


def build_invalid_channel_arg_card(raw_arg: str) -> Card:
    """Card for non-numeric channel arguments (e.g. '!telkkari abc')."""
    return (
        Card(
            title="TV-ohjelmat",
            subtitle="Käyttö: !telkkari <kanavanumero>",
            footer=TELKKARI_CARD_FOOTER,
            accent_color="#eab308",
        )
        .set_badge("VIRHE", BadgeColor.YELLOW)
        .add_callout(
            f"Virheellinen kanavanumero: '{raw_arg}'. Anna kanavanumero pelkkänä lukuna (esim. !telkkari 1).",
            tone="warning",
        )
    )


def build_channel_day_card(
    channel_name: str,
    items: list[tuple[datetime, datetime, str]],
    now: datetime | None = None,
) -> Card:
    """Build a high-signal picture card for a single channel's remaining day schedule."""
    resolved_now = (now.astimezone(HELSINKI_TZ) if now is not None else datetime.now(HELSINKI_TZ))
    date_str = resolved_now.strftime("%d.%m.%Y")

    if not items:
        return build_telkkari_info_card(
            channel_name,
            f"Kanavan {channel_name} ohjelmatietoja ei löytynyt loppupäivälle.",
            badge_text="TYHJÄ",
            badge_color=BadgeColor.GRAY,
            subtitle=f"tänään {date_str}",
        )

    ordered = sorted(items, key=lambda x: x[0])
    live_item: tuple[datetime, datetime, str] | None = None
    upcoming_items: list[tuple[datetime, datetime, str]] = []
    for start_dt, stop_dt, title in ordered:
        if live_item is None and start_dt <= resolved_now < stop_dt:
            live_item = (start_dt, stop_dt, title)
        elif start_dt >= resolved_now or live_item is not None:
            upcoming_items.append((start_dt, stop_dt, title))

    is_live = live_item is not None
    card = Card(
        title=channel_name,
        subtitle=f"tänään {date_str} • {len(ordered)} ohjelmaa",
        footer=TELKKARI_CARD_FOOTER,
        accent_color="#22c55e" if is_live else "#38bdf8",
    )
    if is_live:
        card.set_badge("TÄNÄÄN", BadgeColor.GREEN)
        l_start, l_stop, l_title = live_item
        total_sec = max(60.0, (l_stop - l_start).total_seconds())
        elapsed_sec = max(0.0, min(total_sec, (resolved_now - l_start).total_seconds()))
        elapsed_min = int(elapsed_sec // 60)
        total_min = max(1, int(total_sec // 60))
        rem_min = max(0, total_min - elapsed_min)
        card.add_section(
            "NYT KÄYNNISSÄ",
            right_text=f"{l_start.strftime('%H:%M')} – {l_stop.strftime('%H:%M')}",
            accent_color="#22c55e",
        )
        card.add_callout(
            l_title,
            title=f"Jäljellä {rem_min} min",
            tone="live",
        )
        card.add_progress_bar(
            "Ohjelman eteneminen",
            value=float(elapsed_min),
            max_value=float(total_min),
            unit="min",
            color="#22c55e",
        )
    else:
        card.set_badge("TÄNÄÄN", BadgeColor.BLUE)

    table_items = upcoming_items if is_live else ordered
    if table_items:
        rows: list[list[str]] = [
            [f"{start_dt.strftime('%H:%M')} - {stop_dt.strftime('%H:%M')}", title]
            for start_dt, stop_dt, title in table_items
        ]
        card.add_section(
            "TULOSSA TÄNÄÄN" if is_live else "OHJELMAT TÄNÄÄN",
            right_text=f"{len(rows)} ohjelmaa",
        )
        card.add_table(
            headers=["Aika", "Ohjelma"],
            rows=rows,
            col_widths=[1.4, 4.6],
            primary_col=1,
            bold_cols=[1],
            mono_cols=[0],
            max_rows=22,
            overflow="ellipsis",
        )

    return card


def build_next_hour_card(
    entries: list[tuple[str, datetime, datetime, str]],
    now: datetime | None = None,
    next_hour_end: datetime | None = None,
) -> Card:
    """Build a visual timeline picture card for the next-hour overview across channels."""
    resolved_now = (now.astimezone(HELSINKI_TZ) if now is not None else datetime.now(HELSINKI_TZ))
    resolved_end = (
        next_hour_end.astimezone(HELSINKI_TZ)
        if next_hour_end is not None
        else resolved_now + timedelta(hours=1)
    )
    time_range = f"{resolved_now.strftime('%H:%M')} – {resolved_end.strftime('%H:%M')}"

    if not entries:
        return build_telkkari_info_card(
            "TV-ohjelmat",
            "Seuraavan tunnin aikana ei löytynyt ohjelmatietoja.",
            badge_text="TYHJÄ",
            badge_color=BadgeColor.GRAY,
            subtitle=time_range,
        )

    ordered = sorted(entries, key=lambda x: (x[0], x[1]))
    # Include 15-min lookback so currently-airing shows display live progress to the left of NYT
    window_start = resolved_now - timedelta(minutes=15)
    window_span_s = max(60.0, (resolved_end - window_start).total_seconds())
    now_frac = max(0.0, min(1.0, (resolved_now - window_start).total_seconds() / window_span_s))

    timeline_items: list[TimelineItem] = []
    live_count = 0
    for ch_name, start_dt, stop_dt, title in ordered[:14]:
        is_live = start_dt <= resolved_now < stop_dt
        if is_live:
            live_count += 1
        sf = max(0.0, min(1.0, (start_dt - window_start).total_seconds() / window_span_s))
        ef = max(sf, min(1.0, (stop_dt - window_start).total_seconds() / window_span_s))
        prog: float | None = None
        if is_live:
            dur = max(1.0, (stop_dt - start_dt).total_seconds())
            prog = max(0.0, min(1.0, (resolved_now - start_dt).total_seconds() / dur))
        timeline_items.append(
            TimelineItem(
                label=ch_name,
                title=title,
                start_frac=sf,
                end_frac=ef,
                time_text=f"{start_dt.strftime('%H:%M')}–{stop_dt.strftime('%H:%M')}",
                is_live=is_live,
                color="#22c55e" if is_live else "#38bdf8",
                progress=prog,
            )
        )

    card = (
        Card(
            title="TV-ohjelmat",
            subtitle=f"seuraavan tunnin aikana • {time_range}",
            footer=f"Vihreä = nyt käynnissä ({live_count} kpl) • {TELKKARI_CARD_FOOTER}",
            accent_color="#22c55e" if live_count > 0 else "#38bdf8",
        )
        .set_badge(f"{len(ordered)} OHJELMAA", BadgeColor.GREEN)
        .add_section("OHJELMAKARTTA", right_text=time_range)
        .add_timeline(
            items=timeline_items,
            start_label=window_start.strftime("%H:%M"),
            end_label=resolved_end.strftime("%H:%M"),
            now_fraction=now_frac,
            now_label=f"NYT {resolved_now.strftime('%H:%M')}",
        )
    )
    if len(ordered) > len(timeline_items):
        card.add_text(f"… (+{len(ordered) - len(timeline_items)} muuta ohjelmaa)", muted=True)
    return card


def _collect_channel_day_items(
    channel_num: int,
    config: TelkkariConfig,
    now: datetime | None = None,
    xml_root: ET.Element | None = None,
) -> tuple[str | None, list[tuple[datetime, datetime, str]], str | None]:
    """Collect remaining-day items for card rendering.

    Returns ``(channel_name, items, error)`` where ``error`` is one of
    ``"unknown"``, ``"fetch_failed"``, ``"empty"`` or ``None`` on success.
    Mirrors the filtering rules of :func:`get_channel_day_schedule` without
    building the fallback text.
    """
    if channel_num not in FREE_CHANNELS:
        return None, [], "unknown"

    ch_name, epg_id = FREE_CHANNELS[channel_num]
    resolved_now = (now.astimezone(HELSINKI_TZ) if now is not None else datetime.now(HELSINKI_TZ))
    today_end = resolved_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    try:
        root = xml_root if xml_root is not None else fetch_epg_data(config)
    except Exception as e:
        logger.error("Failed to fetch EPG data: %s", e)
        return ch_name, [], "fetch_failed"

    items: list[tuple[datetime, datetime, str]] = []
    for p in root.findall("programme"):
        if p.get("channel") != epg_id:
            continue
        start_raw = p.get("start")
        stop_raw = p.get("stop")
        if not start_raw or not stop_raw:
            continue
        try:
            start_dt = parse_xmltv_time(start_raw).astimezone(HELSINKI_TZ)
            stop_dt = parse_xmltv_time(stop_raw).astimezone(HELSINKI_TZ)
        except Exception:
            continue
        if stop_dt > resolved_now and start_dt < today_end:
            title_elem = p.find("title")
            title = title_elem.text if title_elem is not None and title_elem.text else "Tuntematon ohjelma"
            items.append((start_dt, stop_dt, title))

    items.sort(key=lambda x: x[0])
    if not items:
        return ch_name, [], "empty"
    return ch_name, items, None


def _collect_next_hour_entries(
    config: TelkkariConfig,
    now: datetime | None = None,
    xml_root: ET.Element | None = None,
) -> tuple[list[tuple[str, datetime, datetime, str]], datetime, datetime, str | None]:
    """Collect next-hour entries for card rendering.

    Returns ``(entries, now, next_hour_end, error)`` where ``error`` is
    ``"fetch_failed"``, ``"empty"`` or ``None`` on success. Mirrors the
    filtering rules of :func:`get_next_hour_schedule`.
    """
    resolved_now = (now.astimezone(HELSINKI_TZ) if now is not None else datetime.now(HELSINKI_TZ))
    next_hour_end = resolved_now + timedelta(hours=1)

    try:
        root = xml_root if xml_root is not None else fetch_epg_data(config)
    except Exception as e:
        logger.error("Failed to fetch EPG data: %s", e)
        return [], resolved_now, next_hour_end, "fetch_failed"

    entries: list[tuple[str, datetime, datetime, str]] = []
    for ch_num in config.default_channels:
        if ch_num not in FREE_CHANNELS:
            continue
        ch_name, epg_id = FREE_CHANNELS[ch_num]
        ch_items: list[tuple[datetime, datetime, str]] = []
        for p in root.findall("programme"):
            if p.get("channel") != epg_id:
                continue
            start_raw = p.get("start")
            stop_raw = p.get("stop")
            if not start_raw or not stop_raw:
                continue
            try:
                start_dt = parse_xmltv_time(start_raw).astimezone(HELSINKI_TZ)
                stop_dt = parse_xmltv_time(stop_raw).astimezone(HELSINKI_TZ)
            except Exception:
                continue
            if stop_dt > resolved_now and start_dt < next_hour_end:
                title_elem = p.find("title")
                title = title_elem.text if title_elem is not None and title_elem.text else "Tuntematon ohjelma"
                ch_items.append((start_dt, stop_dt, title))
        ch_items.sort(key=lambda x: x[0])
        for start_dt, stop_dt, title in ch_items:
            entries.append((ch_name, start_dt, stop_dt, title))

    if not entries:
        return [], resolved_now, next_hour_end, "empty"
    return entries, resolved_now, next_hour_end, None


def get_channel_day_schedule_card(
    channel_num: int,
    config: TelkkariConfig,
    now: datetime | None = None,
    xml_root: ET.Element | None = None,
) -> tuple[str, Card]:
    """Fetch a channel's day schedule and return ``(fallback_text, card)``.

    The fallback text is identical to :func:`get_channel_day_schedule` so
    existing behaviour is preserved when photo delivery fails.
    """
    fallback_text = get_channel_day_schedule(channel_num, config, now=now, xml_root=xml_root)

    if channel_num not in FREE_CHANNELS:
        return fallback_text, build_unknown_channel_card(channel_num)

    ch_name, items, error = _collect_channel_day_items(channel_num, config, now=now, xml_root=xml_root)
    resolved_now = (now.astimezone(HELSINKI_TZ) if now is not None else datetime.now(HELSINKI_TZ))

    if error == "fetch_failed":
        return fallback_text, build_telkkari_error_card(fallback_text)
    if error == "empty" or not items:
        return fallback_text, build_channel_day_card(ch_name or f"Kanava {channel_num}", [], now=resolved_now)
    return fallback_text, build_channel_day_card(ch_name or f"Kanava {channel_num}", items, now=resolved_now)


def get_next_hour_schedule_card(
    config: TelkkariConfig,
    now: datetime | None = None,
    xml_root: ET.Element | None = None,
) -> tuple[str, Card]:
    """Fetch the next-hour overview and return ``(fallback_text, card)``.

    The fallback text is identical to :func:`get_next_hour_schedule` so
    existing behaviour is preserved when photo delivery fails.
    """
    fallback_text = get_next_hour_schedule(config, now=now, xml_root=xml_root)

    entries, resolved_now, resolved_end, error = _collect_next_hour_entries(config, now=now, xml_root=xml_root)

    if error == "fetch_failed":
        return fallback_text, build_telkkari_error_card(fallback_text)
    if error == "empty" or not entries:
        return fallback_text, build_next_hour_card([], now=resolved_now, next_hour_end=resolved_end)
    return fallback_text, build_next_hour_card(entries, now=resolved_now, next_hour_end=resolved_end)
