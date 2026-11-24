from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime, timezone
from typing import Any

from bot.config import TwitchConfig
from bot.tasks.twitch_logic import TwitchClient

_TWITCH_CMD_RE = re.compile(r"^\s*!twitch(?:\s+|$)", re.IGNORECASE)
_TWITCH_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?twitch\.tv/([A-Za-z0-9_]+)", re.IGNORECASE)


def parse_twitch_command(text: str | None) -> tuple[bool, str]:
    if not text:
        return False, ""
    match = _TWITCH_CMD_RE.search(text)
    if not match:
        return False, ""
    subcommand = text[match.end():].strip()
    return True, subcommand


def normalize_twitch_channel(raw: str) -> str:
    """Extracts a clean Twitch channel login from a username, @mention, or URL."""
    token = raw.strip().split()[0] if raw.strip() else ""
    url_match = _TWITCH_URL_RE.match(token)
    if url_match:
        return url_match.group(1).lower()
    return token.lstrip("@").lower()


def _format_uptime(started_at: str) -> str:
    if not started_at:
        return ""
    try:
        started_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError:
        return ""
    seconds = max(0, int((datetime.now(timezone.utc) - started_dt).total_seconds()))
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _format_live_channel(channel: str, stream_info: dict[str, Any]) -> str:
    display_name = html.escape(str(stream_info.get("user_name") or channel))
    game = html.escape(str(stream_info.get("game_name", "")))
    title = html.escape(str(stream_info.get("title", "")))
    url = f"https://twitch.tv/{html.escape(channel)}"

    lines = [f"🔴 <b>{display_name}</b> (LIVE)"]
    if game:
        lines.append(f"   Peli: {game}")
    if title:
        lines.append(f"   Otsikko: {title}")
    viewer_count = stream_info.get("viewer_count")
    if isinstance(viewer_count, int) and not isinstance(viewer_count, bool):
        lines.append(f"   Katsojat: {viewer_count:,}")
    uptime = _format_uptime(str(stream_info.get("started_at", "")))
    if uptime:
        lines.append(f"   Kesto: {uptime}")
    lines.append(f"   Linkki: {url}")
    return "\n".join(lines) + "\n"


def _format_offline_channel(channel: str) -> str:
    safe_channel = html.escape(channel)
    return f"⚪ <b>{safe_channel}</b>: Offline (https://twitch.tv/{safe_channel})"


async def fetch_twitch_status_reply(
    config: TwitchConfig,
    client: TwitchClient | None = None,
    channel_name: str | None = None,
) -> str:
    target_channel = normalize_twitch_channel(channel_name) if channel_name else ""

    if target_channel:
        if not (config.client_id and config.client_secret):
            return "Twitch-integraatiota ei ole määritetty (.env missing TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET)."
    elif not config.is_configured:
        return "Twitch-integraatiota ei ole määritetty (.env missing TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET / TWITCH_CHANNELS)."

    client = client or TwitchClient(config)
    channels = (target_channel,) if target_channel else config.channels

    user_map = await asyncio.to_thread(client.get_user_ids, channels)
    if target_channel:
        user_id = user_map.get(target_channel)
        if not user_id:
            return f"❓ <b>{html.escape(target_channel)}</b>: Ei löydy Twitchistä"

        stream_info = await asyncio.to_thread(client.get_stream_info, user_id)
        if stream_info:
            return _format_live_channel(target_channel, stream_info)
        return _format_offline_channel(target_channel)

    if not user_map:
        return "Ei saatu haettua Twitch-kanavien tietoja."

    status_lines = ["<b>Seurattavat Twitch-kanavat:</b>\n"]

    for channel in channels:
        user_id = user_map.get(channel.lower())
        if not user_id:
            status_lines.append(f"❓ <b>{html.escape(channel)}</b>: Ei löydy Twitchistä")
            continue

        stream_info = await asyncio.to_thread(client.get_stream_info, user_id)
        if stream_info:
            status_lines.append(_format_live_channel(channel, stream_info))
        else:
            status_lines.append(_format_offline_channel(channel))

    return "\n".join(status_lines)
