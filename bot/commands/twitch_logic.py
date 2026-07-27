from __future__ import annotations

import asyncio
import re
from bot.config import TwitchConfig
from bot.tasks.twitch_logic import TwitchClient

_TWITCH_CMD_RE = re.compile(r"^\s*!twitch(?:\s+|$)", re.IGNORECASE)


def parse_twitch_command(text: str | None) -> tuple[bool, str]:
    if not text:
        return False, ""
    match = _TWITCH_CMD_RE.search(text)
    if not match:
        return False, ""
    subcommand = text[match.end():].strip()
    return True, subcommand


async def fetch_twitch_status_reply(
    config: TwitchConfig,
    client: TwitchClient | None = None,
) -> str:
    if not config.is_configured:
        return "Twitch-integraatiota ei ole määritetty (.env missing TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET / TWITCH_CHANNELS)."

    client = client or TwitchClient(config)

    user_map = await asyncio.to_thread(client.get_user_ids, config.channels)
    if not user_map:
        return "Ei saatu haettua Twitch-kanavien tietoja."

    status_lines = ["<b>Seurattavat Twitch-kanavat:</b>\n"]

    for channel in config.channels:
        user_id = user_map.get(channel.lower())
        if not user_id:
            status_lines.append(f"❓ <b>{channel}</b>: Ei löydy Twitchistä")
            continue

        stream_info = await asyncio.to_thread(client.get_stream_info, user_id)
        if stream_info:
            game = stream_info.get("game_name", "")
            title = stream_info.get("title", "")
            url = f"https://twitch.tv/{channel}"
            status_lines.append(
                f"🔴 <b>{channel}</b> (LIVE)\n"
                f"   Peli: {game}\n"
                f"   Otsikko: {title}\n"
                f"   Linkki: {url}\n"
            )
        else:
            status_lines.append(f"⚪ <b>{channel}</b>: Offline (https://twitch.tv/{channel})")

    return "\n".join(status_lines)
