from __future__ import annotations


def is_mumble_status_command(text: str | None) -> bool:
    if text is None:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    normalized = stripped.casefold()
    return normalized in {"mumble", "!mumble"}


def format_duration(seconds: int | None) -> str:
    if seconds is None or seconds < 0:
        return "ei saatavilla"

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_mumble_channel_notice(requested_by: str) -> str:
    requester = requested_by.strip() or "tuntematon käyttäjä"
    return f"📣 Telegramissa !mumble-komennon käytti: {requester}"


def format_mumble_status_report(
    *,
    server_address: str,
    channels: list[dict[str, object]],
) -> str:
    total_users = 0
    lines = [f"🎧 Mumble ({server_address})"]

    for channel in channels:
        channel_name = str(channel.get("name", "Tuntematon kanava"))
        users = channel.get("users", [])
        user_list = users if isinstance(users, list) else []
        total_users += len(user_list)
        lines.append(f"• {channel_name} ({len(user_list)})")

        if not user_list:
            continue

        for user in user_list:
            if not isinstance(user, dict):
                continue

            name = str(user.get("name", "Tuntematon käyttäjä"))
            online_seconds = user.get("online_seconds")
            muted = bool(user.get("muted"))
            deafened = bool(user.get("deafened"))

            lines.append(
                "  - "
                f"{name} | ⏱ {format_duration(online_seconds if isinstance(online_seconds, int) else None)}"
                f" | mute {'kyllä' if muted else 'ei'}"
                f" | deaf {'kyllä' if deafened else 'ei'}"
            )

    lines.insert(1, f"Kanavat: {len(channels)} | Käyttäjät: {total_users}")
    return "\n".join(lines)
