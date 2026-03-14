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


def format_mumble_status_report(
    *,
    server_address: str,
    channels: list[dict[str, object]],
    expected_channel_count: int,
) -> str:
    lines = [f"🎧 Mumble ({server_address})", f"Kanavia seurannassa: {len(channels)}"]

    if len(channels) < expected_channel_count:
        lines.append(
            f"⚠️ Kanavia löytyi vain {len(channels)} / {expected_channel_count}. Tarkista MUMBLE_TARGET_CHANNELS."
        )

    for channel in channels:
        channel_name = str(channel.get("name", "Tuntematon kanava"))
        users = channel.get("users", [])
        user_list = users if isinstance(users, list) else []
        lines.append(f"\n• {channel_name} ({len(user_list)} käyttäjää)")

        if not user_list:
            lines.append("  - Ei käyttäjiä")
            continue

        for user in user_list:
            if not isinstance(user, dict):
                continue

            name = str(user.get("name", "Tuntematon käyttäjä"))
            session = user.get("session")
            user_id = user.get("user_id")
            online_seconds = user.get("online_seconds")
            idle_seconds = user.get("idle_seconds")
            states = {
                "mute": bool(user.get("mute")),
                "deaf": bool(user.get("deaf")),
                "self_mute": bool(user.get("self_mute")),
                "self_deaf": bool(user.get("self_deaf")),
                "suppress": bool(user.get("suppress")),
                "recording": bool(user.get("recording")),
            }
            extras = user.get("extras", {})
            extra_details = (
                ", ".join(f"{key}={value}" for key, value in extras.items())
                if isinstance(extras, dict) and extras
                else "ei lisätietoja"
            )

            lines.append(
                f"  - {name} (session={session}, user_id={user_id})"
            )
            lines.append(
                "    online="
                f"{format_duration(online_seconds if isinstance(online_seconds, int) else None)}, "
                "idle="
                f"{format_duration(idle_seconds if isinstance(idle_seconds, int) else None)}"
            )
            lines.append(
                "    tila: "
                + ", ".join(f"{key}={'kyllä' if value else 'ei'}" for key, value in states.items())
            )
            lines.append(f"    extra: {extra_details}")

    return "\n".join(lines)
