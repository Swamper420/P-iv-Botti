from __future__ import annotations


def is_mumble_status_command(text: str | None) -> bool:
    if text is None:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    normalized = stripped.casefold()
    return normalized in {"mumble", "!mumble"}


def extract_mumble_tele_message(text: str | None) -> str | None:
    if text is None:
        return None

    stripped = text.strip()
    if not stripped:
        return None

    prefix = "!tele"
    if not stripped.casefold().startswith(prefix):
        return None

    message = stripped[len(prefix) :].strip()
    if not message:
        return None
    return message


def format_duration(seconds: int | None) -> str:
    if seconds is None or seconds < 0:
        return "ei saatavilla"

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_mumble_channel_notice(requested_by: str) -> str:
    requester = requested_by.strip() or "tuntematon käyttäjä"
    return f"📣 Telegramissa !mumble-komennon käytti: {requester}"


def build_mumble_user_tracking_key(user: dict[str, object]) -> str | None:
    session = user.get("session")
    if isinstance(session, int):
        return f"session:{session}"

    name = user.get("name")
    if isinstance(name, str):
        normalized_name = name.strip().casefold()
        if normalized_name:
            return f"name:{normalized_name}"

    return None


def update_mumble_connection_tracker(
    *,
    users: list[dict[str, object]],
    own_session: object,
    connected_since_by_key: dict[str, float],
    now_monotonic: float,
) -> None:
    active_keys: set[str] = set()
    for user in users:
        if own_session is not None and user.get("session") == own_session:
            continue
        tracking_key = build_mumble_user_tracking_key(user)
        if tracking_key is None:
            continue
        active_keys.add(tracking_key)
        connected_since_by_key.setdefault(tracking_key, now_monotonic)

    stale_keys = set(connected_since_by_key) - active_keys
    for stale_key in stale_keys:
        connected_since_by_key.pop(stale_key, None)


def resolve_online_seconds(
    *,
    user: dict[str, object],
    connected_since_by_key: dict[str, float | int],
    now_monotonic: float,
) -> int | None:
    tracking_key = build_mumble_user_tracking_key(user)
    if tracking_key is not None:
        connected_at = connected_since_by_key.get(tracking_key)
        if isinstance(connected_at, (float, int)):
            return max(0, int(now_monotonic - float(connected_at)))

    fallback = user.get("onlinesecs")
    if isinstance(fallback, int) and fallback >= 0:
        return fallback
    return None


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
            online = format_duration(online_seconds if isinstance(online_seconds, int) else None)
            mute_status = "🔇" if muted else "🎤"
            deaf_status = "🙉" if deafened else "👂"

            lines.append(
                f"  - {name} | ⏱ {online} | {mute_status} | {deaf_status}"
            )

    lines.insert(1, f"Kanavat: {len(channels)} | Käyttäjät: {total_users}")
    return "\n".join(lines)
