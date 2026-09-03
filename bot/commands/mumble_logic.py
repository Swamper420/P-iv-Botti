from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.config import MumbleConfig
    from bot.tasks.mumble_logic import MumbleManager, MumbleServerSnapshot, MumbleUserInfo


def parse_mumble_command(text: str) -> tuple[bool, str, str | None]:
    """
    Parses !mumble command text.
    Returns: (is_match, action, target)
    action can be: 'summary', 'stats', 'help'
    """
    if not text:
        return False, "", None

    match = re.match(r"(?i)^\s*!mumble(?:\s+(.*))?$", text.strip())
    if not match:
        return False, "", None

    args = (match.group(1) or "").strip()
    if not args:
        return True, "summary", None

    if args.lower() in ("help", "ohje", "?"):
        return True, "help", None

    # Check for '!mumble stats <user>'
    stats_match = re.match(r"(?i)^stats\s+(.+)$", args)
    if stats_match:
        return True, "stats", stats_match.group(1).strip()

    # Otherwise treat the argument as the username for stats
    return True, "stats", args


def format_duration(seconds: int | None) -> str:
    """Formats an elapsed duration in seconds into a friendly human-readable Finnish string."""
    if seconds is None or seconds < 0:
        return "tuntematon"

    if seconds < 60:
        return f"{seconds} s"

    if seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        if secs == 0:
            return f"{minutes} min"
        return f"{minutes} min {secs} s"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    if hours >= 24:
        days = hours // 24
        hours = hours % 24
        if hours == 0 and minutes == 0:
            return f"{days} pv"
        if hours == 0:
            return f"{days} pv {minutes} min"
        return f"{days} pv {hours} t {minutes} min"

    if minutes == 0:
        return f"{hours} t"
    return f"{hours} t {minutes} min"


def format_mumble_summary(
    snapshot: MumbleServerSnapshot, config: MumbleConfig
) -> str:
    """Formats the overview of users and channels on the Mumble server."""
    if not config.is_configured:
        return "⚠️ <b>Mumble ei ole käytössä.</b> Aseta <code>MUMBLE_HOST</code> .env-tiedostoon."

    if not snapshot.is_connected:
        server_str = f"{config.host}:{config.port}"
        err = f" ({snapshot.error_message})" if snapshot.error_message else ""
        return (
            f"⚠️ <b>Ei yhteyttä Mumble-palvelimeen</b> (<code>{server_str}</code>){err}. "
            "Yhdistetään uudelleen..."
        )

    server_display = html.escape(snapshot.server_name or config.host)
    active_users = snapshot.active_users
    total_count = len(active_users)

    lines = [
        f"🔊 <b>Mumble: {server_display}</b>",
        f"Käyttäjiä paikalla: <b>{total_count}</b>",
        "",
    ]

    if total_count == 0:
        lines.append("<i>Ei muita käyttäjiä kanavilla.</i>")
        return "\n".join(lines)

    # Group users by channel
    channels_with_users: dict[int, list[MumbleUserInfo]] = {}
    for user in active_users:
        channels_with_users.setdefault(user.channel_id, []).append(user)

    for ch_id, users in channels_with_users.items():
        ch_info = snapshot.channels.get(ch_id)
        channel_name = html.escape(ch_info.name if ch_info else f"Kanava {ch_id}")
        lines.append(f"📁 <b>{channel_name}</b>")

        for user in sorted(users, key=lambda u: u.name.lower()):
            icons = []
            if user.is_deafened or user.is_self_deafened:
                icons.append("🔕")
            elif user.is_muted or user.is_self_muted:
                icons.append("🔇")
            if user.is_recording:
                icons.append("🔴")
            if user.is_priority_speaker:
                icons.append("🌟")

            badge_str = f" {''.join(icons)}" if icons else ""
            online_str = format_duration(user.online_seconds)

            idle_str = ""
            if user.idle_seconds is not None and user.idle_seconds >= 60:
                idle_str = f" (idle {format_duration(user.idle_seconds)})"

            ping_part = ""
            if user.ping_ms is not None:
                ping_part = f" | 📶 {user.ping_ms:.0f} ms"

            lines.append(f"  • <b>{html.escape(user.name)}</b>{badge_str}")
            lines.append(f"    ⏱ {online_str}{idle_str}{ping_part}")

        lines.append("")

    return "\n".join(lines).rstrip()


def format_user_details(user: MumbleUserInfo, server_name: str) -> str:
    """Formats in-depth statistics for an individual Mumble user."""
    lines = [
        f"👤 <b>Mumble: {html.escape(user.name)}</b>",
        f"• Kanava: <b>{html.escape(user.channel_name)}</b>",
        f"• Paikallaoloaika: <b>{format_duration(user.online_seconds)}</b>",
    ]

    if user.idle_seconds is not None:
        if user.idle_seconds >= 60:
            lines.append(f"• Toimettomana (idle): <b>{format_duration(user.idle_seconds)}</b>")
        else:
            lines.append("• Toimettomana (idle): <b>aktiivinen</b>")

    if user.ping_ms is not None:
        lines.append(f"• Viive (ping): <b>{user.ping_ms:.1f} ms</b>")

    status_parts = []
    if user.is_deafened:
        status_parts.append("🔕 Kuulokkeet pois")
    elif user.is_muted:
        status_parts.append("🔇 Mykistetty")
    else:
        status_parts.append("🎙️ Äänessä")

    if user.is_recording:
        status_parts.append("🔴 Nauhoittaa")
    if user.is_priority_speaker:
        status_parts.append("🌟 Priority speaker")

    lines.append(f"• Tila: <b>{', '.join(status_parts)}</b>")

    if user.client_release or user.client_os:
        client_desc = (
            f"{user.client_release} ({user.client_os})".strip()
            if user.client_os
            else user.client_release
        )
        lines.append(f"• Asiakasohjelma: <b>{html.escape(client_desc)}</b>")

    if user.bandwidth_kbps is not None and user.bandwidth_kbps > 0:
        lines.append(f"• Kaistanleveys: <b>{user.bandwidth_kbps:.0f} kbps</b>")

    if user.packets_good > 0 or user.packets_lost > 0:
        total_p = user.packets_good + user.packets_lost
        loss_pct = (user.packets_lost / total_p * 100.0) if total_p > 0 else 0.0
        lines.append(f"• Paketit: <b>{user.packets_good} kpl</b> ({loss_pct:.1f}% hävikki)")

    return "\n".join(lines)


async def handle_mumble_command(
    manager: MumbleManager | None,
    config: MumbleConfig,
    command_text: str,
) -> str:
    """Main command handler executing parsed action and returning HTML reply."""
    is_match, action, target = parse_mumble_command(command_text)
    if not is_match:
        return ""

    if action == "help":
        return (
            "🔊 <b>Mumble-komennon käyttö:</b>\n\n"
            "• <code>!mumble</code> — Näytä Mumble-palvelimen käyttäjät, kanavat ja kesto\n"
            "• <code>!mumble &lt;käyttäjä&gt;</code> — Näytä tarkemmat tilastot tietystä käyttäjästä\n"
            "• <code>!mumble stats &lt;käyttäjä&gt;</code> — Sama kuin yllä"
        )

    if not config.is_configured:
        return "⚠️ <b>Mumble ei ole käytössä.</b> Aseta <code>MUMBLE_HOST</code> .env-tiedostoon."

    if manager is None:
        server_str = f"{config.host}:{config.port}"
        return (
            f"⚠️ <b>Mumble-taustapalvelu ei ole käynnissä</b> (<code>{server_str}</code>)."
        )

    # Refresh statistics
    refresh_timeout = min(config.stats_timeout_seconds, 2.0)
    await manager.refresh_stats(timeout=refresh_timeout)
    snapshot = manager.get_snapshot()

    if not snapshot.is_connected:
        return format_mumble_summary(snapshot, config)

    if action == "summary":
        return format_mumble_summary(snapshot, config)

    if action == "stats" and target:
        target_lower = target.strip().lower()
        active_users = snapshot.active_users

        # 1. Exact match
        for u in active_users:
            if u.name.lower() == target_lower:
                return format_user_details(u, snapshot.server_name)

        # 2. Case-insensitive substring match
        matches = [u for u in active_users if target_lower in u.name.lower()]
        if len(matches) == 1:
            return format_user_details(matches[0], snapshot.server_name)
        elif len(matches) > 1:
            names = ", ".join(f"<code>{html.escape(m.name)}</code>" for m in matches)
            return f"❓ Useampi käyttäjä vastaa hakua '<b>{html.escape(target)}</b>': {names}"

        return f"❌ Käyttäjää '<b>{html.escape(target)}</b>' ei löytynyt Mumble-palvelimelta."

    return format_mumble_summary(snapshot, config)
