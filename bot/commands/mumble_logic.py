from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

from bot.rendering import Badge, BadgeColor, Card

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


def build_mumble_summary_card(
    snapshot: MumbleServerSnapshot, config: MumbleConfig
) -> Card:
    """Builds a rich structured Card for Mumble server overview."""
    server_name = snapshot.server_name or config.host or "Mumble"
    server_host = f"{config.host}:{config.port}" if config.host else "Mumble"

    if not config.is_configured:
        return (
            Card(title="Mumble", subtitle=server_host, footer="P-iv-Botti Mumble")
            .set_badge("EI KÄYTÖSSÄ", BadgeColor.GRAY)
            .add_text("Mumble ei ole käytössä. Aseta MUMBLE_HOST .env-tiedostoon.", muted=True)
        )

    if not snapshot.is_connected:
        err_msg = f" ({snapshot.error_message})" if snapshot.error_message else ""
        return (
            Card(title=f"Mumble: {server_name}", subtitle=server_host, footer="P-iv-Botti Mumble")
            .set_badge("OFFLINE", BadgeColor.RED)
            .add_text(f"Ei yhteyttä Mumble-palvelimeen{err_msg}. Yhdistetään uudelleen...")
        )

    total_users = len(snapshot.active_users)
    badge_label = f"{total_users} ONLINE"
    badge_color = BadgeColor.GREEN if total_users > 0 else BadgeColor.GRAY

    card = (
        Card(
            title=f"Mumble: {server_name}",
            subtitle=server_host,
            footer=f"Kanavia: {len(snapshot.channels)} • P-iv-Botti Mumble",
        )
        .set_badge(badge_label, badge_color)
    )

    if total_users == 0:
        card.add_text("Ei käyttäjiä kanavilla.", muted=True)
        return card

    rows: list[list[str]] = []
    for u in sorted(snapshot.active_users, key=lambda x: x.name.lower()):
        tags: list[str] = []
        if u.is_deafened or u.is_self_deafened:
            tags.append("[DEAF]")
        elif u.is_muted or u.is_self_muted:
            tags.append("[MUTE]")
        if u.is_recording:
            tags.append("[REC]")
        if u.is_priority_speaker:
            tags.append("[PRIO]")

        tag_str = f" {' '.join(tags)}" if tags else ""
        ch_info = snapshot.channels.get(u.channel_id)
        ch_name = ch_info.name if ch_info else f"Kanava {u.channel_id}"

        time_str = format_duration(u.online_seconds)
        if u.idle_seconds is not None and u.idle_seconds >= 60:
            time_str += f" (idle {format_duration(u.idle_seconds)})"

        ping_str = f"{u.ping_ms:.0f} ms" if u.ping_ms is not None else "-"
        rows.append([f"{u.name}{tag_str}", ch_name, time_str, ping_str])

    card.add_table(
        headers=["Käyttäjä", "Kanava", "Aika", "Viive"],
        rows=rows,
        alignments=["left", "left", "left", "right"],
    )
    return card


def build_mumble_user_card(user: MumbleUserInfo, server_name: str) -> Card:
    """Builds a rich structured Card for a specific Mumble user."""
    card = Card(
        title=f"Mumble: {user.name}",
        subtitle=f"Kanava: {user.channel_name}",
        footer=f"Palvelin: {server_name} • P-iv-Botti Mumble",
    )

    if user.is_deafened:
        card.set_badge("KUULOKKEET POIS", BadgeColor.RED)
    elif user.is_muted:
        card.set_badge("MYKISTETTY", BadgeColor.YELLOW)
    elif user.is_recording:
        card.set_badge("NAUHOITTAA", BadgeColor.RED)
    else:
        card.set_badge("ONLINE", BadgeColor.GREEN)

    card.add_key_value("Paikallaoloaika", format_duration(user.online_seconds))
    if user.idle_seconds is not None and user.idle_seconds >= 60:
        card.add_key_value("Toimettomana (idle)", format_duration(user.idle_seconds))
    else:
        card.add_key_value("Toimettomana (idle)", "aktiivinen")

    ping_val = f"{user.ping_ms:.1f} ms" if user.ping_ms is not None else "-"
    card.add_key_value("Viive (ping)", ping_val)

    status_parts = []
    if user.is_deafened:
        status_parts.append("Kuulokkeet pois")
    elif user.is_muted:
        status_parts.append("Mykistetty")
    else:
        status_parts.append("Äänessä")
    if user.is_recording:
        status_parts.append("Nauhoittaa")
    if user.is_priority_speaker:
        status_parts.append("Priority")
    card.add_key_value("Tila", ", ".join(status_parts))

    card.add_divider()

    client_desc = (
        f"{user.client_release} ({user.client_os})".strip()
        if user.client_os
        else user.client_release
    ) or "-"
    card.add_key_value("Asiakasohjelma", client_desc)

    bw_str = (
        f"{user.bandwidth_kbps:.0f} kbps"
        if user.bandwidth_kbps and user.bandwidth_kbps > 0
        else "-"
    )
    card.add_key_value("Kaistanleveys", bw_str)

    if user.packets_good > 0 or user.packets_lost > 0:
        total_p = user.packets_good + user.packets_lost
        loss_pct = (user.packets_lost / total_p * 100.0) if total_p > 0 else 0.0
        card.add_key_value("Paketit", f"{user.packets_good} kpl ({loss_pct:.1f}% hävikki)")

    return card


async def handle_mumble_card_command(
    manager: MumbleManager | None,
    config: MumbleConfig,
    command_text: str,
) -> tuple[str, Card | None]:
    """Processes Mumble command and returns both (reply_text, card_or_none)."""
    is_match, action, target = parse_mumble_command(command_text)
    if not is_match:
        return "", None

    if action == "help":
        help_text = (
            "🔊 <b>Mumble-komennon käyttö:</b>\n\n"
            "• <code>!mumble</code> — Näytä Mumble-palvelimen käyttäjät, kanavat ja kesto\n"
            "• <code>!mumble &lt;käyttäjä&gt;</code> — Näytä tarkemmat tilastot tietystä käyttäjästä\n"
            "• <code>!mumble stats &lt;käyttäjä&gt;</code> — Sama kuin yllä"
        )
        help_card = (
            Card(title="Mumble Ohje", footer="P-iv-Botti Mumble")
            .set_badge("OHJE", BadgeColor.BLUE)
            .add_text("!mumble — Näytä Mumble-palvelimen käyttäjät ja kanavat")
            .add_text("!mumble <käyttäjä> — Näytä käyttäjän tarkat tilastot")
            .add_code_block("!mumble\n!mumble stats Teppo")
        )
        return help_text, help_card

    if not config.is_configured:
        msg = "⚠️ <b>Mumble ei ole käytössä.</b> Aseta <code>MUMBLE_HOST</code> .env-tiedostoon."
        card = (
            Card(title="Mumble", footer="P-iv-Botti Mumble")
            .set_badge("EI KÄYTÖSSÄ", BadgeColor.GRAY)
            .add_text("Mumble ei ole käytössä. Aseta MUMBLE_HOST .env-tiedostoon.")
        )
        return msg, card

    if manager is None:
        server_str = f"{config.host}:{config.port}"
        msg = f"⚠️ <b>Mumble-taustapalvelu ei ole käynnissä</b> (<code>{server_str}</code>)."
        card = (
            Card(title="Mumble", subtitle=server_str, footer="P-iv-Botti Mumble")
            .set_badge("OFFLINE", BadgeColor.RED)
            .add_text("Mumble-taustapalvelu ei ole käynnissä.")
        )
        return msg, card

    # Refresh statistics
    refresh_timeout = min(config.stats_timeout_seconds, 2.0)
    await manager.refresh_stats(timeout=refresh_timeout)
    snapshot = manager.get_snapshot()

    if not snapshot.is_connected:
        return format_mumble_summary(snapshot, config), build_mumble_summary_card(snapshot, config)

    if action == "summary":
        return format_mumble_summary(snapshot, config), build_mumble_summary_card(snapshot, config)

    if action == "stats" and target:
        target_lower = target.strip().lower()
        active_users = snapshot.active_users

        # 1. Exact match
        for u in active_users:
            if u.name.lower() == target_lower:
                return format_user_details(u, snapshot.server_name), build_mumble_user_card(u, snapshot.server_name)

        # 2. Case-insensitive substring match
        matches = [u for u in active_users if target_lower in u.name.lower()]
        if len(matches) == 1:
            return format_user_details(matches[0], snapshot.server_name), build_mumble_user_card(matches[0], snapshot.server_name)
        elif len(matches) > 1:
            names = ", ".join(f"<code>{html.escape(m.name)}</code>" for m in matches)
            plain_names = ", ".join(m.name for m in matches)
            reply_text = f"❓ Useampi käyttäjä vastaa hakua '<b>{html.escape(target)}</b>': {names}"
            card = (
                Card(title="Mumble Haku", subtitle=f"Hakusana: {target}", footer="P-iv-Botti Mumble")
                .set_badge("USEITA TULOKSIA", BadgeColor.YELLOW)
                .add_text(f"Useampi käyttäjä vastaa hakua:\n{plain_names}")
            )
            return reply_text, card

        not_found_text = f"❌ Käyttäjää '<b>{html.escape(target)}</b>' ei löytynyt Mumble-palvelimelta."
        card = (
            Card(title="Mumble Haku", subtitle=f"Hakusana: {target}", footer="P-iv-Botti Mumble")
            .set_badge("EI LÖYTYNYT", BadgeColor.RED)
            .add_text(f"Käyttäjää '{target}' ei löytynyt Mumble-palvelimelta.")
        )
        return not_found_text, card

    return format_mumble_summary(snapshot, config), build_mumble_summary_card(snapshot, config)


async def handle_mumble_command(
    manager: MumbleManager | None,
    config: MumbleConfig,
    command_text: str,
) -> str:
    """Main command handler executing parsed action and returning HTML reply."""
    text, _ = await handle_mumble_card_command(manager, config, command_text)
    return text
