from __future__ import annotations

from datetime import date
import html
import re

from bot.config import MaratonConfig
from bot.rendering import BadgeColor, Card, render_card
from bot.tasks._maraton_render import (
    render_maraton_briefing_image,
    render_maraton_legs_image,
    render_maraton_week_image,
)
from bot.tasks.maraton_logic import (
    RELAY_LEGS,
    WEEKDAY_NAMES_FI,
    build_maraton_briefing,
    build_maraton_card,
    format_finnish_date,
    format_maraton_text,
    select_phase,
)


def parse_maraton_command(text: str) -> tuple[bool, str]:
    """Parses !maraton command text into (is_match, action).

    Actions: 'status', 'legs', 'week', 'help'
    """
    if not text:
        return False, ""

    match = re.match(r"(?i)^\s*!maraton(?:\s+(.*))?$", text.strip())
    if not match:
        return False, ""

    args = (match.group(1) or "").strip().lower()
    if not args:
        return True, "status"

    if args in ("osuudet", "osuus", "viesti", "reitti", "legs"):
        return True, "legs"

    if args in ("treeni", "vinkki", "viikko", "ohjelma", "plan"):
        return True, "week"

    if args in ("help", "ohje", "?"):
        return True, "help"

    return True, "status"


def _build_legs_response(config: MaratonConfig, today: date | None = None) -> tuple[str, Card]:
    briefing = build_maraton_briefing(config, today=today)
    date_str = format_finnish_date(briefing.event_date)

    text_lines = [
        f"🏃 <b>TAMPERE MARATON 2027 — MARATONVIESTIN OSUUDET ({date_str})</b>",
        "Yhteensä <b>42,195 km</b> (4 juoksijaa, Ratinan stadion):",
        "",
    ]
    rows: list[list[str]] = []
    for leg in RELAY_LEGS:
        text_lines.append(
            f"• <b>{leg.number}. osuus — {leg.distance_km}</b> ({html.escape(leg.role)}, {html.escape(leg.loops)})\n"
            f"  ⚠️ {html.escape(leg.danger_note)}.\n"
            f"  💡 {html.escape(leg.tactical_tip)}"
        )
        rows.append([f"{leg.number}. {leg.role}", leg.distance_km, leg.loops])

    card = (
        Card(
            title="MARATONVIESTIN OSUUDET",
            subtitle=f"TAMPERE MARATON 2027 • {date_str} • 42,195 KM",
            accent_color=briefing.accent_color,
            footer="Valitkaa 14,1 km kolmososuudelle joukkueen kovin kestävyyskone.",
        )
        .set_badge(briefing.badge_text, briefing.badge_color)
        .add_table(
            headers=["Osuus", "Matka", "Reitti"],
            rows=rows,
            alignments=["left", "right", "left"],
        )
        .add_divider()
    )
    for leg in RELAY_LEGS:
        card.add_text(
            f"{leg.number}. osuus ({leg.distance_km}): {leg.danger_note}. {leg.tactical_tip}",
            muted=False if leg.number == 3 else True,
            bold=leg.number == 3,
        )

    return "\n".join(text_lines), card


def _build_week_response(config: MaratonConfig, today: date | None = None) -> tuple[str, Card]:
    briefing = build_maraton_briefing(config, today=today)
    date_str = format_finnish_date(briefing.event_date)
    phase = select_phase(max(1, briefing.days_remaining))

    text_lines = [
        f"📅 <b>VIIKKOTREENIOHJELMA — {html.escape(phase.name)}</b>",
        f"⏱ Tampere Maraton 2027 ({date_str}): <b>{max(0, briefing.days_remaining)} pv jäljellä</b>",
        "",
    ]
    rows: list[list[str]] = []
    for day_name, workout in zip(WEEKDAY_NAMES_FI, phase.weekly_workouts):
        marker = "👉 " if day_name == briefing.weekday_name else "• "
        text_lines.append(f"{marker}<b>{day_name}:</b> {html.escape(workout)}")
        day_label = f"{day_name} *" if day_name == briefing.weekday_name else day_name
        rows.append([day_label, workout])

    text_lines.append("")
    text_lines.append(f"🧠 <b>Valmentajan vinkki:</b> {html.escape(briefing.coach_tip)}")

    card = (
        Card(
            title="VIIKON TREENIOHJELMA",
            subtitle=f"{phase.name.upper()} • {date_str}",
            accent_color=briefing.accent_color,
            footer="* = tämän päivän harjoitus • Päivittäinen muistutus klo " + config.send_time,
        )
        .set_badge(briefing.badge_text, briefing.badge_color)
        .add_text(briefing.headline, bold=True, color="#f4f4f5")
        .add_divider()
    )
    for day_name, workout in zip(WEEKDAY_NAMES_FI, phase.weekly_workouts):
        is_today = day_name == briefing.weekday_name
        prefix = f"> {day_name.upper()} (TÄNÄÄN): " if is_today else f"{day_name}: "
        card.add_text(
            f"{prefix}{workout}",
            bold=is_today,
            color=briefing.accent_color if is_today else None,
            muted=not is_today,
        )

    card.add_divider()
    card.add_text(f"Valmentajan vinkki: {briefing.coach_tip}", muted=True)

    return "\n".join(text_lines), card


def _build_help_response(config: MaratonConfig) -> tuple[str, Card]:
    help_text = (
        "🏃 <b>Tampere Maraton 2027 — Maratonviesti</b>\n\n"
        "• <code>!maraton</code> — Päivän laskuri, pelottelu ja päivän treeniohje\n"
        "• <code>!maraton osuudet</code> — 4-osuuksisen viestin matkat ja taktiikka\n"
        "• <code>!maraton treeni</code> — Koko viikon treeniohjelma nykyiselle vaiheelle\n"
        f"• Päivittäinen automaattimuistutus: klo <b>{html.escape(config.send_time)}</b> ({html.escape(config.timezone)})"
    )
    help_card = (
        Card(
            title="MARATON-KOMENNOT",
            subtitle=f"TAMPERE MARATON 2027 • {config.event_date}",
            accent_color="#ef4444",
            footer=f"Päivittäinen muistutus klo {config.send_time} ({config.timezone})",
        )
        .set_badge("OHJE", BadgeColor.BLUE)
        .add_text("!maraton — Päivän laskuri, tsemppi ja päivän treeni")
        .add_text("!maraton osuudet — Maratonviestin osuudet (11,2 / 7,0 / 14,1 / 9,9 km)")
        .add_text("!maraton treeni — Koko viikon harjoitusohjelma nykyiseen vaiheeseen")
        .add_code_block("!maraton\n!maraton osuudet\n!maraton treeni")
    )
    return help_text, help_card


def handle_maraton_card_command(
    config: MaratonConfig,
    command_text: str,
    today: date | None = None,
) -> tuple[str, Card | None]:
    """Processes !maraton command and returns (html_text, card_or_none)."""
    is_match, action = parse_maraton_command(command_text)
    if not is_match:
        return "", None

    if action == "help":
        return _build_help_response(config)
    if action == "legs":
        return _build_legs_response(config, today=today)
    if action == "week":
        return _build_week_response(config, today=today)

    briefing = build_maraton_briefing(config, today=today)
    return format_maraton_text(briefing), build_maraton_card(briefing)


def render_maraton_command_image(
    config: MaratonConfig,
    command_text: str,
    today: date | None = None,
) -> tuple[str, bytes | None]:
    """Processes !maraton command and returns (html_fallback_text, png_bytes_or_none)."""
    is_match, action = parse_maraton_command(command_text)
    if not is_match:
        return "", None

    if action == "help":
        text, card = _build_help_response(config)
        return text, render_card(card)

    briefing = build_maraton_briefing(config, today=today)
    if action == "legs":
        text, _ = _build_legs_response(config, today=today)
        return text, render_maraton_legs_image(briefing)

    if action == "week":
        text, _ = _build_week_response(config, today=today)
        return text, render_maraton_week_image(config, briefing)

    return format_maraton_text(briefing), render_maraton_briefing_image(briefing)
