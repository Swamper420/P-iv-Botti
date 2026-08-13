from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from bot.storage import load_json_data, save_json_data

REMINDERS_FILE = "reminders.json"

WEEKDAY_MAP = {
    "maanantai": 0, "maanantaina": 0, "ma": 0,
    "tiistai": 1, "tiistaina": 1, "ti": 1,
    "keskiviikko": 2, "keskiviikkona": 2, "ke": 2,
    "torstai": 3, "torstaina": 3, "to": 3,
    "perjantai": 4, "perjantaina": 4, "pe": 4,
    "lauantai": 5, "lauantaina": 5, "la": 5,
    "sunnuntai": 6, "sunnuntaina": 6, "su": 6,
}


def _reminders_path(storage_dir: Path) -> Path:
    return storage_dir / REMINDERS_FILE


def load_all_reminders(storage_dir: Path) -> list[dict[str, Any]]:
    data = load_json_data(_reminders_path(storage_dir), default_factory=list)
    if isinstance(data, list):
        return data
    return []


def save_all_reminders(storage_dir: Path, reminders: list[dict[str, Any]]) -> None:
    save_json_data(_reminders_path(storage_dir), reminders)


def parse_relative_time(token: str, now: datetime) -> datetime | None:
    match = re.match(r"^\+?(\d+)\s*(s|sec|m|min|t|h|d|pv)$", token, re.IGNORECASE)
    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2).lower()

    if unit in ("s", "sec"):
        return now + timedelta(seconds=amount)
    elif unit in ("m", "min"):
        return now + timedelta(minutes=amount)
    elif unit in ("t", "h"):
        return now + timedelta(hours=amount)
    elif unit in ("d", "pv"):
        return now + timedelta(days=amount)

    return None


def parse_finnish_date(token: str, time_obj: datetime, now: datetime) -> datetime | None:
    token_clean = token.lower().strip()

    if token_clean in ("tänään", "tanaan"):
        target_date = now.date()
    elif token_clean in ("huomenna", "huomen"):
        target_date = (now + timedelta(days=1)).date()
    elif token_clean in ("ylihuomenna", "yli-huomenna"):
        target_date = (now + timedelta(days=2)).date()
    elif token_clean in WEEKDAY_MAP:
        target_weekday = WEEKDAY_MAP[token_clean]
        days_ahead = (target_weekday - now.weekday()) % 7
        if days_ahead == 0:
            candidate_dt = now.replace(
                hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0
            )
            if candidate_dt <= now:
                days_ahead = 7
        target_date = (now + timedelta(days=days_ahead)).date()
    else:
        # Check DD.MM.YYYY or DD.MM.
        match_full = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", token_clean)
        match_short = re.match(r"^(\d{1,2})\.(\d{1,2})\.?$", token_clean)

        if match_full:
            day, month, year = map(int, match_full.groups())
            try:
                target_date = datetime(year, month, day).date()
            except ValueError:
                return None
        elif match_short:
            day, month = map(int, match_short.groups())
            year = now.year
            try:
                candidate = datetime(year, month, day, time_obj.hour, time_obj.minute, tzinfo=now.tzinfo)
                if candidate <= now:
                    year += 1
                target_date = datetime(year, month, day).date()
            except ValueError:
                return None
        else:
            return None

    return datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        time_obj.hour,
        time_obj.minute,
        0,
        0,
        tzinfo=now.tzinfo,
    )


def parse_reminder_args(
    args_text: str,
    now: datetime | None = None,
) -> tuple[datetime | None, list[str], str, str | None]:
    """
    Parses command arguments for !muistuta.
    Returns (due_at, target_users, message_text, error_message).
    """
    if now is None:
        now = datetime.now().astimezone()

    tokens = args_text.strip().split()
    if not tokens:
        return None, [], "", "Käyttö: !muistuta <aika> [päivä] [@käyttäjä] [viesti]"

    first_token = tokens[0]
    due_at: datetime | None = None
    consumed_tokens = 1

    # Check relative time first (e.g. +15m, 15min, 2h)
    rel_due = parse_relative_time(first_token, now)
    if rel_due is not None:
        due_at = rel_due
    else:
        # Check HH:MM or H:MM
        time_match = re.match(r"^(\d{1,2})[:.](\d{2})$", first_token)
        if not time_match:
            return (
                None,
                [],
                "",
                "Virheellinen aika. Anna aika muodossa HH:MM (esim. 21:50) tai suhteellisena (+15m, 2h).",
            )

        hour, minute = map(int, time_match.groups())
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None, [], "", "Virheellinen aika (tunnit 0-23, minutit 0-59)."

        temp_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # Check if second token is a Finnish date
        if len(tokens) > 1:
            date_due = parse_finnish_date(tokens[1], temp_time, now)
            if date_due is not None:
                due_at = date_due
                consumed_tokens = 2

        if due_at is None:
            # Default date logic: today if in future, tomorrow if in past
            if temp_time > now:
                due_at = temp_time
            else:
                due_at = temp_time + timedelta(days=1)

    remaining_tokens = tokens[consumed_tokens:]
    target_users: list[str] = []
    message_parts: list[str] = []

    for token in remaining_tokens:
        if token.startswith("@") and len(token) > 1:
            target_users.append(token)
        message_parts.append(token)

    message_text = " ".join(message_parts).strip()
    if not message_text:
        message_text = "Muistutus!"

    return due_at, target_users, message_text, None


def add_reminder(
    storage_dir: Path,
    chat_id: int,
    creator: str,
    due_at: datetime,
    targets: list[str],
    message: str,
    media: dict[str, str] | None = None,
    max_per_chat: int = 50,
) -> tuple[dict[str, Any] | None, str]:
    reminders = load_all_reminders(storage_dir)

    chat_reminders = [r for r in reminders if r.get("chat_id") == chat_id]
    if len(chat_reminders) >= max_per_chat:
        return None, f"Virhe: Chatissa on jo enimmäismäärä muistutuksia ({max_per_chat})."

    next_id = max([int(r.get("id", 0)) for r in reminders], default=0) + 1
    now_iso = datetime.now().astimezone().isoformat()
    due_iso = due_at.isoformat()

    new_item = {
        "id": next_id,
        "chat_id": chat_id,
        "creator": creator,
        "created_at": now_iso,
        "due_at": due_iso,
        "targets": targets,
        "message": message,
        "media": media,
    }

    reminders.append(new_item)
    save_all_reminders(storage_dir, reminders)

    formatted_time = due_at.strftime("%d.%m.%Y klo %H:%M")
    diff_seconds = int((due_at - datetime.now().astimezone()).total_seconds())
    if diff_seconds < 60:
        time_desc = "hetken kuluttua"
    elif diff_seconds < 3600:
        mins = diff_seconds // 60
        time_desc = f"noin {mins} min kuluttua"
    else:
        hours = diff_seconds // 3600
        mins = (diff_seconds % 3600) // 60
        if mins > 0:
            time_desc = f"noin {hours}t {mins}min kuluttua"
        else:
            time_desc = f"noin {hours}t kuluttua"

    reply = f"⏰ Muistutus #{next_id} asetettu: {formatted_time} ({time_desc})."
    return new_item, reply


def list_reminders(storage_dir: Path, chat_id: int) -> str:
    reminders = load_all_reminders(storage_dir)
    chat_reminders = [r for r in reminders if r.get("chat_id") == chat_id]

    if not chat_reminders:
        return "Ei aktiivisia muistutuksia tässä chatissa."

    chat_reminders.sort(key=lambda r: r.get("due_at", ""))
    lines = ["⏰ **Aktiiviset muistutukset:**"]

    for r in chat_reminders:
        rid = r.get("id")
        due_str = r.get("due_at", "")
        try:
            dt = datetime.fromisoformat(due_str)
            due_fmt = dt.strftime("%d.%m.%Y klo %H:%M")
        except (ValueError, TypeError):
            due_fmt = due_str

        msg = r.get("message", "")
        if len(msg) > 40:
            msg = msg[:37] + "..."

        targets = r.get("targets", [])
        target_str = f" ({', '.join(targets)})" if targets else ""
        media_str = " 📷[media]" if r.get("media") else ""

        lines.append(f"• #{rid}: {due_fmt}{target_str}{media_str} — {msg}")

    return "\n".join(lines)


def cancel_reminder(storage_dir: Path, chat_id: int, reminder_id: int) -> str:
    reminders = load_all_reminders(storage_dir)
    new_reminders = []
    found = False

    for r in reminders:
        if r.get("chat_id") == chat_id and r.get("id") == reminder_id:
            found = True
        else:
            new_reminders.append(r)

    if not found:
        return f"Muistutusta #{reminder_id} ei löytynyt tästä chatista."

    save_all_reminders(storage_dir, new_reminders)
    return f"🗑️ Muistutus #{reminder_id} poistettu."


def get_due_reminders(storage_dir: Path, now: datetime | None = None) -> list[dict[str, Any]]:
    if now is None:
        now = datetime.now().astimezone()

    reminders = load_all_reminders(storage_dir)
    due_items = []

    for r in reminders:
        due_str = r.get("due_at", "")
        try:
            dt = datetime.fromisoformat(due_str)
            if dt <= now:
                due_items.append(r)
        except (ValueError, TypeError):
            continue

    return due_items


def remove_reminders(storage_dir: Path, reminder_ids: set[int]) -> None:
    if not reminder_ids:
        return
    reminders = load_all_reminders(storage_dir)
    new_reminders = [r for r in reminders if r.get("id") not in reminder_ids]
    save_all_reminders(storage_dir, new_reminders)
