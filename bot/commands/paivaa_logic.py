from __future__ import annotations

import json
from pathlib import Path

PAIVAA_HISTORY_FILE = "paivaa_recent_replies.json"
PAIVAA_HISTORY_SIZE = 4


def get_paivaa_reply(text: str | None) -> str | None:
    if text is None:
        return None

    if text.strip().casefold() == "päivää".casefold():
        return "Päivää *tips fedora*"

    return None


def _paivaa_history_path(storage_dir: Path) -> Path:
    return storage_dir / PAIVAA_HISTORY_FILE


def load_recent_paivaa_replies(storage_dir: Path) -> list[str]:
    history_path = _paivaa_history_path(storage_dir)
    if not history_path.exists():
        return []

    try:
        raw_data = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(raw_data, list):
        return []

    replies = [value.strip() for value in raw_data if isinstance(value, str) and value.strip()]
    return replies[-PAIVAA_HISTORY_SIZE:]


def store_recent_paivaa_reply(storage_dir: Path, reply: str) -> None:
    history = load_recent_paivaa_replies(storage_dir)
    history.append(reply.strip())
    trimmed_history = [item for item in history if item][-PAIVAA_HISTORY_SIZE:]
    _paivaa_history_path(storage_dir).write_text(
        json.dumps(trimmed_history, ensure_ascii=False), encoding="utf-8"
    )


def build_paivaa_ai_prompt(recent_replies: list[str]) -> str:
    avoid_section = "\n".join(f"- {reply}" for reply in recent_replies)
    if not avoid_section:
        avoid_section = "- (none yet)"

    return (
        "Vastaa käyttäjän viestiin 'päivää'. "
        "Kirjoita yksi ainoa erittäin cringe, ylitsevuotavan uwu-tyylinen suomenkielinen tervehdys. "
        "Ei selityksiä, ei markdownia, ei lainausmerkkejä, ei listoja.\n"
        "Älä toista mitään näistä aiemmista vastauksista täsmälleen:\n"
        f"{avoid_section}"
    )


def ensure_unique_paivaa_reply(reply: str, recent_replies: list[str]) -> str:
    candidate = reply.strip() or "Päiwää~ uwu nyaa >w<"
    candidate = candidate.splitlines()[0].strip() or "Päiwää~ uwu nyaa >w<"

    if candidate not in recent_replies:
        return candidate

    suffix_count = 1
    unique_candidate = candidate
    while unique_candidate in recent_replies:
        unique_candidate = f"{candidate} {'~' * suffix_count} uwu"
        suffix_count += 1

    return unique_candidate
