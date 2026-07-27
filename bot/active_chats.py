from __future__ import annotations

import logging
from pathlib import Path

from telegram import Update

from bot.storage import load_json_data, save_json_data

LOGGER = logging.getLogger(__name__)
ACTIVE_CHAT_IDS_FILE = "active_chat_ids.json"


def _active_chat_ids_path(storage_dir: Path) -> Path:
    return storage_dir / ACTIVE_CHAT_IDS_FILE


def load_active_chat_ids(storage_dir: Path) -> set[int]:
    path = _active_chat_ids_path(storage_dir)
    raw_data = load_json_data(path, default_factory=list)
    if not isinstance(raw_data, list):
        return set()

    chat_ids: set[int] = set()
    for value in raw_data:
        try:
            chat_ids.add(int(value))
        except (TypeError, ValueError):
            continue
    return chat_ids


def _store_active_chat_ids(storage_dir: Path, chat_ids: set[int]) -> None:
    path = _active_chat_ids_path(storage_dir)
    save_json_data(path, sorted(chat_ids))


def track_active_chat(update: Update, storage_dir: Path) -> None:
    chat = update.effective_chat
    if chat is None:
        return

    chat_id = chat.id
    active_chat_ids = load_active_chat_ids(storage_dir)
    if chat_id in active_chat_ids:
        return

    active_chat_ids.add(chat_id)
    try:
        _store_active_chat_ids(storage_dir, active_chat_ids)
    except OSError:
        LOGGER.exception("Failed to persist active chat id %s", chat_id)
