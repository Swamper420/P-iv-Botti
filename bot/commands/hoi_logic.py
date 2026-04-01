from __future__ import annotations

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
HOI_FILE = "hoi_lists.json"


def _get_path(storage_dir: Path) -> Path:
    return storage_dir / HOI_FILE


def _load_data(storage_dir: Path) -> dict[str, dict[str, list[str]]]:
    path = _get_path(storage_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        LOGGER.exception("Failed to load hoi lists from %s", path)
    return {}


def _save_data(storage_dir: Path, data: dict[str, dict[str, list[str]]]) -> None:
    path = _get_path(storage_dir)
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        LOGGER.exception("Failed to save hoi lists to %s", path)


def list_all(storage_dir: Path, chat_id: int) -> str:
    data = _load_data(storage_dir)
    chat_data = data.get(str(chat_id), {})

    if not chat_data:
        return "Tässä chatissa ei ole vielä yhtään listaa. Luo uusi lista komennolla: !hoi @kayttaja listanimi"

    lines = ["**Tallennetut listat:**"]
    for list_name, users in sorted(chat_data.items()):
        lines.append(f"• {list_name}: {', '.join(users)}")
    return "\n".join(lines)


def ping_list(storage_dir: Path, chat_id: int, list_name: str) -> str:
    data = _load_data(storage_dir)
    list_name_lower = list_name.lower()

    chat_data = data.get(str(chat_id), {})
    actual_name = next((k for k in chat_data if k.lower() == list_name_lower), None)

    if not actual_name or not chat_data[actual_name]:
        return f"Listaa '{list_name}' ei löytynyt tai se on tyhjä."

    return f"Huomio {actual_name}! {' '.join(chat_data[actual_name])}"


def add_users(storage_dir: Path, chat_id: int, list_name: str, users: list[str]) -> str:
    data = _load_data(storage_dir)
    chat_id_str = str(chat_id)
    if chat_id_str not in data:
        data[chat_id_str] = {}

    chat_data = data[chat_id_str]
    list_name_lower = list_name.lower()

    # Find existing case-insensitive list or use the provided name
    actual_name = next((k for k in chat_data if k.lower() == list_name_lower), list_name)

    if actual_name not in chat_data:
        chat_data[actual_name] = []

    added = []
    for user in users:
        # QOL: Automatically append @ if missing
        if not user.startswith("@"):
            user = f"@{user}"
        if user not in chat_data[actual_name]:
            chat_data[actual_name].append(user)
            added.append(user)

    _save_data(storage_dir, data)

    if not added:
        return f"Kaikki mainitut käyttäjät olivat jo listalla '{actual_name}'."
    return f"Lisätty {', '.join(added)} listalle '{actual_name}'."


def remove_users(storage_dir: Path, chat_id: int, list_name: str, users: list[str]) -> str:
    data = _load_data(storage_dir)
    chat_id_str = str(chat_id)

    chat_data = data.get(chat_id_str, {})
    list_name_lower = list_name.lower()
    actual_name = next((k for k in chat_data if k.lower() == list_name_lower), None)

    if not actual_name:
        return f"Listaa '{list_name}' ei löytynyt."

    removed = []
    for user in users:
        if not user.startswith("@"):
            user = f"@{user}"
        if user in chat_data[actual_name]:
            chat_data[actual_name].remove(user)
            removed.append(user)

    msg_suffix = ""
    # QOL: Auto-delete list if empty
    if not chat_data[actual_name]:
        del chat_data[actual_name]
        msg_suffix = f"\nLista '{actual_name}' on nyt tyhjä ja poistettiin automaattisesti."

    _save_data(storage_dir, data)

    if not removed:
        return f"Ketään ei poistettu. Käyttäjiä ei löytynyt listalta '{actual_name}'."

    return f"Poistettu {', '.join(removed)} listalta '{actual_name}'.{msg_suffix}"
