from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def load_json_data(
    file_path: Path,
    default_factory: Callable[[], T] | None = None,
) -> T | Any:
    if not file_path.exists():
        return default_factory() if default_factory else None

    try:
        content = file_path.read_text(encoding="utf-8")
        return json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Failed to load JSON data from %s: %s", file_path, exc)
        return default_factory() if default_factory else None


def save_json_data(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(file_path)
    except OSError:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        LOGGER.exception("Failed to save JSON data to %s", file_path)
        raise
