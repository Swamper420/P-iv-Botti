from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        os.environ.setdefault(key, value.strip())


@dataclass(frozen=True)
class BotConfig:
    telegram_bot_token: str
    storage_dir: Path

    @classmethod
    def from_environment(cls) -> "BotConfig":
        project_root = Path(__file__).resolve().parent.parent
        _load_env_file(project_root / ".env")

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        storage_dir = Path(
            os.getenv("STORAGE_DIR", str(project_root / "storage"))
        ).resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)

        return cls(telegram_bot_token=token, storage_dir=storage_dir)
