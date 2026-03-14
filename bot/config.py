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
    ai_backend_url: str
    ai_max_tokens: int
    ai_backend_timeout_seconds: int
    openweather_api_key: str
    weathercam_stations_url: str
    weathercam_image_base_url: str
    openweather_current_url: str
    weather_api_timeout_seconds: int
    digitraffic_user: str
    max_reply_length: int
    steam_cs2_rss_url: str
    steam_rss_poll_interval_seconds: int
    steam_rss_request_timeout_seconds: int
    stt_backend_url: str
    stt_timeout_seconds: int
    stt_max_audio_seconds: int
    mumble_host: str
    mumble_port: int
    mumble_username: str
    mumble_password: str
    mumble_connect_timeout_seconds: int
    mumble_status_wait_seconds: int
    mumble_monitor_interval_seconds: int

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

        return cls(
            telegram_bot_token=token,
            storage_dir=storage_dir,
            ai_backend_url=os.getenv("AI_BACKEND_URL", "http://127.0.0.1:8080/query").strip(),
            ai_max_tokens=int(os.getenv("AI_MAX_TOKENS", "650")),
            ai_backend_timeout_seconds=int(os.getenv("AI_BACKEND_TIMEOUT_SECONDS", "30")),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY", "").strip(),
            weathercam_stations_url=os.getenv(
                "WEATHERCAM_STATIONS_URL",
                "https://tie.digitraffic.fi/api/weathercam/v1/stations",
            ).strip(),
            weathercam_image_base_url=os.getenv(
                "WEATHERCAM_IMAGE_BASE_URL", "https://weathercam.digitraffic.fi"
            ).strip(),
            openweather_current_url=os.getenv(
                "OPENWEATHER_CURRENT_URL",
                "https://api.openweathermap.org/data/2.5/weather",
            ).strip(),
            weather_api_timeout_seconds=int(
                os.getenv("WEATHER_API_TIMEOUT_SECONDS", "30")
            ),
            digitraffic_user=os.getenv("DIGITRAFFIC_USER", "telegram-bot-1.0").strip(),
            max_reply_length=int(os.getenv("MAX_REPLY_LENGTH", "5000")),
            steam_cs2_rss_url=os.getenv(
                "STEAM_CS2_RSS_URL", "https://steamcommunity.com/games/csgo/rss/"
            ).strip(),
            steam_rss_poll_interval_seconds=int(
                os.getenv("STEAM_RSS_POLL_INTERVAL_SECONDS", "300")
            ),
            steam_rss_request_timeout_seconds=int(
                os.getenv("STEAM_RSS_REQUEST_TIMEOUT_SECONDS", "30")
            ),
            stt_backend_url=os.getenv(
                "STT_BACKEND_URL", "http://127.0.0.1:8081/transcribe"
            ).strip(),
            stt_timeout_seconds=int(os.getenv("STT_TIMEOUT_SECONDS", "30")),
            stt_max_audio_seconds=int(os.getenv("STT_MAX_AUDIO_SECONDS", "600")),
            mumble_host=os.getenv("MUMBLE_HOST", "127.0.0.1").strip(),
            mumble_port=int(os.getenv("MUMBLE_PORT", "64738")),
            mumble_username=os.getenv("MUMBLE_USERNAME", "telegram-status-bot").strip(),
            mumble_password=os.getenv("MUMBLE_PASSWORD", "").strip(),
            mumble_connect_timeout_seconds=int(
                os.getenv("MUMBLE_CONNECT_TIMEOUT_SECONDS", "10")
            ),
            mumble_status_wait_seconds=int(os.getenv("MUMBLE_STATUS_WAIT_SECONDS", "1")),
            mumble_monitor_interval_seconds=int(
                os.getenv("MUMBLE_MONITOR_INTERVAL_SECONDS", "10")
            ),
        )
