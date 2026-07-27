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
    naama_model_name: str = "yolo26n-seg.pt"
    naama_confidence_threshold: float = 0.15
    naama_mask_threshold: float = 0.35
    naama_max_image_bytes: int = 10_000_000

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
        naama_confidence_threshold = float(
            os.getenv("NAAMA_CONFIDENCE_THRESHOLD", "0.15")
        )
        if not 0 <= naama_confidence_threshold <= 1:
            raise ValueError("NAAMA_CONFIDENCE_THRESHOLD must be between 0 and 1")
        naama_mask_threshold = float(
            os.getenv("NAAMA_MASK_THRESHOLD", "0.35")
        )
        if not 0 <= naama_mask_threshold <= 1:
            raise ValueError("NAAMA_MASK_THRESHOLD must be between 0 and 1")
        naama_max_image_bytes = int(
            os.getenv("NAAMA_MAX_IMAGE_BYTES", "10000000")
        )
        if naama_max_image_bytes < 1:
            raise ValueError("NAAMA_MAX_IMAGE_BYTES must be >= 1")

        return cls(
            telegram_bot_token=token,
            storage_dir=storage_dir,
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
            naama_model_name=os.getenv("NAAMA_MODEL_NAME", "yolo26n-seg.pt").strip(),
            naama_confidence_threshold=naama_confidence_threshold,
            naama_mask_threshold=naama_mask_threshold,
            naama_max_image_bytes=naama_max_image_bytes,
        )
