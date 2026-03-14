from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)


def parse_weather_camera_location(text: str | None) -> tuple[bool, str | None]:
    if text is None:
        return False, None

    stripped = text.strip()
    lowered = stripped.casefold()

    for prefix in ("!sääkuva", "!saakuva"):
        if not lowered.startswith(prefix):
            continue

        rest = stripped[len(prefix) :]
        if rest and not rest[0].isspace():
            continue

        location = rest.strip()
        return True, location or None

    return False, None


def _fetch_json(url: str, timeout_seconds: int, headers: dict[str, str] | None = None) -> dict:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_bytes(
    url: str, timeout_seconds: int, headers: dict[str, str] | None = None
) -> bytes:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def get_weather_cam_data(location_query: str, config: BotConfig) -> tuple[bytes | None, str]:
    headers = {"Digitraffic-User": config.digitraffic_user}
    try:
        data = _fetch_json(
            config.weathercam_stations_url,
            config.weather_api_timeout_seconds,
            headers=headers,
        )
    except TimeoutError:
        LOGGER.exception("Weather camera station fetch timed out")
        return None, "API-yhteys aikakatkesi"
    except json.JSONDecodeError:
        LOGGER.exception("Weather camera station response could not be parsed")
        return None, "API vastasi virheellisellä datalla"
    except (HTTPError, URLError, OSError):
        LOGGER.exception("Weather camera station fetch failed")
        return None, "API-yhteys epäonnistui"

    location_json = None
    for feature in data.get("features", []):
        name = feature.get("properties", {}).get("name", "")
        if location_query.casefold() in name.casefold():
            location_json = feature
            break

    if not location_json:
        return None, "Sijaintia ei löytynyt"

    try:
        presets = location_json["properties"]["presets"]
        camera_id = presets[0]["id"]
    except (KeyError, IndexError, TypeError):
        return None, "Kamera ei ole saatavilla"

    image_url = f"{config.weathercam_image_base_url.rstrip('/')}/{camera_id}.jpg"
    try:
        img_data = _download_bytes(
            image_url, config.weather_api_timeout_seconds, headers=headers
        )
    except TimeoutError:
        LOGGER.exception("Weather camera image download timed out")
        return None, "Kuvan lataus aikakatkesi"
    except (HTTPError, URLError, OSError):
        LOGGER.exception("Weather camera image download failed")
        return None, "Kuvan lataus epäonnistui"

    return img_data, f"{camera_id}.jpg"


def get_openweather_summary(location_query: str, config: BotConfig) -> str | None:
    if not config.openweather_api_key:
        return None

    params = urlencode(
        {
            "q": location_query,
            "appid": config.openweather_api_key,
            "units": "metric",
            "lang": "fi",
        }
    )
    url = f"{config.openweather_current_url}?{params}"

    try:
        data = _fetch_json(url, config.weather_api_timeout_seconds)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        LOGGER.warning("OpenWeather fetch failed for location %s", location_query)
        return None

    weather = (data.get("weather") or [{}])[0]
    main = data.get("main", {})
    temp = main.get("temp")
    feels_like = main.get("feels_like")
    if temp is None or feels_like is None:
        return None

    location_name = data.get("name", location_query)
    description = weather.get("description", "ei kuvausta")
    return (
        f"🌡️ {location_name}: {description}, "
        f"{float(temp):.1f}°C (tuntuu kuin {float(feels_like):.1f}°C)"
    )
