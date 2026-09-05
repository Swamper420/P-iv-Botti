from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bot.config import BotConfig, WeatherConfig
from bot.rendering import BadgeColor, Card

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
    req_headers = dict(headers) if headers else {}
    req_headers["Accept-Encoding"] = "gzip"

    request = Request(url, headers=req_headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        data = response.read()
        if response.info().get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return json.loads(data.decode("utf-8"))


def _download_bytes(
    url: str, timeout_seconds: int, headers: dict[str, str] | None = None
) -> bytes:
    req_headers = dict(headers) if headers else {}
    req_headers["Accept-Encoding"] = "gzip"

    request = Request(url, headers=req_headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        data = response.read()
        if response.info().get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return data


def _extract_weather_config(config: WeatherConfig | BotConfig) -> WeatherConfig:
    if isinstance(config, WeatherConfig):
        return config
    return config.weather


# ---------------------------------------------------------------------------
# Structured weather models
# ---------------------------------------------------------------------------


@dataclass
class WeatherInfo:
    location_name: str
    country: str | None = None
    description: str = "ei kuvausta"
    main_condition: str = ""
    condition_id: int = 800
    icon: str = ""
    temp_c: float = 0.0
    feels_like_c: float = 0.0
    temp_min_c: float | None = None
    temp_max_c: float | None = None
    humidity_pct: int | None = None
    pressure_hpa: int | None = None
    wind_speed_ms: float | None = None
    wind_deg: int | None = None
    wind_gust_ms: float | None = None
    clouds_pct: int | None = None
    visibility_m: int | None = None
    observation_ts: int | None = None
    timezone_offset_s: int = 0
    sunrise_ts: int | None = None
    sunset_ts: int | None = None


@dataclass
class WeatherCamResult:
    image_bytes: bytes | None = None
    camera_id: str | None = None
    station_name: str | None = None
    error: str | None = None


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_openweather_payload(data: dict, fallback_query: str) -> WeatherInfo | None:
    """Parse OpenWeather current-weather JSON into a WeatherInfo.

    Returns None when mandatory temperature fields are missing.
    """
    if not isinstance(data, dict):
        return None
    weather_list = data.get("weather") or [{}]
    weather = weather_list[0] if isinstance(weather_list, list) else {}
    if not isinstance(weather, dict):
        weather = {}
    main = data.get("main", {})
    if not isinstance(main, dict):
        main = {}
    wind = data.get("wind", {})
    if not isinstance(wind, dict):
        wind = {}
    clouds = data.get("clouds", {})
    if not isinstance(clouds, dict):
        clouds = {}
    sys_info = data.get("sys", {})
    if not isinstance(sys_info, dict):
        sys_info = {}

    temp = _as_float(main.get("temp"))
    feels_like = _as_float(main.get("feels_like"))
    if temp is None or feels_like is None:
        return None

    try:
        condition_id = int(weather.get("id", 800))
    except (TypeError, ValueError):
        condition_id = 800

    location_name = str(data.get("name") or fallback_query)
    country_raw = sys_info.get("country")
    country = str(country_raw).strip() if country_raw else None

    try:
        tz_offset = int(data.get("timezone", 0))
    except (TypeError, ValueError):
        tz_offset = 0

    return WeatherInfo(
        location_name=location_name,
        country=country,
        description=str(weather.get("description") or "ei kuvausta"),
        main_condition=str(weather.get("main") or ""),
        condition_id=condition_id,
        icon=str(weather.get("icon") or ""),
        temp_c=temp,
        feels_like_c=feels_like,
        temp_min_c=_as_float(main.get("temp_min")),
        temp_max_c=_as_float(main.get("temp_max")),
        humidity_pct=_as_int(main.get("humidity")),
        pressure_hpa=_as_int(main.get("pressure")),
        wind_speed_ms=_as_float(wind.get("speed")),
        wind_deg=_as_int(wind.get("deg")),
        wind_gust_ms=_as_float(wind.get("gust")),
        clouds_pct=_as_int(clouds.get("all")),
        visibility_m=_as_int(data.get("visibility")),
        observation_ts=_as_int(data.get("dt")),
        timezone_offset_s=tz_offset,
        sunrise_ts=_as_int(sys_info.get("sunrise")),
        sunset_ts=_as_int(sys_info.get("sunset")),
    )


def get_openweather_details(
    location_query: str, config: WeatherConfig | BotConfig
) -> WeatherInfo | None:
    """Fetch full OpenWeather details. Returns None when unavailable."""
    cfg = _extract_weather_config(config)
    if not cfg.openweather_api_key:
        return None

    params = urlencode(
        {
            "q": location_query,
            "appid": cfg.openweather_api_key,
            "units": "metric",
            "lang": "fi",
        }
    )
    url = f"{cfg.openweather_current_url}?{params}"

    try:
        data = _fetch_json(url, cfg.timeout_seconds)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        LOGGER.warning("OpenWeather fetch failed for location %s", location_query)
        return None

    try:
        return parse_openweather_payload(data, location_query)
    except Exception:
        LOGGER.warning("OpenWeather payload parsing failed for %s", location_query)
        return None


def get_openweather_summary(
    location_query: str, config: WeatherConfig | BotConfig
) -> str | None:
    info = get_openweather_details(location_query, config)
    if info is None:
        return None
    return (
        f"🌡️ {info.location_name}: {info.description}, "
        f"{float(info.temp_c):.1f}°C (tuntuu kuin {float(info.feels_like_c):.1f}°C)"
    )


def _find_station_feature(data: dict, location_query: str) -> dict | None:
    for feature in data.get("features", []):
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties", {})
        if not isinstance(props, dict):
            continue
        name = str(props.get("name", ""))
        if location_query.casefold() in name.casefold():
            return feature
    return None


def get_weather_cam_details(
    location_query: str, config: WeatherConfig | BotConfig
) -> WeatherCamResult:
    """Fetch Digitraffic station + camera image with rich metadata."""
    cfg = _extract_weather_config(config)
    station_headers = {
        "Digitraffic-User": cfg.digitraffic_user,
        "If-None-Match": "",
    }
    try:
        data = _fetch_json(
            cfg.weathercam_stations_url,
            cfg.timeout_seconds,
            headers=station_headers,
        )
    except TimeoutError:
        LOGGER.exception("Weather camera station fetch timed out")
        return WeatherCamResult(error="API-yhteys aikakatkesi")
    except json.JSONDecodeError:
        LOGGER.exception("Weather camera station response could not be parsed")
        return WeatherCamResult(error="API vastasi virheellisellä datalla")
    except (HTTPError, URLError, OSError):
        LOGGER.exception("Weather camera station fetch failed")
        return WeatherCamResult(error="API-yhteys epäonnistui")

    location_json = _find_station_feature(data, location_query)
    if not location_json:
        return WeatherCamResult(error="Sijaintia ei löytynyt")

    props = location_json.get("properties", {})
    station_name = str(props.get("name", "")).strip() or None
    try:
        presets = props["presets"]
        camera_id = str(presets[0]["id"])
    except (KeyError, IndexError, TypeError, AttributeError):
        return WeatherCamResult(
            station_name=station_name, error="Kamera ei ole saatavilla"
        )

    image_url = f"{cfg.weathercam_image_base_url.rstrip('/')}/{camera_id}.jpg"
    image_headers = {
        "Digitraffic-User": cfg.digitraffic_user,
        "If-None-Match": "",
    }
    try:
        img_data = _download_bytes(
            image_url, cfg.timeout_seconds, headers=image_headers
        )
    except TimeoutError:
        LOGGER.exception("Weather camera image download timed out")
        return WeatherCamResult(
            camera_id=camera_id, station_name=station_name, error="Kuvan lataus aikakatkesi"
        )
    except (HTTPError, URLError, OSError):
        LOGGER.exception("Weather camera image download failed")
        return WeatherCamResult(
            camera_id=camera_id, station_name=station_name, error="Kuvan lataus epäonnistui"
        )

    if not img_data:
        return WeatherCamResult(
            camera_id=camera_id, station_name=station_name, error="Kuvan lataus epäonnistui"
        )

    return WeatherCamResult(
        image_bytes=img_data, camera_id=camera_id, station_name=station_name
    )


def get_weather_cam_data(
    location_query: str, config: WeatherConfig | BotConfig
) -> tuple[bytes | None, str]:
    """Backward-compatible wrapper returning (image_bytes, filename_or_error)."""
    result = get_weather_cam_details(location_query, config)
    if result.image_bytes is None:
        return None, result.error or "Kuvan lataus epäonnistui"
    return result.image_bytes, f"{result.camera_id}.jpg"


# ---------------------------------------------------------------------------
# Presentation helpers (pure, unit-testable)
# ---------------------------------------------------------------------------


def weather_emoji(condition_id: int, icon: str = "") -> str:
    if 200 <= condition_id <= 232:
        return "⛈️"
    if 300 <= condition_id <= 321:
        return "🌦️"
    if 500 <= condition_id <= 531:
        return "🌧️"
    if 600 <= condition_id <= 622:
        return "❄️"
    if 701 <= condition_id <= 781:
        return "🌫️"
    if condition_id == 800:
        return "🌙" if icon.endswith("n") else "☀️"
    if condition_id in (801, 802):
        return "⛅"
    if 803 <= condition_id <= 804:
        return "☁️"
    return "🌡️"


def badge_color_for_condition(condition_id: int, icon: str = "") -> BadgeColor:
    if 200 <= condition_id <= 232:
        return BadgeColor.PURPLE
    if 300 <= condition_id <= 622:
        return BadgeColor.BLUE
    if 701 <= condition_id <= 781:
        return BadgeColor.GRAY
    if condition_id == 800:
        return BadgeColor.YELLOW
    if 801 <= condition_id <= 804:
        return BadgeColor.GRAY
    return BadgeColor.BLUE


def accent_color_for_condition(condition_id: int, icon: str = "") -> str:
    if condition_id == 800:
        return "#818cf8" if icon.endswith("n") else "#facc15"
    if 200 <= condition_id <= 232:
        return "#a855f7"
    if 300 <= condition_id <= 531:
        return "#38bdf8"
    if 600 <= condition_id <= 622:
        return "#7dd3fc"
    if 701 <= condition_id <= 781:
        return "#a1a1aa"
    if 801 <= condition_id <= 804:
        return "#94a3b8"
    return "#38bdf8"


def wind_direction_label(deg: int | None) -> str | None:
    """Map wind degrees to Finnish compass label with arrow, e.g. '↙ Lounas'."""
    if deg is None:
        return None
    try:
        d = int(deg) % 360
    except (TypeError, ValueError):
        return None
    idx = int((d + 22.5) // 45) % 8
    labels = [
        ("↑", "Pohjoinen"),
        ("↗", "Koillinen"),
        ("→", "Itä"),
        ("↘", "Kaakko"),
        ("↓", "Etelä"),
        ("↙", "Lounas"),
        ("←", "Länsi"),
        ("↖", "Luode"),
    ]
    arrow, label = labels[idx]
    return f"{arrow} {label}"


def format_wind(speed_ms: float | None, deg: int | None) -> str | None:
    if speed_ms is None:
        return None
    base = f"{float(speed_ms):.1f} m/s"
    direction = wind_direction_label(deg)
    if direction:
        return f"{base} {direction}"
    if deg is not None:
        return f"{base} ({int(deg)}°)"
    return base


def format_visibility(visibility_m: int | None) -> str | None:
    if visibility_m is None:
        return None
    try:
        v = int(visibility_m)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    if v >= 1000:
        return f"{v / 1000:.1f} km"
    return f"{v} m"


def _local_time(ts: int | None, tz_offset_s: int, fmt: str) -> str | None:
    if ts is None:
        return None
    try:
        tz = timezone(timedelta(seconds=int(tz_offset_s)))
        dt = datetime.fromtimestamp(int(ts), tz=tz)
        return dt.strftime(fmt)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def format_observation_time(info: WeatherInfo) -> str | None:
    return _local_time(info.observation_ts, info.timezone_offset_s, "%d.%m. %H:%M")


def format_sun_time(ts: int | None, tz_offset_s: int) -> str | None:
    return _local_time(ts, tz_offset_s, "%H:%M")


def _capitalize_fi(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def build_weather_fallback_text(
    location_query: str,
    cam: WeatherCamResult | None = None,
    weather: WeatherInfo | None = None,
) -> str:
    """Plain-text fallback mirroring the picture card content."""
    lines: list[str] = []
    if weather is not None:
        emoji = weather_emoji(weather.condition_id, weather.icon)
        title = weather.location_name
        if weather.country:
            title = f"{title}, {weather.country}"
        desc = _capitalize_fi(weather.description)
        lines.append(
            f"{emoji} {title}: {desc}, "
            f"{weather.temp_c:.1f}°C (tuntuu {weather.feels_like_c:.1f}°C)"
        )
        details: list[str] = []
        if weather.temp_min_c is not None and weather.temp_max_c is not None:
            details.append(f"Min {weather.temp_min_c:.1f} / Max {weather.temp_max_c:.1f}°C")
        if weather.humidity_pct is not None:
            details.append(f"Kosteus {weather.humidity_pct}%")
        wind_str = format_wind(weather.wind_speed_ms, weather.wind_deg)
        if wind_str:
            details.append(f"Tuuli {wind_str}")
        if weather.wind_gust_ms is not None:
            details.append(f"Puuskat {weather.wind_gust_ms:.1f} m/s")
        if weather.pressure_hpa is not None:
            details.append(f"Paine {weather.pressure_hpa} hPa")
        if weather.clouds_pct is not None:
            details.append(f"Pilvisyys {weather.clouds_pct}%")
        vis = format_visibility(weather.visibility_m)
        if vis:
            details.append(f"Näkyvyys {vis}")
        if details:
            lines.append(" • ".join(details))
        sun_parts: list[str] = []
        sunrise = format_sun_time(weather.sunrise_ts, weather.timezone_offset_s)
        sunset = format_sun_time(weather.sunset_ts, weather.timezone_offset_s)
        if sunrise:
            sun_parts.append(f"nousu {sunrise}")
        if sunset:
            sun_parts.append(f"lasku {sunset}")
        obs = format_observation_time(weather)
        if sun_parts:
            lines.append(f"Aurinko: {', '.join(sun_parts)}" + (f" • Havainto {obs}" if obs else ""))
        elif obs:
            lines.append(f"Havainto {obs}")
    else:
        lines.append(f"🌡️ {location_query}: säätiedot eivät saatavilla")

    if cam is not None:
        if cam.station_name or cam.camera_id:
            cam_label = cam.station_name or location_query
            if cam.camera_id:
                lines.append(f"Kamera: {cam_label} ({cam.camera_id})")
            else:
                lines.append(f"Kamera: {cam_label}")
    return "\n".join(lines)


def build_weather_error_card(message: str, location_query: str | None = None) -> Card:
    subtitle = f"Haku: {location_query}" if location_query else None
    return (
        Card(
            title="Sääkuva",
            subtitle=subtitle,
            footer="Digitraffic / Fintraffic • P-iv-Botti",
        )
        .set_badge("VIRHE", BadgeColor.RED)
        .add_text(f"⚠️ {message}")
        .add_text("Kokeile toista läheistä paikkaa (esim. !sääkuva Helsinki).", muted=True)
    )


def build_weather_card(
    location_query: str,
    cam: WeatherCamResult | None = None,
    weather: WeatherInfo | None = None,
) -> Card:
    """Build a pleasing, informative picture card combining cam photo + weather."""
    station_name = cam.station_name if cam and cam.station_name else None
    camera_id = cam.camera_id if cam and cam.camera_id else None
    cam_bytes = cam.image_bytes if cam and cam.image_bytes else None

    if weather is not None:
        emoji = weather_emoji(weather.condition_id, weather.icon)
        badge_color = badge_color_for_condition(weather.condition_id, weather.icon)
        accent = accent_color_for_condition(weather.condition_id, weather.icon)

        title = weather.location_name
        if weather.country:
            title = f"{title}, {weather.country}"

        subtitle_parts: list[str] = []
        if station_name:
            subtitle_parts.append(station_name)
        if camera_id:
            subtitle_parts.append(f"kamera {camera_id}")
        obs = format_observation_time(weather)
        if obs:
            subtitle_parts.append(f"havainto {obs}")
        if not subtitle_parts:
            subtitle_parts.append(f"haku: {location_query}")
        subtitle = " • ".join(subtitle_parts)

        desc_cap = _capitalize_fi(weather.description)
        badge_text = f"{emoji} {weather.temp_c:.1f}°C {desc_cap}"

        card = Card(
            title=title,
            subtitle=subtitle,
            footer="OpenWeather • Digitraffic / Fintraffic • P-iv-Botti",
            accent_color=accent,
        ).set_badge(badge_text, badge_color)

        # Hero headline: big readable summary
        card.add_text(
            f"{emoji} {desc_cap} — {weather.temp_c:.1f}°C "
            f"(tuntuu {weather.feels_like_c:.1f}°C)",
            bold=True,
        )

        if cam_bytes:
            caption_parts: list[str] = []
            if station_name:
                caption_parts.append(station_name)
            if camera_id:
                caption_parts.append(camera_id)
            caption = " • ".join(caption_parts) if caption_parts else None
            card.add_image(cam_bytes, caption=caption, max_height=420)

        # Key facts grid (2 columns keeps it dense but readable)
        card.add_key_value("Lämpötila", f"{weather.temp_c:.1f}°C")
        card.add_key_value("Tuntuu", f"{weather.feels_like_c:.1f}°C")
        if weather.temp_min_c is not None and weather.temp_max_c is not None:
            card.add_key_value(
                "Alin / Ylin", f"{weather.temp_min_c:.1f} / {weather.temp_max_c:.1f}°C"
            )
        if weather.humidity_pct is not None:
            card.add_key_value("Kosteus", f"{weather.humidity_pct}%")
        wind_str = format_wind(weather.wind_speed_ms, weather.wind_deg)
        if wind_str:
            card.add_key_value("Tuuli", wind_str)
        if weather.wind_gust_ms is not None:
            card.add_key_value("Puuskat", f"{weather.wind_gust_ms:.1f} m/s")
        if weather.pressure_hpa is not None:
            card.add_key_value("Paine", f"{weather.pressure_hpa} hPa")
        if weather.clouds_pct is not None:
            card.add_key_value("Pilvisyys", f"{weather.clouds_pct}%")
        vis = format_visibility(weather.visibility_m)
        if vis:
            card.add_key_value("Näkyvyys", vis)

        # Visual meters for humidity / cloudiness
        if weather.humidity_pct is not None:
            try:
                card.add_progress_bar(
                    "Kosteus",
                    value=float(weather.humidity_pct),
                    max_value=100.0,
                    unit="%",
                    color="#38bdf8",
                )
            except (TypeError, ValueError):
                pass
        if weather.clouds_pct is not None:
            try:
                card.add_progress_bar(
                    "Pilvisyys",
                    value=float(weather.clouds_pct),
                    max_value=100.0,
                    unit="%",
                    color="#94a3b8",
                )
            except (TypeError, ValueError):
                pass

        sunrise = format_sun_time(weather.sunrise_ts, weather.timezone_offset_s)
        sunset = format_sun_time(weather.sunset_ts, weather.timezone_offset_s)
        if sunrise or sunset or obs:
            card.add_divider()
            if sunrise:
                card.add_key_value("Auringonnousu", sunrise)
            if sunset:
                card.add_key_value("Auringonlasku", sunset)
            if obs:
                card.add_key_value("Havainto", obs)

        return card

    # No weather data: camera-focused card with helpful note
    title = station_name or location_query
    subtitle = f"kamera {camera_id}" if camera_id else f"haku: {location_query}"
    card = Card(
        title=title,
        subtitle=subtitle,
        footer="Digitraffic / Fintraffic • P-iv-Botti",
    ).set_badge("KAMERA", BadgeColor.BLUE)
    if cam_bytes:
        card.add_image(cam_bytes, caption=subtitle, max_height=460)
    card.add_text(
        "Säätiedot eivät ole saatavilla (OPENWEATHER_API_KEY puuttuu tai haku epäonnistui).",
        muted=True,
    )
    return card
