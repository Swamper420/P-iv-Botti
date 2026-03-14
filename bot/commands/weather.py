from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from telegram import InputFile, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.commands.message_utils import reply_in_chunks
from bot.commands.weather_logic import parse_weather_camera_location
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)


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


def _get_weather_cam_data(
    location_query: str, config: BotConfig
) -> tuple[bytes | None, str]:
    headers = {"Digitraffic-User": config.digitraffic_user, "If-None-Match": ""}
    try:
        data = _fetch_json(
            config.weathercam_stations_url,
            config.weather_api_timeout_seconds,
            headers=headers,
        )
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
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
    except (HTTPError, URLError, TimeoutError, OSError):
        LOGGER.exception("Weather camera image download failed")
        return None, "Kuvan lataus epäonnistui"

    return img_data, f"{camera_id}.jpg"


def _get_openweather_summary(location_query: str, config: BotConfig) -> str | None:
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


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_weather(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return

        matched, location = parse_weather_camera_location(message.text)
        if not matched:
            return

        if not location:
            await reply_in_chunks(
                update, "Käyttö: `!sääkuva <kaupunki>`", config.max_reply_length
            )
            return

        if update.effective_chat is not None:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

        weather_summary = await asyncio.to_thread(
            _get_openweather_summary, location, config
        )
        img_data, result = await asyncio.to_thread(_get_weather_cam_data, location, config)

        if img_data is None:
            await reply_in_chunks(update, f"⚠️ {result}", config.max_reply_length)
            return

        photo = InputFile(BytesIO(img_data), filename=result)
        await message.reply_photo(photo=photo)
        if weather_summary:
            await reply_in_chunks(update, weather_summary, config.max_reply_length)

    return handle_weather


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(
            filters.Regex(r"(?i)^\s*!(?:sääkuva|saakuva)(?:\s|$)"),
            _build_handler(config),
        )
    )
