import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from bot.commands.weather_logic import (
    MAX_CAM_ANGLES_GRID,
    CameraFrame,
    WeatherCamResult,
    WeatherInfo,
    _select_preset_index,
    accent_color_for_condition,
    badge_color_for_condition,
    build_weather_card,
    build_weather_error_card,
    build_weather_fallback_text,
    format_visibility,
    format_wind,
    get_openweather_details,
    get_openweather_summary,
    get_weather_cam_data,
    get_weather_cam_details,
    parse_openweather_payload,
    parse_weather_camera_location,
    weather_emoji,
    wind_direction_label,
)
from bot.config import BotConfig, Cs2RssConfig, NaamaConfig, WeatherConfig
from bot.rendering import BadgeColor, ImageElement, ImageGridElement, render_card


def _make_config(api_key: str = "") -> BotConfig:
    return BotConfig(
        telegram_bot_token="token",
        storage_dir=Path("."),
        max_reply_length=5000,
        weather=WeatherConfig(
            openweather_api_key=api_key,
            weathercam_stations_url="https://stations.invalid",
            weathercam_image_base_url="https://images.invalid",
            openweather_current_url="https://api.openweathermap.org/data/2.5/weather",
            timeout_seconds=30,
            digitraffic_user="telegram-bot-1.0",
        ),
        cs2_rss=Cs2RssConfig(
            url="https://steamcommunity.com/games/csgo/rss/",
            poll_interval_seconds=300,
            request_timeout_seconds=30,
        ),
        naama=NaamaConfig(),
    )


def _make_jpeg_bytes(width: int = 640, height: int = 480, color: str = "#334155") -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_weather_info(**overrides) -> WeatherInfo:
    base = WeatherInfo(
        location_name="Helsinki",
        country="FI",
        description="selkeää",
        main_condition="Clear",
        condition_id=800,
        icon="01d",
        temp_c=18.5,
        feels_like_c=17.2,
        temp_min_c=16.0,
        temp_max_c=20.1,
        humidity_pct=55,
        pressure_hpa=1015,
        wind_speed_ms=3.5,
        wind_deg=210,
        wind_gust_ms=6.2,
        clouds_pct=10,
        visibility_m=10000,
        observation_ts=1720000000,
        timezone_offset_s=7200,
        sunrise_ts=1719990000,
        sunset_ts=1720040000,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class WeatherLogicTests(unittest.TestCase):
    def test_extracts_location_from_weather_command(self) -> None:
        self.assertEqual(
            parse_weather_camera_location("!sääkuva Helsinki"), (True, "Helsinki", None)
        )

    def test_extracts_location_from_ascii_alias(self) -> None:
        self.assertEqual(
            parse_weather_camera_location("  !saakuva  Oulu "), (True, "Oulu", None)
        )

    def test_matches_command_without_location(self) -> None:
        self.assertEqual(parse_weather_camera_location("!sääkuva"), (True, None, None))

    def test_ignores_non_command_text(self) -> None:
        self.assertEqual(parse_weather_camera_location("hello test"), (False, None, None))

    def test_extracts_angle_number(self) -> None:
        self.assertEqual(
            parse_weather_camera_location("!sääkuva Helsinki 2"), (True, "Helsinki", 2)
        )
        self.assertEqual(
            parse_weather_camera_location("!sääkuva Uusi Kaupunki 3"),
            (True, "Uusi Kaupunki", 3),
        )
        self.assertEqual(
            parse_weather_camera_location("  !saakuva Oulu  02  "), (True, "Oulu", 2)
        )

    def test_lone_number_is_missing_location(self) -> None:
        self.assertEqual(parse_weather_camera_location("!sääkuva 2"), (True, None, None))

    def test_weather_image_fetch_sends_digitraffic_headers(self) -> None:
        location_data = {
            "features": [{"properties": {"name": "Helsinki", "presets": [{"id": "CAM123"}]}}]
        }
        config = _make_config()

        with (
            patch("bot.commands.weather_logic._fetch_json", return_value=location_data) as fetch,
            patch("bot.commands.weather_logic._download_bytes", return_value=b"jpg") as download,
        ):
            image, filename = get_weather_cam_data("helsinki", config)

        self.assertEqual(image, b"jpg")
        self.assertEqual(filename, "CAM123.jpg")
        self.assertEqual(
            fetch.call_args.kwargs["headers"],
            {"Digitraffic-User": "telegram-bot-1.0", "If-None-Match": ""},
        )
        self.assertEqual(
            download.call_args.kwargs["headers"],
            {"Digitraffic-User": "telegram-bot-1.0", "If-None-Match": ""},
        )

    def test_cam_details_returns_station_metadata(self) -> None:
        location_data = {
            "features": [
                {"properties": {"name": "Kaisaniemi Helsinki", "presets": [{"id": "C12345"}]}}
            ]
        }
        config = _make_config()
        with (
            patch("bot.commands.weather_logic._fetch_json", return_value=location_data),
            patch("bot.commands.weather_logic._download_bytes", return_value=b"imgbytes"),
        ):
            result = get_weather_cam_details("helsinki", config)
        self.assertEqual(result.image_bytes, b"imgbytes")
        self.assertEqual(result.camera_id, "C12345")
        self.assertEqual(result.station_name, "Kaisaniemi Helsinki")
        self.assertIsNone(result.error)

    def test_cam_details_location_not_found(self) -> None:
        config = _make_config()
        with patch("bot.commands.weather_logic._fetch_json", return_value={"features": []}):
            result = get_weather_cam_details("atlantis", config)
        self.assertIsNone(result.image_bytes)
        self.assertEqual(result.error, "Sijaintia ei löytynyt")

    def test_cam_details_missing_presets(self) -> None:
        location_data = {"features": [{"properties": {"name": "Helsinki"}}]}
        config = _make_config()
        with patch("bot.commands.weather_logic._fetch_json", return_value=location_data):
            result = get_weather_cam_details("helsinki", config)
        self.assertIsNone(result.image_bytes)
        self.assertEqual(result.error, "Kamera ei ole saatavilla")

    def test_parse_openweather_payload_full(self) -> None:
        payload = {
            "weather": [{"id": 800, "main": "Clear", "description": "selkeää", "icon": "01d"}],
            "main": {
                "temp": 18.5,
                "feels_like": 17.2,
                "temp_min": 16.0,
                "temp_max": 20.1,
                "pressure": 1015,
                "humidity": 55,
            },
            "visibility": 10000,
            "wind": {"speed": 3.5, "deg": 210, "gust": 6.2},
            "clouds": {"all": 10},
            "dt": 1720000000,
            "sys": {"country": "FI", "sunrise": 1719990000, "sunset": 1720040000},
            "timezone": 7200,
            "name": "Helsinki",
        }
        info = parse_openweather_payload(payload, "Helsinki")
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.location_name, "Helsinki")
        self.assertEqual(info.country, "FI")
        self.assertAlmostEqual(info.temp_c, 18.5)
        self.assertEqual(info.humidity_pct, 55)
        self.assertEqual(info.wind_deg, 210)
        self.assertEqual(info.visibility_m, 10000)

    def test_parse_openweather_payload_missing_temp_returns_none(self) -> None:
        payload = {"weather": [{"id": 800, "description": "selkeää"}], "main": {}}
        self.assertIsNone(parse_openweather_payload(payload, "Helsinki"))

    def test_openweather_details_none_without_api_key(self) -> None:
        config = _make_config(api_key="")
        self.assertIsNone(get_openweather_details("Helsinki", config))

    def test_openweather_summary_format_preserved(self) -> None:
        payload = {
            "weather": [{"id": 800, "main": "Clear", "description": "selkeää", "icon": "01d"}],
            "main": {"temp": 18.5, "feels_like": 17.2},
            "name": "Helsinki",
        }
        config = _make_config(api_key="key")
        with patch("bot.commands.weather_logic._fetch_json", return_value=payload):
            summary = get_openweather_summary("Helsinki", config)
        self.assertEqual(summary, "🌡️ Helsinki: selkeää, 18.5°C (tuntuu kuin 17.2°C)")

    def test_openweather_details_uses_full_payload(self) -> None:
        payload = {
            "weather": [{"id": 500, "main": "Rain", "description": "heikkoa sadetta", "icon": "10d"}],
            "main": {"temp": 12.0, "feels_like": 10.5, "humidity": 90, "pressure": 1002},
            "wind": {"speed": 5.0, "deg": 180},
            "clouds": {"all": 100},
            "visibility": 5000,
            "dt": 1720000000,
            "sys": {"country": "FI"},
            "timezone": 7200,
            "name": "Oulu",
        }
        config = _make_config(api_key="key")
        with patch("bot.commands.weather_logic._fetch_json", return_value=payload):
            info = get_openweather_details("Oulu", config)
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.condition_id, 500)
        self.assertEqual(info.clouds_pct, 100)

    def test_weather_emoji_mapping(self) -> None:
        self.assertEqual(weather_emoji(211), "⛈️")
        self.assertEqual(weather_emoji(301), "🌦️")
        self.assertEqual(weather_emoji(501), "🌧️")
        self.assertEqual(weather_emoji(601), "❄️")
        self.assertEqual(weather_emoji(741), "🌫️")
        self.assertEqual(weather_emoji(800, "01d"), "☀️")
        self.assertEqual(weather_emoji(800, "01n"), "🌙")
        self.assertEqual(weather_emoji(801), "⛅")
        self.assertEqual(weather_emoji(804), "☁️")

    def test_badge_and_accent_mapping(self) -> None:
        self.assertEqual(badge_color_for_condition(211), BadgeColor.PURPLE)
        self.assertEqual(badge_color_for_condition(501), BadgeColor.BLUE)
        self.assertEqual(badge_color_for_condition(800, "01d"), BadgeColor.YELLOW)
        self.assertEqual(badge_color_for_condition(803), BadgeColor.GRAY)
        accent = accent_color_for_condition(800, "01d")
        self.assertTrue(accent.startswith("#"))
        self.assertEqual(accent_color_for_condition(800, "01n"), "#818cf8")

    def test_wind_direction_labels(self) -> None:
        self.assertEqual(wind_direction_label(0), "↑ Pohjoinen")
        self.assertEqual(wind_direction_label(90), "→ Itä")
        self.assertEqual(wind_direction_label(180), "↓ Etelä")
        self.assertEqual(wind_direction_label(270), "← Länsi")
        self.assertIsNone(wind_direction_label(None))
        self.assertIn("Lounas", format_wind(3.5, 225) or "")
        self.assertEqual(format_wind(None, 180), None)
        self.assertEqual(format_visibility(10000), "10.0 km")
        self.assertEqual(format_visibility(500), "500 m")
        self.assertIsNone(format_visibility(None))

    def test_build_weather_card_with_photo_and_facts_renders_png(self) -> None:
        cam = WeatherCamResult(
            image_bytes=_make_jpeg_bytes(), camera_id="C12345", station_name="Kaisaniemi"
        )
        weather = _make_weather_info()
        card = build_weather_card("Helsinki", cam=cam, weather=weather)
        self.assertIn("Helsinki", card.title)
        self.assertIsNotNone(card.badge)
        self.assertTrue(any(isinstance(el, ImageElement) for el in card.elements))
        raw = render_card(card)
        self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
        img = Image.open(BytesIO(raw))
        self.assertEqual(img.width, 800)
        self.assertGreater(img.height, 500)

    def test_build_weather_card_without_weather_still_pleasing(self) -> None:
        cam = WeatherCamResult(
            image_bytes=_make_jpeg_bytes(), camera_id="C99", station_name="Oulu"
        )
        card = build_weather_card("Oulu", cam=cam, weather=None)
        self.assertIn("Oulu", card.title)
        raw = render_card(card)
        img = Image.open(BytesIO(raw))
        self.assertEqual(img.width, 800)

    def test_build_weather_card_without_photo_renders(self) -> None:
        weather = _make_weather_info()
        card = build_weather_card("Tampere", cam=None, weather=weather)
        self.assertFalse(any(isinstance(el, ImageElement) for el in card.elements))
        raw = render_card(card)
        img = Image.open(BytesIO(raw))
        self.assertEqual(img.width, 800)

    def test_build_weather_card_night_and_snow_accents(self) -> None:
        cam = WeatherCamResult(image_bytes=_make_jpeg_bytes(), camera_id="C1", station_name="Ruka")
        night = _make_weather_info(condition_id=800, icon="01n", description="selkeää")
        snow = _make_weather_info(condition_id=601, icon="13d", description="lumisadetta")
        for info in (night, snow):
            card = build_weather_card("Ruka", cam=cam, weather=info)
            raw = render_card(card)
            self.assertTrue(raw.startswith(b"\x89PNG"))

    def test_fallback_text_contains_key_facts(self) -> None:
        cam = WeatherCamResult(image_bytes=b"x", camera_id="C12345", station_name="Kaisaniemi")
        weather = _make_weather_info()
        text = build_weather_fallback_text("Helsinki", cam=cam, weather=weather)
        self.assertIn("Helsinki", text)
        self.assertIn("18.5°C", text)
        self.assertIn("Kosteus 55%", text)
        self.assertIn("Kamera: Kaisaniemi (C12345)", text)

    def test_error_card_has_red_badge(self) -> None:
        card = build_weather_error_card("Sijaintia ei löytynyt", "Atlantis")
        self.assertIsNotNone(card.badge)
        assert card.badge is not None
        self.assertEqual(card.badge.text, "VIRHE")
        raw = render_card(card)
        self.assertTrue(raw.startswith(b"\x89PNG"))

    def test_select_preset_index_trailing_digit(self) -> None:
        self.assertEqual(_select_preset_index(["C01501", "C01502", "C01503"], 2), 1)
        self.assertEqual(_select_preset_index(["C01501", "C01502"], 2), 1)
        # Fallback to positional index when ids do not end with the angle.
        self.assertEqual(_select_preset_index(["CAM_A", "CAM_B"], 2), 1)
        self.assertIsNone(_select_preset_index(["C01501", "C01502"], 5))
        self.assertIsNone(_select_preset_index([], 1))

    def test_cam_details_single_angle(self) -> None:
        location_data = {
            "features": [
                {
                    "properties": {
                        "name": "Helsinki",
                        "presets": [{"id": "C01501"}, {"id": "C01502"}, {"id": "C01503"}],
                    }
                }
            ]
        }
        config = _make_config()
        with (
            patch("bot.commands.weather_logic._fetch_json", return_value=location_data),
            patch(
                "bot.commands.weather_logic._download_bytes",
                side_effect=lambda url, *a, **k: f"bytes-for-{url}".encode(),
            ) as download,
        ):
            result = get_weather_cam_details("helsinki", config, angle=2)
        self.assertEqual(result.camera_id, "C01502")
        self.assertEqual(result.selected_angle, 2)
        self.assertEqual(result.total_angles, 3)
        self.assertEqual(len(result.frames), 1)
        self.assertTrue(download.call_args[0][0].endswith("/C01502.jpg"))

    def test_cam_details_angle_out_of_range(self) -> None:
        location_data = {
            "features": [
                {"properties": {"name": "Helsinki", "presets": [{"id": "C1"}, {"id": "C2"}]}}
            ]
        }
        config = _make_config()
        with patch("bot.commands.weather_logic._fetch_json", return_value=location_data):
            result = get_weather_cam_details("helsinki", config, angle=5)
        self.assertIsNone(result.image_bytes)
        self.assertIn("1–2", result.error or "")
        self.assertEqual(result.total_angles, 2)

    def test_cam_details_grid_fetches_up_to_four(self) -> None:
        location_data = {
            "features": [
                {
                    "properties": {
                        "name": "Helsinki",
                        "presets": [
                            {"id": "C1"},
                            {"id": "C2"},
                            {"id": "C3"},
                            {"id": "C4"},
                            {"id": "C5"},
                        ],
                    }
                }
            ]
        }
        config = _make_config()
        with (
            patch("bot.commands.weather_logic._fetch_json", return_value=location_data),
            patch(
                "bot.commands.weather_logic._download_bytes", return_value=b"img"
            ) as download,
        ):
            result = get_weather_cam_details("helsinki", config)
        self.assertEqual(len(result.frames), MAX_CAM_ANGLES_GRID)
        self.assertEqual(result.total_angles, 5)
        self.assertEqual(download.call_count, MAX_CAM_ANGLES_GRID)
        self.assertEqual([f.angle_number for f in result.frames], [1, 2, 3, 4])

    def test_cam_details_grid_skips_failed_downloads(self) -> None:
        location_data = {
            "features": [
                {
                    "properties": {
                        "name": "Helsinki",
                        "presets": [{"id": "C1"}, {"id": "C2"}, {"id": "C3"}],
                    }
                }
            ]
        }
        config = _make_config()

        def _fail_c2(url: str, *args, **kwargs) -> bytes:
            if url.endswith("/C2.jpg"):
                raise OSError("boom")
            return b"img"

        with (
            patch("bot.commands.weather_logic._fetch_json", return_value=location_data),
            patch("bot.commands.weather_logic._download_bytes", side_effect=_fail_c2),
        ):
            result = get_weather_cam_details("helsinki", config)
        self.assertEqual([f.camera_id for f in result.frames], ["C1", "C3"])
        self.assertIsNotNone(result.image_bytes)

    def test_build_card_grid_uses_image_grid_element(self) -> None:
        frames = [
            CameraFrame(camera_id=f"C{i}", image_bytes=_make_jpeg_bytes(), angle_number=i)
            for i in (1, 2, 3)
        ]
        cam = WeatherCamResult(
            station_name="Kaisaniemi", frames=frames, total_angles=3
        )
        card = build_weather_card("Helsinki", cam=cam, weather=_make_weather_info())
        self.assertTrue(any(isinstance(el, ImageGridElement) for el in card.elements))
        self.assertFalse(any(isinstance(el, ImageElement) for el in card.elements))
        raw = render_card(card)
        img = Image.open(BytesIO(raw))
        self.assertEqual(img.width, 800)
        self.assertGreater(img.height, 600)

    def test_build_card_single_angle_uses_single_image(self) -> None:
        frame = CameraFrame(
            camera_id="C01502", image_bytes=_make_jpeg_bytes(), angle_number=2
        )
        cam = WeatherCamResult(
            station_name="Helsinki",
            frames=[frame],
            selected_angle=2,
            total_angles=3,
            image_bytes=frame.image_bytes,
            camera_id=frame.camera_id,
        )
        card = build_weather_card("Helsinki", cam=cam, weather=_make_weather_info())
        self.assertTrue(any(isinstance(el, ImageElement) for el in card.elements))
        self.assertIn("2/3", card.subtitle or "")
        raw = render_card(card)
        self.assertTrue(raw.startswith(b"\x89PNG"))

    def test_fallback_grid_lists_all_cameras(self) -> None:
        frames = [
            CameraFrame(camera_id="C1", image_bytes=b"a", angle_number=1),
            CameraFrame(camera_id="C2", image_bytes=b"b", angle_number=2),
        ]
        cam = WeatherCamResult(station_name="Helsinki", frames=frames, total_angles=2)
        text = build_weather_fallback_text("Helsinki", cam=cam, weather=_make_weather_info())
        self.assertIn("Kamerat (2)", text)
        self.assertIn("C1", text)
        self.assertIn("!sääkuva Helsinki", text)


if __name__ == "__main__":
    unittest.main()
