import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from bot.commands.telkkari_logic import (
    build_channel_day_card,
    build_invalid_channel_arg_card,
    build_next_hour_card,
    build_unknown_channel_card,
    clear_epg_cache,
    fetch_epg_data,
    get_channel_day_schedule,
    get_channel_day_schedule_card,
    get_next_hour_schedule,
    get_next_hour_schedule_card,
    parse_xmltv_time,
)
from bot.config import TelkkariConfig
from bot.rendering import render_card

SAMPLE_XMLTV = """<?xml version="1.0" encoding="UTF-8"?>
<tv>
  <channel id="YLE.TV1.fi">
    <display-name>YLE TV1</display-name>
  </channel>
  <channel id="YLE.TV2.fi">
    <display-name>YLE TV2</display-name>
  </channel>
  <programme start="20260802080000 +0300" stop="20260802090000 +0300" channel="YLE.TV1.fi">
    <title>Aamuuutiset</title>
  </programme>
  <programme start="20260802130000 +0300" stop="20260802140000 +0300" channel="YLE.TV1.fi">
    <title>Uutiset 13:00</title>
  </programme>
  <programme start="20260802133000 +0300" stop="20260802143000 +0300" channel="YLE.TV2.fi">
    <title>Pikku Kakkonen</title>
  </programme>
  <programme start="20260802200000 +0300" stop="20260802203000 +0300" channel="YLE.TV1.fi">
    <title>Iltauutiset</title>
  </programme>
</tv>
"""


class TelkkariLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_epg_cache()
        self.config = TelkkariConfig(
            epg_url="https://example.com/epg.xml",
            default_channels=(1, 2),
            cache_timeout_seconds=1800,
            timeout_seconds=30,
        )

    def tearDown(self) -> None:
        clear_epg_cache()

    def test_parse_xmltv_time(self) -> None:
        dt1 = parse_xmltv_time("20260802040000 +0000")
        self.assertEqual(dt1, datetime(2026, 8, 2, 4, 0, 0, tzinfo=timezone.utc))

        dt2 = parse_xmltv_time("20260802130000 +0300")
        expected_tz = timezone(timedelta(hours=3))
        self.assertEqual(dt2, datetime(2026, 8, 2, 13, 0, 0, tzinfo=expected_tz))

    def test_get_channel_day_schedule_success(self) -> None:
        root = ET.fromstring(SAMPLE_XMLTV)
        now = datetime(2026, 8, 2, 13, 15, 0, tzinfo=ZoneInfo("Europe/Helsinki"))

        res = get_channel_day_schedule(1, self.config, now=now, xml_root=root)
        self.assertIn("📺 YLE TV1 (tänään):", res)
        # Past programs (08:00 - 09:00) should be excluded
        self.assertNotIn("08:00 - 09:00: Aamuuutiset", res)
        # Currently airing and future programs for today should be included
        self.assertIn("13:00 - 14:00: Uutiset 13:00", res)
        self.assertIn("20:00 - 20:30: Iltauutiset", res)


    def test_get_channel_day_schedule_unknown_channel(self) -> None:
        res = get_channel_day_schedule(99, self.config)
        self.assertIn("⚠️ Tuntematon kanavanumero: 99.", res)
        self.assertIn("1: YLE TV1", res)

    def test_get_channel_day_schedule_no_programmes(self) -> None:
        root = ET.fromstring(SAMPLE_XMLTV)
        # Channel 3 (MTV3) has no programmes in sample
        now = datetime(2026, 8, 2, 13, 15, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        res = get_channel_day_schedule(3, self.config, now=now, xml_root=root)
        self.assertIn("ohjelmatietoja ei löytynyt", res)


    def test_get_next_hour_schedule(self) -> None:
        root = ET.fromstring(SAMPLE_XMLTV)
        now = datetime(2026, 8, 2, 13, 15, 0, tzinfo=ZoneInfo("Europe/Helsinki"))

        res = get_next_hour_schedule(self.config, now=now, xml_root=root)
        self.assertIn("📺 TV-ohjelmat seuraavan tunnin aikana:", res)
        self.assertIn("YLE TV1:", res)
        self.assertIn("13:00 - 14:00: Uutiset 13:00", res)
        self.assertIn("YLE TV2:", res)
        self.assertIn("13:30 - 14:30: Pikku Kakkonen", res)
        # Iltauutiset at 20:00 should not be included
        self.assertNotIn("Iltauutiset", res)

    def test_get_next_hour_schedule_empty(self) -> None:
        root = ET.fromstring(SAMPLE_XMLTV)
        # Current time at 03:00 has no shows in next hour
        now = datetime(2026, 8, 2, 3, 0, 0, tzinfo=ZoneInfo("Europe/Helsinki"))

        res = get_next_hour_schedule(self.config, now=now, xml_root=root)
        self.assertIn("Seuraavan tunnin aikana ei löytynyt ohjelmatietoja.", res)

    @patch("urllib.request.urlopen")
    def test_fetch_epg_data_caching(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = SAMPLE_XMLTV.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        root1 = fetch_epg_data(self.config)
        root2 = fetch_epg_data(self.config)

        self.assertIs(root1, root2)
        # Verify network request was made only once due to cache
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("urllib.request.urlopen", side_effect=Exception("Network error"))
    def test_fetch_epg_data_failure_handled(self, mock_urlopen: MagicMock) -> None:
        res = get_channel_day_schedule(1, self.config)
        self.assertIn("TV-ohjelmatietojen haku epäonnistui", res)

    def test_channel_day_card_matches_fallback_and_renders(self) -> None:
        root = ET.fromstring(SAMPLE_XMLTV)
        now = datetime(2026, 8, 2, 13, 15, 0, tzinfo=ZoneInfo("Europe/Helsinki"))

        fallback = get_channel_day_schedule(1, self.config, now=now, xml_root=root)
        card_fallback, card = get_channel_day_schedule_card(
            1, self.config, now=now, xml_root=root
        )
        self.assertEqual(fallback, card_fallback)
        self.assertEqual(card.title, "YLE TV1")
        self.assertEqual(card.badge.text, "TÄNÄÄN")
        png = render_card(card)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_channel_day_card_unknown_and_empty_render(self) -> None:
        root = ET.fromstring(SAMPLE_XMLTV)
        now = datetime(2026, 8, 2, 13, 15, 0, tzinfo=ZoneInfo("Europe/Helsinki"))

        fallback, card = get_channel_day_schedule_card(
            99, self.config, now=now, xml_root=root
        )
        self.assertIn("Tuntematon kanavanumero", fallback)
        self.assertEqual(card.title, "Tuntematon kanava")
        self.assertTrue(render_card(card).startswith(b"\x89PNG\r\n\x1a\n"))

        fallback_empty, empty_card = get_channel_day_schedule_card(
            3, self.config, now=now, xml_root=root
        )
        self.assertIn("ei löytynyt", fallback_empty)
        self.assertEqual(empty_card.badge.text, "TYHJÄ")
        self.assertTrue(render_card(empty_card).startswith(b"\x89PNG\r\n\x1a\n"))

    def test_next_hour_card_matches_fallback_and_renders(self) -> None:
        root = ET.fromstring(SAMPLE_XMLTV)
        now = datetime(2026, 8, 2, 13, 15, 0, tzinfo=ZoneInfo("Europe/Helsinki"))

        fallback = get_next_hour_schedule(self.config, now=now, xml_root=root)
        card_fallback, card = get_next_hour_schedule_card(
            self.config, now=now, xml_root=root
        )
        self.assertEqual(fallback, card_fallback)
        self.assertEqual(card.title, "TV-ohjelmat")
        self.assertIn("OHJELMAA", card.badge.text)
        png = render_card(card)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_next_hour_card_empty_renders(self) -> None:
        root = ET.fromstring(SAMPLE_XMLTV)
        now = datetime(2026, 8, 2, 3, 0, 0, tzinfo=ZoneInfo("Europe/Helsinki"))

        fallback, card = get_next_hour_schedule_card(
            self.config, now=now, xml_root=root
        )
        self.assertIn("ei löytynyt ohjelmatietoja", fallback)
        self.assertEqual(card.badge.text, "TYHJÄ")
        self.assertTrue(render_card(card).startswith(b"\x89PNG\r\n\x1a\n"))

    def test_card_fetch_failure_renders_error(self) -> None:
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            fallback, card = get_channel_day_schedule_card(1, self.config)
            self.assertIn("haku epäonnistui", fallback)
            self.assertEqual(card.badge.text, "VIRHE")
            self.assertTrue(render_card(card).startswith(b"\x89PNG\r\n\x1a\n"))

            fallback_next, card_next = get_next_hour_schedule_card(self.config)
            self.assertIn("haku epäonnistui", fallback_next)
            self.assertEqual(card_next.badge.text, "VIRHE")
            self.assertTrue(render_card(card_next).startswith(b"\x89PNG\r\n\x1a\n"))

    def test_pure_builders_render(self) -> None:
        now = datetime(2026, 8, 2, 13, 15, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        start = datetime(2026, 8, 2, 13, 0, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        stop = datetime(2026, 8, 2, 14, 0, 0, tzinfo=ZoneInfo("Europe/Helsinki"))

        day_card = build_channel_day_card("YLE TV1", [(start, stop, "Uutiset")], now=now)
        self.assertTrue(render_card(day_card).startswith(b"\x89PNG\r\n\x1a\n"))

        hour_card = build_next_hour_card(
            [("YLE TV1", start, stop, "Uutiset")], now=now, next_hour_end=stop
        )
        self.assertTrue(render_card(hour_card).startswith(b"\x89PNG\r\n\x1a\n"))

        self.assertTrue(
            render_card(build_unknown_channel_card(99)).startswith(b"\x89PNG\r\n\x1a\n")
        )
        self.assertTrue(
            render_card(build_invalid_channel_arg_card("abc")).startswith(
                b"\x89PNG\r\n\x1a\n"
            )
        )


if __name__ == "__main__":
    unittest.main()
