import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from bot.commands.telkkari_logic import (
    clear_epg_cache,
    fetch_epg_data,
    get_channel_day_schedule,
    get_next_hour_schedule,
    parse_xmltv_time,
)
from bot.config import TelkkariConfig

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
        self.assertIn("08:00 - 09:00: Aamuuutiset", res)
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
        self.assertIn("ohjelmatietoja ei löytynyt tälle päivälle", res)

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


if __name__ == "__main__":
    unittest.main()
