from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bot.commands.muistuta_logic import (
    add_reminder,
    cancel_reminder,
    get_due_reminders,
    list_reminders,
    parse_reminder_args,
    remove_reminders,
)


class MuistutaLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tz = ZoneInfo("Europe/Helsinki")
        self.now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=self.tz)  # Thursday 12:00

    def test_parse_time_only_future(self) -> None:
        due_at, targets, message, err = parse_reminder_args("21:50 Ota lääkkeet", now=self.now)
        self.assertIsNone(err)
        self.assertIsNotNone(due_at)
        self.assertEqual(due_at.strftime("%Y-%m-%d %H:%M"), "2026-08-13 21:50")
        self.assertEqual(message, "Ota lääkkeet")
        self.assertEqual(targets, [])

    def test_parse_time_only_past(self) -> None:
        due_at, targets, message, err = parse_reminder_args("09:00 Aamukahvi", now=self.now)
        self.assertIsNone(err)
        self.assertIsNotNone(due_at)
        self.assertEqual(due_at.strftime("%Y-%m-%d %H:%M"), "2026-08-14 09:00")
        self.assertEqual(message, "Aamukahvi")

    def test_parse_finnish_full_date(self) -> None:
        due_at, targets, message, err = parse_reminder_args("21:50 25.12.2026 Joulu", now=self.now)
        self.assertIsNone(err)
        self.assertIsNotNone(due_at)
        self.assertEqual(due_at.strftime("%Y-%m-%d %H:%M"), "2026-12-25 21:50")
        self.assertEqual(message, "Joulu")

    def test_parse_huomenna(self) -> None:
        due_at, targets, message, err = parse_reminder_args("21:50 huomenna Testi", now=self.now)
        self.assertIsNone(err)
        self.assertIsNotNone(due_at)
        self.assertEqual(due_at.strftime("%Y-%m-%d %H:%M"), "2026-08-14 21:50")

    def test_parse_ylihuomenna(self) -> None:
        due_at, targets, message, err = parse_reminder_args("21:50 ylihuomenna Testi", now=self.now)
        self.assertIsNone(err)
        self.assertIsNotNone(due_at)
        self.assertEqual(due_at.strftime("%Y-%m-%d %H:%M"), "2026-08-15 21:50")

    def test_parse_weekday_perjantai(self) -> None:
        # self.now is Thursday 2026-08-13
        due_at, targets, message, err = parse_reminder_args("18:00 perjantaina Viikonloppu", now=self.now)
        self.assertIsNone(err)
        self.assertIsNotNone(due_at)
        self.assertEqual(due_at.strftime("%Y-%m-%d %H:%M"), "2026-08-14 18:00")

    def test_parse_relative_time(self) -> None:
        due_at, targets, message, err = parse_reminder_args("+15m Pitsa uuniin", now=self.now)
        self.assertIsNone(err)
        self.assertIsNotNone(due_at)
        self.assertEqual(due_at.strftime("%Y-%m-%d %H:%M"), "2026-08-13 12:15")
        self.assertEqual(message, "Pitsa uuniin")

    def test_parse_mentions(self) -> None:
        due_at, targets, message, err = parse_reminder_args("21:50 @matti @teppo Pelit alkaa", now=self.now)
        self.assertIsNone(err)
        self.assertEqual(targets, ["@matti", "@teppo"])
        self.assertEqual(message, "@matti @teppo Pelit alkaa")

    def test_invalid_time_format(self) -> None:
        due_at, targets, message, err = parse_reminder_args("invalid_time text", now=self.now)
        self.assertIsNotNone(err)
        self.assertIsNone(due_at)

    def test_storage_crud(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_dir = Path(tmp_dir)

            # Initially empty list
            self.assertIn("Ei aktiivisia muistutuksia", list_reminders(storage_dir, 12345))

            due_at = datetime(2026, 8, 13, 11, 0, 0, tzinfo=self.tz)  # past relative to now
            item, reply = add_reminder(
                storage_dir=storage_dir,
                chat_id=12345,
                creator="@creator",
                due_at=due_at,
                targets=["@target"],
                message="Testiviesti",
                media={"file_id": "test_file_id", "media_type": "photo"},
            )
            self.assertIsNotNone(item)
            self.assertIn("Muistutus #1 asetettu", reply)

            # Check listing
            lst = list_reminders(storage_dir, 12345)
            self.assertIn("#1", lst)
            self.assertIn("Testiviesti", lst)

            # Check due reminders
            due_items = get_due_reminders(storage_dir, now=self.now)
            self.assertEqual(len(due_items), 1)
            self.assertEqual(due_items[0]["id"], 1)

            # Remove due reminders
            remove_reminders(storage_dir, {1})
            self.assertEqual(len(get_due_reminders(storage_dir, now=self.now)), 0)

            # Test cancel non-existent
            cancel_msg = cancel_reminder(storage_dir, 12345, 99)
            self.assertIn("ei löytynyt", cancel_msg)


if __name__ == "__main__":
    unittest.main()
