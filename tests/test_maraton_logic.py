from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from bot.commands.maraton_logic import (
    handle_maraton_card_command,
    parse_maraton_command,
    render_maraton_command_image,
)
from bot.config import (
    BotConfig,
    Cs2RssConfig,
    MaratonConfig,
    NaamaConfig,
    WeatherConfig,
)
from bot.rendering import BadgeColor, render_card
from bot.tasks._maraton_render import (
    render_maraton_briefing_image,
    render_maraton_legs_image,
    render_maraton_week_image,
)
from bot.tasks.maraton import MaratonNotifier, register
from bot.tasks.maraton_logic import (
    build_maraton_briefing,
    build_maraton_card,
    format_maraton_caption,
    format_maraton_text,
    mark_reminder_sent,
    should_send_daily_reminder,
)


def _make_bot_config(
    storage_dir: Path,
    maraton: MaratonConfig | None = None,
) -> BotConfig:
    return BotConfig(
        telegram_bot_token="token",
        storage_dir=storage_dir,
        max_reply_length=5000,
        weather=WeatherConfig(
            openweather_api_key="",
            weathercam_stations_url="https://tie.digitraffic.fi/api/weathercam/v1/stations",
            weathercam_image_base_url="https://weathercam.digitraffic.fi",
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
        maraton=maraton or MaratonConfig(),
    )


class MaratonLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = MaratonConfig(event_date="2027-09-05", send_time="09:00", timezone="Europe/Helsinki")

    def test_phases_escalate_in_dread_and_badge_color_toward_race_day(self) -> None:
        checkpoints = [
            (date(2026, 11, 23), "shadow", BadgeColor.BLUE),         # 286 days
            (date(2027, 2, 1), "winter_grind", BadgeColor.YELLOW),   # 216 days
            (date(2027, 4, 15), "engine_up", BadgeColor.YELLOW),     # 143 days
            (date(2027, 6, 15), "lactate_truth", BadgeColor.RED),    # 82 days
            (date(2027, 7, 20), "critical_sharpen", BadgeColor.RED), # 47 days
            (date(2027, 8, 15), "pre_storm", BadgeColor.RED),        # 21 days
            (date(2027, 8, 30), "final_countdown", BadgeColor.RED),  # 6 days
            (date(2027, 9, 5), "race_day", BadgeColor.RED),          # 0 days
        ]

        previous_dread = -1
        for day, expected_key, expected_color in checkpoints:
            briefing = build_maraton_briefing(self.cfg, today=day)
            self.assertEqual(briefing.phase_key, expected_key)
            self.assertEqual(briefing.badge_color, expected_color)
            self.assertGreater(briefing.dread_percent, previous_dread)
            previous_dread = briefing.dread_percent

    def test_final_days_and_race_day_and_post_race(self) -> None:
        eve = build_maraton_briefing(self.cfg, today=date(2027, 9, 4))
        self.assertEqual(eve.days_remaining, 1)
        self.assertEqual(eve.badge_text, "1 PV JÄLJELLÄ")
        self.assertIn("HUOMENNA", eve.headline)
        self.assertIsNotNone(eve.milestone_banner)

        race_day = build_maraton_briefing(self.cfg, today=date(2027, 9, 5))
        self.assertTrue(race_day.is_race_day)
        self.assertFalse(race_day.is_post_race)
        self.assertEqual(race_day.days_remaining, 0)
        self.assertEqual(race_day.badge_text, "KISAPÄIVÄ")
        self.assertEqual(race_day.progress_percent, 100.0)

        after = build_maraton_briefing(self.cfg, today=date(2027, 9, 7))
        self.assertTrue(after.is_post_race)
        self.assertEqual(after.days_remaining, -2)
        self.assertEqual(after.badge_color, BadgeColor.GREEN)
        self.assertIn("2 pv sitten", after.headline)

    def test_milestones_trigger_on_key_days(self) -> None:
        event = date(2027, 9, 5)
        for days in (300, 100, 42, 30, 7, 1, 0):
            briefing = build_maraton_briefing(self.cfg, today=event - timedelta(days=days))
            self.assertIsNotNone(briefing.milestone_banner, msg=f"Missing milestone for {days}d")

        regular = build_maraton_briefing(self.cfg, today=event - timedelta(days=111))
        self.assertIsNone(regular.milestone_banner)

    def test_consecutive_days_vary_content_and_match_weekday_workouts(self) -> None:
        start = date(2027, 6, 7)  # Monday
        headlines = set()
        workouts = set()
        for offset in range(7):
            briefing = build_maraton_briefing(self.cfg, today=start + timedelta(days=offset))
            headlines.add(briefing.headline)
            workouts.add(briefing.workout_detail)

        self.assertGreaterEqual(len(headlines), 3)
        self.assertEqual(len(workouts), 7)

    def test_caption_and_text_formatting_constraints(self) -> None:
        for day in (date(2026, 11, 23), date(2027, 7, 25), date(2027, 9, 5), date(2027, 9, 6)):
            briefing = build_maraton_briefing(self.cfg, today=day)
            caption = format_maraton_caption(briefing)
            full_text = format_maraton_text(briefing)

            self.assertLess(len(caption), 1024)
            self.assertIn("TAMPERE MARATON 2027", caption)
            self.assertIn("TAMPERE MARATON 2027", full_text)

    def test_should_send_daily_reminder_schedule_and_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)

            # Before 09:00 Helsinki time (06:59 UTC == 08:59 EET in November)
            before_dt = datetime(2026, 11, 23, 6, 59, tzinfo=timezone.utc)
            self.assertFalse(should_send_daily_reminder(self.cfg, storage, now=before_dt))

            # At 09:00 Helsinki time (07:00 UTC == 09:00 EET)
            at_dt = datetime(2026, 11, 23, 7, 0, tzinfo=timezone.utc)
            self.assertTrue(should_send_daily_reminder(self.cfg, storage, now=at_dt))

            # Mark sent -> should not send again same day
            mark_reminder_sent(storage, date(2026, 11, 23), now=at_dt)
            later_same_day = datetime(2026, 11, 23, 18, 0, tzinfo=timezone.utc)
            self.assertFalse(should_send_daily_reminder(self.cfg, storage, now=later_same_day))

            # Next day at 09:00 -> should send again
            next_day = datetime(2026, 11, 24, 7, 0, tzinfo=timezone.utc)
            self.assertTrue(should_send_daily_reminder(self.cfg, storage, now=next_day))

            # Disabled config -> never sends
            disabled_cfg = MaratonConfig(enabled=False)
            self.assertFalse(should_send_daily_reminder(disabled_cfg, storage, now=next_day))

            # Race day -> still sends!
            race_day_dt = datetime(2027, 9, 5, 6, 0, tzinfo=timezone.utc)  # 09:00 EEST
            self.assertTrue(should_send_daily_reminder(self.cfg, storage, now=race_day_dt))

            # Day after race -> stops automatically
            after_race_dt = datetime(2027, 9, 6, 10, 0, tzinfo=timezone.utc)
            self.assertFalse(should_send_daily_reminder(self.cfg, storage, now=after_race_dt))

    def test_command_parser_and_all_cards_render_to_png(self) -> None:
        self.assertEqual(parse_maraton_command("!maraton"), (True, "status"))
        self.assertEqual(parse_maraton_command("!MARATON osuudet"), (True, "legs"))
        self.assertEqual(parse_maraton_command("!maraton treeni"), (True, "week"))
        self.assertEqual(parse_maraton_command("!maraton vinkki"), (True, "week"))
        self.assertEqual(parse_maraton_command("!maraton ohje"), (True, "help"))
        self.assertEqual(parse_maraton_command("moi"), (False, ""))

        for cmd in ("!maraton", "!maraton osuudet", "!maraton treeni", "!maraton ohje"):
            text, card = handle_maraton_card_command(self.cfg, cmd, today=date(2026, 11, 23))
            self.assertTrue(text)
            self.assertIsNotNone(card)
            assert card is not None
            png_bytes = render_card(card)
            self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

        # Also verify race-day and post-race cards render cleanly
        for day in (date(2027, 9, 5), date(2027, 9, 8)):
            card = build_maraton_card(build_maraton_briefing(self.cfg, today=day))
            png_bytes = render_card(card)
            self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_bespoke_posters_render_cleanly_without_rounded_corners(self) -> None:
        from PIL import ImageDraw
        from unittest.mock import patch

        with patch.object(
            ImageDraw.ImageDraw,
            "rounded_rectangle",
            side_effect=AssertionError("rounded_rectangle must never be called"),
        ):
            for day in (date(2026, 11, 23), date(2027, 7, 25), date(2027, 9, 5), date(2027, 9, 7)):
                briefing = build_maraton_briefing(self.cfg, today=day)
                png = render_maraton_briefing_image(briefing)
                self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

            b = build_maraton_briefing(self.cfg, today=date(2027, 7, 25))
            self.assertTrue(render_maraton_legs_image(b).startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(render_maraton_week_image(self.cfg, b).startswith(b"\x89PNG\r\n\x1a\n"))

            for cmd in ("!maraton", "!maraton osuudet", "!maraton treeni", "!maraton ohje"):
                text, img_bytes = render_maraton_command_image(
                    self.cfg, cmd, today=date(2027, 7, 25)
                )
                self.assertTrue(text)
                self.assertIsNotNone(img_bytes)
                assert img_bytes is not None
                self.assertTrue(img_bytes.startswith(b"\x89PNG\r\n\x1a\n"))


class MaratonTaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_notifier_broadcasts_card_once_per_day_and_falls_back_on_photo_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            cfg = _make_bot_config(
                storage,
                maraton=MaratonConfig(
                    enabled=True,
                    event_date="2027-09-05",
                    send_time="09:00",
                    timezone="Europe/Helsinki",
                    chat_ids=(111, 222),
                    use_card=True,
                ),
            )
            notifier = MaratonNotifier(cfg)

            app = MagicMock()
            app.bot = MagicMock()
            # First chat succeeds with send_photo, second chat fails send_photo -> falls back to send_message
            app.bot.send_photo = AsyncMock(side_effect=[None, RuntimeError("upload failed")])
            app.bot.send_message = AsyncMock()

            now = datetime(2026, 11, 23, 7, 5, tzinfo=timezone.utc)  # 09:05 EET
            sent_first = await notifier.check_and_send(app, now=now)
            self.assertTrue(sent_first)
            self.assertEqual(app.bot.send_photo.await_count, 2)
            self.assertEqual(app.bot.send_message.await_count, 1)
            self.assertEqual(app.bot.send_message.await_args.kwargs["chat_id"], 222)

            # Second call same day does nothing
            sent_second = await notifier.check_and_send(app, now=now + timedelta(hours=2))
            self.assertFalse(sent_second)
            self.assertEqual(app.bot.send_photo.await_count, 2)

    async def test_notifier_text_mode_and_registration_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp)
            cfg = _make_bot_config(
                storage,
                maraton=MaratonConfig(
                    enabled=True,
                    event_date="2027-09-05",
                    send_time="09:00",
                    timezone="Europe/Helsinki",
                    chat_ids=(333,),
                    use_card=False,
                ),
            )
            notifier = MaratonNotifier(cfg)

            app = MagicMock()
            app.bot = MagicMock()
            app.bot.send_message = AsyncMock()
            app.bot.send_photo = AsyncMock()

            now = datetime(2027, 8, 29, 6, 30, tzinfo=timezone.utc)  # 09:30 EEST (7 days left)
            sent = await notifier.check_and_send(app, now=now)
            self.assertTrue(sent)
            app.bot.send_photo.assert_not_awaited()
            app.bot.send_message.assert_awaited_once()
            sent_text = app.bot.send_message.await_args.kwargs["text"]
            self.assertIn("7 päivää", sent_text)
            self.assertIn("VIIKKO", sent_text)

            # Register attaches post_init and post_shutdown
            dummy_app = MagicMock()
            dummy_app.post_init = None
            dummy_app.post_shutdown = None
            register(dummy_app, cfg)
            self.assertIsNotNone(dummy_app.post_init)
            self.assertIsNotNone(dummy_app.post_shutdown)


if __name__ == "__main__":
    unittest.main()
