import unittest

from bot.commands.mumble_logic import (
    build_mumble_user_tracking_key,
    format_duration,
    format_mumble_channel_notice,
    format_mumble_status_report,
    is_mumble_status_command,
    resolve_online_seconds,
    update_mumble_connection_tracker,
)


class MumbleLogicTests(unittest.TestCase):
    def test_matches_mumble_command_with_or_without_prefix(self) -> None:
        self.assertTrue(is_mumble_status_command("mumble"))
        self.assertTrue(is_mumble_status_command(" !mumble "))
        self.assertFalse(is_mumble_status_command("mumble now"))

    def test_formats_duration_hms(self) -> None:
        self.assertEqual(format_duration(3723), "01:02:03")
        self.assertEqual(format_duration(None), "ei saatavilla")

    def test_formats_channel_notice_with_requester(self) -> None:
        self.assertEqual(
            format_mumble_channel_notice("@swamper"),
            "📣 Telegramissa !mumble-komennon käytti: @swamper",
        )

    def test_formats_status_report_for_channels_and_users(self) -> None:
        report = format_mumble_status_report(
            server_address="127.0.0.1:64738",
            channels=[
                {
                    "name": "Alpha",
                    "users": [
                        {
                            "name": "Tester",
                            "online_seconds": 65,
                            "muted": True,
                            "deafened": False,
                        }
                    ],
                },
                {"name": "Beta", "users": []},
            ],
        )

        self.assertIn("Mumble (127.0.0.1:64738)", report)
        self.assertIn("Kanavat: 2 | Käyttäjät: 1", report)
        self.assertIn("• Alpha (1)", report)
        self.assertIn("Tester | ⏱ 00:01:05 | mute kyllä | deaf ei", report)
        self.assertIn("• Beta (0)", report)
        self.assertNotIn("MUMBLE_TARGET_CHANNELS", report)
        self.assertNotIn("session=", report)
        self.assertNotIn("extra:", report)

    def test_builds_tracking_key_from_session_or_name(self) -> None:
        self.assertEqual(
            build_mumble_user_tracking_key({"session": 42, "name": "Tester"}),
            "session:42",
        )
        self.assertEqual(
            build_mumble_user_tracking_key({"name": "  TeSter  "}),
            "name:tester",
        )
        self.assertIsNone(build_mumble_user_tracking_key({}))

    def test_updates_tracker_adds_and_removes_users(self) -> None:
        tracker = {"session:100": 10.0}
        users = [{"session": 1, "name": "Alice"}, {"session": 2, "name": "Bob"}]

        update_mumble_connection_tracker(
            users=users,
            own_session=None,
            connected_since_by_key=tracker,
            now_monotonic=100.0,
        )

        self.assertEqual(tracker["session:1"], 100.0)
        self.assertEqual(tracker["session:2"], 100.0)
        self.assertNotIn("session:100", tracker)

    def test_resolves_online_seconds_from_tracker(self) -> None:
        tracker = {"session:1": 90.0}
        seconds = resolve_online_seconds(
            user={"session": 1, "onlinesecs": None},
            connected_since_by_key=tracker,
            now_monotonic=100.0,
        )
        self.assertEqual(seconds, 10)

    def test_resolve_online_seconds_falls_back_to_onlinesecs(self) -> None:
        seconds = resolve_online_seconds(
            user={"name": "Alice", "onlinesecs": 77},
            connected_since_by_key={},
            now_monotonic=100.0,
        )
        self.assertEqual(seconds, 77)


if __name__ == "__main__":
    unittest.main()
