import unittest

from bot.commands.mumble_logic import (
    format_duration,
    format_mumble_channel_notice,
    format_mumble_status_report,
    is_mumble_status_command,
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


if __name__ == "__main__":
    unittest.main()
