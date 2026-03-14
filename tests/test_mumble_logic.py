import unittest

from bot.commands.mumble_logic import (
    format_duration,
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

    def test_formats_status_report_for_channels_and_users(self) -> None:
        report = format_mumble_status_report(
            server_address="127.0.0.1:64738",
            channels=[
                {
                    "name": "Alpha",
                    "users": [
                        {
                            "name": "Tester",
                            "session": 9,
                            "user_id": 7,
                            "online_seconds": 65,
                            "idle_seconds": 5,
                            "mute": False,
                            "deaf": False,
                            "self_mute": True,
                            "self_deaf": False,
                            "suppress": False,
                            "recording": False,
                            "extras": {"hash": "abc"},
                        }
                    ],
                },
                {"name": "Beta", "users": []},
            ],
        )

        self.assertIn("Mumble (127.0.0.1:64738)", report)
        self.assertIn("Kanavia seurannassa: 2", report)
        self.assertIn("• Alpha (1 käyttäjää)", report)
        self.assertIn("online=00:01:05, idle=00:00:05", report)
        self.assertIn("extra: hash=abc", report)
        self.assertIn("• Beta (0 käyttäjää)", report)
        self.assertNotIn("MUMBLE_TARGET_CHANNELS", report)


if __name__ == "__main__":
    unittest.main()
