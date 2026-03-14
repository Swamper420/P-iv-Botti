import unittest

from bot.commands.weather_logic import parse_weather_camera_location


class WeatherLogicTests(unittest.TestCase):
    def test_extracts_location_from_weather_command(self) -> None:
        self.assertEqual(
            parse_weather_camera_location("!sääkuva Helsinki"), (True, "Helsinki")
        )

    def test_extracts_location_from_ascii_alias(self) -> None:
        self.assertEqual(
            parse_weather_camera_location("  !saakuva  Oulu "), (True, "Oulu")
        )

    def test_matches_command_without_location(self) -> None:
        self.assertEqual(parse_weather_camera_location("!sääkuva"), (True, None))

    def test_ignores_non_command_text(self) -> None:
        self.assertEqual(parse_weather_camera_location("aih: test"), (False, None))


if __name__ == "__main__":
    unittest.main()
