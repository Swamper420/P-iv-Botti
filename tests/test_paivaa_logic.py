import unittest

from bot.commands.paivaa_logic import get_paivaa_reply, split_message


class PaivaaLogicTests(unittest.TestCase):
    def test_matches_paivaa_with_whitespace(self) -> None:
        self.assertEqual(get_paivaa_reply("  Päivää  "), "Päivää *tips fedora*")

    def test_does_not_match_other_text(self) -> None:
        self.assertIsNone(get_paivaa_reply("hello"))

    def test_does_not_match_none(self) -> None:
        self.assertIsNone(get_paivaa_reply(None))

    def test_split_message_under_limit(self) -> None:
        self.assertEqual(split_message("hello", 5), ["hello"])

    def test_split_message_over_limit(self) -> None:
        self.assertEqual(split_message("abcdefghij", 4), ["abcd", "efgh", "ij"])


if __name__ == "__main__":
    unittest.main()
