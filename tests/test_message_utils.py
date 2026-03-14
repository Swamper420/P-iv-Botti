import unittest

from bot.commands.message_utils import split_message


class MessageUtilsTests(unittest.TestCase):
    def test_split_message_under_limit(self) -> None:
        self.assertEqual(split_message("hello", 5), ["hello"])

    def test_split_message_over_limit(self) -> None:
        self.assertEqual(split_message("abcdefghij", 4), ["abcd", "efgh", "ij"])

    def test_split_message_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(ValueError):
            split_message("hello", 0)


if __name__ == "__main__":
    unittest.main()
