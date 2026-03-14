import unittest

from bot.commands.aih_logic import get_aih_prompt


class AihLogicTests(unittest.TestCase):
    def test_extracts_aih_prompt(self) -> None:
        self.assertEqual(
            get_aih_prompt("aih: Who was the president of Finland in 2012"),
            "Who was the president of Finland in 2012",
        )

    def test_extracts_aih_prompt_case_insensitive(self) -> None:
        self.assertEqual(get_aih_prompt("  AIH:   test prompt  "), "test prompt")

    def test_does_not_extract_empty_aih_prompt(self) -> None:
        self.assertIsNone(get_aih_prompt("aih:   "))


if __name__ == "__main__":
    unittest.main()
