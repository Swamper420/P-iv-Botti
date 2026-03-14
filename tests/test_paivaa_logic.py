import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bot.commands.paivaa_logic import (
    PAIVAA_HISTORY_FILE,
    PAIVAA_HISTORY_SIZE,
    build_paivaa_ai_prompt,
    ensure_unique_paivaa_reply,
    get_paivaa_reply,
    load_recent_paivaa_replies,
    store_recent_paivaa_reply,
)


class PaivaaLogicTests(unittest.TestCase):
    def test_matches_paivaa_with_whitespace(self) -> None:
        self.assertEqual(get_paivaa_reply("  Päivää  "), "Päivää *tips fedora*")

    def test_does_not_match_other_text(self) -> None:
        self.assertIsNone(get_paivaa_reply("hello"))

    def test_does_not_match_none(self) -> None:
        self.assertIsNone(get_paivaa_reply(None))

    def test_store_recent_paivaa_reply_keeps_last_four(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            storage_dir = Path(tmp_dir)
            for index in range(6):
                store_recent_paivaa_reply(storage_dir, f"reply-{index}")

            history = load_recent_paivaa_replies(storage_dir)
            self.assertEqual(history, ["reply-2", "reply-3", "reply-4", "reply-5"])
            self.assertLessEqual(len(history), PAIVAA_HISTORY_SIZE)

    def test_load_recent_paivaa_replies_returns_empty_for_invalid_data(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            storage_dir = Path(tmp_dir)
            (storage_dir / PAIVAA_HISTORY_FILE).write_text("{bad-json", encoding="utf-8")
            self.assertEqual(load_recent_paivaa_replies(storage_dir), [])

    def test_ensure_unique_paivaa_reply_rewrites_duplicate(self) -> None:
        unique = ensure_unique_paivaa_reply("rawr uwu", ["rawr uwu", "other"])
        self.assertNotEqual(unique, "rawr uwu")
        self.assertTrue(unique.startswith("rawr uwu"))

    def test_build_paivaa_ai_prompt_contains_avoid_list(self) -> None:
        prompt = build_paivaa_ai_prompt(["a", "b"])
        self.assertIn("Älä toista", prompt)
        self.assertIn("- a", prompt)
        self.assertIn("- b", prompt)

if __name__ == "__main__":
    unittest.main()
