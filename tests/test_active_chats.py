import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bot.active_chats import load_active_chat_ids, track_active_chat


class ActiveChatsTests(unittest.TestCase):
    def test_track_active_chat_persists_chat_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            update = SimpleNamespace(effective_chat=SimpleNamespace(id=12345))

            track_active_chat(update, storage_dir)

            self.assertEqual(load_active_chat_ids(storage_dir), {12345})

    def test_track_active_chat_ignores_missing_chat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            update = SimpleNamespace(effective_chat=None)

            track_active_chat(update, storage_dir)

            self.assertEqual(load_active_chat_ids(storage_dir), set())


if __name__ == "__main__":
    unittest.main()
