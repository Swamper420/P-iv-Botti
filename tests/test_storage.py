import tempfile
import unittest
from pathlib import Path

from bot.storage import load_json_data, save_json_data


class StorageTests(unittest.TestCase):
    def test_saves_and_loads_json_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "sub" / "data.json"
            save_json_data(file_path, {"key": "value", "numbers": [1, 2, 3]})

            loaded = load_json_data(file_path, default_factory=dict)
            self.assertEqual(loaded, {"key": "value", "numbers": [1, 2, 3]})

    def test_load_json_data_returns_default_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "nonexistent.json"
            loaded = load_json_data(file_path, default_factory=list)
            self.assertEqual(loaded, [])

    def test_load_json_data_returns_default_if_corrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "corrupt.json"
            file_path.write_text("invalid json...", encoding="utf-8")
            loaded = load_json_data(file_path, default_factory=set)
            self.assertEqual(loaded, set())


if __name__ == "__main__":
    unittest.main()
