import unittest

from bot.commands.stt_logic import (
    is_audio_duration_allowed,
    parse_transcription_response,
)


class SttLogicTests(unittest.TestCase):
    def test_accepts_duration_within_limit(self) -> None:
        self.assertTrue(is_audio_duration_allowed(600, 600))

    def test_rejects_missing_or_too_long_duration(self) -> None:
        self.assertFalse(is_audio_duration_allowed(None, 600))
        self.assertFalse(is_audio_duration_allowed(601, 600))

    def test_parses_json_text_response(self) -> None:
        self.assertEqual(
            parse_transcription_response('{"transcript":"  terve maailma  "}'),
            "terve maailma",
        )

    def test_parses_plain_text_response(self) -> None:
        self.assertEqual(parse_transcription_response("  hei  "), "hei")

    def test_returns_none_for_unusable_payload(self) -> None:
        self.assertIsNone(parse_transcription_response("{}"))


if __name__ == "__main__":
    unittest.main()
