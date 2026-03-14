import unittest

from bot.commands.link_video_logic import build_yt_dlp_format_selector, extract_urls


class LinkVideoLogicTests(unittest.TestCase):
    def test_extract_urls_returns_empty_for_none(self) -> None:
        self.assertEqual(extract_urls(None), [])

    def test_extract_urls_finds_multiple_urls(self) -> None:
        text = "katso https://example.com/test ja https://youtu.be/test123"
        self.assertEqual(
            extract_urls(text), ["https://example.com/test", "https://youtu.be/test123"]
        )

    def test_extract_urls_strips_trailing_punctuation(self) -> None:
        text = "video: https://example.com/video.mp4)."
        self.assertEqual(extract_urls(text), ["https://example.com/video.mp4"])

    def test_build_yt_dlp_format_selector_prefers_vp9_then_avc(self) -> None:
        selector = build_yt_dlp_format_selector(360)
        self.assertEqual(
            selector,
            "bv*[height<=360][vcodec^=vp09]+ba/"
            "bv*[height<=360][vcodec^=avc]+ba/"
            "b[height<=360]",
        )


if __name__ == "__main__":
    unittest.main()
