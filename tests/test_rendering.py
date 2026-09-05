from __future__ import annotations

import unittest
from unittest.mock import patch
from io import BytesIO
from PIL import Image

from bot.rendering import (
    DARK_THEME,
    Badge,
    BadgeColor,
    Card,
    CodeBlockElement,
    DividerElement,
    KeyValuesElement,
    ProgressBarElement,
    TableElement,
    TextElement,
    Theme,
    render_card,
    render_table_card,
    render_text_card,
)


class RenderingTests(unittest.TestCase):
    def _assert_valid_png(self, data: bytes, min_width: int = 300, min_height: int = 50) -> Image.Image:
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"), "Output must start with PNG signature")
        img = Image.open(BytesIO(data))
        self.assertGreaterEqual(img.width, min_width)
        self.assertGreaterEqual(img.height, min_height)
        return img

    def test_render_text_card_produces_valid_png(self) -> None:
        raw_bytes = render_text_card(
            title="Palvelimen tila",
            text="Kaikki järjestelmät toimivat normaalisti.\nEi havaittuja häiriöitä.",
            subtitle="palvelin.example.com",
            badge="ONLINE",
            badge_color="green",
            footer="Päivitetty 18:25",
        )
        img = self._assert_valid_png(raw_bytes)
        self.assertEqual(img.width, 800)

    def test_render_text_card_with_code_block(self) -> None:
        text = "Tässä on komento:\n```\n!mine stats bedrock-1\n```\nJa toinen huomio."
        raw_bytes = render_text_card(
            title="Ohje",
            text=text,
        )
        self._assert_valid_png(raw_bytes)

    def test_render_table_card(self) -> None:
        raw_bytes = render_table_card(
            title="Pelaajatilastot",
            headers=["Pelaaja", "Pisteet", "Taso"],
            rows=[
                ["Matti", "1250", "12"],
                ["Teppo", "980", "9"],
                ["Seppo", "450", "4"],
            ],
            subtitle="Viimeiset 7 päivää",
            badge="TOP 3",
            badge_color="yellow",
        )
        img = self._assert_valid_png(raw_bytes)
        self.assertEqual(img.width, 800)

    def test_render_card_with_all_elements(self) -> None:
        card = (
            Card(
                title="Monipuolinen Kortti",
                subtitle="Kaikki elementtityypit mukana",
                footer="P-iv-Botti Render System",
                accent_color="#22c55e",
            )
            .set_badge("AKTIIVINEN", BadgeColor.GREEN)
            .add_text("Tämä on tavallinen tekstikappale.")
            .add_text("Tämä on lihavoitu teksti.", bold=True)
            .add_text("Tämä on kooditeksti.", is_code=True)
            .add_text("Tämä on himmennetty teksti.", muted=True)
            .add_key_value("Versio", "1.21.1")
            .add_key_value("Tila", Badge("VALMIS", BadgeColor.BLUE))
            .add_divider()
            .add_progress_bar("Muistinkäyttö", value=768, max_value=1024, unit="MB")
            .add_table(
                headers=["Kanava", "Käyttäjiä"],
                rows=[["Aula", "4"], ["Pelihuone", "2"]],
                alignments=["left", "right"],
            )
            .add_code_block("def hello():\n    return 'world'")
        )

        raw_bytes = render_card(card)
        img = self._assert_valid_png(raw_bytes)
        self.assertEqual(img.width, 800)
        self.assertGreater(img.height, 300)

    def test_custom_theme(self) -> None:
        custom_theme = Theme(
            card_width=600,
            card_bg="#0f172a",
            accent_color="#ec4899",
            text_primary="#ffffff",
        )
        card = Card(title="Mukautettu teema").add_text("Sisältöä mukautetulla leveydellä")
        raw_bytes = render_card(card, theme=custom_theme)
        img = self._assert_valid_png(raw_bytes, min_width=600)
        self.assertEqual(img.width, 600)

    def test_badge_colors(self) -> None:
        for color in [
            BadgeColor.GREEN,
            BadgeColor.RED,
            BadgeColor.YELLOW,
            BadgeColor.BLUE,
            BadgeColor.PURPLE,
            BadgeColor.GRAY,
            "custom_unknown",
        ]:
            badge = Badge(text="TEST", color=color)
            style = badge.get_style()
            self.assertTrue(style.bg_color.startswith("#"))
            self.assertTrue(style.fg_color.startswith("#"))
            self.assertTrue(style.border_color.startswith("#"))

    def test_no_rounded_corners_called(self) -> None:
        """Verify that rounded_rectangle is never called anywhere during rendering."""
        card = (
            Card(title="Kulmatarkistus")
            .set_badge("TEST", "green")
            .add_progress_bar("RAM", 50, 100)
            .add_code_block("print('test')")
            .add_table(["A", "B"], [["1", "2"]])
        )

        with patch("PIL.ImageDraw.ImageDraw.rounded_rectangle") as mock_rounded:
            render_card(card)
            mock_rounded.assert_not_called()

    def test_table_overflow_and_max_rows(self) -> None:
        """Verify table renders cleanly with text overflow and max_rows."""
        long_name = "SuperExtremelyLongPlayerNameThatExceedsTheColumnBoundaryByFar123456789"
        rows = [[f"Player_{i}", long_name, "1000"] for i in range(20)]
        card = Card(title="Overflow Test").add_table(
            headers=["Pelaaja", "Kuvaus", "Pisteet"],
            rows=rows,
            col_widths=[1, 3, 1],
            max_rows=5,
            overflow="ellipsis",
        )
        raw_bytes = render_card(card)
        img = self._assert_valid_png(raw_bytes)
        self.assertEqual(img.width, 800)

    def test_keyvalues_and_codeblock_overflow(self) -> None:
        """Verify key-values and code blocks truncate gracefully on overflow."""
        card = (
            Card(title="KeyValues Overflow")
            .add_key_value(
                "ErittäinPitkäAvainNimiJokaOnPitkä",
                "ErittäinPitkäArvoTekstiJokaMuutenYlittäisiPalstanLeveydenJaAiheuttaisiSotkua",
            )
            .add_code_block(
                "x = '" + "A" * 200 + "'\n" * 15,
                max_lines=4,
                overflow="ellipsis",
            )
        )
        raw_bytes = render_card(card)
        img = self._assert_valid_png(raw_bytes)
        self.assertEqual(img.width, 800)

    def test_text_element_max_lines_and_card_max_height(self) -> None:
        """Verify TextElement max_lines and Card max_height constraints."""
        card = (
            Card(title="Max Height Card", max_height=400)
            .add_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6", max_lines=3)
            .add_text("Extra long content " * 30)
        )
        raw_bytes = render_card(card)
        img = self._assert_valid_png(raw_bytes)
        self.assertLessEqual(img.height, 400)

    def _make_test_jpeg(self, width: int = 640, height: int = 480, color: str = "#334155") -> bytes:
        jpeg_img = Image.new("RGB", (width, height), color=color)
        buf = BytesIO()
        jpeg_img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_image_element_renders_valid_png(self) -> None:
        """Verify photo element embeds with sharp border and caption."""
        img_bytes = self._make_test_jpeg()
        card = (
            Card(title="Sääkuva", subtitle="Kaisaniemi • C12345", footer="Testi")
            .set_badge("18.5°C", "yellow")
            .add_text("Selkeää — 18.5°C", bold=True)
            .add_image(img_bytes, caption="Kaisaniemi • C12345", max_height=420)
            .add_key_value("Kosteus", "55%")
        )
        raw_bytes = render_card(card)
        img = self._assert_valid_png(raw_bytes)
        self.assertEqual(img.width, 800)
        self.assertGreater(img.height, 500)

    def test_image_element_tall_photo_respects_max_height(self) -> None:
        """Tall photos must be constrained to max_height and stay centered."""
        tall_bytes = self._make_test_jpeg(width=640, height=1000)
        card = Card(title="Pitkä kuva").add_image(tall_bytes, max_height=300)
        raw_bytes = render_card(card)
        img = self._assert_valid_png(raw_bytes)
        # 300px photo + header/footer chrome should stay compact
        self.assertLess(img.height, 600)

    def test_image_element_corrupt_bytes_renders_placeholder(self) -> None:
        """Corrupt image bytes must not crash rendering."""
        card = Card(title="Rikki").add_image(b"not-an-image", caption="kuva")
        raw_bytes = render_card(card)
        self._assert_valid_png(raw_bytes)

    def test_image_element_no_rounded_corners(self) -> None:
        """Photo rendering must never use rounded rectangles."""
        img_bytes = self._make_test_jpeg()
        card = Card(title="Kulmat").add_image(img_bytes, caption="cap")
        with patch("PIL.ImageDraw.ImageDraw.rounded_rectangle") as mock_rounded:
            render_card(card)
            mock_rounded.assert_not_called()

    def test_image_grid_renders_2x2(self) -> None:
        """Grid element must render up to 4 photos with labels and caption."""
        imgs = [self._make_test_jpeg(color=c) for c in ("#334155", "#475569", "#64748b", "#94a3b8")]
        for n in (1, 2, 3, 4):
            card = Card(title=f"Ruudukko {n}").add_image_grid(
                imgs[:n], labels=[str(i + 1) for i in range(n)], caption="Kaisaniemi"
            )
            raw_bytes = render_card(card)
            img = self._assert_valid_png(raw_bytes)
            self.assertEqual(img.width, 800)

    def test_image_grid_caps_at_four(self) -> None:
        """More than 4 images must be capped to a 2x2 grid."""
        imgs = [self._make_test_jpeg() for _ in range(6)]
        card = Card(title="Paljon").add_image_grid(imgs, labels=[str(i + 1) for i in range(6)])
        raw_bytes = render_card(card)
        img = self._assert_valid_png(raw_bytes)
        self.assertEqual(img.width, 800)
        # 2 rows of capped cells stay compact (well below 3-row height).
        self.assertLess(img.height, 900)

    def test_image_grid_corrupt_and_empty(self) -> None:
        """Corrupt bytes render as placeholders; empty grid renders caption only."""
        card = Card(title="Rikki ruudukko").add_image_grid(
            [b"not-an-image", self._make_test_jpeg()], labels=["1", "2"]
        )
        self._assert_valid_png(render_card(card))
        empty = Card(title="Tyhjä").add_image_grid([], caption="ei kuvia")
        self._assert_valid_png(render_card(empty))

    def test_image_grid_no_rounded_corners(self) -> None:
        """Grid rendering must never use rounded rectangles."""
        imgs = [self._make_test_jpeg(), self._make_test_jpeg()]
        card = Card(title="Kulmat").add_image_grid(imgs, labels=["1", "2"])
        with patch("PIL.ImageDraw.ImageDraw.rounded_rectangle") as mock_rounded:
            render_card(card)
            mock_rounded.assert_not_called()


if __name__ == "__main__":
    unittest.main()

