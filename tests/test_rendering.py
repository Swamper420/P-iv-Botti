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


if __name__ == "__main__":
    unittest.main()
