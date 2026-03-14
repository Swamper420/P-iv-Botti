import unittest

from bot.cs2_rss import Cs2UpdateItem, format_cs2_update, parse_cs2_rss_items


class Cs2RssTests(unittest.TestCase):
    def test_parse_cs2_rss_items_extracts_plain_text_description(self) -> None:
        rss = """<?xml version="1.0"?>
<rss>
  <channel>
    <item>
      <title>Patch Notes</title>
      <author>Valve</author>
      <link>https://example.invalid/update</link>
      <guid>guid-1</guid>
      <description><![CDATA[<p>Hello &amp; welcome <b>players</b></p>]]></description>
    </item>
  </channel>
</rss>"""

        items = parse_cs2_rss_items(rss)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].update_id, "guid-1")
        self.assertEqual(items[0].title, "Patch Notes")
        self.assertEqual(items[0].author, "Valve")
        self.assertEqual(items[0].link, "https://example.invalid/update")
        self.assertEqual(items[0].description, "Hello & welcome players")

    def test_parse_cs2_rss_items_falls_back_to_link_or_title_for_id(self) -> None:
        rss = """<?xml version="1.0"?>
<rss>
  <channel>
    <item>
      <title>Update title</title>
      <link>https://example.invalid/no-guid</link>
      <description>desc</description>
    </item>
  </channel>
</rss>"""

        items = parse_cs2_rss_items(rss)

        self.assertEqual(items[0].update_id, "https://example.invalid/no-guid")

    def test_format_cs2_update_is_telegram_friendly(self) -> None:
        item = Cs2UpdateItem(
            update_id="id-1",
            title="Patch Notes",
            author="Valve",
            link="https://example.invalid/update",
            description="A short summary",
        )

        message = format_cs2_update(item)

        self.assertEqual(
            message,
            "\n".join(
                [
                    "🎮 CS2 update",
                    "Patch Notes",
                    "👤 Valve",
                    "📝 A short summary",
                    "🔗 https://example.invalid/update",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
