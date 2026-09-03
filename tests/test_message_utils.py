from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from bot.commands.message_utils import (
    reply_in_chunks,
    reply_with_card,
    reply_with_image,
    split_message,
)
from bot.rendering import Card


class MessageUtilsTests(unittest.IsolatedAsyncioTestCase):
    def test_split_message_under_limit(self) -> None:
        self.assertEqual(split_message("hello", 5), ["hello"])

    def test_split_message_over_limit(self) -> None:
        self.assertEqual(split_message("abcdefghij", 4), ["abcd", "efgh", "ij"])

    def test_split_message_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(ValueError):
            split_message("hello", 0)

    async def test_reply_in_chunks_sends_text(self) -> None:
        update = MagicMock()
        update.effective_message.reply_text = AsyncMock()
        await reply_in_chunks(update, "hello world", max_reply_length=5)
        self.assertEqual(update.effective_message.reply_text.call_count, 3)

    async def test_reply_with_image_success(self) -> None:
        update = MagicMock()
        update.effective_message.reply_photo = AsyncMock()

        dummy_png = b"\x89PNG\r\n\x1a\nfake_image_data"
        await reply_with_image(update, dummy_png, caption="Test Caption")

        update.effective_message.reply_photo.assert_awaited_once()
        call_kwargs = update.effective_message.reply_photo.call_args.kwargs
        self.assertEqual(call_kwargs["caption"], "Test Caption")

    async def test_reply_with_image_fallback_on_error(self) -> None:
        update = MagicMock()
        update.effective_message.reply_photo = AsyncMock(side_effect=RuntimeError("Upload failed"))
        update.effective_message.reply_text = AsyncMock()

        dummy_png = b"\x89PNG\r\n\x1a\nfake_image_data"
        await reply_with_image(
            update,
            dummy_png,
            fallback_text="Fallback text message",
            max_reply_length=100,
        )

        update.effective_message.reply_photo.assert_awaited_once()
        update.effective_message.reply_text.assert_awaited_once_with("Fallback text message")

    async def test_reply_with_image_raises_without_fallback(self) -> None:
        update = MagicMock()
        update.effective_message.reply_photo = AsyncMock(side_effect=RuntimeError("Upload failed"))

        dummy_png = b"\x89PNG\r\n\x1a\nfake_image_data"
        with self.assertRaises(RuntimeError):
            await reply_with_image(update, dummy_png)

    async def test_reply_with_card(self) -> None:
        update = MagicMock()
        update.effective_message.reply_photo = AsyncMock()

        card = Card(title="Testikortti").add_text("Sisältöä")
        await reply_with_card(update, card, caption="Otsikko")

        update.effective_message.reply_photo.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
