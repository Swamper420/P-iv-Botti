from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.request import urlopen

from bot.active_chats import load_active_chat_ids
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cs2UpdateItem:
    update_id: str
    title: str
    author: str
    link: str
    description: str


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_data(self) -> str:
        return "".join(self._parts).strip()


def _plain_text_from_html(value: str) -> str:
    if not value:
        return ""

    parser = _HTMLStripper()
    parser.feed(value)
    return unescape(parser.get_data())


def parse_cs2_rss_items(xml_text: str) -> list[Cs2UpdateItem]:
    root = ET.fromstring(xml_text)
    items: list[Cs2UpdateItem] = []

    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        author = (item.findtext("author") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = _plain_text_from_html((item.findtext("description") or "").strip())
        update_id = (item.findtext("guid") or link or title).strip()
        items.append(
            Cs2UpdateItem(
                update_id=update_id,
                title=title,
                author=author,
                link=link,
                description=description,
            )
        )

    return items


def format_cs2_update(item: Cs2UpdateItem) -> str:
    parts = ["🎮 CS2 update", item.title]
    if item.author:
        parts.append(f"👤 {item.author}")
    if item.description:
        parts.append(f"📝 {item.description}")
    if item.link:
        parts.append(f"🔗 {item.link}")
    return "\n".join(parts)


class Cs2RssNotifier:
    def __init__(self, config: BotConfig) -> None:
        self._config = config
        self._seen_update_ids: set[str] = set()
        self._seen_initialized = False
        self._task: asyncio.Task[None] | None = None

    def start(self, app) -> None:
        if self._task is not None:
            return
        self._task = app.create_task(self._run(app))

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self, app) -> None:
        while True:
            await self.check_updates(app)
            await asyncio.sleep(self._config.steam_rss_poll_interval_seconds)

    async def _fetch_rss(self) -> str | None:
        def _fetch() -> str:
            with urlopen(
                self._config.steam_cs2_rss_url,
                timeout=self._config.steam_rss_request_timeout_seconds,
            ) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise RuntimeError(f"Steam RSS returned status {status}")
                return response.read().decode("utf-8")

        try:
            return await asyncio.to_thread(_fetch)
        except (URLError, TimeoutError, OSError, RuntimeError):
            LOGGER.exception("Failed to fetch Steam RSS feed")
            return None

    async def check_updates(self, app) -> None:
        rss_xml = await self._fetch_rss()
        if rss_xml is None:
            return

        try:
            items = parse_cs2_rss_items(rss_xml)
        except ET.ParseError:
            LOGGER.exception("Failed to parse Steam RSS feed")
            return

        if not items:
            return

        if not self._seen_initialized:
            self._seen_update_ids = {item.update_id for item in items if item.update_id}
            self._seen_initialized = True
            return

        new_items = [item for item in items if item.update_id not in self._seen_update_ids]
        active_chat_ids = load_active_chat_ids(self._config.storage_dir)
        if not active_chat_ids:
            for item in new_items:
                if item.update_id:
                    self._seen_update_ids.add(item.update_id)
            return

        for item in reversed(new_items):
            message = format_cs2_update(item)
            for chat_id in active_chat_ids:
                try:
                    await app.bot.send_message(chat_id=chat_id, text=message)
                except Exception:
                    LOGGER.exception("Failed to send CS2 update to chat %s", chat_id)
            if item.update_id:
                self._seen_update_ids.add(item.update_id)
