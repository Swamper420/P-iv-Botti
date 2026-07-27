from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine
import urllib.parse
import urllib.request

import websockets

from bot.config import TwitchConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TwitchStreamNotification:
    broadcaster_user_id: str
    broadcaster_login: str
    broadcaster_name: str
    title: str
    game_name: str
    stream_url: str
    thumbnail_url: str
    started_at: str

    def format_telegram_message(self) -> str:
        lines = [
            f"🔴 <b>{self.broadcaster_name} is LIVE on Twitch!</b>",
            "",
            f"<b>Title:</b> {self.title}" if self.title else "",
            f"<b>Category:</b> {self.game_name}" if self.game_name else "",
            f"<b>Link:</b> <a href=\"{self.stream_url}\">{self.stream_url}</a>",
        ]
        return "\n".join([line for line in lines if line])


class TwitchClient:
    def __init__(self, config: TwitchConfig) -> None:
        self.config = config
        self._access_token: str | None = None
        self._token_expires_at: float = 0

    def get_app_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._token_expires_at - 60:
            return self._access_token

        params = urllib.parse.urlencode({
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "grant_type": "client_credentials",
        })
        url = f"{self.config.token_url}?{params}"
        req = urllib.request.Request(url, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self._access_token = data["access_token"]
                expires_in = data.get("expires_in", 3600)
                self._token_expires_at = now + expires_in
                return self._access_token
        except Exception as err:
            LOGGER.error("Failed to acquire Twitch app token: %s", err)
            raise

    def get_user_ids(self, logins: tuple[str, ...]) -> dict[str, str]:
        """Resolves login names to Twitch user IDs. Returns dict {login: user_id}."""
        if not logins:
            return {}

        token = self.get_app_token()
        query = "&".join(f"login={urllib.parse.quote(login)}" for login in logins)
        url = f"{self.config.helix_base_url}/users?{query}"
        req = urllib.request.Request(
            url,
            headers={
                "Client-ID": self.config.client_id,
                "Authorization": f"Bearer {token}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result: dict[str, str] = {}
                for user in data.get("data", []):
                    result[user["login"].lower()] = user["id"]
                return result
        except Exception as err:
            LOGGER.error("Failed to fetch Twitch user IDs for %s: %s", logins, err)
            return {}

    def get_stream_info(self, user_id: str) -> dict[str, Any] | None:
        """Fetches live stream metadata for a user_id."""
        token = self.get_app_token()
        url = f"{self.config.helix_base_url}/streams?user_id={urllib.parse.quote(user_id)}"
        req = urllib.request.Request(
            url,
            headers={
                "Client-ID": self.config.client_id,
                "Authorization": f"Bearer {token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                streams = data.get("data", [])
                if streams:
                    return streams[0]
                return None
        except Exception as err:
            LOGGER.error("Failed to fetch Twitch stream info for user %s: %s", user_id, err)
            return None

    def subscribe_eventsub_websocket(self, session_id: str, broadcaster_user_id: str) -> bool:
        token = self.get_app_token()
        url = f"{self.config.helix_base_url}/eventsub/subscriptions"
        payload = {
            "type": "stream.online",
            "version": "1",
            "condition": {
                "broadcaster_user_id": broadcaster_user_id,
            },
            "transport": {
                "method": "websocket",
                "session_id": session_id,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Client-ID": self.config.client_id,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 202)
        except Exception as err:
            LOGGER.error("Failed to subscribe EventSub for broadcaster %s: %s", broadcaster_user_id, err)
            return False


OnStreamOnlineCallback = Callable[[TwitchStreamNotification], Coroutine[Any, Any, None]]


class TwitchEventSubNotifier:
    def __init__(
        self,
        config: TwitchConfig,
        on_stream_online: OnStreamOnlineCallback,
        client: TwitchClient | None = None,
    ) -> None:
        self.config = config
        self.on_stream_online = on_stream_online
        self.client = client or TwitchClient(config)
        self._user_map: dict[str, str] = {}  # {user_id: login}
        self._running = False

    async def start(self) -> None:
        if not self.config.is_configured:
            LOGGER.info("Twitch integration is not fully configured. Skipping background task.")
            return

        self._running = True
        ws_url = self.config.websocket_url

        while self._running:
            try:
                # Resolve channel user IDs before connecting
                user_id_map = await asyncio.to_thread(self.client.get_user_ids, self.config.channels)
                self._user_map = {uid: login for login, uid in user_id_map.items()}

                if not self._user_map:
                    LOGGER.warning("Could not resolve any Twitch channels: %s. Retrying in %ds...", self.config.channels, self.config.reconnect_delay_seconds)
                    await asyncio.sleep(self.config.reconnect_delay_seconds)
                    continue

                LOGGER.info("Connecting to Twitch EventSub WebSocket at %s...", ws_url)
                ws_url = await self._run_websocket_session(ws_url)
            except asyncio.CancelledError:
                LOGGER.info("Twitch EventSub task cancelled.")
                self._running = False
                break
            except Exception as err:
                LOGGER.error("Error in Twitch EventSub WebSocket loop: %s", err, exc_info=True)
                ws_url = self.config.websocket_url
                await asyncio.sleep(self.config.reconnect_delay_seconds)

    async def _run_websocket_session(self, ws_url: str) -> str:
        """Runs single WebSocket session loop. Returns next websocket URL (for reconnect) or base URL."""
        next_ws_url = self.config.websocket_url

        async with websockets.connect(ws_url) as ws:
            async for raw_msg in ws:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue

                metadata = data.get("metadata", {})
                message_type = metadata.get("message_type")
                payload = data.get("payload", {})

                if message_type == "session_welcome":
                    session_id = payload.get("session", {}).get("id")
                    LOGGER.info("Twitch EventSub WebSocket connected (session: %s). Subscribing channels...", session_id)
                    if session_id:
                        for uid in self._user_map.keys():
                            ok = await asyncio.to_thread(
                                self.client.subscribe_eventsub_websocket, session_id, uid
                            )
                            if ok:
                                LOGGER.info("Subscribed stream.online for Twitch user %s", uid)

                elif message_type == "session_reconnect":
                    reconnect_url = payload.get("session", {}).get("reconnect_url")
                    if reconnect_url:
                        LOGGER.info("Received Twitch session_reconnect URL: %s", reconnect_url)
                        next_ws_url = reconnect_url
                        break

                elif message_type == "notification":
                    subscription = payload.get("subscription", {})
                    sub_type = subscription.get("type")
                    if sub_type == "stream.online":
                        await self._handle_stream_online_event(payload.get("event", {}))

                elif message_type == "session_keepalive":
                    pass

        return next_ws_url

    async def _handle_stream_online_event(self, event: dict[str, Any]) -> None:
        user_id = str(event.get("broadcaster_user_id", ""))
        login = str(event.get("broadcaster_user_login", "")).lower()
        name = str(event.get("broadcaster_user_name", login))
        started_at = str(event.get("started_at", ""))

        LOGGER.info("Received stream.online event for %s (%s)", name, login)

        # Query Helix streams API for title & game category
        stream_info = await asyncio.to_thread(self.client.get_stream_info, user_id)

        title = stream_info.get("title", "") if stream_info else ""
        game_name = stream_info.get("game_name", "") if stream_info else ""
        thumbnail_template = stream_info.get("thumbnail_url", "") if stream_info else ""
        thumbnail_url = thumbnail_template.replace("{width}", "1280").replace("{height}", "720") if thumbnail_template else ""

        notification = TwitchStreamNotification(
            broadcaster_user_id=user_id,
            broadcaster_login=login,
            broadcaster_name=name,
            title=title,
            game_name=game_name,
            stream_url=f"https://twitch.tv/{login}",
            thumbnail_url=thumbnail_url,
            started_at=started_at,
        )

        try:
            await self.on_stream_online(notification)
        except Exception as err:
            LOGGER.error("Failed executing on_stream_online callback for %s: %s", login, err, exc_info=True)
