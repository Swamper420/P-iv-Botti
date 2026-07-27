from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine
import urllib.error
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


@dataclass(frozen=True)
class TwitchStreamSummaryNotification:
    broadcaster_user_id: str
    broadcaster_login: str
    broadcaster_name: str
    duration_seconds: int
    peak_viewers: int
    title: str
    game_name: str
    stream_url: str
    vod_url: str | None = None

    def format_telegram_message(self) -> str:
        hours = self.duration_seconds // 3600
        minutes = (self.duration_seconds % 3600) // 60
        if hours > 0:
            duration_str = f"{hours}h {minutes}m"
        else:
            duration_str = f"{max(1, minutes)}m"

        lines = [
            f"🏁 <b>{self.broadcaster_name} stream ended!</b>",
            "",
            f"⏱ <b>Duration:</b> {duration_str}",
            f"📊 <b>Peak Viewers:</b> {self.peak_viewers:,}" if self.peak_viewers > 0 else "",
            f"🎮 <b>Category:</b> {self.game_name}" if self.game_name else "",
            f"📝 <b>Title:</b> {self.title}" if self.title else "",
            f"📹 <b>VOD:</b> <a href=\"{self.vod_url}\">{self.vod_url}</a>" if self.vod_url else f"🔗 <b>Channel:</b> <a href=\"{self.stream_url}\">{self.stream_url}</a>",
        ]
        return "\n".join([line for line in lines if line])


@dataclass
class StreamTracker:
    broadcaster_user_id: str
    broadcaster_login: str
    broadcaster_name: str
    start_time: float
    title: str
    game_name: str
    peak_viewers: int = 0


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

    def get_latest_vod_url(self, user_id: str) -> str | None:
        """Fetches latest archive VOD URL for a user_id."""
        token = self.get_app_token()
        url = f"{self.config.helix_base_url}/videos?user_id={urllib.parse.quote(user_id)}&type=archive&first=1"
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
                videos = data.get("data", [])
                if videos:
                    return videos[0].get("url")
                return None
        except Exception as err:
            LOGGER.error("Failed to fetch Twitch latest VOD for user %s: %s", user_id, err)
            return None

    def subscribe_eventsub_websocket(
        self, session_id: str, broadcaster_user_id: str, event_type: str = "stream.online"
    ) -> bool:
        if not self.config.user_access_token:
            LOGGER.warning(
                "Cannot subscribe EventSub via WebSocket: TWITCH_USER_ACCESS_TOKEN is not configured "
                "(Twitch requires a User Access Token for WebSocket subscriptions)."
            )
            return False

        token = self.config.user_access_token
        url = f"{self.config.helix_base_url}/eventsub/subscriptions"
        payload = {
            "type": event_type,
            "version": "1",
            "condition": {
                "broadcaster_user_id": str(broadcaster_user_id),
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
        except urllib.error.HTTPError as err:
            try:
                body_text = err.read().decode("utf-8", errors="ignore")
            except Exception:
                body_text = ""
            if "invalid transport and auth combination" in body_text:
                LOGGER.warning(
                    "Twitch rejected WebSocket EventSub subscription for broadcaster %s "
                    "(invalid transport and auth combination: User Access Token required). Fallback to polling.",
                    broadcaster_user_id,
                )
            else:
                LOGGER.error(
                    "Failed to subscribe EventSub for broadcaster %s: %s - Body: %s",
                    broadcaster_user_id,
                    err,
                    body_text,
                )
            return False
        except Exception as err:
            LOGGER.error("Failed to subscribe EventSub for broadcaster %s: %s", broadcaster_user_id, err)
            return False



OnStreamOnlineCallback = Callable[[TwitchStreamNotification], Coroutine[Any, Any, None]]
OnStreamOfflineCallback = Callable[[TwitchStreamSummaryNotification], Coroutine[Any, Any, None]]
OnTokenExpiredCallback = Callable[[str], Coroutine[Any, Any, None]]


class TwitchEventSubNotifier:
    def __init__(
        self,
        config: TwitchConfig,
        on_stream_online: OnStreamOnlineCallback,
        client: TwitchClient | None = None,
        on_token_expired: OnTokenExpiredCallback | None = None,
        on_stream_offline: OnStreamOfflineCallback | None = None,
    ) -> None:
        self.config = config
        self.on_stream_online = on_stream_online
        self.on_stream_offline = on_stream_offline
        self.on_token_expired = on_token_expired
        self.client = client or TwitchClient(config)
        self._user_map: dict[str, str] = {}  # {user_id: login}
        self._live_user_ids: set[str] = set()
        self._active_stream_trackers: dict[str, StreamTracker] = {}
        self._use_polling_fallback = False
        self._running = False
        self._token_expired_notified = False

    async def start(self) -> None:
        if not self.config.is_configured:
            LOGGER.info("Twitch integration is not fully configured. Skipping background task.")
            return

        self._running = True
        ws_url = self.config.websocket_url

        if not self.config.user_access_token:
            LOGGER.info(
                "TWITCH_USER_ACCESS_TOKEN is not configured. Twitch EventSub WebSockets require a User Access Token. "
                "Using periodic API polling (interval: %ds)...",
                self.config.poll_interval_seconds,
            )
            self._use_polling_fallback = True

        while self._running:
            if self._use_polling_fallback:
                await self._run_polling_loop()
                break

            try:
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
                LOGGER.error("Error in Twitch EventSub WebSocket loop: %s. Switching to polling fallback.", err, exc_info=True)
                self._use_polling_fallback = True
                ws_url = self.config.websocket_url

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
                    success_count = 0
                    if session_id:
                        for uid in self._user_map.keys():
                            ok_online = await asyncio.to_thread(
                                self.client.subscribe_eventsub_websocket, session_id, uid, "stream.online"
                            )
                            ok_offline = await asyncio.to_thread(
                                self.client.subscribe_eventsub_websocket, session_id, uid, "stream.offline"
                            )
                            if ok_online and ok_offline:
                                success_count += 1
                                LOGGER.info("Subscribed stream events for Twitch user %s", uid)

                    if success_count == 0 and self._user_map:
                        LOGGER.warning(
                            "EventSub WebSocket subscriptions failed (Twitch requires a User Access Token for WebSockets). "
                            "Switching automatically to periodic API polling (interval: %ds)...",
                            self.config.poll_interval_seconds,
                        )
                        self._use_polling_fallback = True
                        if self.config.user_access_token and self.on_token_expired and not self._token_expired_notified:
                            self._token_expired_notified = True
                            try:
                                await self.on_token_expired(
                                    "Twitch WebSocket subscription failed. Your TWITCH_USER_ACCESS_TOKEN may be expired or invalid."
                                )
                            except Exception as err:
                                LOGGER.error("Failed executing on_token_expired callback: %s", err, exc_info=True)
                        break

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
                    elif sub_type == "stream.offline":
                        await self._handle_stream_offline_event(payload.get("event", {}))

                elif message_type == "session_keepalive":
                    pass

        return next_ws_url

    async def _handle_stream_online_event(self, event: dict[str, Any]) -> None:
        user_id = str(event.get("broadcaster_user_id", ""))
        login = str(event.get("broadcaster_user_login", "")).lower()
        name = str(event.get("broadcaster_user_name", login))
        started_at = str(event.get("started_at", ""))

        LOGGER.info("Received stream.online event for %s (%s)", name, login)

        stream_info = await asyncio.to_thread(self.client.get_stream_info, user_id)

        title = stream_info.get("title", "") if stream_info else ""
        game_name = stream_info.get("game_name", "") if stream_info else ""
        thumbnail_template = stream_info.get("thumbnail_url", "") if stream_info else ""
        thumbnail_url = thumbnail_template.replace("{width}", "1280").replace("{height}", "720") if thumbnail_template else ""
        viewer_count = int(stream_info.get("viewer_count", 0)) if stream_info else 0

        start_ts = time.time()
        if started_at:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                start_ts = dt.timestamp()
            except Exception:
                pass

        self._active_stream_trackers[user_id] = StreamTracker(
            broadcaster_user_id=user_id,
            broadcaster_login=login,
            broadcaster_name=name,
            start_time=start_ts,
            title=title,
            game_name=game_name,
            peak_viewers=viewer_count,
        )

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

    async def _handle_stream_offline_event(self, event: dict[str, Any]) -> None:
        user_id = str(event.get("broadcaster_user_id", ""))
        login = str(event.get("broadcaster_user_login", "")).lower()
        name = str(event.get("broadcaster_user_name", login))

        LOGGER.info("Received stream.offline event for %s (%s)", name, login)

        tracker = self._active_stream_trackers.pop(user_id, None)
        now = time.time()
        start_ts = tracker.start_time if tracker else now
        duration_seconds = max(0, int(now - start_ts))

        vod_url = await asyncio.to_thread(self.client.get_latest_vod_url, user_id)

        summary = TwitchStreamSummaryNotification(
            broadcaster_user_id=user_id,
            broadcaster_login=tracker.broadcaster_login if tracker else login,
            broadcaster_name=tracker.broadcaster_name if tracker else name,
            duration_seconds=duration_seconds,
            peak_viewers=tracker.peak_viewers if tracker else 0,
            title=tracker.title if tracker else "",
            game_name=tracker.game_name if tracker else "",
            stream_url=f"https://twitch.tv/{tracker.broadcaster_login if tracker else login}",
            vod_url=vod_url,
        )

        if self.on_stream_offline:
            try:
                await self.on_stream_offline(summary)
            except Exception as err:
                LOGGER.error("Failed executing on_stream_offline callback for %s: %s", login, err, exc_info=True)

    async def _run_polling_loop(self) -> None:
        LOGGER.info("Starting Twitch polling fallback loop (checking every %ds)...", self.config.poll_interval_seconds)
        while self._running:
            try:
                user_id_map = await asyncio.to_thread(self.client.get_user_ids, self.config.channels)
                currently_live: set[str] = set()

                for login, uid in user_id_map.items():
                    stream_info = await asyncio.to_thread(self.client.get_stream_info, uid)
                    if stream_info:
                        currently_live.add(uid)
                        viewer_count = int(stream_info.get("viewer_count", 0))
                        title = stream_info.get("title", "")
                        game_name = stream_info.get("game_name", "")
                        name = stream_info.get("user_name", login)

                        if uid not in self._live_user_ids:
                            thumbnail_template = stream_info.get("thumbnail_url", "")
                            thumbnail_url = thumbnail_template.replace("{width}", "1280").replace("{height}", "720") if thumbnail_template else ""
                            started_at = stream_info.get("started_at", "")

                            start_ts = time.time()
                            if started_at:
                                try:
                                    from datetime import datetime
                                    dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                                    start_ts = dt.timestamp()
                                except Exception:
                                    pass

                            self._active_stream_trackers[uid] = StreamTracker(
                                broadcaster_user_id=uid,
                                broadcaster_login=login,
                                broadcaster_name=name,
                                start_time=start_ts,
                                title=title,
                                game_name=game_name,
                                peak_viewers=viewer_count,
                            )

                            notification = TwitchStreamNotification(
                                broadcaster_user_id=uid,
                                broadcaster_login=login,
                                broadcaster_name=name,
                                title=title,
                                game_name=game_name,
                                stream_url=f"https://twitch.tv/{login}",
                                thumbnail_url=thumbnail_url,
                                started_at=started_at,
                            )
                            await self.on_stream_online(notification)
                        else:
                            tracker = self._active_stream_trackers.get(uid)
                            if tracker:
                                tracker.peak_viewers = max(tracker.peak_viewers, viewer_count)
                                if title:
                                    tracker.title = title
                                if game_name:
                                    tracker.game_name = game_name

                # Check for channels that went offline
                went_offline = self._live_user_ids - currently_live
                for uid in went_offline:
                    tracker = self._active_stream_trackers.pop(uid, None)
                    now = time.time()
                    start_ts = tracker.start_time if tracker else now
                    duration_seconds = max(0, int(now - start_ts))
                    login = tracker.broadcaster_login if tracker else uid

                    vod_url = await asyncio.to_thread(self.client.get_latest_vod_url, uid)
                    summary = TwitchStreamSummaryNotification(
                        broadcaster_user_id=uid,
                        broadcaster_login=login,
                        broadcaster_name=tracker.broadcaster_name if tracker else login,
                        duration_seconds=duration_seconds,
                        peak_viewers=tracker.peak_viewers if tracker else 0,
                        title=tracker.title if tracker else "",
                        game_name=tracker.game_name if tracker else "",
                        stream_url=f"https://twitch.tv/{login}",
                        vod_url=vod_url,
                    )
                    if self.on_stream_offline:
                        try:
                            await self.on_stream_offline(summary)
                        except Exception as err:
                            LOGGER.error("Failed executing on_stream_offline callback for %s: %s", login, err, exc_info=True)

                self._live_user_ids = currently_live
            except asyncio.CancelledError:
                LOGGER.info("Twitch polling task cancelled.")
                break
            except Exception as err:
                LOGGER.error("Error in Twitch polling fallback loop: %s", err, exc_info=True)

            await asyncio.sleep(self.config.poll_interval_seconds)
