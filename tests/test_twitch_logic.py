import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.commands.twitch_logic import fetch_twitch_status_reply, parse_twitch_command
from bot.config import BotConfig, TwitchConfig
from bot.tasks.twitch_logic import (
    TwitchClient,
    TwitchEventSubNotifier,
    TwitchStreamNotification,
    TwitchStreamSummaryNotification,
)


class TestTwitchLogic(unittest.IsolatedAsyncioTestCase):
    def test_twitch_config_is_configured(self) -> None:
        cfg = TwitchConfig(client_id="cid", client_secret="csecret", channels=("shroud",))
        self.assertTrue(cfg.is_configured)

        cfg_unconfigured = TwitchConfig(client_id="", client_secret="", channels=())
        self.assertFalse(cfg_unconfigured.is_configured)

    def test_parse_twitch_command(self) -> None:
        matched, sub = parse_twitch_command("!twitch")
        self.assertTrue(matched)
        self.assertEqual(sub, "")

        matched, sub = parse_twitch_command("  !TWITCH status ")
        self.assertTrue(matched)
        self.assertEqual(sub, "status")

        matched, _ = parse_twitch_command("hello world")
        self.assertFalse(matched)

    def test_stream_notification_formatting(self) -> None:
        notif = TwitchStreamNotification(
            broadcaster_user_id="123",
            broadcaster_login="shroud",
            broadcaster_name="shroud",
            title="Valorant Ranked",
            game_name="Valorant",
            stream_url="https://twitch.tv/shroud",
            thumbnail_url="https://example.com/thumb.jpg",
            started_at="2026-07-27T17:00:00Z",
        )
        msg = notif.format_telegram_message()
        self.assertIn("shroud is LIVE on Twitch!", msg)
        self.assertIn("Valorant Ranked", msg)
        self.assertIn("Valorant", msg)
        self.assertIn("https://twitch.tv/shroud", msg)

    def test_stream_summary_notification_formatting(self) -> None:
        summary = TwitchStreamSummaryNotification(
            broadcaster_user_id="123",
            broadcaster_login="shroud",
            broadcaster_name="shroud",
            duration_seconds=16320,  # 4h 32m
            peak_viewers=24510,
            title="Valorant Ranked",
            game_name="Valorant",
            stream_url="https://twitch.tv/shroud",
            vod_url="https://twitch.tv/videos/99999",
        )
        msg = summary.format_telegram_message()
        self.assertIn("shroud stream ended!", msg)
        self.assertIn("4h 32m", msg)
        self.assertIn("24,510", msg)
        self.assertIn("Valorant", msg)
        self.assertIn("https://twitch.tv/videos/99999", msg)

    @patch("urllib.request.urlopen")
    def test_twitch_client_get_latest_vod_url(self, mock_urlopen: MagicMock) -> None:
        mock_token_resp = MagicMock()
        mock_token_resp.read.return_value = json.dumps({"access_token": "t", "expires_in": 3600}).encode("utf-8")
        mock_token_resp.__enter__.return_value = mock_token_resp

        mock_vod_resp = MagicMock()
        mock_vod_resp.read.return_value = json.dumps({
            "data": [{"url": "https://twitch.tv/videos/12345"}]
        }).encode("utf-8")
        mock_vod_resp.__enter__.return_value = mock_vod_resp

        mock_urlopen.side_effect = [mock_token_resp, mock_vod_resp]

        cfg = TwitchConfig(client_id="cid", client_secret="csecret", channels=("shroud",))
        client = TwitchClient(cfg)
        vod_url = client.get_latest_vod_url("1001")

        self.assertEqual(vod_url, "https://twitch.tv/videos/12345")

    @patch("urllib.request.urlopen")
    def test_twitch_client_get_app_token(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "access_token": "mock_token_123",
            "expires_in": 3600,
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        cfg = TwitchConfig(client_id="cid", client_secret="csecret", channels=("shroud",))
        client = TwitchClient(cfg)
        token = client.get_app_token()

        self.assertEqual(token, "mock_token_123")
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_twitch_client_get_user_ids(self, mock_urlopen: MagicMock) -> None:
        mock_token_resp = MagicMock()
        mock_token_resp.read.return_value = json.dumps({"access_token": "t", "expires_in": 3600}).encode("utf-8")
        mock_token_resp.__enter__.return_value = mock_token_resp

        mock_users_resp = MagicMock()
        mock_users_resp.read.return_value = json.dumps({
            "data": [
                {"id": "1001", "login": "shroud"},
                {"id": "1002", "login": "tarik"},
            ]
        }).encode("utf-8")
        mock_users_resp.__enter__.return_value = mock_users_resp

        mock_urlopen.side_effect = [mock_token_resp, mock_users_resp]

        cfg = TwitchConfig(client_id="cid", client_secret="csecret", channels=("shroud", "tarik"))
        client = TwitchClient(cfg)
        user_ids = client.get_user_ids(("shroud", "tarik"))

        self.assertEqual(user_ids, {"shroud": "1001", "tarik": "1002"})

    async def test_handle_stream_online_event_triggers_callback(self) -> None:
        cfg = TwitchConfig(client_id="cid", client_secret="csecret", channels=("shroud",))
        mock_client = MagicMock(spec=TwitchClient)
        mock_client.get_stream_info.return_value = {
            "title": "Pro CS2",
            "game_name": "Counter-Strike 2",
            "thumbnail_url": "https://example.com/{width}x{height}.jpg",
        }

        notifications: list[TwitchStreamNotification] = []

        async def callback(n: TwitchStreamNotification) -> None:
            notifications.append(n)

        notifier = TwitchEventSubNotifier(config=cfg, on_stream_online=callback, client=mock_client)

        event_payload = {
            "broadcaster_user_id": "1001",
            "broadcaster_user_login": "shroud",
            "broadcaster_user_name": "shroud",
            "started_at": "2026-07-27T17:00:00Z",
        }

        await notifier._handle_stream_online_event(event_payload)

        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0].broadcaster_login, "shroud")
        self.assertEqual(notifications[0].title, "Pro CS2")
        self.assertEqual(notifications[0].game_name, "Counter-Strike 2")
        self.assertEqual(notifications[0].thumbnail_url, "https://example.com/1280x720.jpg")

    async def test_fetch_twitch_status_reply_unconfigured(self) -> None:
        cfg = TwitchConfig()
        reply = await fetch_twitch_status_reply(cfg)
        self.assertIn("ei ole määritetty", reply)

    async def test_fetch_twitch_status_reply_configured(self) -> None:
        cfg = TwitchConfig(client_id="cid", client_secret="csecret", channels=("shroud", "tarik"))
        mock_client = MagicMock(spec=TwitchClient)
        mock_client.get_user_ids.return_value = {"shroud": "1001", "tarik": "1002"}
        mock_client.get_stream_info.side_effect = lambda uid: (
            {"title": "CS2 Major", "game_name": "Counter-Strike 2"} if uid == "1001" else None
        )

        reply = await fetch_twitch_status_reply(cfg, client=mock_client)

        self.assertIn("shroud", reply)
        self.assertIn("LIVE", reply)
        self.assertIn("CS2 Major", reply)
        self.assertIn("tarik", reply)
        self.assertIn("Offline", reply)

    def test_subscribe_eventsub_websocket_without_user_token(self) -> None:
        cfg = TwitchConfig(client_id="cid", client_secret="csecret", user_access_token="", channels=("shroud",))
        client = TwitchClient(cfg)
        ok = client.subscribe_eventsub_websocket("session123", "1001")
        self.assertFalse(ok)

    @patch("urllib.request.urlopen")
    def test_subscribe_eventsub_websocket_invalid_auth_400(self, mock_urlopen: MagicMock) -> None:
        import io
        import urllib.error
        err_resp = urllib.error.HTTPError(
            url="http://example.com",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=io.BytesIO(b'{"error":"Bad Request","status":400,"message":"invalid transport and auth combination"}'),
        )
        mock_urlopen.side_effect = err_resp

        cfg = TwitchConfig(client_id="cid", client_secret="csecret", user_access_token="user_tok", channels=("shroud",))
        client = TwitchClient(cfg)
        ok = client.subscribe_eventsub_websocket("session123", "1001")
        self.assertFalse(ok)

    async def test_notifier_defaults_to_polling_without_user_token(self) -> None:
        cfg = TwitchConfig(client_id="cid", client_secret="csecret", user_access_token="", channels=("shroud",))
        mock_client = MagicMock(spec=TwitchClient)

        async def callback(_: TwitchStreamNotification) -> None:
            pass

        notifier = TwitchEventSubNotifier(config=cfg, on_stream_online=callback, client=mock_client)

    async def test_notifier_calls_on_token_expired_when_user_token_fails(self) -> None:
        cfg = TwitchConfig(client_id="cid", client_secret="csecret", user_access_token="invalid_tok", channels=("shroud",))
        mock_client = MagicMock(spec=TwitchClient)
        mock_client.subscribe_eventsub_websocket.return_value = False

        token_errors: list[str] = []

        async def callback(_: TwitchStreamNotification) -> None:
            pass

        async def on_token_expired(reason: str) -> None:
            token_errors.append(reason)

        notifier = TwitchEventSubNotifier(
            config=cfg,
            on_stream_online=callback,
            client=mock_client,
            on_token_expired=on_token_expired,
        )
        notifier._user_map = {"1001": "shroud"}

        mock_ws = AsyncMock()
        mock_ws.__aiter__.return_value = [
            json.dumps({
                "metadata": {"message_type": "session_welcome"},
                "payload": {"session": {"id": "sess123"}},
            })
        ]

        with patch("websockets.connect") as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_ws
            await notifier._run_websocket_session("wss://example.com")

        self.assertTrue(notifier._use_polling_fallback)
        self.assertEqual(len(token_errors), 1)
        self.assertIn("TWITCH_USER_ACCESS_TOKEN", token_errors[0])

    async def test_offline_detection_triggers_summary_callback(self) -> None:
        cfg = TwitchConfig(client_id="cid", client_secret="csecret", channels=("shroud",))
        mock_client = MagicMock(spec=TwitchClient)
        mock_client.get_user_ids.return_value = {"shroud": "1001"}
        mock_client.get_latest_vod_url.return_value = "https://twitch.tv/videos/12345"

        summaries: list[TwitchStreamSummaryNotification] = []

        async def on_online(_: TwitchStreamNotification) -> None:
            pass

        async def on_offline(s: TwitchStreamSummaryNotification) -> None:
            summaries.append(s)

        notifier = TwitchEventSubNotifier(
            config=cfg,
            on_stream_online=on_online,
            on_stream_offline=on_offline,
            client=mock_client,
        )

        call_count = 0

        def mock_get_stream_info(uid: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "title": "CS2 Major",
                    "game_name": "Counter-Strike 2",
                    "user_name": "shroud",
                    "viewer_count": 15000,
                    "started_at": "2026-07-27T17:00:00Z",
                }
            return None

        mock_client.get_stream_info.side_effect = mock_get_stream_info

        with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
            try:
                notifier._running = True
                await notifier._run_polling_loop()
            except asyncio.CancelledError:
                pass

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].broadcaster_login, "shroud")
        self.assertEqual(summaries[0].vod_url, "https://twitch.tv/videos/12345")


if __name__ == "__main__":
    unittest.main()



