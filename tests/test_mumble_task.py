from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from bot.config import BotConfig, MumbleConfig
from bot.tasks.mumble import (
    MumbleTask,
    get_mumble_manager,
    register,
    set_mumble_manager,
)
from bot.tasks.mumble_logic import (
    MumbleManager,
    MumbleServerSnapshot,
    UserStatsData,
)


class TestMumbleTask(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        set_mumble_manager(None)

    def test_manager_not_configured_start(self) -> None:
        cfg = MumbleConfig(host="")
        manager = MumbleManager(cfg)
        self.assertFalse(manager.is_connected)
        manager.start()
        self.assertFalse(manager.is_connected)
        manager.stop()

    def test_manager_handle_user_stats(self) -> None:
        cfg = MumbleConfig(host="localhost")
        manager = MumbleManager(cfg)

        # Mock incoming protobuf UserStats message
        mess = MagicMock()
        mess.session = 42
        mess.HasField.side_effect = lambda field: field in (
            "onlinesecs",
            "idlesecs",
            "tcp_ping_avg",
            "udp_ping_avg",
            "bandwidth",
            "version",
            "from_client",
        )
        mess.onlinesecs = 1200
        mess.idlesecs = 60
        mess.tcp_ping_avg = 15.5
        mess.udp_ping_avg = 12.3
        mess.bandwidth = 72000
        mess.version.HasField.side_effect = lambda field: field in ("release", "os", "os_version")
        mess.version.release = "1.4.287"
        mess.version.os = "Linux"
        mess.version.os_version = "6.8.0"
        mess.from_client.HasField.side_effect = lambda field: field in ("good", "lost")
        mess.from_client.good = 500
        mess.from_client.lost = 1

        manager._handle_user_stats(mess)

        stats = manager._user_stats.get(42)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.session, 42)
        self.assertEqual(stats.online_seconds, 1200)
        self.assertEqual(stats.idle_seconds, 60)
        self.assertEqual(stats.tcp_ping_avg, 15.5)
        self.assertEqual(stats.udp_ping_avg, 12.3)
        self.assertEqual(stats.client_release, "1.4.287")
        self.assertEqual(stats.client_os, "Linux")
        self.assertEqual(stats.client_os_version, "6.8.0")
        self.assertEqual(stats.packets_good, 500)
        self.assertEqual(stats.packets_lost, 1)

    def test_manager_user_lifecycle(self) -> None:
        cfg = MumbleConfig(host="localhost")
        manager = MumbleManager(cfg)

        # Simulate user created
        user_created = {"session": 10, "name": "Matti"}
        manager._on_user_created(user_created)
        self.assertIn(10, manager._user_join_times)

        # Store dummy stats
        manager._user_stats[10] = UserStatsData(session=10, online_seconds=50)

        # Simulate user removed
        user_removed = {"session": 10}
        manager._on_user_removed(user_removed)
        self.assertNotIn(10, manager._user_join_times)
        self.assertNotIn(10, manager._user_stats)

    def test_manager_get_snapshot_disconnected(self) -> None:
        cfg = MumbleConfig(host="mumble.test", port=64738)
        manager = MumbleManager(cfg)
        snap = manager.get_snapshot()
        self.assertFalse(snap.is_connected)
        self.assertEqual(snap.host, "mumble.test")
        self.assertEqual(snap.port, 64738)

    def test_manager_get_snapshot_connected_mock(self) -> None:
        cfg = MumbleConfig(host="mumble.test", port=64738, user="Botti")
        manager = MumbleManager(cfg)
        manager._connected_at = time.time()

        # Mock client
        client = MagicMock()
        client.is_alive.return_value = True
        from pymumble_py3 import constants
        client.connected = constants.PYMUMBLE_CONN_STATE_CONNECTED
        class MockUsers(dict):
            myself_session = 1

        client.users = MockUsers({
            1: {"session": 1, "name": "Botti", "channel_id": 0},
            2: {
                "session": 2,
                "name": "Pekka",
                "channel_id": 1,
                "mute": True,
                "self_deaf": True,
            },
        })
        client.channels = {
            0: {"channel_id": 0, "name": "Root", "parent": 0},
            1: {"channel_id": 1, "name": "Sauna", "parent": 0},
        }
        manager._client = client

        # Add stats for user 2
        manager._user_stats[2] = UserStatsData(
            session=2,
            online_seconds=300,
            idle_seconds=70,
            udp_ping_avg=14.0,
            client_release="1.5.0",
            client_os="Windows",
            bandwidth=64000,
        )

        snap = manager.get_snapshot()
        self.assertTrue(snap.is_connected)
        self.assertEqual(snap.total_active_users, 1)
        self.assertEqual(len(snap.active_users), 1)

        pekka = snap.active_users[0]
        self.assertEqual(pekka.name, "Pekka")
        self.assertEqual(pekka.channel_name, "Sauna")
        self.assertTrue(pekka.is_muted)
        self.assertTrue(pekka.is_deafened)
        self.assertGreaterEqual(pekka.online_seconds or 0, 300)
        self.assertGreaterEqual(pekka.idle_seconds or 0, 70)
        self.assertEqual(pekka.ping_ms, 14.0)
        self.assertEqual(pekka.client_os, "Windows")
        self.assertEqual(pekka.client_release, "1.5.0")
        self.assertEqual(pekka.bandwidth_kbps, 512.0)

    async def test_task_lifecycle_and_registration(self) -> None:
        app = MagicMock()
        app.bot_data = {}
        app.post_init = None
        app.post_shutdown = None

        config = MagicMock(spec=BotConfig)
        config.mumble = MumbleConfig(host="")

        task = MumbleTask(app, config)
        self.assertIn("mumble_manager", app.bot_data)
        self.assertIs(get_mumble_manager(app), task.manager)

        task.start()
        await task.stop()

        # Test task register function
        register(app, config)
        self.assertIsNotNone(app.post_init)
        self.assertIsNotNone(app.post_shutdown)


if __name__ == "__main__":
    unittest.main()
