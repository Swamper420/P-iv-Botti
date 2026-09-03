from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from bot.commands.mumble_logic import (
    build_mumble_summary_card,
    build_mumble_user_card,
    format_duration,
    format_mumble_summary,
    format_user_details,
    handle_mumble_card_command,
    handle_mumble_command,
    parse_mumble_command,
)
from bot.rendering import render_card
from bot.config import MumbleConfig
from bot.tasks.mumble_logic import (
    MumbleChannelInfo,
    MumbleServerSnapshot,
    MumbleUserInfo,
)


class TestMumbleLogic(unittest.IsolatedAsyncioTestCase):
    def test_parse_mumble_command(self) -> None:
        matched, action, target = parse_mumble_command("!mumble")
        self.assertTrue(matched)
        self.assertEqual(action, "summary")
        self.assertIsNone(target)

        matched, action, target = parse_mumble_command("   !mumble   ")
        self.assertTrue(matched)
        self.assertEqual(action, "summary")

        matched, action, target = parse_mumble_command("!mumble help")
        self.assertTrue(matched)
        self.assertEqual(action, "help")

        matched, action, target = parse_mumble_command("!mumble stats Teppo")
        self.assertTrue(matched)
        self.assertEqual(action, "stats")
        self.assertEqual(target, "Teppo")

        matched, action, target = parse_mumble_command("!mumble Teppo")
        self.assertTrue(matched)
        self.assertEqual(action, "stats")
        self.assertEqual(target, "Teppo")

        matched, action, target = parse_mumble_command("!notmumble")
        self.assertFalse(matched)

        matched, action, target = parse_mumble_command("")
        self.assertFalse(matched)

    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(None), "tuntematon")
        self.assertEqual(format_duration(-5), "tuntematon")
        self.assertEqual(format_duration(0), "0 s")
        self.assertEqual(format_duration(45), "45 s")
        self.assertEqual(format_duration(60), "1 min")
        self.assertEqual(format_duration(125), "2 min 5 s")
        self.assertEqual(format_duration(3600), "1 t")
        self.assertEqual(format_duration(3660), "1 t 1 min")
        self.assertEqual(format_duration(7200 + 15 * 60), "2 t 15 min")
        self.assertEqual(format_duration(86400 * 2 + 3600 * 3 + 60 * 10), "2 pv 3 t 10 min")
        self.assertEqual(format_duration(86400 * 3), "3 pv")

    def test_format_mumble_summary_not_configured(self) -> None:
        cfg = MumbleConfig(host="")
        snap = MumbleServerSnapshot(is_connected=False, host="", port=64738, server_name="", bot_user="Botti")
        text = format_mumble_summary(snap, cfg)
        self.assertIn("Mumble ei ole käytössä", text)

    def test_format_mumble_summary_disconnected(self) -> None:
        cfg = MumbleConfig(host="mumble.example.com", port=64738)
        snap = MumbleServerSnapshot(
            is_connected=False,
            host="mumble.example.com",
            port=64738,
            server_name="mumble.example.com",
            bot_user="Botti",
            error_message="Yhteysvirhe",
        )
        text = format_mumble_summary(snap, cfg)
        self.assertIn("Ei yhteyttä Mumble-palvelimeen", text)
        self.assertIn("mumble.example.com:64738", text)

    def test_format_mumble_summary_zero_users(self) -> None:
        cfg = MumbleConfig(host="mumble.example.com")
        bot_user = MumbleUserInfo(
            session=1,
            name="P-iv-Botti",
            channel_id=0,
            channel_name="Root",
            is_myself=True,
        )
        snap = MumbleServerSnapshot(
            is_connected=True,
            host="mumble.example.com",
            port=64738,
            server_name="Mumble Server",
            bot_user="P-iv-Botti",
            users={1: bot_user},
        )
        text = format_mumble_summary(snap, cfg)
        self.assertIn("Käyttäjiä paikalla: <b>0</b>", text)
        self.assertIn("Ei muita käyttäjiä kanavilla", text)

    def test_format_mumble_summary_with_users(self) -> None:
        cfg = MumbleConfig(host="mumble.example.com")
        u1 = MumbleUserInfo(
            session=2,
            name="Alice",
            channel_id=1,
            channel_name="Pelikanava",
            online_seconds=3600,
            idle_seconds=120,
            ping_ms=18.4,
            is_muted=True,
        )
        u2 = MumbleUserInfo(
            session=3,
            name="Bob",
            channel_id=1,
            channel_name="Pelikanava",
            online_seconds=1800,
            idle_seconds=10,
            ping_ms=25.0,
            is_deafened=True,
            is_recording=True,
        )
        ch1 = MumbleChannelInfo(channel_id=1, name="Pelikanava")
        snap = MumbleServerSnapshot(
            is_connected=True,
            host="mumble.example.com",
            port=64738,
            server_name="Pelipalvelin",
            bot_user="P-iv-Botti",
            channels={1: ch1},
            users={2: u1, 3: u2},
        )
        text = format_mumble_summary(snap, cfg)
        self.assertIn("Käyttäjiä paikalla: <b>2</b>", text)
        self.assertIn("Pelikanava", text)
        self.assertIn("Alice", text)
        self.assertIn("Bob", text)
        self.assertIn("🔇", text)  # Alice muted
        self.assertIn("🔕", text)  # Bob deafened
        self.assertIn("🔴", text)  # Bob recording
        self.assertIn("idle 2 min", text)  # Alice idle
        self.assertIn("18 ms", text)

    def test_format_user_details(self) -> None:
        user = MumbleUserInfo(
            session=5,
            name="Teppo",
            channel_id=2,
            channel_name="Yleinen",
            online_seconds=5000,
            idle_seconds=300,
            ping_ms=14.2,
            bandwidth_kbps=72.0,
            client_release="1.5.634",
            client_os="Linux x86_64",
            packets_good=1000,
            packets_lost=2,
            is_priority_speaker=True,
        )
        text = format_user_details(user, "Testipalvelin")
        self.assertIn("Teppo", text)
        self.assertIn("Yleinen", text)
        self.assertIn("1 t 23 min", text)
        self.assertIn("idle", text)
        self.assertIn("14.2 ms", text)
        self.assertIn("1.5.634 (Linux x86_64)", text)
        self.assertIn("72 kbps", text)
        self.assertIn("Priority speaker", text)
        self.assertIn("0.2% hävikki", text)

    async def test_handle_mumble_command_not_configured(self) -> None:
        cfg = MumbleConfig(host="")
        res = await handle_mumble_command(None, cfg, "!mumble")
        self.assertIn("Mumble ei ole käytössä", res)

    async def test_handle_mumble_command_help(self) -> None:
        cfg = MumbleConfig(host="localhost")
        res = await handle_mumble_command(None, cfg, "!mumble help")
        self.assertIn("Mumble-komennon käyttö", res)

    async def test_handle_mumble_command_manager_none(self) -> None:
        cfg = MumbleConfig(host="localhost")
        res = await handle_mumble_command(None, cfg, "!mumble")
        self.assertIn("Mumble-taustapalvelu ei ole käynnissä", res)

    async def test_handle_mumble_command_stats_found_and_not_found(self) -> None:
        cfg = MumbleConfig(host="localhost")
        u = MumbleUserInfo(
            session=10,
            name="Kalle",
            channel_id=1,
            channel_name="Huone",
            online_seconds=100,
        )
        snap = MumbleServerSnapshot(
            is_connected=True,
            host="localhost",
            port=64738,
            server_name="Localhost",
            bot_user="Botti",
            channels={1: MumbleChannelInfo(channel_id=1, name="Huone")},
            users={10: u},
        )
        manager = MagicMock()
        manager.refresh_stats = AsyncMock()
        manager.get_snapshot.return_value = snap

        # Exact match
        res = await handle_mumble_command(manager, cfg, "!mumble Kalle")
        self.assertIn("Mumble: Kalle", res)

        # Case-insensitive substring match
        res = await handle_mumble_command(manager, cfg, "!mumble kal")
        self.assertIn("Mumble: Kalle", res)

        # Not found
        res = await handle_mumble_command(manager, cfg, "!mumble Tuntematon")
        self.assertIn("ei löytynyt Mumble-palvelimelta", res)

    async def test_handle_mumble_card_command_summary_and_stats(self) -> None:
        cfg = MumbleConfig(host="localhost", port=64738)
        u = MumbleUserInfo(
            session=10,
            name="Kalle",
            channel_id=1,
            channel_name="Huone",
            online_seconds=3660,
            ping_ms=15.0,
            idle_seconds=120,
        )
        snap = MumbleServerSnapshot(
            is_connected=True,
            host="localhost",
            port=64738,
            server_name="Localhost",
            bot_user="Botti",
            channels={1: MumbleChannelInfo(channel_id=1, name="Huone")},
            users={10: u},
        )
        manager = MagicMock()
        manager.refresh_stats = AsyncMock()
        manager.get_snapshot.return_value = snap

        # Summary card
        text, card = await handle_mumble_card_command(manager, cfg, "!mumble")
        self.assertIn("Mumble: Localhost", text)
        self.assertIsNotNone(card)
        self.assertEqual(card.title, "Mumble: Localhost")
        png_bytes = render_card(card)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

        # User detail card
        text, card = await handle_mumble_card_command(manager, cfg, "!mumble Kalle")
        self.assertIn("Mumble: Kalle", text)
        self.assertIsNotNone(card)
        self.assertEqual(card.title, "Mumble: Kalle")
        png_bytes = render_card(card)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

        # Empty users summary card
        empty_snap = MumbleServerSnapshot(
            is_connected=True,
            host="localhost",
            port=64738,
            server_name="Localhost",
            bot_user="Botti",
            channels={},
            users={},
        )
        manager.get_snapshot.return_value = empty_snap
        text, card = await handle_mumble_card_command(manager, cfg, "!mumble")
        self.assertIn("Käyttäjiä paikalla: <b>0</b>", text)
        self.assertIsNotNone(card)
        png_bytes = render_card(card)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
