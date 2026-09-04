import json
import ssl
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from bot.commands.mine_logic import (
    CraftyClient,
    _format_death_causes,
    _format_duration,
    _format_memory,
    _parse_players_field,
    add_mine_allowlist,
    fetch_mine_allowlist,
    fetch_mine_stats,
    fetch_mine_status,
    handle_mine_card_command,
    handle_mine_command,
    parse_bds_player_stats,
    parse_mine_command,
)
from bot.config import CraftyConfig
from bot.rendering import render_card


class MineLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CraftyConfig(
            base_url="https://localhost:8443",
            api_token="test-secret-token",
            timeout_seconds=10,
            verify_ssl=False,
        )

    def test_parse_mine_command(self) -> None:
        self.assertEqual(parse_mine_command("!mine"), (True, "status", "", ""))
        self.assertEqual(parse_mine_command("!mine   "), (True, "status", "", ""))
        self.assertEqual(parse_mine_command("  !mine   "), (True, "status", "", ""))
        self.assertEqual(parse_mine_command("!MINE survival"), (True, "status", "survival", ""))
        self.assertEqual(parse_mine_command("!mine 1"), (True, "status", "1", ""))
        self.assertEqual(
            parse_mine_command("!mine server-uuid-123"), (True, "status", "server-uuid-123", "")
        )

        # Allowlist list subcommands
        self.assertEqual(parse_mine_command("!mine allowlist"), (True, "allowlist_list", "", ""))
        self.assertEqual(parse_mine_command("!mine whitelist"), (True, "allowlist_list", "", ""))
        self.assertEqual(parse_mine_command("!mine sallitut"), (True, "allowlist_list", "", ""))
        self.assertEqual(
            parse_mine_command("!mine allowlist bedrock"), (True, "allowlist_list", "bedrock", "")
        )
        self.assertEqual(
            parse_mine_command("!mine allowlist list survival"),
            (True, "allowlist_list", "survival", ""),
        )

        # Allowlist add subcommands
        self.assertEqual(
            parse_mine_command("!mine allowlist add Notch"),
            (True, "allowlist_add", "", "Notch"),
        )
        self.assertEqual(
            parse_mine_command("!mine whitelist lisaa Steve"),
            (True, "allowlist_add", "", "Steve"),
        )
        self.assertEqual(
            parse_mine_command("!mine allowlist add bedrock Alex"),
            (True, "allowlist_add", "bedrock", "Alex"),
        )
        self.assertEqual(
            parse_mine_command("!mine allowlist add Xbox Gamertag 123"),
            (True, "allowlist_add", "Xbox", "Gamertag 123"),
        )
        self.assertEqual(
            parse_mine_command("!mine allowlist add"),
            (True, "allowlist_add", "", ""),
        )

        # Stats subcommands
        self.assertEqual(parse_mine_command("!mine stats"), (True, "stats", "", ""))
        self.assertEqual(parse_mine_command("!mine tilastot"), (True, "stats", "", ""))
        self.assertEqual(parse_mine_command("!mine statistiikka"), (True, "stats", "", ""))
        self.assertEqual(parse_mine_command("!mine statit"), (True, "stats", "", ""))
        self.assertEqual(parse_mine_command("!mine pelaajat"), (True, "stats", "", ""))
        self.assertEqual(parse_mine_command("!mine stats bedrock"), (True, "stats", "bedrock", ""))
        self.assertEqual(parse_mine_command("!mine stats Steve"), (True, "stats", "Steve", ""))
        self.assertEqual(
            parse_mine_command("!mine stats bedrock Steve"), (True, "stats", "bedrock", "Steve")
        )
        self.assertEqual(
            parse_mine_command("!mine stats Bedrock Realm Steve Jobs"),
            (True, "stats", "Bedrock", "Realm Steve Jobs"),
        )

        # Invalid matches
        self.assertEqual(parse_mine_command("!miner"), (False, "", "", ""))
        self.assertEqual(parse_mine_command("!mineraali"), (False, "", "", ""))
        self.assertEqual(parse_mine_command("mine"), (False, "", "", ""))
        self.assertEqual(parse_mine_command(""), (False, "", "", ""))
        self.assertEqual(parse_mine_command(None), (False, "", "", ""))

    def test_format_memory(self) -> None:
        self.assertIsNone(_format_memory(None))
        self.assertEqual(_format_memory(331476992.0), "316.1 MB")
        self.assertEqual(_format_memory(1073741824), "1.00 GB")
        self.assertEqual(_format_memory("331476992.0"), "316.1 MB")
        self.assertEqual(_format_memory("2.4 GB"), "2.4 GB")
        self.assertEqual(_format_memory(0), "0 MB")

    def test_parse_players_field(self) -> None:
        # None / empty
        self.assertEqual(_parse_players_field(None), [])
        self.assertEqual(_parse_players_field(""), [])
        self.assertEqual(_parse_players_field("[]"), [])

        # JSON-encoded string (as Crafty API actually returns)
        self.assertEqual(
            _parse_players_field('["Steve", "Alex"]'),
            ["Steve", "Alex"],
        )
        self.assertEqual(
            _parse_players_field('[{"name": "Notch"}]'),
            ["Notch"],
        )

        # Already a Python list
        self.assertEqual(
            _parse_players_field(["Steve", "Alex"]),
            ["Steve", "Alex"],
        )
        self.assertEqual(
            _parse_players_field([{"name": "Notch"}]),
            ["Notch"],
        )

        # Invalid JSON string
        self.assertEqual(_parse_players_field("not json"), [])

    def test_fetch_mine_status_not_configured(self) -> None:
        unconfigured = CraftyConfig(api_token="")
        reply = fetch_mine_status(unconfigured)
        self.assertIn("Crafty Controller -integraatiota ei ole määritetty", reply)
        self.assertIn("CRAFTY_API_TOKEN", reply)

    def test_fetch_mine_status_online_server(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {
                "server_id": "uuid-1",
                "server_name": "Survival SMP",
                "server_port": 25565,
                "version": "1.20.4",
            }
        ]
        client.get_server_stats.return_value = {
            "running": True,
            "online": 3,
            "max_players": 20,
            "players": ["Steve", "Alex", "Notch"],
            "cpu": 15.42,
            "mem_percent": 48.6,
            "mem": 331476992.0,
            "world_name": "world_survival",
            "motd": "Tervetuloa!",
        }

        reply = fetch_mine_status(self.config, client=client)

        self.assertNotIn("Minecraft-palvelimet", reply)
        self.assertNotIn("⛏️", reply)
        self.assertNotIn("🟢", reply)
        self.assertIn("<b>Survival SMP</b>", reply)
        self.assertIn("» TILA:", reply)
        self.assertIn("<b>PÄÄLLÄ</b>", reply)
        self.assertIn("» PELAAJAT:", reply)
        self.assertIn("<b>3</b> / <b>20</b> (Steve, Alex, Notch)", reply)
        self.assertIn("» CPU:", reply)
        self.assertIn("<b>15.4 %</b>", reply)
        self.assertIn("» RAM:", reply)
        self.assertIn("<b>316.1 MB</b> (48.6 %)", reply)
        self.assertIn("» VERSIO:", reply)
        self.assertIn("<b>1.20.4</b>", reply)
        self.assertIn("» PORTTI:", reply)
        self.assertIn("<code>25565</code>", reply)
        self.assertIn("» MAAILMA:", reply)
        self.assertIn("<b>world_survival</b>", reply)
        self.assertIn("» KUVAUS:", reply)
        self.assertIn("<i>Tervetuloa!</i>", reply)

    def test_fetch_mine_status_online_players_from_separate_fetch(self) -> None:
        """When stats lack a players list, get_online_players is called as fallback."""
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {
                "server_id": "uuid-1",
                "server_name": "Survival SMP",
                "server_port": 25565,
            }
        ]
        # Stats have an online count but players is empty JSON string (as Crafty returns)
        client.get_server_stats.return_value = {
            "running": True,
            "online": 2,
            "max_players": 20,
            "players": "[]",
        }
        client.get_online_players.return_value = ["Steve", "Alex"]

        reply = fetch_mine_status(self.config, client=client)

        client.get_online_players.assert_called_once_with("uuid-1")
        self.assertIn("(Steve, Alex)", reply)

    def test_fetch_mine_status_players_json_string(self) -> None:
        """Crafty returns players as a JSON-encoded string; verify it gets parsed."""
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "uuid-1", "server_name": "Survival SMP"}
        ]
        client.get_server_stats.return_value = {
            "running": True,
            "online": 2,
            "max_players": 20,
            "players": '["Steve", "Alex"]',
        }

        reply = fetch_mine_status(self.config, client=client)

        # Players came from stats, so get_online_players should NOT be called
        client.get_online_players.assert_not_called()
        self.assertIn("(Steve, Alex)", reply)

    def test_fetch_mine_status_skips_online_players_when_stats_have_them(self) -> None:
        """When stats already include players (as list), get_online_players is not called."""
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "uuid-1", "server_name": "Survival SMP"}
        ]
        client.get_server_stats.return_value = {
            "running": True,
            "online": 1,
            "max_players": 20,
            "players": ["Notch"],
        }

        reply = fetch_mine_status(self.config, client=client)

        client.get_online_players.assert_not_called()
        self.assertIn("(Notch)", reply)

    def test_fetch_mine_status_offline_server(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {
                "server_id": "uuid-2",
                "server_name": "Creative Test",
                "server_port": 25566,
            }
        ]
        client.get_server_stats.return_value = {
            "running": False,
            "status": "stopped",
            "online": 0,
            "max_players": 10,
        }

        reply = fetch_mine_status(self.config, client=client)

        self.assertNotIn("🔴", reply)
        self.assertIn("<b>Creative Test</b>", reply)
        self.assertIn("» TILA:", reply)
        self.assertIn("<b>POIS PÄÄLTÄ</b>", reply)
        self.assertIn("» PELAAJAT:", reply)
        self.assertIn("<b>0</b> / <b>10</b>", reply)
        self.assertNotIn("» CPU:", reply)
        self.assertNotIn("» RAM:", reply)

    def test_fetch_mine_status_multiple_servers(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "1", "server_name": "Server One"},
            {"server_id": "2", "server_name": "Server Two"},
        ]
        client.get_server_stats.side_effect = [
            {"running": True, "online": 1, "max_players": 10},
            {"running": False, "online": 0, "max_players": 10},
        ]

        reply = fetch_mine_status(self.config, client=client)

        self.assertIn("<b>Server One</b>", reply)
        self.assertIn("<b>Server Two</b>", reply)

    def test_fetch_mine_status_filtering_by_name(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "1", "server_name": "Survival SMP"},
            {"server_id": "2", "server_name": "Creative Test"},
        ]
        client.get_server_stats.return_value = {
            "running": True,
            "online": 2,
            "max_players": 20,
        }

        reply = fetch_mine_status(self.config, server_query="creative", client=client)

        self.assertIn("Creative Test", reply)
        self.assertNotIn("Survival SMP", reply)

    def test_fetch_mine_status_filtering_not_found(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "1", "server_name": "Survival SMP"},
        ]

        reply = fetch_mine_status(self.config, server_query="hardcore", client=client)
        self.assertIn("Palvelinta 'hardcore' ei löytynyt Crafty Controllerista.", reply)

    def test_fetch_mine_status_default_server_id(self) -> None:
        cfg = CraftyConfig(
            base_url="https://localhost:8443",
            api_token="token",
            default_server_id="uuid-default",
        )
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "uuid-other", "server_name": "Other Server"},
            {"server_id": "uuid-default", "server_name": "Default Server"},
        ]
        client.get_server_stats.return_value = {
            "running": True,
            "online": 4,
            "max_players": 10,
        }

        reply = fetch_mine_status(cfg, server_query="", client=client)

        self.assertIn("Default Server", reply)
        self.assertNotIn("Other Server", reply)

    def test_fetch_mine_status_empty_server_list(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = []

        reply = fetch_mine_status(self.config, client=client)
        self.assertIn("Crafty Controllerista ei löytynyt yhtään palvelinta", reply)

    def test_fetch_mine_status_auth_error(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.side_effect = HTTPError(
            url="https://localhost:8443/api/v2/servers",
            code=401,
            msg="Unauthorized",
            hdrs=MagicMock(),
            fp=None,
        )

        reply = fetch_mine_status(self.config, client=client)
        self.assertIn("autentikointivirhe (HTTP 401)", reply)
        self.assertIn("Tarkista CRAFTY_API_TOKEN", reply)

    def test_fetch_mine_status_http_error(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.side_effect = HTTPError(
            url="https://localhost:8443/api/v2/servers",
            code=500,
            msg="Internal Error",
            hdrs=MagicMock(),
            fp=None,
        )

        reply = fetch_mine_status(self.config, client=client)
        self.assertIn("Crafty Controller API -virhe (HTTP 500)", reply)

    def test_fetch_mine_status_connection_error(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.side_effect = URLError("Connection refused")

        reply = fetch_mine_status(self.config, client=client)
        self.assertIn("Yhteysvirhe Crafty Controlleriin: Connection refused", reply)

    def test_fetch_mine_allowlist_success(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bedrock-1", "server_name": "Bedrock Realm"}
        ]
        client.get_server_allowlist.return_value = ["PlayerOne", "PlayerTwo"]

        reply = fetch_mine_allowlist(self.config, client=client)
        self.assertIn("<b>Bedrock Realm</b>", reply)
        self.assertIn("Sallitut pelaajat (2):", reply)
        self.assertIn("• <b>PlayerOne</b>", reply)
        self.assertIn("• <b>PlayerTwo</b>", reply)

    def test_fetch_mine_allowlist_empty(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bedrock-1", "server_name": "Bedrock Realm"}
        ]
        client.get_server_allowlist.return_value = []

        reply = fetch_mine_allowlist(self.config, client=client)
        self.assertIn("<b>Bedrock Realm</b>", reply)
        self.assertIn("Allowlist on tyhjä tai sitä ei saatu luettua", reply)
        self.assertIn("!mine allowlist add", reply)

    def test_add_mine_allowlist_success(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bedrock-1", "server_name": "Bedrock Realm"}
        ]

        reply = add_mine_allowlist(self.config, player_name="GamerTag 123", client=client)
        client.add_to_allowlist.assert_called_once_with("bedrock-1", "GamerTag 123")
        self.assertIn("✅ Pelaaja <b>GamerTag 123</b> lisätty palvelimen <b>Bedrock Realm</b>", reply)
        self.assertIn("allowlist add", reply)

    def test_add_mine_allowlist_missing_name(self) -> None:
        reply = add_mine_allowlist(self.config, player_name="")
        self.assertIn("Määritä pelaajanimi", reply)

    def test_add_mine_allowlist_invalid_characters(self) -> None:
        reply = add_mine_allowlist(self.config, player_name="Invalid;Name!")
        self.assertIn("Virheellinen pelaajanimi", reply)

    def test_add_mine_allowlist_command_failure(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bedrock-1", "server_name": "Bedrock Realm"}
        ]
        client.add_to_allowlist.side_effect = HTTPError(
            url="http://localhost", code=500, msg="Server error", hdrs=MagicMock(), fp=None
        )

        reply = add_mine_allowlist(self.config, player_name="ValidPlayer", client=client)
        self.assertIn("❌ Komennon suoritus epäonnistui palvelimella <b>Bedrock Realm</b>", reply)

    def test_handle_mine_command_routing(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "1", "server_name": "Bedrock Realm"}
        ]
        client.get_server_stats.return_value = {"running": True, "online": 0}
        client.get_server_allowlist.return_value = ["Steve"]

        status_reply = handle_mine_command(self.config, "!mine", client=client)
        self.assertIn("<b>Bedrock Realm</b>", status_reply)
        self.assertIn("» TILA:", status_reply)

        list_reply = handle_mine_command(self.config, "!mine allowlist", client=client)
        self.assertIn("Sallitut pelaajat (1):", list_reply)
        self.assertIn("Steve", list_reply)

        add_reply = handle_mine_command(self.config, "!mine allowlist add Alex", client=client)
        self.assertIn("✅ Pelaaja <b>Alex</b> lisätty", add_reply)

    def test_html_escaping_in_status_output(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {
                "server_id": "1",
                "server_name": "<Test & Fun>",
            }
        ]
        client.get_server_stats.return_value = {
            "running": True,
            "online": 1,
            "max_players": 10,
            "players": ["<script>alert(1)</script>"],
            "motd": "A & B < C > D",
        }

        reply = fetch_mine_status(self.config, client=client)
        self.assertIn("&lt;Test &amp; Fun&gt;", reply)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", reply)
        self.assertIn("A &amp; B &lt; C &gt; D", reply)
        self.assertNotIn("<script>", reply)

    def test_crafty_client_get_servers_formats(self) -> None:
        client = CraftyClient(self.config)

        # Test list in "data"
        with patch.object(
            client, "_request_json", return_value={"status": "ok", "data": [{"server_id": "1"}]}
        ):
            self.assertEqual(client.get_servers(), [{"server_id": "1"}])

        # Test direct list
        with patch.object(client, "_request_json", return_value=[{"server_id": "2"}]):
            self.assertEqual(client.get_servers(), [{"server_id": "2"}])

        # Test servers in dict
        with patch.object(
            client,
            "_request_json",
            return_value={"status": "ok", "data": {"servers": [{"server_id": "3"}]}},
        ):
            self.assertEqual(client.get_servers(), [{"server_id": "3"}])

    def test_crafty_client_get_server_stats_formats(self) -> None:
        client = CraftyClient(self.config)

        # Test data dict
        with patch.object(
            client,
            "_request_json",
            return_value={"status": "ok", "data": {"running": True, "cpu": 10.0}},
        ):
            self.assertEqual(
                client.get_server_stats("1"), {"running": True, "cpu": 10.0}
            )

        # Test raw dict
        with patch.object(
            client, "_request_json", return_value={"running": False}
        ):
            self.assertEqual(client.get_server_stats("2"), {"running": False})

    def test_crafty_client_send_server_command(self) -> None:
        client = CraftyClient(self.config)
        with patch.object(client, "_request_json", return_value={"status": "ok"}) as mock_req:
            client.send_server_command("1", "/allowlist add Steve")
            mock_req.assert_called_once_with(
                "/api/v2/servers/1/stdin",
                method="POST",
                payload="allowlist add Steve",
                content_type="text/plain",
            )

    def test_crafty_client_send_server_command_raises_on_error(self) -> None:
        client = CraftyClient(self.config)
        err = HTTPError(url="", code=500, msg="Server Error", hdrs=MagicMock(), fp=None)
        with patch.object(client, "_request_json", side_effect=err):
            with self.assertRaises(HTTPError):
                client.send_server_command("1", "allowlist add Steve")

    def test_crafty_client_get_server_allowlist(self) -> None:
        client = CraftyClient(self.config)
        # Crafty API response: {"status": "ok", "data": {"content": "<raw json text>"}}
        allowlist_json = json.dumps([
            {"name": "BedrockGamer1", "xuid": "123456"},
            {"name": "BedrockGamer2", "xuid": "789101"},
        ])
        with patch.object(
            client,
            "get_server_file",
            return_value={"status": "ok", "data": {"content": allowlist_json}},
        ):
            names = client.get_server_allowlist("1")
            self.assertEqual(names, ["BedrockGamer1", "BedrockGamer2"])


    def test_crafty_client_request_json_with_urlopen(self) -> None:
        client = CraftyClient(
            CraftyConfig(
                base_url="https://localhost:8443",
                api_token="my-token",
                timeout_seconds=5,
                verify_ssl=False,
            )
        )

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"status": "ok", "data": []}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("bot.commands.mine_logic.urlopen", return_value=mock_resp) as mock_urlopen:
            result = client._request_json("/api/v2/servers")
            self.assertEqual(result, {"status": "ok", "data": []})
            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]
            self.assertEqual(req.full_url, "https://localhost:8443/api/v2/servers")
            self.assertEqual(req.headers.get("Authorization"), "Bearer my-token")

    def test_crafty_client_get_online_players_from_stats_json_string(self) -> None:
        """get_online_players parses players from a JSON-encoded string in stats."""
        client = CraftyClient(self.config)
        with patch.object(
            client,
            "get_server_stats",
            return_value={"running": True, "online": 2, "players": '["Steve", "Alex"]'},
        ):
            names = client.get_online_players("1")
            self.assertEqual(names, ["Steve", "Alex"])

    def test_crafty_client_get_online_players_falls_back_to_list_command(self) -> None:
        """When stats players is empty, sends 'list' command and parses logs."""
        client = CraftyClient(self.config)
        with patch.object(
            client,
            "get_server_stats",
            return_value={"running": True, "online": 1, "players": "[]"},
        ), patch.object(
            client, "send_server_command"
        ) as mock_cmd, patch.object(
            client,
            "get_server_logs",
            return_value=[
                "[INFO] Some other log line",
                "[INFO] There are 1/10 players online:",
                "[INFO] Gamer123",
            ],
        ), patch("bot.commands.mine_logic.time.sleep"):
            names = client.get_online_players("1")
            mock_cmd.assert_called_once_with("1", "list")
            self.assertEqual(names, ["Gamer123"])

    def test_crafty_client_get_server_logs(self) -> None:
        """get_server_logs returns a list of log line strings."""
        client = CraftyClient(self.config)
        with patch.object(
            client,
            "_request_json",
            return_value={
                "status": "ok",
                "data": ["[INFO] Line 1", "[INFO] Line 2"],
            },
        ):
            logs = client.get_server_logs("1")
            self.assertEqual(logs, ["[INFO] Line 1", "[INFO] Line 2"])

    def test_format_duration(self) -> None:
        self.assertEqual(_format_duration(0), "0 min")
        self.assertEqual(_format_duration(-5), "0 min")
        self.assertEqual(_format_duration(45), "45 s")
        self.assertEqual(_format_duration(60), "1 min")
        self.assertEqual(_format_duration(150), "2 min")
        self.assertEqual(_format_duration(3600), "1 h")
        self.assertEqual(_format_duration(3720), "1 h 2 min")
        self.assertEqual(_format_duration(7200), "2 h")
        self.assertEqual(_format_duration(90000), "1 pv 1 h")
        self.assertEqual(_format_duration(86400 * 2), "2 pv")

    def test_format_death_causes(self) -> None:
        self.assertEqual(_format_death_causes([]), "")
        self.assertEqual(
            _format_death_causes(["was slain by Zombie", "fell from a high place", "was slain by Zombie"]),
            "was slain by Zombie (2x), fell from a high place (1x)",
        )

    def test_crafty_client_get_server_allowlist_entries(self) -> None:
        client = CraftyClient(self.config)
        allowlist_json = json.dumps([
            {"name": "PlayerOne", "xuid": "11111", "ignoresPlayerLimit": False},
            {"name": "PlayerTwo", "xuid": "22222", "ignoresPlayerLimit": True},
        ])
        with patch.object(
            client,
            "get_server_file",
            return_value={"status": "ok", "data": {"content": allowlist_json}},
        ):
            entries = client.get_server_allowlist_entries("1")
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["name"], "PlayerOne")
            self.assertEqual(entries[0]["xuid"], "11111")
            self.assertFalse(entries[0]["ignoresPlayerLimit"])

    def test_crafty_client_get_server_permissions(self) -> None:
        client = CraftyClient(self.config)
        perms_json = json.dumps([
            {"permission": "operator", "xuid": "11111"},
            {"permission": "member", "xuid": "22222"},
            {"permission": "visitor", "name": "VisitorBob"},
        ])
        with patch.object(
            client,
            "get_server_file",
            return_value={"status": "ok", "data": {"content": perms_json}},
        ):
            perms = client.get_server_permissions("1")
            self.assertEqual(perms["11111"], "operator")
            self.assertEqual(perms["22222"], "member")
            self.assertEqual(perms["visitorbob"], "visitor")

    def test_parse_bds_player_stats(self) -> None:
        logs = [
            "[2026-08-27 10:00:00 INFO] Player connected: Steve, xuid: 1001",
            "[2026-08-27 10:15:00 INFO] Steve fell from a high place",
            "[2026-08-27 10:20:00 INFO] <Steve> Watch out for cliffs!",
            "[2026-08-27 10:30:00 INFO] Player disconnected: Steve, xuid: 1001",
            "[2026-08-27 11:00:00 INFO] Player connected: Alex, xuid: 1002",
            "[2026-08-27 11:45:00 INFO] Alex was slain by Zombie",
        ]
        allowlist = [
            {"name": "Steve", "xuid": "1001"},
            {"name": "Alex", "xuid": "1002"},
            {"name": "InactiveBob", "xuid": "1003"},
        ]
        permissions = {"1001": "operator", "1002": "member"}
        online_players = ["Alex"]

        stats = parse_bds_player_stats(
            log_lines=logs,
            allowlist_entries=allowlist,
            permissions=permissions,
            online_player_names=online_players,
        )

        self.assertEqual(len(stats), 3)

        # Alex is online -> sorted first
        alex = stats[0]
        self.assertEqual(alex.name, "Alex")
        self.assertTrue(alex.is_online)
        self.assertEqual(alex.role, "Jäsen")
        self.assertEqual(alex.deaths, 1)
        self.assertEqual(alex.death_causes, ["was slain by Zombie"])

        # Steve is offline, had 30m playtime
        steve = stats[1]
        self.assertEqual(steve.name, "Steve")
        self.assertFalse(steve.is_online)
        self.assertEqual(steve.role, "Ylläpitäjä (OP)")
        self.assertEqual(steve.total_playtime_seconds, 1800.0)
        self.assertEqual(steve.session_count, 1)
        self.assertEqual(steve.deaths, 1)
        self.assertEqual(steve.chat_count, 1)

        # InactiveBob was only in allowlist
        bob = stats[2]
        self.assertEqual(bob.name, "InactiveBob")
        self.assertFalse(bob.is_online)
        self.assertEqual(bob.session_count, 0)
        self.assertEqual(bob.total_playtime_seconds, 0.0)

    def test_fetch_mine_stats_unconfigured(self) -> None:
        unconfigured = CraftyConfig(api_token="")
        reply = fetch_mine_stats(unconfigured)
        self.assertIn("Crafty Controller -integraatiota ei ole määritetty", reply)

    def test_fetch_mine_stats_server_overview(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bds-1", "server_name": "Bedrock SMP"}
        ]
        client.get_server_logs.return_value = [
            "[2026-08-27 10:00:00 INFO] Player connected: Steve, xuid: 1001",
            "[2026-08-27 10:30:00 INFO] Steve fell from a high place",
            "[2026-08-27 11:00:00 INFO] Player disconnected: Steve, xuid: 1001",
        ]
        client.get_server_allowlist_entries.return_value = [
            {"name": "Steve", "xuid": "1001"}
        ]
        client.get_server_permissions.return_value = {"1001": "operator"}
        client.get_online_players.return_value = []

        reply = fetch_mine_stats(self.config, client=client)

        self.assertIn("📊 <b>Bedrock SMP</b> — <i>Pelaajatilastot (1 pelaajaa):</i>", reply)
        self.assertIn("⚪ <b>Steve</b> <i>(Ylläpitäjä (OP))</i>", reply)
        self.assertIn("<code>» TILA:     </code> Poissa linjoilta", reply)
        self.assertIn("<code>» PELIAIKA: </code> <b>1 h</b> (1 istuntoa)", reply)
        self.assertIn("<code>» KUOLEMAT: </code> <b>1 kpl</b> (viimeisin: <i>fell from a high place</i>)", reply)

    def test_fetch_mine_stats_single_player_found(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bds-1", "server_name": "Bedrock SMP"}
        ]
        client.get_server_logs.return_value = [
            "[2026-08-27 10:00:00 INFO] Player connected: Steve, xuid: 1001",
            "[2026-08-27 10:15:00 INFO] Steve fell from a high place",
            "[2026-08-27 10:20:00 INFO] <Steve> Hello world",
            "[2026-08-27 10:30:00 INFO] Player disconnected: Steve, xuid: 1001",
        ]
        client.get_server_allowlist_entries.return_value = [{"name": "Steve", "xuid": "1001"}]
        client.get_server_permissions.return_value = {"1001": "operator"}
        client.get_online_players.return_value = []

        reply = fetch_mine_stats(self.config, player_query="Steve", client=client)

        self.assertIn("📊 <b>Steve</b> — <i>Pelaajatilastot (Bedrock SMP)</i>", reply)
        self.assertIn("<code>» TILA:       </code> ⚪ <b>Poissa linjoilta</b>", reply)
        self.assertIn("<code>» ROOLI:      </code> <b>Ylläpitäjä (OP)</b>", reply)
        self.assertIn("<code>» PELIAIKA:   </code> <b>30 min</b>", reply)
        self.assertIn("<code>» ISTUNNOT:   </code> <b>1 kpl</b>", reply)
        self.assertIn("<code>» KUOLEMAT:   </code> <b>1 kpl</b>", reply)
        self.assertIn("<code>» KUOLINSYYT: </code> <i>fell from a high place (1x)</i>", reply)
        self.assertIn("<code>» CHAT:       </code> <b>1 viestiä</b>", reply)
        self.assertIn("<code>» XUID:       </code> <code>1001</code>", reply)

    def test_fetch_mine_stats_single_player_not_found(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bds-1", "server_name": "Bedrock SMP"}
        ]
        client.get_server_logs.return_value = []
        client.get_server_allowlist_entries.return_value = []
        client.get_server_permissions.return_value = {}
        client.get_online_players.return_value = []

        reply = fetch_mine_stats(self.config, player_query="NonExistentPlayer", client=client)

        self.assertIn("Pelaajaa '<b>NonExistentPlayer</b>' ei löytynyt palvelimen <b>Bedrock SMP</b>", reply)

    def test_fetch_mine_stats_smart_server_resolution(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bds-1", "server_name": "Bedrock SMP"}
        ]
        client.get_server_logs.return_value = [
            "[2026-08-27 10:00:00 INFO] Player connected: Xbox Gamer 123, xuid: 5555",
            "[2026-08-27 10:45:00 INFO] Player disconnected: Xbox Gamer 123, xuid: 5555",
        ]
        client.get_server_allowlist_entries.return_value = [{"name": "Xbox Gamer 123", "xuid": "5555"}]
        client.get_server_permissions.return_value = {}
        client.get_online_players.return_value = []

        # User ran "!mine stats Xbox Gamer 123" where "Xbox" was put in server_query and "Gamer 123" in player_query
        reply = fetch_mine_stats(
            self.config, server_query="Xbox", player_query="Gamer 123", client=client
        )

        self.assertIn("📊 <b>Xbox Gamer 123</b> — <i>Pelaajatilastot (Bedrock SMP)</i>", reply)
        self.assertIn("<code>» PELIAIKA:   </code> <b>45 min</b>", reply)

    def test_handle_mine_command_stats_routing(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bds-1", "server_name": "Bedrock SMP"}
        ]
        client.get_server_logs.return_value = []
        client.get_server_allowlist_entries.return_value = [{"name": "Steve", "xuid": "1001"}]
        client.get_server_permissions.return_value = {}
        client.get_online_players.return_value = []

        reply = handle_mine_command(self.config, "!mine stats", client=client)
        self.assertIn("📊 <b>Bedrock SMP</b> — <i>Pelaajatilastot (1 pelaajaa):</i>", reply)
        self.assertIn("Steve", reply)

    def test_handle_mine_card_command_status(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {
                "server_id": "bds-1",
                "server_name": "Bedrock SMP",
                "running": True,
                "server_port": 19132,
                "world_name": "Bedrock World",
            }
        ]
        client.get_server_stats.return_value = {
            "running": True,
            "online": 2,
            "max_players": 10,
            "players": ["Steve", "Alex"],
            "cpu": 12.5,
            "mem_percent": 35.0,
            "mem": 1024 * 1024 * 512,
            "version": "1.21.20",
        }

        reply_text, card = handle_mine_card_command(self.config, "!mine", client=client)
        self.assertIn("PÄÄLLÄ", reply_text)
        self.assertIsNotNone(card)
        self.assertEqual(card.title, "Bedrock SMP")
        self.assertEqual(card.badge.text, "ONLINE")
        # Verify card can be rendered into valid PNG image
        png_bytes = render_card(card)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_handle_mine_card_command_allowlist(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bds-1", "server_name": "Bedrock SMP"}
        ]
        client.get_server_allowlist.return_value = ["Steve", "Alex", "Notch"]

        reply_text, card = handle_mine_card_command(self.config, "!mine allowlist", client=client)
        self.assertIn("Sallitut pelaajat (3)", reply_text)
        self.assertIsNotNone(card)
        self.assertEqual(card.badge.text, "3 PELAAJAA")
        png_bytes = render_card(card)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_handle_mine_card_command_allowlist_add(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bds-1", "server_name": "Bedrock SMP"}
        ]
        client.add_to_allowlist.return_value = True

        reply_text, card = handle_mine_card_command(self.config, "!mine allowlist add Jeb", client=client)
        self.assertIn("lisätty", reply_text)
        self.assertIsNotNone(card)
        self.assertEqual(card.badge.text, "LISÄTTY")
        png_bytes = render_card(card)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_handle_mine_card_command_stats_single_player(self) -> None:
        client = MagicMock(spec=CraftyClient)
        client.get_servers.return_value = [
            {"server_id": "bds-1", "server_name": "Bedrock SMP"}
        ]
        client.get_server_logs.return_value = [
            "[2026-08-27 12:00:00 INFO] Player connected: Steve, xuid: 1111",
            "[2026-08-27 12:05:00 INFO] Steve fell from a high place",
            "[2026-08-27 13:00:00 INFO] Player disconnected: Steve, xuid: 1111",
        ]
        client.get_server_allowlist_entries.return_value = [{"name": "Steve", "xuid": "1111"}]
        client.get_server_permissions.return_value = {"1111": "operator"}
        client.get_online_players.return_value = []

        reply_text, card = handle_mine_card_command(self.config, "!mine stats Steve", client=client)
        self.assertIn("Pelaajatilastot", reply_text)
        self.assertIsNotNone(card)
        self.assertEqual(card.title, "Steve")
        png_bytes = render_card(card)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_handle_mine_card_command_unconfigured(self) -> None:
        unconfigured = CraftyConfig()
        reply_text, card = handle_mine_card_command(unconfigured, "!mine")
        self.assertIn("ei ole määritetty", reply_text)
        self.assertIsNotNone(card)
        self.assertEqual(card.badge.text, "EI KÄYTÖSSÄ")


if __name__ == "__main__":
    unittest.main()



