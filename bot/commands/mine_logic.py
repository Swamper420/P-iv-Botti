from __future__ import annotations

import html
import json
import logging
import re
import ssl
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bot.config import CraftyConfig

LOGGER = logging.getLogger(__name__)

_MINE_CMD_RE = re.compile(r"^\s*!mine(?:\s+|$)", re.IGNORECASE)


def parse_mine_command(text: str | None) -> tuple[bool, str]:
    """Parse !mine command and optional server name/id argument."""
    if not text:
        return False, ""
    match = _MINE_CMD_RE.search(text)
    if not match:
        return False, ""
    server_query = text[match.end() :].strip()
    return True, server_query


class CraftyClient:
    """Client for interacting with Crafty Controller REST API."""

    def __init__(self, config: CraftyConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.api_token = config.api_token
        self.timeout_seconds = config.timeout_seconds
        self.verify_ssl = config.verify_ssl

    def _create_ssl_context(self) -> ssl.SSLContext | None:
        if self.base_url.startswith("https://"):
            if not self.verify_ssl:
                ctx = ssl._create_unverified_context()
                return ctx
            return ssl.create_default_context()
        return None

    def _request_json(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
            "User-Agent": "P-iv-Botti/1.0",
        }
        req = Request(url, headers=headers, method="GET")
        ssl_ctx = self._create_ssl_context()

        with urlopen(req, timeout=self.timeout_seconds, context=ssl_ctx) as response:
            data = response.read().decode("utf-8")
            if not data.strip():
                return {}
            return json.loads(data)

    def get_servers(self) -> list[dict[str, Any]]:
        """Fetch list of all servers managed by Crafty Controller."""
        response = self._request_json("/api/v2/servers")
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            data = response.get("data")
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                servers = data.get("servers")
                if isinstance(servers, list):
                    return servers
                return [data]
        return []

    def get_server_stats(self, server_id: str | int) -> dict[str, Any]:
        """Fetch stats for a specific server by ID."""
        response = self._request_json(f"/api/v2/servers/{server_id}/stats")
        if isinstance(response, dict):
            data = response.get("data")
            if isinstance(data, dict):
                return data
            return response
        return {}


def _format_server_block(server: dict[str, Any], stats: dict[str, Any]) -> str:
    """Format individual server info and stats into HTML-formatted lines."""
    server_id = (
        server.get("server_id")
        or server.get("server_uuid")
        or server.get("id")
        or stats.get("server_id")
        or "?"
    )
    raw_name = (
        server.get("server_name")
        or server.get("name")
        or stats.get("server_name")
        or f"Palvelin {server_id}"
    )
    server_name = html.escape(str(raw_name))

    # Determine running status
    running_val = stats.get("running")
    if running_val is None:
        running_val = server.get("running")

    status_val = stats.get("status") or server.get("status")

    is_running = False
    if isinstance(running_val, bool):
        is_running = running_val
    elif isinstance(running_val, str):
        is_running = running_val.strip().lower() in ("true", "1", "running", "started")
    elif status_val and isinstance(status_val, str):
        is_running = status_val.strip().lower() in ("running", "started", "online")

    status_lower = str(status_val).strip().lower() if status_val else ""
    if is_running:
        status_icon = "🟢"
        status_text = "Päällä"
    elif status_lower in ("starting", "restarting", "käynnistyy"):
        status_icon = "🟡"
        status_text = "Käynnistyy"
    else:
        status_icon = "🔴"
        status_text = "Pois päältä"

    # Players
    online_players = (
        stats.get("online")
        if stats.get("online") is not None
        else stats.get("players_online", stats.get("online_players"))
    )
    max_players = (
        stats.get("max_players")
        if stats.get("max_players") is not None
        else stats.get("maxplayers", stats.get("players_max"))
    )

    player_list = stats.get("players") or stats.get("player_list") or []
    player_names: list[str] = []
    if isinstance(player_list, list):
        for p in player_list:
            if isinstance(p, dict):
                p_name = p.get("name") or p.get("username")
                if p_name:
                    player_names.append(str(p_name))
            elif isinstance(p, str) and p.strip():
                player_names.append(p.strip())

    if max_players is not None:
        players_display = f"{online_players or 0} / {max_players}"
    elif online_players is not None:
        players_display = str(online_players)
    else:
        players_display = "0"

    if player_names:
        escaped_names = [html.escape(n) for n in player_names]
        players_display += f" ({', '.join(escaped_names)})"

    lines = [
        f"{status_icon} <b>{server_name}</b>",
        f"   • Tila: {status_text}",
        f"   • Pelaajat: {players_display}",
    ]

    if is_running:
        cpu = stats.get("cpu")
        if cpu is not None:
            try:
                cpu_float = float(cpu)
                lines.append(f"   • CPU: {cpu_float:.1f} %")
            except (ValueError, TypeError):
                lines.append(f"   • CPU: {html.escape(str(cpu))}")

        mem_percent = stats.get("mem_percent") or stats.get("memory_percent")
        mem_usage = stats.get("mem") or stats.get("memory") or stats.get("mem_usage")
        if mem_percent is not None:
            try:
                mem_float = float(mem_percent)
                if mem_usage:
                    lines.append(
                        f"   • RAM: {mem_float:.1f} % ({html.escape(str(mem_usage))})"
                    )
                else:
                    lines.append(f"   • RAM: {mem_float:.1f} %")
            except (ValueError, TypeError):
                lines.append(f"   • RAM: {html.escape(str(mem_percent))}")
        elif mem_usage:
            lines.append(f"   • RAM: {html.escape(str(mem_usage))}")

        version = (
            stats.get("version")
            or server.get("version")
            or stats.get("server_version")
            or server.get("server_version")
        )
        if version:
            lines.append(f"   • Versio: {html.escape(str(version))}")

        port = (
            server.get("server_port")
            or server.get("port")
            or stats.get("server_port")
            or stats.get("port")
        )
        if port:
            lines.append(f"   • Portti: {html.escape(str(port))}")

        world = (
            stats.get("world_name")
            or stats.get("world")
            or server.get("world_name")
            or server.get("world")
        )
        if world:
            lines.append(f"   • Maailma: {html.escape(str(world))}")

        motd = (
            stats.get("motd")
            or stats.get("desc")
            or stats.get("description")
            or server.get("motd")
            or server.get("desc")
        )
        if motd:
            lines.append(f"   • Kuvaus: {html.escape(str(motd))}")

    return "\n".join(lines)


def fetch_mine_status(
    config: CraftyConfig,
    server_query: str = "",
    client: CraftyClient | None = None,
) -> str:
    """Fetch status of Crafty Controller Minecraft servers and return formatted reply."""
    if not config.is_configured:
        return (
            "⚠️ Crafty Controller -integraatiota ei ole määritetty "
            "(.env puuttuu CRAFTY_API_TOKEN)."
        )

    client = client or CraftyClient(config)

    try:
        servers = client.get_servers()
    except HTTPError as exc:
        LOGGER.warning("Crafty API HTTP error: %s", exc)
        if exc.code in (401, 403):
            return (
                f"⚠️ Crafty Controller API -autentikointivirhe (HTTP {exc.code}): "
                "Tarkista CRAFTY_API_TOKEN."
            )
        return f"⚠️ Crafty Controller API -virhe (HTTP {exc.code}): {exc.reason}"
    except URLError as exc:
        LOGGER.warning("Crafty API connection error: %s", exc)
        return f"⚠️ Yhteysvirhe Crafty Controlleriin: {exc.reason}"
    except Exception as exc:
        LOGGER.exception("Unexpected error querying Crafty API: %s", exc)
        return f"⚠️ Virhe haettaessa tietoja Crafty Controllerista: {exc}"

    if not servers:
        return "⚠️ Crafty Controllerista ei löytynyt yhtään palvelinta."

    # Filter servers if query is specified
    selected_servers: list[dict[str, Any]] = []
    if server_query:
        query_norm = server_query.strip().casefold()
        for s in servers:
            s_id = str(s.get("server_id", s.get("server_uuid", s.get("id", "")))).casefold()
            s_name = str(s.get("server_name", s.get("name", ""))).casefold()
            if query_norm == s_id or query_norm in s_name or query_norm in s_id:
                selected_servers.append(s)

        if not selected_servers:
            return f"⚠️ Palvelinta '{server_query}' ei löytynyt Crafty Controllerista."
    elif config.default_server_id:
        def_id = config.default_server_id.strip().casefold()
        for s in servers:
            s_id = str(s.get("server_id", s.get("server_uuid", s.get("id", "")))).casefold()
            s_name = str(s.get("server_name", s.get("name", ""))).casefold()
            if def_id == s_id or def_id == s_name:
                selected_servers.append(s)
        if not selected_servers:
            selected_servers = servers
    else:
        selected_servers = servers

    server_blocks: list[str] = []
    for server in selected_servers:
        s_id = (
            server.get("server_id")
            or server.get("server_uuid")
            or server.get("id")
        )
        stats: dict[str, Any] = {}
        if s_id is not None:
            try:
                stats = client.get_server_stats(s_id)
            except Exception as exc:
                LOGGER.warning("Failed to fetch stats for server %s: %s", s_id, exc)

        server_blocks.append(_format_server_block(server, stats))

    header = "⛏️ <b>Minecraft-palvelimet (Crafty Controller):</b>\n\n"
    return header + "\n\n".join(server_blocks)
