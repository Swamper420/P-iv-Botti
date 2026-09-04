from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import json
import logging
import re
import ssl
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bot.config import CraftyConfig
from bot.rendering import Badge, BadgeColor, Card

LOGGER = logging.getLogger(__name__)

_MINE_CMD_RE = re.compile(r"^\s*!mine(?:\s+|$)", re.IGNORECASE)
_PLAYER_NAME_RE = re.compile(r"^[a-zA-Z0-9_ ]{1,32}$")

# Bedrock ``list`` command output:
#   "There are 2/10 players online:"
#   "Steve, Alex"
# (names may appear on the same line or the next line)
_BEDROCK_LIST_RE = re.compile(
    r"There are \d+/\d+ players online:\s*(.*)",
    re.IGNORECASE,
)

# Common log-line prefix patterns to strip, e.g.:
#   "[18:27:57 INFO]: ", "[INFO] ", "[2024-01-01 12:00:00 INFO]: "
_LOG_PREFIX_RE = re.compile(
    r"^\[.*?\]\s*:?\s*",
)

_BDS_TIMESTAMP_RE = re.compile(
    r"^(?:\[(?P<date>\d{4}-\d{2}-\d{2})?[ T]?(?P<time>\d{2}:\d{2}:\d{2})(?:\.\d+|\:\d+)?(?:Z)?\s*(?P<level>[A-Z]+)?\]\s*:?\s*|(?P<iso>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}))?"
)
_BDS_CONNECT_RE = re.compile(
    r"Player\s+connected:\s*(?P<name>[^,]+)(?:,\s*xuid:\s*(?P<xuid>\w+))?",
    re.IGNORECASE,
)
_BDS_DISCONNECT_RE = re.compile(
    r"Player\s+disconnected:\s*(?P<name>[^,]+)(?:,\s*xuid:\s*(?P<xuid>\w+))?",
    re.IGNORECASE,
)
_BDS_CHAT_RE = re.compile(
    r"^<(?P<name>[^>]+)>\s+(?P<msg>.*)$"
)

_BDS_DEATH_PATTERNS = [
    "was slain by",
    "was blown up by",
    "was shot by",
    "was killed by",
    "was squashed by",
    "was pricked to death",
    "was impaled by",
    "was roasted in",
    "was struck by",
    "fell from a high place",
    "fell from",
    "fell off",
    "fell out of",
    "drowned",
    "burned to death",
    "tried to swim in lava",
    "suffocated in a wall",
    "suffocated",
    "hit the ground too hard",
    "withered away",
    "starved to death",
    "experienced kinetic energy",
    "discovered the floor was lava",
    "went up in flames",
    "walked into danger zone",
    "walked into a cactus",
    "walked into fire",
    "blew up",
    "died",
    "froze to death",
]


@dataclass
class PlayerStatInfo:
    name: str
    xuid: str | None = None
    role: str | None = None
    is_online: bool = False
    current_session_seconds: float = 0.0
    total_playtime_seconds: float = 0.0
    session_count: int = 0
    first_seen: str | None = None
    last_seen: str | None = None
    deaths: int = 0
    death_causes: list[str] = field(default_factory=list)
    chat_count: int = 0
    ignores_player_limit: bool = False


def _parse_players_field(raw: Any) -> list[str]:
    """Extract player names from a Crafty stats ``players`` field.

    The field may be a Python list, a JSON-encoded string (``"[]"``), or
    ``None``.  Returns a flat list of player name strings.
    """
    if raw is None:
        return []

    # If it's already a Python list, use it directly
    entries: list[Any] | None = None
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return []
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                entries = parsed
        except (json.JSONDecodeError, ValueError):
            pass

    if entries is None:
        return []

    names: list[str] = []
    for item in entries:
        if isinstance(item, dict):
            name = item.get("name") or item.get("username")
            if name:
                names.append(str(name))
        elif isinstance(item, str) and item.strip():
            names.append(item.strip())
    return names


def parse_mine_command(
    text: str | None,
) -> tuple[bool, str, str, str]:
    """Parse !mine command into (is_match, subcommand, target_server, target_player).

    Subcommands:
      - 'status' (default): target_server is the server query
      - 'allowlist_list': target_server is optional server query
      - 'allowlist_add': target_server is optional server query, target_player is the player gamertag/name
      - 'stats': target_server is optional server query, target_player is optional player query
    """
    if not text:
        return False, "", "", ""
    match = _MINE_CMD_RE.search(text)
    if not match:
        return False, "", "", ""

    rest = text[match.end() :].strip()
    if not rest:
        return True, "status", "", ""

    tokens = rest.split()
    first_token_lower = tokens[0].lower()

    if first_token_lower in ("allowlist", "whitelist", "sallitut"):
        sub_args = tokens[1:]
        if not sub_args:
            return True, "allowlist_list", "", ""

        sub_action_lower = sub_args[0].lower()
        if sub_action_lower in ("add", "lisaa", "lisää", "+"):
            player_tokens = sub_args[1:]
            if not player_tokens:
                return True, "allowlist_add", "", ""
            # If 2+ tokens, check if the first token is a known server specifier or part of player name.
            # Example: "!mine allowlist add Notch" -> server="", player="Notch"
            # Example: "!mine allowlist add bedrock Steve" -> server="bedrock", player="Steve"
            # Example: "!mine allowlist add Xbox Gamertag 123" (Bedrock names can have spaces)
            # If length == 1: player = player_tokens[0]
            if len(player_tokens) == 1:
                return True, "allowlist_add", "", player_tokens[0]
            return True, "allowlist_add", player_tokens[0], " ".join(player_tokens[1:])

        if sub_action_lower in ("list", "lista", "nayta", "näytä"):
            server_query = " ".join(sub_args[1:]).strip()
            return True, "allowlist_list", server_query, ""

        # If !mine allowlist <palvelin>
        return True, "allowlist_list", " ".join(sub_args).strip(), ""

    if first_token_lower in ("stats", "tilastot", "statistiikka", "statit", "pelaajat"):
        sub_args = tokens[1:]
        if not sub_args:
            return True, "stats", "", ""
        if len(sub_args) == 1:
            return True, "stats", sub_args[0], ""
        return True, "stats", sub_args[0], " ".join(sub_args[1:])

    return True, "status", rest, ""


def _format_memory(raw_mem: Any) -> str | None:
    """Format memory value in bytes or numeric into human-readable MB/GB."""
    if raw_mem is None:
        return None
    if isinstance(raw_mem, (int, float)):
        val = float(raw_mem)
        if val > 1024 * 100:  # Value is in bytes
            mb = val / (1024 * 1024)
            if mb >= 1024:
                return f"{mb / 1024:.2f} GB"
            return f"{mb:.1f} MB"
        if val > 0:
            return f"{val:.1f} MB"
        return "0 MB"
    if isinstance(raw_mem, str):
        cleaned = raw_mem.strip()
        try:
            val = float(cleaned)
            return _format_memory(val)
        except ValueError:
            return cleaned
    return str(raw_mem)


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

    def _request(
        self,
        path: str,
        method: str = "GET",
        payload: Any = None,
        content_type: str = "application/json",
    ) -> tuple[int, str, Any]:
        """Perform HTTP request against Crafty Controller API."""
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "P-iv-Botti/1.0",
        }
        data_bytes: bytes | None = None
        if payload is not None:
            if isinstance(payload, (dict, list)):
                headers["Content-Type"] = "application/json"
                data_bytes = json.dumps(payload).encode("utf-8")
            elif isinstance(payload, str):
                headers["Content-Type"] = content_type
                data_bytes = payload.encode("utf-8")
            elif isinstance(payload, bytes):
                headers["Content-Type"] = content_type
                data_bytes = payload

        req = Request(url, headers=headers, data=data_bytes, method=method)
        ssl_ctx = self._create_ssl_context()

        with urlopen(req, timeout=self.timeout_seconds, context=ssl_ctx) as response:
            status = getattr(response, "status", 200)
            raw_text = response.read().decode("utf-8", errors="replace")
            parsed_json: Any = None
            if raw_text.strip():
                try:
                    parsed_json = json.loads(raw_text)
                except Exception:
                    parsed_json = None
            return status, raw_text, parsed_json

    def _request_json(
        self, path: str, method: str = "GET", payload: Any = None, content_type: str = "application/json"
    ) -> Any:
        _, raw_text, parsed_json = self._request(path, method=method, payload=payload, content_type=content_type)
        if parsed_json is not None:
            return parsed_json
        return raw_text

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

    def send_server_command(self, server_id: str | int, command: str) -> dict[str, Any]:
        """Send a console command to a running server via Crafty Controller stdin API.

        The /api/v2/servers/{id}/stdin endpoint expects the raw command as plain
        text in the request body (no JSON wrapper). This matches the Crafty 4 source:
        svr.send_command(self.request.body.decode("utf-8"))
        """
        cmd = command.strip()
        if cmd.startswith("/"):
            cmd = cmd[1:]

        return self._request_json(
            f"/api/v2/servers/{server_id}/stdin",
            method="POST",
            payload=cmd,
            content_type="text/plain",
        )

    def get_server_file(self, server_id: str | int, file_path: str) -> Any:
        """Fetch file content from server via Crafty Controller files API.

        The /api/v2/servers/{id}/files endpoint is a POST that accepts a JSON
        body {"path": "<relative_or_absolute_path>"} and returns:
        {"status": "ok", "data": {"content": "<file text>", "attributes": {...}}}
        """
        clean_path = file_path.lstrip("/")
        try:
            return self._request_json(
                f"/api/v2/servers/{server_id}/files",
                method="POST",
                payload={"path": clean_path},
            )
        except Exception as exc:
            LOGGER.debug("get_server_file failed for %s on server %s: %s", clean_path, server_id, exc)
            return None

    def _extract_file_content(self, file_data: Any) -> str | None:
        """Extract raw string content from Crafty file API response."""
        if not file_data:
            return None
        if isinstance(file_data, dict):
            data_field = file_data.get("data")
            if isinstance(data_field, dict):
                content = data_field.get("content")
                if content is not None:
                    return str(content)
            elif isinstance(data_field, str):
                return data_field
            if "content" in file_data:
                return str(file_data["content"])
        elif isinstance(file_data, str):
            return file_data
        return None

    def get_server_allowlist_entries(self, server_id: str | int) -> list[dict[str, Any]]:
        """Retrieve allowlist player entries (with name, xuid, etc.) for a server."""
        for filename in ("allowlist.json", "whitelist.json"):
            try:
                file_data = self.get_server_file(server_id, filename)
                raw_content = self._extract_file_content(file_data)
                if not raw_content:
                    continue

                entries: list[Any] = []
                try:
                    parsed = json.loads(raw_content)
                    if isinstance(parsed, list):
                        entries = parsed
                except Exception:
                    pass

                valid_entries: list[dict[str, Any]] = []
                for item in entries:
                    if isinstance(item, dict):
                        valid_entries.append(item)
                    elif isinstance(item, str) and item.strip():
                        valid_entries.append({"name": item.strip()})
                if valid_entries:
                    return valid_entries
            except Exception as exc:
                LOGGER.debug("Could not read %s for server %s: %s", filename, server_id, exc)

        return []

    def get_server_allowlist(self, server_id: str | int) -> list[str]:
        """Retrieve allowlist player names for a server."""
        entries = self.get_server_allowlist_entries(server_id)
        names: list[str] = []
        for item in entries:
            name = item.get("name") or item.get("username")
            if name and str(name).strip():
                names.append(str(name).strip())
        return names

    def get_server_permissions(self, server_id: str | int) -> dict[str, str]:
        """Retrieve permissions mapping {xuid_or_name: permission_level}."""
        for filename in ("permissions.json", "ops.json"):
            try:
                file_data = self.get_server_file(server_id, filename)
                raw_content = self._extract_file_content(file_data)
                if not raw_content:
                    continue

                entries: list[Any] = []
                try:
                    parsed = json.loads(raw_content)
                    if isinstance(parsed, list):
                        entries = parsed
                except Exception:
                    pass

                perms: dict[str, str] = {}
                for item in entries:
                    if isinstance(item, dict):
                        perm = item.get("permission") or item.get("level") or "operator"
                        xuid = item.get("xuid")
                        name = item.get("name")
                        if xuid:
                            perms[str(xuid)] = str(perm)
                        if name:
                            perms[str(name).casefold()] = str(perm)
                if perms:
                    return perms
            except Exception as exc:
                LOGGER.debug("Could not read %s for server %s: %s", filename, server_id, exc)

        return {}



    def get_server_logs(self, server_id: str | int) -> list[str]:
        """Fetch recent server log lines via the Crafty logs API.

        Returns a list of log-line strings (most recent lines last).
        """
        try:
            response = self._request_json(
                f"/api/v2/servers/{server_id}/logs?colors=false&raw=false&html=false"
            )
        except Exception as exc:
            LOGGER.debug("get_server_logs failed for server %s: %s", server_id, exc)
            return []

        if isinstance(response, dict):
            data = response.get("data")
            if isinstance(data, list):
                return [str(line) for line in data]
        if isinstance(response, list):
            return [str(line) for line in response]
        return []

    def get_online_players(self, server_id: str | int) -> list[str]:
        """Fetch online player names for a running Bedrock server.

        Strategy:
        1. Check stats ``players`` field (may be a JSON string like ``"[]"``).
        2. If empty, send the Bedrock ``list`` console command, wait briefly,
           then parse the server logs for the response.
        """
        # 1. Try stats players field
        try:
            stats = self.get_server_stats(server_id)
            raw_players = stats.get("players") or stats.get("player_list")
            names = _parse_players_field(raw_players)
            if names:
                return names
        except Exception as exc:
            LOGGER.debug("get_online_players stats lookup failed for server %s: %s", server_id, exc)

        # 2. Send Bedrock ``list`` command and read the response from logs
        #    Bedrock output format:
        #      line 1: "There are 1/10 players online:"
        #      line 2: "Gamer123, Gamer456"
        try:
            self.send_server_command(server_id, "list")
            time.sleep(1)  # give the server a moment to produce the response
            log_lines = self.get_server_logs(server_id)
            # Scan from newest to oldest for the Bedrock ``list`` response
            for i, line in enumerate(reversed(log_lines)):
                m = _BEDROCK_LIST_RE.search(line)
                if m:
                    tail = m.group(1).strip()
                    if tail:
                        return [
                            n.strip()
                            for n in tail.split(",")
                            if n.strip()
                        ]
                    # Names are on the *next* log line (chronologically
                    # after this one).
                    real_idx = len(log_lines) - 1 - i
                    if real_idx + 1 < len(log_lines):
                        next_line = log_lines[real_idx + 1].strip()
                        # Strip common log prefixes like "[INFO] " or
                        # "[18:27:57 INFO]: "
                        next_line = _LOG_PREFIX_RE.sub("", next_line).strip()
                        if next_line:
                            return [
                                n.strip()
                                for n in next_line.split(",")
                                if n.strip()
                            ]
                    return []
        except Exception as exc:
            LOGGER.debug("get_online_players list-command failed for server %s: %s", server_id, exc)

        return []

        return []

    def add_to_allowlist(self, server_id: str | int, player_name: str) -> bool:
        """Add a player to Bedrock allowlist using allowlist add."""
        # For Bedrock Dedicated Server / Crafty Bedrock:
        # Command is 'allowlist add <name>' or 'allowlist add "<name>"'
        escaped_name = f'"{player_name}"' if " " in player_name else player_name
        self.send_server_command(server_id, f"allowlist add {escaped_name}")
        # Also reload allowlist to ensure it takes effect immediately
        try:
            self.send_server_command(server_id, "allowlist reload")
        except Exception:
            pass
        return True


def _format_server_block(
    server: dict[str, Any],
    stats: dict[str, Any],
    online_player_names: list[str] | None = None,
) -> str:
    """Format individual server info and stats with stylized crazy formatting and gamer flavor."""
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
        status_text = "<b>PÄÄLLÄ</b> — <i>Kuutiot tulilla!</i>"
    elif status_lower in ("starting", "restarting", "käynnistyy"):
        status_text = "<b>KÄYNNISTYY</b> — <i>Palikat asettuvat paikoilleen...</i>"
    else:
        status_text = "<b>POIS PÄÄLTÄ</b> — <i>Kuutiot unessa, servu offline.</i>"

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

    player_list = stats.get("players") or stats.get("player_list")
    player_names: list[str] = _parse_players_field(player_list)

    # Use explicitly fetched online player names when stats didn't include them
    if not player_names and online_player_names:
        player_names = list(online_player_names)

    count_val = online_players or 0
    if max_players is not None:
        players_display = f"<b>{count_val}</b> / <b>{max_players}</b>"
    elif online_players is not None:
        players_display = f"<b>{count_val}</b>"
    else:
        players_display = "<b>0</b>"

    if player_names:
        escaped_names = [html.escape(n) for n in player_names]
        players_display += f" ({', '.join(escaped_names)})"
    elif count_val == 0 and is_running:
        players_display += " — <i>Aavemaisen hiljaista...</i>"

    lines = [
        f"<b>{server_name}</b>",
        f"<code>» TILA:    </code> {status_text}",
        f"<code>» PELAAJAT:</code> {players_display}",
    ]

    if is_running:
        cpu = stats.get("cpu")
        if cpu is not None:
            try:
                cpu_float = float(cpu)
                lines.append(f"<code>» CPU:     </code> <b>{cpu_float:.1f} %</b>")
            except (ValueError, TypeError):
                lines.append(f"<code>» CPU:     </code> <b>{html.escape(str(cpu))}</b>")

        mem_percent = stats.get("mem_percent") or stats.get("memory_percent")
        mem_usage = stats.get("mem") or stats.get("memory") or stats.get("mem_usage")
        formatted_mem = _format_memory(mem_usage)

        if formatted_mem and mem_percent is not None:
            try:
                mem_float = float(mem_percent)
                lines.append(
                    f"<code>» RAM:     </code> <b>{formatted_mem}</b> ({mem_float:.1f} %)"
                )
            except (ValueError, TypeError):
                lines.append(f"<code>» RAM:     </code> <b>{formatted_mem}</b>")
        elif formatted_mem:
            lines.append(f"<code>» RAM:     </code> <b>{formatted_mem}</b>")
        elif mem_percent is not None:
            try:
                mem_float = float(mem_percent)
                lines.append(f"<code>» RAM:     </code> <b>{mem_float:.1f} %</b>")
            except (ValueError, TypeError):
                lines.append(f"<code>» RAM:     </code> <b>{html.escape(str(mem_percent))}</b>")

        version = (
            stats.get("version")
            or server.get("version")
            or stats.get("server_version")
            or server.get("server_version")
        )
        if version:
            lines.append(f"<code>» VERSIO:  </code> <b>{html.escape(str(version))}</b>")

        port = (
            server.get("server_port")
            or server.get("port")
            or stats.get("server_port")
            or stats.get("port")
        )
        if port:
            lines.append(f"<code>» PORTTI:  </code> <code>{html.escape(str(port))}</code>")

        world = (
            stats.get("world_name")
            or stats.get("world")
            or server.get("world_name")
            or server.get("world")
        )
        if world and str(world).strip() != str(raw_name).strip():
            lines.append(f"<code>» MAAILMA: </code> <b>{html.escape(str(world))}</b>")

        motd = (
            stats.get("motd")
            or stats.get("desc")
            or stats.get("description")
            or server.get("motd")
            or server.get("desc")
        )
        if (
            motd
            and str(motd).strip() != str(raw_name).strip()
            and str(motd).strip() != str(world).strip()
        ):
            lines.append(f"<code>» KUVAUS:  </code> <i>{html.escape(str(motd))}</i>")

    return "\n".join(lines)


def _resolve_servers(
    servers: list[dict[str, Any]],
    server_query: str,
    default_server_id: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Resolve target server(s) from a query or default."""
    if server_query:
        query_norm = server_query.strip().casefold()
        selected: list[dict[str, Any]] = []
        for s in servers:
            s_id = str(s.get("server_id", s.get("server_uuid", s.get("id", "")))).casefold()
            s_name = str(s.get("server_name", s.get("name", ""))).casefold()
            if query_norm == s_id or query_norm in s_name or query_norm in s_id:
                selected.append(s)
        if not selected:
            return [], f"Palvelinta '{server_query}' ei löytynyt Crafty Controllerista."
        return selected, None

    if default_server_id:
        def_id = default_server_id.strip().casefold()
        selected = []
        for s in servers:
            s_id = str(s.get("server_id", s.get("server_uuid", s.get("id", "")))).casefold()
            s_name = str(s.get("server_name", s.get("name", ""))).casefold()
            if def_id == s_id or def_id == s_name:
                selected.append(s)
        if selected:
            return selected, None

    return servers, None


def build_mine_info_card(
    title: str,
    text: str,
    badge_text: str = "TIETO",
    badge_color: BadgeColor | str = BadgeColor.BLUE,
    subtitle: str | None = None,
) -> Card:
    """Build a simple informational Card for errors, notices, and help."""
    return (
        Card(title=title, subtitle=subtitle, footer="Crafty Controller • P-iv-Botti")
        .set_badge(badge_text, badge_color)
        .add_text(text)
    )


def build_mine_status_card(
    servers_data: list[tuple[dict[str, Any], dict[str, Any], list[str]]],
) -> Card:
    """Build a rich Card for Minecraft server(s) status."""
    if not servers_data:
        return build_mine_info_card(
            "Minecraft Palvelimet",
            "Ei palvelimia saatavilla.",
            badge_text="TYHJÄ",
            badge_color=BadgeColor.GRAY,
        )

    if len(servers_data) == 1:
        server, stats, online_player_names = servers_data[0]
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
        is_starting = (not is_running) and (status_lower in ("starting", "restarting", "käynnistyy"))

        if is_running:
            badge_label, badge_color = "ONLINE", BadgeColor.GREEN
        elif is_starting:
            badge_label, badge_color = "KÄYNNISTYY", BadgeColor.YELLOW
        else:
            badge_label, badge_color = "OFFLINE", BadgeColor.RED

        port = (
            server.get("server_port")
            or server.get("port")
            or stats.get("server_port")
            or stats.get("port")
        )
        world = (
            stats.get("world_name")
            or stats.get("world")
            or server.get("world_name")
            or server.get("world")
        )
        motd = (
            stats.get("motd")
            or stats.get("desc")
            or stats.get("description")
            or server.get("motd")
            or server.get("desc")
        )
        version = (
            stats.get("version")
            or server.get("version")
            or stats.get("server_version")
            or server.get("server_version")
        )

        sub_parts = []
        if port:
            sub_parts.append(f"Portti: {port}")
        if world and str(world).strip() != str(raw_name).strip():
            sub_parts.append(f"Maailma: {world}")
        sub_title = " • ".join(sub_parts) if sub_parts else None

        card = (
            Card(title=str(raw_name), subtitle=sub_title, footer="Crafty Controller • P-iv-Botti")
            .set_badge(badge_label, badge_color)
        )

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
        player_list = stats.get("players") or stats.get("player_list")
        player_names = _parse_players_field(player_list)
        if not player_names and online_player_names:
            player_names = list(online_player_names)

        count_val = online_players or 0
        if max_players is not None:
            p_disp = f"{count_val} / {max_players}"
        elif online_players is not None:
            p_disp = str(count_val)
        else:
            p_disp = "0"

        status_disp = "Päällä" if is_running else ("Käynnistyy" if is_starting else "Pois päältä")
        card.add_key_value("Tila", status_disp)
        card.add_key_value("Pelaajat", p_disp)

        if is_running:
            if version:
                card.add_key_value("Versio", str(version))
            if port and not sub_title:
                card.add_key_value("Portti", str(port))

            cpu = stats.get("cpu")
            if cpu is not None:
                try:
                    cpu_float = float(cpu)
                    card.add_progress_bar("CPU", value=round(cpu_float, 1), max_value=100.0, unit="%", color="#38bdf8")
                except (ValueError, TypeError):
                    card.add_key_value("CPU", str(cpu))

            mem_percent = stats.get("mem_percent") or stats.get("memory_percent")
            mem_usage = stats.get("mem") or stats.get("memory") or stats.get("mem_usage")
            formatted_mem = _format_memory(mem_usage)
            if mem_percent is not None:
                try:
                    mem_float = float(mem_percent)
                    label = f"RAM ({formatted_mem})" if formatted_mem else "RAM"
                    card.add_progress_bar(label, value=round(mem_float, 1), max_value=100.0, unit="%", color="#a855f7")
                except (ValueError, TypeError):
                    if formatted_mem:
                        card.add_key_value("RAM", formatted_mem)
            elif formatted_mem:
                card.add_key_value("RAM", formatted_mem)

        if player_names:
            card.add_divider()
            rows = [[str(i + 1), name] for i, name in enumerate(player_names)]
            card.add_table(
                headers=["#", "Pelaaja"],
                rows=rows,
                col_widths=[1, 6],
                primary_col=1,
                bold_cols=[1],
                max_rows=10,
                overflow="ellipsis",
            )
        elif count_val == 0 and is_running:
            card.add_text("Ei pelaajia paikalla (aavemaisen hiljaista).", muted=True)

        if motd and str(motd).strip() != str(raw_name).strip() and str(motd).strip() != str(world).strip():
            card.add_text(f"Kuvaus: {motd}", muted=True)

        return card

    # Multiple servers
    any_running = False
    for server, stats, _ in servers_data:
        running_val = stats.get("running")
        if running_val is None:
            running_val = server.get("running")
        status_val = stats.get("status") or server.get("status")
        if isinstance(running_val, bool) and running_val:
            any_running = True
            break
        if isinstance(running_val, str) and running_val.strip().lower() in ("true", "1", "running", "started"):
            any_running = True
            break
        if status_val and isinstance(status_val, str) and status_val.strip().lower() in ("running", "started", "online"):
            any_running = True
            break

    card = (
        Card(
            title="Minecraft Palvelimet",
            subtitle=f"{len(servers_data)} palvelinta",
            footer="Crafty Controller • P-iv-Botti",
        )
        .set_badge("ONLINE" if any_running else "OFFLINE", BadgeColor.GREEN if any_running else BadgeColor.RED)
    )

    for idx, (server, stats, online_player_names) in enumerate(servers_data):
        if idx > 0:
            card.add_divider()
        server_id = server.get("server_id") or server.get("server_uuid") or server.get("id") or stats.get("server_id") or "?"
        raw_name = server.get("server_name") or server.get("name") or stats.get("server_name") or f"Palvelin {server_id}"

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
        is_starting = (not is_running) and (status_lower in ("starting", "restarting", "käynnistyy"))

        if is_running:
            s_badge = Badge("PÄÄLLÄ", BadgeColor.GREEN)
        elif is_starting:
            s_badge = Badge("KÄYNNISTYY", BadgeColor.YELLOW)
        else:
            s_badge = Badge("OFFLINE", BadgeColor.RED)

        online_players = stats.get("online") if stats.get("online") is not None else stats.get("players_online", stats.get("online_players"))
        max_players = stats.get("max_players") if stats.get("max_players") is not None else stats.get("maxplayers", stats.get("players_max"))
        player_list = stats.get("players") or stats.get("player_list")
        player_names = _parse_players_field(player_list)
        if not player_names and online_player_names:
            player_names = list(online_player_names)

        count_val = online_players or 0
        p_disp = f"{count_val} / {max_players}" if max_players is not None else str(count_val)
        version = stats.get("version") or server.get("version") or stats.get("server_version") or server.get("server_version")

        card.add_text(str(raw_name), bold=True)
        items: list[tuple[str, str | Badge]] = [
            ("Tila", s_badge),
            ("Pelaajat", p_disp),
        ]
        if version:
            items.append(("Versio", str(version)))
        card.add_key_values(items, columns=2)
        if player_names:
            card.add_text(f"Pelaajat: {', '.join(player_names)}")

    return card


def build_mine_allowlist_card(servers_allowlist: list[tuple[str, list[str]]]) -> Card:
    """Build a rich Card for Minecraft server allowlist."""
    if not servers_allowlist:
        return build_mine_info_card(
            "Minecraft Allowlist",
            "Ei palvelimia saatavilla.",
            badge_text="TYHJÄ",
            badge_color=BadgeColor.GRAY,
        )

    if len(servers_allowlist) == 1:
        server_name, names = servers_allowlist[0]
        card = Card(
            title=f"Minecraft: {server_name}",
            subtitle=f"Sallitut pelaajat ({len(names)})",
            footer="Crafty Controller • P-iv-Botti",
        )
        if names:
            card.set_badge(f"{len(names)} PELAAJAA", BadgeColor.BLUE)
            rows = [[str(i + 1), name] for i, name in enumerate(names)]
            card.add_table(
                headers=["#", "Pelaaja"],
                rows=rows,
                col_widths=[1, 5],
                primary_col=1,
                bold_cols=[1],
                max_rows=25,
                overflow="ellipsis",
            )
        else:
            card.set_badge("TYHJÄ", BadgeColor.GRAY)
            card.add_text("Allowlist on tyhjä tai sitä ei saatu luettua.", muted=True)
            card.add_code_block("!mine allowlist add <pelaaja>")
        return card

    total_players = sum(len(names) for _, names in servers_allowlist)
    card = Card(
        title="Minecraft Allowlist",
        subtitle=f"{len(servers_allowlist)} palvelinta • {total_players} pelaajaa yhteensä",
        footer="Crafty Controller • P-iv-Botti",
    ).set_badge(f"{total_players} PELAAJAA", BadgeColor.BLUE)

    for idx, (server_name, names) in enumerate(servers_allowlist):
        if idx > 0:
            card.add_divider()
        card.add_text(f"Palvelin: {server_name} ({len(names)} pelaajaa)", bold=True)
        if names:
            rows = [[str(i + 1), name] for i, name in enumerate(names)]
            card.add_table(
                headers=["#", "Pelaaja"],
                rows=rows,
                col_widths=[1, 5],
                primary_col=1,
                bold_cols=[1],
                max_rows=10,
                overflow="ellipsis",
            )
        else:
            card.add_text("Allowlist on tyhjä.", muted=True)

    return card


def build_mine_allowlist_add_card(results: list[tuple[str, str, bool, str]]) -> Card:
    """Build a rich Card for allowlist add result."""
    if not results:
        return build_mine_info_card(
            "Minecraft Allowlist",
            "Ei palvelimia valittuna.",
            badge_text="VIRHE",
            badge_color=BadgeColor.RED,
        )

    all_ok = all(r[2] for r in results)
    server_name = results[0][0]
    player_name = results[0][1]

    if len(results) == 1:
        if all_ok:
            return (
                Card(
                    title=player_name,
                    subtitle=f"Minecraft Allowlist • {server_name}",
                    footer="Crafty Controller • P-iv-Botti",
                )
                .set_badge("LISÄTTY", BadgeColor.GREEN)
                .add_key_value("Palvelin", server_name)
                .add_key_value("Toiminto", "allowlist add")
                .add_text(f"Pelaaja {player_name} lisätty sallittujen listalle!")
            )
        else:
            return (
                Card(
                    title=player_name,
                    subtitle=f"Minecraft Allowlist Virhe • {server_name}",
                    footer="Crafty Controller • P-iv-Botti",
                )
                .set_badge("VIRHE", BadgeColor.RED)
                .add_text(results[0][3])
            )

    badge = Badge("LISÄTTY", BadgeColor.GREEN) if all_ok else Badge(
        "OSITTAINEN" if any(r[2] for r in results) else "VIRHE",
        BadgeColor.YELLOW if any(r[2] for r in results) else BadgeColor.RED,
    )
    card = Card(
        title=player_name,
        subtitle=f"Minecraft Allowlist ({len(results)} palvelinta)",
        footer="Crafty Controller • P-iv-Botti",
    ).set_badge(badge.text, badge.color)
    for s_name, _, ok, msg in results:
        status_txt = "✅ Lisätty" if ok else f"❌ {msg}"
        card.add_text(f"{s_name}: {status_txt}")
    return card


def fetch_mine_status_card(
    config: CraftyConfig,
    server_query: str = "",
    client: CraftyClient | None = None,
) -> tuple[str, Card | None]:
    """Fetch status of Crafty Controller Minecraft servers and return (formatted_text, card)."""
    if not config.is_configured:
        text = (
            "Crafty Controller -integraatiota ei ole määritetty "
            "(.env puuttuu CRAFTY_API_TOKEN)."
        )
        return text, build_mine_info_card("Minecraft Palvelin", text, badge_text="EI KÄYTÖSSÄ", badge_color=BadgeColor.GRAY)

    client = client or CraftyClient(config)

    try:
        servers = client.get_servers()
    except HTTPError as exc:
        LOGGER.warning("Crafty API HTTP error: %s", exc)
        if exc.code in (401, 403):
            text = (
                f"Crafty Controller API -autentikointivirhe (HTTP {exc.code}): "
                "Tarkista CRAFTY_API_TOKEN."
            )
            return text, build_mine_info_card("Minecraft Palvelin", text, badge_text="AUTH VIRHE", badge_color=BadgeColor.RED)
        text = f"Crafty Controller API -virhe (HTTP {exc.code}): {exc.reason}"
        return text, build_mine_info_card("Minecraft Palvelin", text, badge_text="API VIRHE", badge_color=BadgeColor.RED)
    except URLError as exc:
        LOGGER.warning("Crafty API connection error: %s", exc)
        text = f"Yhteysvirhe Crafty Controlleriin: {exc.reason}"
        return text, build_mine_info_card("Minecraft Palvelin", text, badge_text="YHTEYSVIRHE", badge_color=BadgeColor.RED)
    except Exception as exc:
        LOGGER.exception("Unexpected error querying Crafty API: %s", exc)
        text = f"Virhe haettaessa tietoja Crafty Controllerista: {exc}"
        return text, build_mine_info_card("Minecraft Palvelin", text, badge_text="VIRHE", badge_color=BadgeColor.RED)

    if not servers:
        text = "Crafty Controllerista ei löytynyt yhtään palvelinta."
        return text, build_mine_info_card("Minecraft Palvelin", text, badge_text="TYHJÄ", badge_color=BadgeColor.YELLOW)

    selected_servers, err_msg = _resolve_servers(
        servers, server_query, config.default_server_id
    )
    if err_msg:
        return err_msg, build_mine_info_card("Minecraft Palvelin", err_msg, badge_text="EI LÖYTYNYT", badge_color=BadgeColor.YELLOW)

    server_blocks: list[str] = []
    servers_data: list[tuple[dict[str, Any], dict[str, Any], list[str]]] = []
    for server in selected_servers:
        s_id = (
            server.get("server_id")
            or server.get("server_uuid")
            or server.get("id")
        )
        stats: dict[str, Any] = {}
        online_player_names: list[str] = []
        if s_id is not None:
            try:
                stats = client.get_server_stats(s_id)
            except Exception as exc:
                LOGGER.warning("Failed to fetch stats for server %s: %s", s_id, exc)

            has_players_in_stats = bool(
                _parse_players_field(
                    stats.get("players") or stats.get("player_list")
                )
            )
            if not has_players_in_stats:
                try:
                    online_player_names = client.get_online_players(s_id)
                except Exception as exc:
                    LOGGER.debug("Failed to fetch online players for server %s: %s", s_id, exc)

        server_blocks.append(
            _format_server_block(server, stats, online_player_names)
        )
        servers_data.append((server, stats, online_player_names))

    reply_text = "\n\n".join(server_blocks)
    card = build_mine_status_card(servers_data)
    return reply_text, card


def fetch_mine_status(
    config: CraftyConfig,
    server_query: str = "",
    client: CraftyClient | None = None,
) -> str:
    """Fetch status of Crafty Controller Minecraft servers and return formatted reply."""
    reply_text, _ = fetch_mine_status_card(config, server_query=server_query, client=client)
    return reply_text


def fetch_mine_allowlist_card(
    config: CraftyConfig,
    server_query: str = "",
    client: CraftyClient | None = None,
) -> tuple[str, Card | None]:
    """Fetch allowlist (whitelist) of Minecraft players for server(s) and return (text, card)."""
    if not config.is_configured:
        text = (
            "Crafty Controller -integraatiota ei ole määritetty "
            "(.env puuttuu CRAFTY_API_TOKEN)."
        )
        return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="EI KÄYTÖSSÄ", badge_color=BadgeColor.GRAY)

    client = client or CraftyClient(config)

    try:
        servers = client.get_servers()
    except HTTPError as exc:
        LOGGER.warning("Crafty API HTTP error: %s", exc)
        if exc.code in (401, 403):
            text = (
                f"Crafty Controller API -autentikointivirhe (HTTP {exc.code}): "
                "Tarkista CRAFTY_API_TOKEN."
            )
            return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="AUTH VIRHE", badge_color=BadgeColor.RED)
        text = f"Crafty Controller API -virhe (HTTP {exc.code}): {exc.reason}"
        return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="API VIRHE", badge_color=BadgeColor.RED)
    except URLError as exc:
        LOGGER.warning("Crafty API connection error: %s", exc)
        text = f"Yhteysvirhe Crafty Controlleriin: {exc.reason}"
        return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="YHTEYSVIRHE", badge_color=BadgeColor.RED)
    except Exception as exc:
        LOGGER.exception("Unexpected error querying Crafty API: %s", exc)
        text = f"Virhe haettaessa tietoja Crafty Controllerista: {exc}"
        return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="VIRHE", badge_color=BadgeColor.RED)

    if not servers:
        text = "Crafty Controllerista ei löytynyt yhtään palvelinta."
        return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="TYHJÄ", badge_color=BadgeColor.YELLOW)

    selected_servers, err_msg = _resolve_servers(
        servers, server_query, config.default_server_id
    )
    if err_msg:
        return err_msg, build_mine_info_card("Minecraft Allowlist", err_msg, badge_text="EI LÖYTYNYT", badge_color=BadgeColor.YELLOW)

    blocks: list[str] = []
    servers_allowlist: list[tuple[str, list[str]]] = []
    for server in selected_servers:
        s_id = (
            server.get("server_id")
            or server.get("server_uuid")
            or server.get("id")
        )
        raw_name = (
            server.get("server_name")
            or server.get("name")
            or f"Palvelin {s_id}"
        )
        server_name = html.escape(str(raw_name))

        if s_id is None:
            continue

        names = client.get_server_allowlist(s_id)
        servers_allowlist.append((str(raw_name), names))
        if names:
            escaped_names = [f"• <b>{html.escape(n)}</b>" for n in names]
            names_str = "\n".join(escaped_names)
            blocks.append(
                f"📋 <b>{server_name}</b> — <i>Sallitut pelaajat ({len(names)}):</i>\n{names_str}"
            )
        else:
            blocks.append(
                f"📋 <b>{server_name}</b> — <i>Allowlist on tyhjä tai sitä ei saatu luettua.</i>\n"
                f"Lisää pelaaja komennolla: <code>!mine allowlist add &lt;pelaaja&gt;</code>"
            )

    reply_text = "\n\n".join(blocks)
    card = build_mine_allowlist_card(servers_allowlist)
    return reply_text, card


def fetch_mine_allowlist(
    config: CraftyConfig,
    server_query: str = "",
    client: CraftyClient | None = None,
) -> str:
    """Fetch allowlist (whitelist) of Minecraft players for server(s)."""
    reply_text, _ = fetch_mine_allowlist_card(config, server_query=server_query, client=client)
    return reply_text


def add_mine_allowlist_card(
    config: CraftyConfig,
    player_name: str,
    server_query: str = "",
    client: CraftyClient | None = None,
) -> tuple[str, Card | None]:
    """Add a player to the allowlist on the selected Minecraft server and return (text, card)."""
    if not config.is_configured:
        text = (
            "Crafty Controller -integraatiota ei ole määritetty "
            "(.env puuttuu CRAFTY_API_TOKEN)."
        )
        return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="EI KÄYTÖSSÄ", badge_color=BadgeColor.GRAY)

    clean_name = player_name.strip()
    if not clean_name:
        text = "Määritä pelaajanimi: <code>!mine allowlist add &lt;pelaajanimi&gt;</code>"
        return text, build_mine_info_card("Minecraft Allowlist", "Määritä pelaajanimi: !mine allowlist add <pelaaja>", badge_text="OHJE", badge_color=BadgeColor.BLUE)

    if not _PLAYER_NAME_RE.fullmatch(clean_name):
        text = (
            f"Virheellinen pelaajanimi '<b>{html.escape(clean_name)}</b>'. "
            "Bedrock-gamertag voi sisältää kirjaimia, numeroita, välilyöntejä ja alaviivoja (1-32 merkkiä)."
        )
        return text, build_mine_info_card("Minecraft Allowlist", f"Virheellinen pelaajanimi '{clean_name}'.", badge_text="VIRHE", badge_color=BadgeColor.RED)

    client = client or CraftyClient(config)

    try:
        servers = client.get_servers()
    except HTTPError as exc:
        LOGGER.warning("Crafty API HTTP error: %s", exc)
        if exc.code in (401, 403):
            text = (
                f"Crafty Controller API -autentikointivirhe (HTTP {exc.code}): "
                "Tarkista CRAFTY_API_TOKEN."
            )
            return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="AUTH VIRHE", badge_color=BadgeColor.RED)
        text = f"Crafty Controller API -virhe (HTTP {exc.code}): {exc.reason}"
        return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="API VIRHE", badge_color=BadgeColor.RED)
    except URLError as exc:
        LOGGER.warning("Crafty API connection error: %s", exc)
        text = f"Yhteysvirhe Crafty Controlleriin: {exc.reason}"
        return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="YHTEYSVIRHE", badge_color=BadgeColor.RED)
    except Exception as exc:
        LOGGER.exception("Unexpected error querying Crafty API: %s", exc)
        text = f"Virhe haettaessa tietoja Crafty Controllerista: {exc}"
        return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="VIRHE", badge_color=BadgeColor.RED)

    if not servers:
        text = "Crafty Controllerista ei löytynyt yhtään palvelinta."
        return text, build_mine_info_card("Minecraft Allowlist", text, badge_text="TYHJÄ", badge_color=BadgeColor.YELLOW)

    selected_servers, err_msg = _resolve_servers(
        servers, server_query, config.default_server_id
    )

    if err_msg and len(servers) == 1 and not config.default_server_id:
        selected_servers = servers
        clean_name = f"{server_query} {clean_name}".strip()
        err_msg = None

    if err_msg:
        return err_msg, build_mine_info_card("Minecraft Allowlist", err_msg, badge_text="EI LÖYTYNYT", badge_color=BadgeColor.YELLOW)

    results_text: list[str] = []
    card_results: list[tuple[str, str, bool, str]] = []
    for server in selected_servers:
        s_id = (
            server.get("server_id")
            or server.get("server_uuid")
            or server.get("id")
        )
        raw_name = (
            server.get("server_name")
            or server.get("name")
            or f"Palvelin {s_id}"
        )
        server_name = html.escape(str(raw_name))

        if s_id is None:
            continue

        try:
            client.add_to_allowlist(s_id, clean_name)
            results_text.append(
                f"✅ Pelaaja <b>{html.escape(clean_name)}</b> lisätty palvelimen <b>{server_name}</b> sallittujen listalle (<code>allowlist add</code>)!"
            )
            card_results.append((str(raw_name), clean_name, True, "Lisätty sallittujen listalle"))
        except HTTPError as exc:
            LOGGER.warning("Failed to execute allowlist command on server %s: %s", s_id, exc)
            results_text.append(
                f"❌ Komennon suoritus epäonnistui palvelimella <b>{server_name}</b> (HTTP {exc.code}): {exc.reason}"
            )
            card_results.append((str(raw_name), clean_name, False, f"HTTP {exc.code}: {exc.reason}"))
        except Exception as exc:
            LOGGER.warning("Failed to execute allowlist command on server %s: %s", s_id, exc)
            results_text.append(
                f"❌ Virhe lisättäessä pelaajaa palvelimelle <b>{server_name}</b>: {exc}"
            )
            card_results.append((str(raw_name), clean_name, False, str(exc)))

    reply_text = "\n\n".join(results_text)
    card = build_mine_allowlist_add_card(card_results)
    return reply_text, card


def add_mine_allowlist(
    config: CraftyConfig,
    player_name: str,
    server_query: str = "",
    client: CraftyClient | None = None,
) -> str:
    """Add a player to the allowlist on the selected Minecraft server."""
    reply_text, _ = add_mine_allowlist_card(
        config, player_name=player_name, server_query=server_query, client=client
    )
    return reply_text


def _format_duration(seconds: float | int) -> str:
    """Format seconds into human-readable Finnish duration (e.g. 1 h 25 min)."""
    if seconds <= 0:
        return "0 min"
    sec = int(seconds)
    if sec < 60:
        return f"{sec} s"
    mins = (sec // 60) % 60
    hours = sec // 3600
    days = hours // 24
    if days > 0:
        rem_hours = hours % 24
        if rem_hours > 0:
            return f"{days} pv {rem_hours} h"
        return f"{days} pv"
    if hours > 0:
        if mins > 0:
            return f"{hours} h {mins} min"
        return f"{hours} h"
    return f"{mins} min"


def _format_death_causes(causes: list[str]) -> str:
    """Summarize list of death causes with counts, e.g. 'fell from a high place (2x), was slain by Zombie (1x)'."""
    if not causes:
        return ""
    counts: dict[str, int] = {}
    for c in causes:
        cleaned = c.strip()
        if cleaned:
            counts[cleaned] = counts.get(cleaned, 0) + 1

    parts = []
    # Sort by frequency descending
    for cause, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        if count > 1:
            parts.append(f"{cause} ({count}x)")
        else:
            parts.append(f"{cause} (1x)")
    return ", ".join(parts)


def parse_bds_player_stats(
    log_lines: list[str],
    allowlist_entries: list[dict[str, Any]] | None = None,
    permissions: dict[str, str] | None = None,
    online_player_names: list[str] | None = None,
) -> list[PlayerStatInfo]:
    """Parse Bedrock Dedicated Server log lines, allowlist and permissions into player statistics."""
    stats_by_name: dict[str, PlayerStatInfo] = {}
    online_names_set = {n.casefold(): n for n in (online_player_names or [])}
    perms_dict = permissions or {}

    # Initialize players from allowlist
    for entry in allowlist_entries or []:
        raw_name = entry.get("name") or entry.get("username")
        if not raw_name:
            continue
        p_name = str(raw_name).strip()
        if not p_name:
            continue
        pxuid = entry.get("xuid")
        stat = PlayerStatInfo(
            name=p_name,
            xuid=str(pxuid) if pxuid else None,
            ignores_player_limit=bool(entry.get("ignoresPlayerLimit", False)),
        )
        stats_by_name[p_name.casefold()] = stat

    # Track open sessions: player_key -> (start_epoch, start_ts_str)
    open_sessions: dict[str, tuple[float, str]] = {}
    last_known_epoch: float = 0.0
    simulated_second_counter: float = 0.0

    for line in log_lines:
        line_str = line.strip()
        if not line_str:
            continue

        ts_str: str | None = None
        current_epoch: float = 0.0

        ts_match = _BDS_TIMESTAMP_RE.match(line_str)
        if ts_match:
            date_val = ts_match.group("date")
            time_val = ts_match.group("time")
            iso_val = ts_match.group("iso")
            if iso_val:
                try:
                    dt = datetime.fromisoformat(iso_val.replace("Z", "+00:00"))
                    current_epoch = dt.timestamp()
                    ts_str = dt.strftime("%d.%m. %H:%M")
                except Exception:
                    pass
            elif date_val and time_val:
                try:
                    dt = datetime.strptime(f"{date_val} {time_val}", "%Y-%m-%d %H:%M:%S")
                    current_epoch = dt.timestamp()
                    ts_str = f"{date_val} {time_val}"
                except Exception:
                    pass
            elif time_val:
                try:
                    parts = [int(p) for p in time_val.split(":")]
                    current_epoch = parts[0] * 3600 + parts[1] * 60 + parts[2]
                    # Handle midnight rollover if necessary
                    if current_epoch < last_known_epoch and (last_known_epoch - current_epoch) > 3600 * 12:
                        current_epoch += 86400
                    ts_str = time_val
                except Exception:
                    pass

        if current_epoch > 0:
            last_known_epoch = max(last_known_epoch, current_epoch)
        else:
            simulated_second_counter += 1.0
            current_epoch = last_known_epoch + simulated_second_counter

        # Clean line to extract message
        cleaned_msg = _LOG_PREFIX_RE.sub("", line_str).strip()

        # 1. Connect
        conn_match = _BDS_CONNECT_RE.search(cleaned_msg)
        if conn_match:
            pname = conn_match.group("name").strip()
            pxuid = conn_match.group("xuid")
            p_fold = pname.casefold()

            if p_fold not in stats_by_name:
                stats_by_name[p_fold] = PlayerStatInfo(name=pname)
            p_stat = stats_by_name[p_fold]
            p_stat.name = pname
            if pxuid and not p_stat.xuid:
                p_stat.xuid = pxuid

            # If previous session was not closed, close it now
            if p_fold in open_sessions:
                prev_epoch, _ = open_sessions.pop(p_fold)
                duration = max(0.0, current_epoch - prev_epoch)
                p_stat.total_playtime_seconds += duration

            open_sessions[p_fold] = (current_epoch, ts_str or "")
            p_stat.session_count += 1
            if not p_stat.first_seen and ts_str:
                p_stat.first_seen = ts_str
            if ts_str:
                p_stat.last_seen = ts_str
            continue

        # 2. Disconnect
        disc_match = _BDS_DISCONNECT_RE.search(cleaned_msg)
        if disc_match:
            pname = disc_match.group("name").strip()
            pxuid = disc_match.group("xuid")
            p_fold = pname.casefold()

            if p_fold not in stats_by_name:
                stats_by_name[p_fold] = PlayerStatInfo(name=pname)
            p_stat = stats_by_name[p_fold]
            p_stat.name = pname
            if pxuid and not p_stat.xuid:
                p_stat.xuid = pxuid

            if p_fold in open_sessions:
                prev_epoch, _ = open_sessions.pop(p_fold)
                duration = max(0.0, current_epoch - prev_epoch)
                p_stat.total_playtime_seconds += duration

            if ts_str:
                p_stat.last_seen = ts_str
            continue

        # 3. Chat
        chat_match = _BDS_CHAT_RE.match(cleaned_msg)
        if chat_match:
            pname = chat_match.group("name").strip()
            p_fold = pname.casefold()
            if p_fold in stats_by_name:
                p_stat = stats_by_name[p_fold]
                p_stat.chat_count += 1
                if ts_str:
                    p_stat.last_seen = ts_str
            continue

        # 4. Deaths
        for pattern in _BDS_DEATH_PATTERNS:
            pattern_with_space = f" {pattern}"
            if pattern_with_space in cleaned_msg or cleaned_msg.startswith(pattern):
                idx = cleaned_msg.find(pattern)
                candidate_name = cleaned_msg[:idx].strip()
                if candidate_name:
                    p_fold = candidate_name.casefold()
                    if p_fold not in stats_by_name:
                        stats_by_name[p_fold] = PlayerStatInfo(name=candidate_name)
                    p_stat = stats_by_name[p_fold]
                    p_stat.deaths += 1
                    cause = cleaned_msg[idx:].strip()
                    p_stat.death_causes.append(cause)
                    if not p_stat.first_seen and ts_str:
                        p_stat.first_seen = ts_str
                    if ts_str:
                        p_stat.last_seen = ts_str
                    break

    # Reconcile currently online players
    for p_fold, orig_name in online_names_set.items():
        if p_fold not in stats_by_name:
            stats_by_name[p_fold] = PlayerStatInfo(name=orig_name)
        p_stat = stats_by_name[p_fold]
        p_stat.is_online = True
        p_stat.last_seen = "Paikalla nyt"
        if p_fold in open_sessions:
            start_epoch, _ = open_sessions.pop(p_fold)
            curr_duration = max(0.0, last_known_epoch - start_epoch)
            p_stat.current_session_seconds = curr_duration
            p_stat.total_playtime_seconds += curr_duration
        else:
            p_stat.session_count = max(p_stat.session_count, 1)

    # For offline players with an unclosed open session in logs:
    for p_fold, (start_epoch, _) in open_sessions.items():
        p_stat = stats_by_name.get(p_fold)
        if p_stat and not p_stat.is_online:
            duration = max(0.0, last_known_epoch - start_epoch)
            p_stat.total_playtime_seconds += duration

    # Assign roles & permissions
    for p_stat in stats_by_name.values():
        perm: str | None = None
        if p_stat.xuid and str(p_stat.xuid) in perms_dict:
            perm = perms_dict[str(p_stat.xuid)]
        elif p_stat.name.casefold() in perms_dict:
            perm = perms_dict[p_stat.name.casefold()]

        if perm:
            perm_lower = perm.lower()
            if "op" in perm_lower or "admin" in perm_lower:
                p_stat.role = "Ylläpitäjä (OP)"
            elif "member" in perm_lower or "jäsen" in perm_lower:
                p_stat.role = "Jäsen"
            elif "visitor" in perm_lower:
                p_stat.role = "Vierailija"
            else:
                p_stat.role = perm.title()
        else:
            p_stat.role = "Pelaaja"

    # Sort players: online first, then by total playtime descending, then by session count, then by name
    return sorted(
        stats_by_name.values(),
        key=lambda s: (not s.is_online, -s.total_playtime_seconds, -s.session_count, s.name.casefold()),
    )


def _format_single_player_stat(stat: PlayerStatInfo, server_name: str) -> str:
    """Format detailed single player statistics profile."""
    name_esc = html.escape(stat.name)
    server_esc = html.escape(server_name)

    if stat.is_online:
        session_str = _format_duration(stat.current_session_seconds)
        status_line = f"🟢 <b>Paikalla</b> (istunto {session_str})"
    else:
        status_line = "⚪ <b>Poissa linjoilta</b>"

    playtime_str = _format_duration(stat.total_playtime_seconds)
    lines = [
        f"📊 <b>{name_esc}</b> — <i>Pelaajatilastot ({server_esc})</i>",
        "",
        f"<code>» TILA:       </code> {status_line}",
        f"<code>» ROOLI:      </code> <b>{html.escape(stat.role or 'Pelaaja')}</b>",
        f"<code>» PELIAIKA:   </code> <b>{playtime_str}</b>",
        f"<code>» ISTUNNOT:   </code> <b>{stat.session_count} kpl</b>",
        f"<code>» KUOLEMAT:   </code> <b>{stat.deaths} kpl</b>",
    ]

    if stat.death_causes:
        causes_summary = _format_death_causes(stat.death_causes)
        lines.append(f"<code>» KUOLINSYYT: </code> <i>{html.escape(causes_summary)}</i>")

    if stat.chat_count > 0:
        lines.append(f"<code>» CHAT:       </code> <b>{stat.chat_count} viestiä</b>")

    if stat.first_seen:
        lines.append(f"<code>» ENSIKÄYNTI: </code> <b>{html.escape(stat.first_seen)}</b>")

    if stat.last_seen:
        lines.append(f"<code>» VIIMEKSI:   </code> <b>{html.escape(stat.last_seen)}</b>")

    if stat.xuid:
        lines.append(f"<code>» XUID:       </code> <code>{html.escape(str(stat.xuid))}</code>")

    return "\n".join(lines)


def _format_server_player_stats(server_name: str, stats: list[PlayerStatInfo]) -> str:
    """Format overview of all players for a server."""
    server_esc = html.escape(server_name)
    if not stats:
        return (
            f"📊 <b>{server_esc}</b> — <i>Pelaajatilastot</i>\n\n"
            "<i>Ei vielä tallennettuja pelaajatilastoja tai lokitapahtumia.</i>"
        )

    sections = [f"📊 <b>{server_esc}</b> — <i>Pelaajatilastot ({len(stats)} pelaajaa):</i>"]

    for stat in stats:
        name_esc = html.escape(stat.name)
        role_tag = f" <i>({html.escape(stat.role)})</i>" if stat.role and stat.role != "Pelaaja" else ""
        icon = "🟢" if stat.is_online else "⚪"

        if stat.is_online:
            session_str = _format_duration(stat.current_session_seconds)
            state_text = f"Paikalla (istunto {session_str})"
        else:
            state_text = "Poissa linjoilta"

        playtime_str = _format_duration(stat.total_playtime_seconds)
        last_seen_str = stat.last_seen or "Tuntematon"

        player_lines = [
            f"{icon} <b>{name_esc}</b>{role_tag}",
            f"<code>» TILA:     </code> {state_text}",
            f"<code>» PELIAIKA: </code> <b>{playtime_str}</b> ({stat.session_count} istuntoa)",
        ]

        if stat.deaths > 0:
            latest_death = stat.death_causes[-1] if stat.death_causes else ""
            death_tail = f" (viimeisin: <i>{html.escape(latest_death)}</i>)" if latest_death else ""
            player_lines.append(f"<code>» KUOLEMAT: </code> <b>{stat.deaths} kpl</b>{death_tail}")

        player_lines.append(f"<code>» VIIMEKSI: </code> {html.escape(last_seen_str)}")
        sections.append("\n".join(player_lines))

    return "\n\n".join(sections)


def build_mine_single_player_stat_card(stat: PlayerStatInfo, server_name: str) -> Card:
    """Build a rich Card for a single player profile."""
    card = Card(
        title=stat.name,
        subtitle=f"Minecraft • {server_name}",
        footer=f"{server_name} • P-iv-Botti Minecraft",
    )
    if stat.is_online:
        card.set_badge("PAIKALLA", BadgeColor.GREEN)
        session_str = _format_duration(stat.current_session_seconds)
        status_disp = f"Paikalla (istunto {session_str})"
    else:
        card.set_badge("OFFLINE", BadgeColor.GRAY)
        status_disp = "Poissa linjoilta"

    card.add_key_value("Tila", status_disp)
    card.add_key_value("Rooli", stat.role or "Pelaaja")
    card.add_key_value("Peliaika", _format_duration(stat.total_playtime_seconds))
    card.add_key_value("Istunnot", f"{stat.session_count} kpl")
    card.add_key_value("Kuolemat", f"{stat.deaths} kpl")
    card.add_key_value("Chat", f"{stat.chat_count} viestiä" if stat.chat_count > 0 else "0")
    if stat.first_seen:
        card.add_key_value("Ensikäynti", stat.first_seen)
    if stat.last_seen:
        card.add_key_value("Viimeksi", stat.last_seen)
    if stat.xuid:
        card.add_key_value("XUID", str(stat.xuid))

    if stat.death_causes:
        card.add_divider()
        card.add_text(f"Kuolinsyyt: {_format_death_causes(stat.death_causes)}")

    return card


def build_mine_server_player_stats_card(server_name: str, stats: list[PlayerStatInfo]) -> Card:
    """Build a rich Card overview of all player stats on a server."""
    card = Card(
        title="Minecraft Pelaajatilastot",
        subtitle=f"Palvelin: {server_name} • {len(stats)} pelaajaa",
        footer=f"{server_name} • P-iv-Botti Minecraft",
    )
    if not stats:
        card.set_badge("TYHJÄ", BadgeColor.GRAY)
        card.add_text("Ei vielä tallennettuja pelaajatilastoja tai lokitapahtumia.", muted=True)
        return card

    card.set_badge(f"{len(stats)} PELAAJAA", BadgeColor.BLUE)
    rows = []
    for s in stats:
        role_tag = f" ({s.role})" if s.role and s.role != "Pelaaja" else ""
        p_name = f"{s.name}{role_tag}"
        status_str = "Paikalla" if s.is_online else "Poissa"
        playtime_str = _format_duration(s.total_playtime_seconds)
        deaths_str = str(s.deaths)
        last_seen_str = s.last_seen or "-"
        rows.append([p_name, status_str, playtime_str, deaths_str, last_seen_str])

    card.add_table(
        headers=["Pelaaja", "Tila", "Peliaika", "Kuolemat", "Viimeksi"],
        rows=rows,
        alignments=["left", "left", "right", "right", "left"],
        col_widths=[2.2, 1.2, 1.8, 1.2, 2.0],
        primary_col=0,
        bold_cols=[0, 2],
        max_rows=25,
        overflow="ellipsis",
    )
    return card


def fetch_mine_stats_card(
    config: CraftyConfig,
    server_query: str = "",
    player_query: str = "",
    client: CraftyClient | None = None,
) -> tuple[str, Card | None]:
    """Fetch player statistics for Minecraft Bedrock server(s) and return (text, card)."""
    if not config.is_configured:
        text = (
            "Crafty Controller -integraatiota ei ole määritetty "
            "(.env puuttuu CRAFTY_API_TOKEN)."
        )
        return text, build_mine_info_card("Minecraft Tilastot", text, badge_text="EI KÄYTÖSSÄ", badge_color=BadgeColor.GRAY)

    client = client or CraftyClient(config)

    try:
        servers = client.get_servers()
    except HTTPError as exc:
        LOGGER.warning("Crafty API HTTP error: %s", exc)
        if exc.code in (401, 403):
            text = (
                f"Crafty Controller API -autentikointivirhe (HTTP {exc.code}): "
                "Tarkista CRAFTY_API_TOKEN."
            )
            return text, build_mine_info_card("Minecraft Tilastot", text, badge_text="AUTH VIRHE", badge_color=BadgeColor.RED)
        text = f"Crafty Controller API -virhe (HTTP {exc.code}): {exc.reason}"
        return text, build_mine_info_card("Minecraft Tilastot", text, badge_text="API VIRHE", badge_color=BadgeColor.RED)
    except URLError as exc:
        LOGGER.warning("Crafty API connection error: %s", exc)
        text = f"Yhteysvirhe Crafty Controlleriin: {exc.reason}"
        return text, build_mine_info_card("Minecraft Tilastot", text, badge_text="YHTEYSVIRHE", badge_color=BadgeColor.RED)
    except Exception as exc:
        LOGGER.exception("Unexpected error querying Crafty API: %s", exc)
        text = f"Virhe haettaessa tietoja Crafty Controllerista: {exc}"
        return text, build_mine_info_card("Minecraft Tilastot", text, badge_text="VIRHE", badge_color=BadgeColor.RED)

    if not servers:
        text = "Crafty Controllerista ei löytynyt yhtään palvelinta."
        return text, build_mine_info_card("Minecraft Tilastot", text, badge_text="TYHJÄ", badge_color=BadgeColor.YELLOW)

    # Smart server & player resolution
    target_player = player_query.strip()
    selected_servers, err_msg = _resolve_servers(
        servers, server_query, config.default_server_id
    )

    # If server_query did not match a server, check if server_query was actually the player name (or part of it)
    if err_msg:
        if len(servers) == 1 or config.default_server_id:
            fallback_servers, _ = _resolve_servers(servers, "", config.default_server_id)
            if fallback_servers:
                selected_servers = fallback_servers
                if target_player:
                    target_player = f"{server_query} {target_player}".strip()
                else:
                    target_player = server_query.strip()
                err_msg = None

    if err_msg:
        return err_msg, build_mine_info_card("Minecraft Tilastot", err_msg, badge_text="EI LÖYTYNYT", badge_color=BadgeColor.YELLOW)

    results: list[str] = []
    card: Card | None = None

    for server in selected_servers:
        s_id = (
            server.get("server_id")
            or server.get("server_uuid")
            or server.get("id")
        )
        raw_name = (
            server.get("server_name")
            or server.get("name")
            or f"Palvelin {s_id}"
        )
        server_name = str(raw_name)

        if s_id is None:
            continue

        try:
            logs = client.get_server_logs(s_id)
            allowlist_entries = client.get_server_allowlist_entries(s_id)
            permissions = client.get_server_permissions(s_id)
            online_players = client.get_online_players(s_id)

            player_stats = parse_bds_player_stats(
                log_lines=logs,
                allowlist_entries=allowlist_entries,
                permissions=permissions,
                online_player_names=online_players,
            )

            if target_player:
                target_norm = target_player.casefold()
                matched_stat = next(
                    (s for s in player_stats if s.name.casefold() == target_norm or target_norm in s.name.casefold()),
                    None,
                )
                if matched_stat:
                    results.append(_format_single_player_stat(matched_stat, server_name))
                    card = build_mine_single_player_stat_card(matched_stat, server_name)
                else:
                    results.append(
                        f"Pelaajaa '<b>{html.escape(target_player)}</b>' ei löytynyt "
                        f"palvelimen <b>{html.escape(server_name)}</b> tilastoista tai sallittujen listalta."
                    )
                    card = build_mine_info_card(
                        "Minecraft Haku",
                        f"Pelaajaa '{target_player}' ei löytynyt palvelimen {server_name} tilastoista.",
                        badge_text="EI LÖYTYNYT",
                        badge_color=BadgeColor.RED,
                        subtitle=f"Hakusana: {target_player}",
                    )
            else:
                results.append(_format_server_player_stats(server_name, player_stats))
                card = build_mine_server_player_stats_card(server_name, player_stats)

        except Exception as exc:
            LOGGER.exception("Failed to fetch player stats for server %s: %s", s_id, exc)
            results.append(
                f"❌ Virhe haettaessa pelaajatilastoja palvelimelta <b>{html.escape(server_name)}</b>: {exc}"
            )
            card = build_mine_info_card(
                "Minecraft Tilastot",
                f"Virhe haettaessa pelaajatilastoja: {exc}",
                badge_text="VIRHE",
                badge_color=BadgeColor.RED,
            )

    return "\n\n".join(results), card


def fetch_mine_stats(
    config: CraftyConfig,
    server_query: str = "",
    player_query: str = "",
    client: CraftyClient | None = None,
) -> str:
    """Fetch player statistics for Minecraft Bedrock server(s)."""
    reply_text, _ = fetch_mine_stats_card(
        config, server_query=server_query, player_query=player_query, client=client
    )
    return reply_text


def handle_mine_card_command(
    config: CraftyConfig,
    text: str | None,
    client: CraftyClient | None = None,
) -> tuple[str, Card | None]:
    """Entrypoint for processing !mine command strings and routing appropriately."""
    is_match, subcommand, server_query, player_name = parse_mine_command(text)
    if not is_match:
        return "", None

    if subcommand == "allowlist_list":
        return fetch_mine_allowlist_card(config, server_query=server_query, client=client)
    if subcommand == "allowlist_add":
        return add_mine_allowlist_card(
            config, player_name=player_name, server_query=server_query, client=client
        )
    if subcommand == "stats":
        return fetch_mine_stats_card(
            config, server_query=server_query, player_query=player_name, client=client
        )
    return fetch_mine_status_card(config, server_query=server_query, client=client)


def handle_mine_command(
    config: CraftyConfig,
    text: str | None,
    client: CraftyClient | None = None,
) -> str:
    """Entrypoint for processing !mine command strings and routing appropriately."""
    reply_text, _ = handle_mine_card_command(config, text, client=client)
    return reply_text

