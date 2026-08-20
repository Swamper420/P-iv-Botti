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
_PLAYER_NAME_RE = re.compile(r"^[a-zA-Z0-9_ ]{1,32}$")


def parse_mine_command(
    text: str | None,
) -> tuple[bool, str, str, str]:
    """Parse !mine command into (is_match, subcommand, target_server, target_player).

    Subcommands:
      - 'status' (default): target_server is the server query
      - 'allowlist_list': target_server is optional server query
      - 'allowlist_add': target_server is optional server query, target_player is the player gamertag/name
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

    def _request_json(
        self, path: str, method: str = "GET", payload: dict[str, Any] | None = None
    ) -> Any:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
            "User-Agent": "P-iv-Botti/1.0",
        }
        data_bytes: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data_bytes = json.dumps(payload).encode("utf-8")

        req = Request(url, headers=headers, data=data_bytes, method=method)
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

    def send_server_command(self, server_id: str | int, command: str) -> dict[str, Any]:
        """Send a console command to a running server."""
        cmd = command.strip()
        if cmd.startswith("/"):
            cmd = cmd[1:]

        # Try POST /api/v2/servers/{server_id}/action/send_command first
        payload = {"command": cmd}
        try:
            return self._request_json(
                f"/api/v2/servers/{server_id}/action/send_command",
                method="POST",
                payload=payload,
            )
        except HTTPError as exc:
            if exc.code == 404:
                # Fallback to stdin action endpoint if available
                return self._request_json(
                    f"/api/v2/servers/{server_id}/action/stdin",
                    method="POST",
                    payload=payload,
                )
            raise

    def get_server_file(self, server_id: str | int, file_path: str) -> Any:
        """Fetch file content or metadata from a server via Crafty files endpoint."""
        clean_path = file_path.lstrip("/")
        return self._request_json(f"/api/v2/servers/{server_id}/files/{clean_path}")

    def get_server_allowlist(self, server_id: str | int) -> list[str]:
        """Retrieve allowlist player names for a server.

        Checks allowlist.json (Bedrock) or whitelist.json.
        """
        for filename in ("allowlist.json", "whitelist.json"):
            try:
                file_data = self.get_server_file(server_id, filename)
                entries = []
                if isinstance(file_data, list):
                    entries = file_data
                elif isinstance(file_data, dict):
                    if isinstance(file_data.get("data"), list):
                        entries = file_data["data"]
                    elif isinstance(file_data.get("content"), str):
                        try:
                            parsed = json.loads(file_data["content"])
                            if isinstance(parsed, list):
                                entries = parsed
                        except Exception:
                            pass

                names: list[str] = []
                for item in entries:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("username")
                        if name:
                            names.append(str(name))
                    elif isinstance(item, str) and item.strip():
                        names.append(item.strip())
                if names:
                    return names
            except Exception as exc:
                LOGGER.debug("Could not read %s for server %s: %s", filename, server_id, exc)

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


def _format_server_block(server: dict[str, Any], stats: dict[str, Any]) -> str:
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


def fetch_mine_status(
    config: CraftyConfig,
    server_query: str = "",
    client: CraftyClient | None = None,
) -> str:
    """Fetch status of Crafty Controller Minecraft servers and return formatted reply."""
    if not config.is_configured:
        return (
            "Crafty Controller -integraatiota ei ole määritetty "
            "(.env puuttuu CRAFTY_API_TOKEN)."
        )

    client = client or CraftyClient(config)

    try:
        servers = client.get_servers()
    except HTTPError as exc:
        LOGGER.warning("Crafty API HTTP error: %s", exc)
        if exc.code in (401, 403):
            return (
                f"Crafty Controller API -autentikointivirhe (HTTP {exc.code}): "
                "Tarkista CRAFTY_API_TOKEN."
            )
        return f"Crafty Controller API -virhe (HTTP {exc.code}): {exc.reason}"
    except URLError as exc:
        LOGGER.warning("Crafty API connection error: %s", exc)
        return f"Yhteysvirhe Crafty Controlleriin: {exc.reason}"
    except Exception as exc:
        LOGGER.exception("Unexpected error querying Crafty API: %s", exc)
        return f"Virhe haettaessa tietoja Crafty Controllerista: {exc}"

    if not servers:
        return "Crafty Controllerista ei löytynyt yhtään palvelinta."

    selected_servers, err_msg = _resolve_servers(
        servers, server_query, config.default_server_id
    )
    if err_msg:
        return err_msg

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

    return "\n\n".join(server_blocks)


def fetch_mine_allowlist(
    config: CraftyConfig,
    server_query: str = "",
    client: CraftyClient | None = None,
) -> str:
    """Fetch allowlist (whitelist) of Minecraft players for server(s)."""
    if not config.is_configured:
        return (
            "Crafty Controller -integraatiota ei ole määritetty "
            "(.env puuttuu CRAFTY_API_TOKEN)."
        )

    client = client or CraftyClient(config)

    try:
        servers = client.get_servers()
    except HTTPError as exc:
        LOGGER.warning("Crafty API HTTP error: %s", exc)
        if exc.code in (401, 403):
            return (
                f"Crafty Controller API -autentikointivirhe (HTTP {exc.code}): "
                "Tarkista CRAFTY_API_TOKEN."
            )
        return f"Crafty Controller API -virhe (HTTP {exc.code}): {exc.reason}"
    except URLError as exc:
        LOGGER.warning("Crafty API connection error: %s", exc)
        return f"Yhteysvirhe Crafty Controlleriin: {exc.reason}"
    except Exception as exc:
        LOGGER.exception("Unexpected error querying Crafty API: %s", exc)
        return f"Virhe haettaessa tietoja Crafty Controllerista: {exc}"

    if not servers:
        return "Crafty Controllerista ei löytynyt yhtään palvelinta."

    selected_servers, err_msg = _resolve_servers(
        servers, server_query, config.default_server_id
    )
    if err_msg:
        return err_msg

    blocks: list[str] = []
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

    return "\n\n".join(blocks)


def add_mine_allowlist(
    config: CraftyConfig,
    player_name: str,
    server_query: str = "",
    client: CraftyClient | None = None,
) -> str:
    """Add a player to the allowlist on the selected Minecraft server."""
    if not config.is_configured:
        return (
            "Crafty Controller -integraatiota ei ole määritetty "
            "(.env puuttuu CRAFTY_API_TOKEN)."
        )

    clean_name = player_name.strip()
    if not clean_name:
        return "Määritä pelaajanimi: <code>!mine allowlist add &lt;pelaajanimi&gt;</code>"

    if not _PLAYER_NAME_RE.match(clean_name):
        return (
            f"Virheellinen pelaajanimi '<b>{html.escape(clean_name)}</b>'. "
            "Bedrock-gamertag voi sisältää kirjaimia, numeroita, välilyöntejä ja alaviivoja (1-32 merkkiä)."
        )

    client = client or CraftyClient(config)

    try:
        servers = client.get_servers()
    except HTTPError as exc:
        LOGGER.warning("Crafty API HTTP error: %s", exc)
        if exc.code in (401, 403):
            return (
                f"Crafty Controller API -autentikointivirhe (HTTP {exc.code}): "
                "Tarkista CRAFTY_API_TOKEN."
            )
        return f"Crafty Controller API -virhe (HTTP {exc.code}): {exc.reason}"
    except URLError as exc:
        LOGGER.warning("Crafty API connection error: %s", exc)
        return f"Yhteysvirhe Crafty Controlleriin: {exc.reason}"
    except Exception as exc:
        LOGGER.exception("Unexpected error querying Crafty API: %s", exc)
        return f"Virhe haettaessa tietoja Crafty Controllerista: {exc}"

    if not servers:
        return "Crafty Controllerista ei löytynyt yhtään palvelinta."

    # If server_query was passed, try to match. If it didn't match and was 1 server total, fallback
    selected_servers, err_msg = _resolve_servers(
        servers, server_query, config.default_server_id
    )

    # If server_query failed to match any server, check if server_query was actually part of the player name when single server exists
    if err_msg and len(servers) == 1 and not config.default_server_id:
        selected_servers = servers
        clean_name = f"{server_query} {clean_name}".strip()
        err_msg = None

    if err_msg:
        return err_msg

    results: list[str] = []
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
            results.append(
                f"✅ Pelaaja <b>{html.escape(clean_name)}</b> lisätty palvelimen <b>{server_name}</b> sallittujen listalle (<code>allowlist add</code>)!"
            )
        except HTTPError as exc:
            LOGGER.warning("Failed to execute allowlist command on server %s: %s", s_id, exc)
            results.append(
                f"❌ Komennon suoritus epäonnistui palvelimella <b>{server_name}</b> (HTTP {exc.code}): {exc.reason}"
            )
        except Exception as exc:
            LOGGER.warning("Failed to execute allowlist command on server %s: %s", s_id, exc)
            results.append(
                f"❌ Virhe lisättäessä pelaajaa palvelimelle <b>{server_name}</b>: {exc}"
            )

    return "\n\n".join(results)


def handle_mine_command(
    config: CraftyConfig,
    text: str | None,
    client: CraftyClient | None = None,
) -> str:
    """Entrypoint for processing !mine command strings and routing appropriately."""
    is_match, subcommand, server_query, player_name = parse_mine_command(text)
    if not is_match:
        return ""

    if subcommand == "allowlist_list":
        return fetch_mine_allowlist(config, server_query=server_query, client=client)
    if subcommand == "allowlist_add":
        return add_mine_allowlist(
            config, player_name=player_name, server_query=server_query, client=client
        )
    return fetch_mine_status(config, server_query=server_query, client=client)

