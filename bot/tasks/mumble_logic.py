from __future__ import annotations

import asyncio
import logging
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    import pymumble_py3 as pymumble
    from pymumble_py3 import constants, mumble_pb2
    from pymumble_py3.constants import (
        PYMUMBLE_CLBK_CONNECTED,
        PYMUMBLE_CLBK_DISCONNECTED,
        PYMUMBLE_CLBK_USERCREATED,
        PYMUMBLE_CLBK_USERREMOVED,
        PYMUMBLE_CLBK_USERUPDATED,
        PYMUMBLE_CONN_STATE_AUTHENTICATING,
        PYMUMBLE_CONN_STATE_FAILED,
        PYMUMBLE_CONN_STATE_NOT_CONNECTED,
        PYMUMBLE_MSG_TYPES_AUTHENTICATE,
        PYMUMBLE_MSG_TYPES_USERSTATS,
        PYMUMBLE_MSG_TYPES_VERSION,
        PYMUMBLE_OS_STRING,
        PYMUMBLE_OS_VERSION_STRING,
        PYMUMBLE_PROTOCOL_VERSION,
    )
    HAVE_PYMUMBLE = True
except ImportError:  # pragma: no cover
    pymumble = None  # type: ignore
    constants = None  # type: ignore
    mumble_pb2 = None  # type: ignore
    HAVE_PYMUMBLE = False

if TYPE_CHECKING:
    from bot.config import MumbleConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class UserStatsData:
    session: int
    online_seconds: int | None = None
    idle_seconds: int | None = None
    tcp_ping_avg: float | None = None
    udp_ping_avg: float | None = None
    bandwidth: int | None = None
    client_release: str = ""
    client_os: str = ""
    client_os_version: str = ""
    packets_good: int = 0
    packets_lost: int = 0
    updated_at: float = field(default_factory=time.time)


@dataclass
class MumbleUserInfo:
    session: int
    name: str
    channel_id: int
    channel_name: str
    is_muted: bool = False
    is_deafened: bool = False
    is_self_muted: bool = False
    is_self_deafened: bool = False
    is_recording: bool = False
    is_priority_speaker: bool = False
    is_myself: bool = False
    online_seconds: int | None = None
    idle_seconds: int | None = None
    ping_ms: float | None = None
    bandwidth_kbps: float | None = None
    client_os: str = ""
    client_release: str = ""
    packets_good: int = 0
    packets_lost: int = 0
    joined_at: float | None = None


@dataclass
class MumbleChannelInfo:
    channel_id: int
    name: str
    parent_id: int | None = None
    user_sessions: list[int] = field(default_factory=list)


@dataclass
class MumbleServerSnapshot:
    is_connected: bool
    host: str
    port: int
    server_name: str
    bot_user: str
    channels: dict[int, MumbleChannelInfo] = field(default_factory=dict)
    users: dict[int, MumbleUserInfo] = field(default_factory=dict)
    error_message: str | None = None
    connected_at: float | None = None

    @property
    def active_users(self) -> list[MumbleUserInfo]:
        return [u for u in self.users.values() if not u.is_myself]

    @property
    def total_active_users(self) -> int:
        return len(self.active_users)


if HAVE_PYMUMBLE:
    class ModernMumbleClient(pymumble.Mumble):
        """
        Subclass of pymumble.Mumble that fixes Python 3.12+ ssl.wrap_socket removal
        and intercepts UserStats messages.
        """

        def __init__(self, *args, on_user_stats=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._on_user_stats = on_user_stats

        def connect(self) -> int:
            """Override connect using modern ssl.SSLContext compatible with Python 3.12+."""
            try:
                server_info = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
                self.Log.debug("connecting to %s (%s) on port %i.", self.host, server_info[0][1], self.port)
                std_sock = socket.socket(server_info[0][0], socket.SOCK_STREAM)
                std_sock.settimeout(10)
            except (socket.error, OSError) as err:
                self.Log.error("Socket error connecting to %s:%s: %s", self.host, self.port, err)
                self.connected = PYMUMBLE_CONN_STATE_FAILED
                return self.connected

            try:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                if self.certfile:
                    context.load_cert_chain(certfile=self.certfile, keyfile=self.keyfile)

                self.control_socket = context.wrap_socket(
                    std_sock, server_hostname=self.host if self.host else None
                )
            except Exception as err:
                self.Log.error("SSL context wrapping error: %s", err)
                self.connected = PYMUMBLE_CONN_STATE_FAILED
                return self.connected

            try:
                self.control_socket.connect((self.host, self.port))
                self.control_socket.setblocking(False)

                # Send Version
                version = mumble_pb2.Version()
                version.version = (
                    (PYMUMBLE_PROTOCOL_VERSION[0] << 16)
                    + (PYMUMBLE_PROTOCOL_VERSION[1] << 8)
                    + PYMUMBLE_PROTOCOL_VERSION[2]
                )
                version.release = self.application
                version.os = PYMUMBLE_OS_STRING
                version.os_version = PYMUMBLE_OS_VERSION_STRING
                self.Log.debug("sending: version: %s", version)
                self.send_message(PYMUMBLE_MSG_TYPES_VERSION, version)

                # Send Authenticate
                authenticate = mumble_pb2.Authenticate()
                authenticate.username = self.user
                authenticate.password = self.password
                authenticate.tokens.extend(self.tokens)
                authenticate.opus = True
                self.Log.debug("sending: authenticate: %s", authenticate)
                self.send_message(PYMUMBLE_MSG_TYPES_AUTHENTICATE, authenticate)
            except (socket.error, OSError) as err:
                self.Log.error("Connection or handshake error: %s", err)
                self.connected = PYMUMBLE_CONN_STATE_FAILED
                return self.connected

            self.connected = PYMUMBLE_CONN_STATE_AUTHENTICATING
            return self.connected

        def dispatch_control_message(self, msg_type: int, message: bytes) -> None:
            if msg_type == PYMUMBLE_MSG_TYPES_USERSTATS:
                try:
                    mess = mumble_pb2.UserStats()
                    mess.ParseFromString(message)
                    self.Log.debug("message: UserStats : %s", mess)
                    if callable(self._on_user_stats):
                        self._on_user_stats(mess)
                except Exception as err:
                    self.Log.error("Error parsing UserStats message: %s", err)
                return
            super().dispatch_control_message(msg_type, message)
else:
    ModernMumbleClient = None  # type: ignore


class MumbleManager:
    """
    Manages the lifecycle of a Mumble connection and maintains up-to-date
    server, channel, user, and statistics information.
    """

    def __init__(self, config: MumbleConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._client: ModernMumbleClient | None = None
        self._is_running = False
        self._connected_at: float | None = None
        self._last_error: str | None = None

        # Session tracking
        self._user_join_times: dict[int, float] = {}
        self._user_stats: dict[int, UserStatsData] = {}
        self._stats_updated_event = threading.Event()

    @property
    def is_connected(self) -> bool:
        with self._lock:
            if self._client is None:
                return False
            return getattr(self._client, "is_alive", lambda: False)() and getattr(
                self._client, "connected", None
            ) == constants.PYMUMBLE_CONN_STATE_CONNECTED

    def start(self) -> None:
        if not HAVE_PYMUMBLE:
            LOGGER.error("pymumble is not installed. Mumble task cannot start.")
            self._last_error = "pymumble ei ole asennettu"
            return

        if not self.config.is_configured:
            LOGGER.info("Mumble is not configured (MUMBLE_HOST empty). Task disabled.")
            return

        with self._lock:
            if self._is_running:
                return
            self._is_running = True

        self._spawn_client()

    def _spawn_client(self) -> None:
        with self._lock:
            if not self._is_running:
                return

            certfile = self.config.certfile or None
            keyfile = self.config.keyfile or None

            self._client = ModernMumbleClient(
                host=self.config.host,
                user=self.config.user,
                port=self.config.port,
                password=self.config.password,
                certfile=certfile,
                keyfile=keyfile,
                reconnect=True,
                on_user_stats=self._handle_user_stats,
            )

            self._client.callbacks.set_callback(
                PYMUMBLE_CLBK_CONNECTED, self._on_connected
            )
            self._client.callbacks.set_callback(
                PYMUMBLE_CLBK_DISCONNECTED, self._on_disconnected
            )
            self._client.callbacks.set_callback(
                PYMUMBLE_CLBK_USERCREATED, self._on_user_created
            )
            self._client.callbacks.set_callback(
                PYMUMBLE_CLBK_USERUPDATED, self._on_user_updated
            )
            self._client.callbacks.set_callback(
                PYMUMBLE_CLBK_USERREMOVED, self._on_user_removed
            )

            try:
                self._client.start()
                LOGGER.info(
                    "Mumble client started for %s:%s as %s",
                    self.config.host,
                    self.config.port,
                    self.config.user,
                )
            except Exception as err:
                LOGGER.error("Failed to start Mumble client: %s", err)
                self._last_error = str(err)

    def stop(self) -> None:
        with self._lock:
            self._is_running = False
            client = self._client
            self._client = None

        if client is not None:
            try:
                client.stop()
            except Exception as err:
                LOGGER.debug("Error stopping Mumble client: %s", err)
        LOGGER.info("Mumble client stopped.")

    def _on_connected(self) -> None:
        with self._lock:
            self._connected_at = time.time()
            self._last_error = None
            client = self._client

        LOGGER.info(
            "Mumble client successfully connected to %s:%s",
            self.config.host,
            self.config.port,
        )

        if client is None:
            return

        # Auto mute and deafen if configured
        try:
            myself = client.users.myself
            if myself:
                if self.config.auto_mute:
                    myself.mute()
                if self.config.auto_deafen:
                    myself.deafen()
        except Exception as err:
            LOGGER.warning("Failed to auto-mute/deafen bot on Mumble: %s", err)

        # Move to designated channel if configured
        if self.config.channel:
            try:
                target_channel = client.channels.find_by_name(self.config.channel)
                if target_channel:
                    client.users.myself.move_in(target_channel["channel_id"])
                    LOGGER.info("Mumble bot moved to channel %s", self.config.channel)
            except Exception as err:
                LOGGER.warning(
                    "Failed to move Mumble bot to channel %s: %s",
                    self.config.channel,
                    err,
                )

        # Record join times and request stats for already connected users
        with self._lock:
            current_time = time.time()
            for session in list(client.users.keys()):
                if session not in self._user_join_times:
                    self._user_join_times[session] = current_time
                self._request_user_stats_locked(client, session)

    def _on_disconnected(self) -> None:
        with self._lock:
            self._connected_at = None
        LOGGER.warning("Mumble client disconnected from %s:%s", self.config.host, self.config.port)

    def _on_user_created(self, user: dict) -> None:
        session = user.get("session")
        if session is None:
            return

        with self._lock:
            if session not in self._user_join_times:
                self._user_join_times[session] = time.time()
            client = self._client
            if client is not None:
                self._request_user_stats_locked(client, session)

    def _on_user_updated(self, user: dict, actions: dict) -> None:
        session = user.get("session")
        if session is not None and "channel_id" in actions:
            LOGGER.debug("User %s moved to channel %s", user.get("name"), actions.get("channel_id"))

    def _on_user_removed(self, user: dict, *args) -> None:
        session = user.get("session")
        if session is None:
            return

        with self._lock:
            self._user_join_times.pop(session, None)
            self._user_stats.pop(session, None)

    def _request_user_stats_locked(self, client: ModernMumbleClient, session: int) -> None:
        try:
            req = mumble_pb2.UserStats()
            req.session = session
            req.stats_only = True
            client.send_message(PYMUMBLE_MSG_TYPES_USERSTATS, req)
        except Exception as err:
            LOGGER.debug("Failed to send UserStats request for session %s: %s", session, err)

    def _handle_user_stats(self, mess) -> None:
        session = getattr(mess, "session", None)
        if session is None:
            return

        online_seconds = mess.onlinesecs if mess.HasField("onlinesecs") else None
        idle_seconds = mess.idlesecs if mess.HasField("idlesecs") else None
        tcp_ping_avg = mess.tcp_ping_avg if mess.HasField("tcp_ping_avg") else None
        udp_ping_avg = mess.udp_ping_avg if mess.HasField("udp_ping_avg") else None
        bandwidth = mess.bandwidth if mess.HasField("bandwidth") else None

        client_release = ""
        client_os = ""
        client_os_version = ""
        if mess.HasField("version"):
            v = mess.version
            if v.HasField("release"):
                client_release = v.release
            if v.HasField("os"):
                client_os = v.os
            if v.HasField("os_version"):
                client_os_version = v.os_version

        packets_good = 0
        packets_lost = 0
        if mess.HasField("from_client"):
            fc = mess.from_client
            if fc.HasField("good"):
                packets_good = fc.good
            if fc.HasField("lost"):
                packets_lost = fc.lost

        stats_entry = UserStatsData(
            session=session,
            online_seconds=online_seconds,
            idle_seconds=idle_seconds,
            tcp_ping_avg=tcp_ping_avg,
            udp_ping_avg=udp_ping_avg,
            bandwidth=bandwidth,
            client_release=client_release,
            client_os=client_os,
            client_os_version=client_os_version,
            packets_good=packets_good,
            packets_lost=packets_lost,
            updated_at=time.time(),
        )

        with self._lock:
            self._user_stats[session] = stats_entry
            self._stats_updated_event.set()

    async def refresh_stats(self, timeout: float = 1.5) -> None:
        """
        Request fresh user statistics for all connected users and wait briefly for updates.
        """
        if not self.is_connected:
            return

        with self._lock:
            client = self._client
            if client is None:
                return
            sessions = list(client.users.keys())
            self._stats_updated_event.clear()
            for s in sessions:
                self._request_user_stats_locked(client, s)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    def get_snapshot(self) -> MumbleServerSnapshot:
        """
        Generate an immutable snapshot of the current server state,
        including channels, users, durations, and statistics.
        """
        with self._lock:
            is_connected = self.is_connected
            host = self.config.host
            port = self.config.port
            bot_user = self.config.user
            client = self._client
            connected_at = self._connected_at
            last_err = self._last_error

            if not is_connected or client is None:
                return MumbleServerSnapshot(
                    is_connected=False,
                    host=host,
                    port=port,
                    server_name=host,
                    bot_user=bot_user,
                    error_message=last_err,
                )

            server_name = getattr(client, "host", host) or host
            myself_session = getattr(client.users, "myself_session", None)

            # Build channels map
            channels_map: dict[int, MumbleChannelInfo] = {}
            for ch_id, ch in client.channels.items():
                name = ch.get("name", f"Channel {ch_id}")
                parent_id = ch.get("parent")
                channels_map[ch_id] = MumbleChannelInfo(
                    channel_id=ch_id,
                    name=name,
                    parent_id=parent_id if parent_id != ch_id else None,
                )

            # Build users map
            now = time.time()
            users_map: dict[int, MumbleUserInfo] = {}
            for session, user in client.users.items():
                name = user.get("name", f"User {session}")
                channel_id = user.get("channel_id", 0)
                channel_info = channels_map.get(channel_id)
                channel_name = channel_info.name if channel_info else "Juuri"

                if channel_info:
                    channel_info.user_sessions.append(session)

                is_myself = session == myself_session
                is_muted = bool(user.get("mute") or user.get("self_mute"))
                is_deafened = bool(user.get("deaf") or user.get("self_deaf"))
                is_self_muted = bool(user.get("self_mute"))
                is_self_deafened = bool(user.get("self_deaf"))
                is_recording = bool(user.get("recording"))
                is_priority_speaker = bool(user.get("priority_speaker"))

                stats = self._user_stats.get(session)
                joined_at = self._user_join_times.get(session)

                online_seconds: int | None = None
                idle_seconds: int | None = None
                ping_ms: float | None = None
                bandwidth_kbps: float | None = None
                client_os = ""
                client_release = ""
                packets_good = 0
                packets_lost = 0

                if stats:
                    age = now - stats.updated_at
                    if stats.online_seconds is not None:
                        online_seconds = int(stats.online_seconds + age)
                    elif joined_at:
                        online_seconds = int(now - joined_at)

                    if stats.idle_seconds is not None:
                        idle_seconds = int(stats.idle_seconds + age)

                    if stats.udp_ping_avg is not None and stats.udp_ping_avg > 0:
                        ping_ms = stats.udp_ping_avg
                    elif stats.tcp_ping_avg is not None and stats.tcp_ping_avg > 0:
                        ping_ms = stats.tcp_ping_avg

                    if stats.bandwidth is not None and stats.bandwidth > 0:
                        bandwidth_kbps = (stats.bandwidth * 8) / 1000.0

                    client_os = stats.client_os
                    if stats.client_os_version:
                        client_os = f"{client_os} {stats.client_os_version}".strip()
                    client_release = stats.client_release
                    packets_good = stats.packets_good
                    packets_lost = stats.packets_lost
                elif joined_at:
                    online_seconds = int(now - joined_at)

                users_map[session] = MumbleUserInfo(
                    session=session,
                    name=name,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    is_muted=is_muted,
                    is_deafened=is_deafened,
                    is_self_muted=is_self_muted,
                    is_self_deafened=is_self_deafened,
                    is_recording=is_recording,
                    is_priority_speaker=is_priority_speaker,
                    is_myself=is_myself,
                    online_seconds=online_seconds,
                    idle_seconds=idle_seconds,
                    ping_ms=ping_ms,
                    bandwidth_kbps=bandwidth_kbps,
                    client_os=client_os,
                    client_release=client_release,
                    packets_good=packets_good,
                    packets_lost=packets_lost,
                    joined_at=joined_at,
                )

            return MumbleServerSnapshot(
                is_connected=True,
                host=host,
                port=port,
                server_name=server_name,
                bot_user=bot_user,
                channels=channels_map,
                users=users_map,
                connected_at=connected_at,
            )
