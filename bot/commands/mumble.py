from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from socket import timeout as socket_timeout
from threading import Lock

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.message_utils import reply_in_chunks
from bot.commands.mumble_logic import (
    extract_mumble_tele_message,
    format_mumble_channel_notice,
    format_mumble_status_report,
    is_mumble_status_command,
    resolve_online_seconds,
    update_mumble_connection_tracker,
)
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)
COMMAND_USAGE = "!mumble"
_MONITORED_SNAPSHOT_LOCK = Lock()
_MONITORED_SNAPSHOT: dict[str, object] | None = None
_USER_CONNECTED_AT_LOCK = Lock()
_USER_CONNECTED_AT: dict[str, float] = {}
_PERSISTENT_MUMBLE_LOCK = Lock()
_PERSISTENT_MUMBLE: object | None = None
_PERSISTENT_TELE_MESSAGES_LOCK = Lock()
_PERSISTENT_TELE_MESSAGES: list[tuple[str, str]] = []


def _resolve_requester_name(update: Update) -> str:
    user = update.effective_user
    if user is None:
        return "tuntematon käyttäjä"

    if user.username:
        return f"@{user.username}"

    full_name = (user.full_name or "").strip()
    if full_name:
        return full_name

    return str(user.id)


def _store_monitored_snapshot(snapshot: dict[str, object]) -> None:
    global _MONITORED_SNAPSHOT
    with _MONITORED_SNAPSHOT_LOCK:
        _MONITORED_SNAPSHOT = snapshot


def _get_monitored_snapshot() -> dict[str, object] | None:
    with _MONITORED_SNAPSHOT_LOCK:
        if _MONITORED_SNAPSHOT is None:
            return None
        return dict(_MONITORED_SNAPSHOT)


def _queue_persistent_tele_message(sender_name: str, body: str) -> None:
    with _PERSISTENT_TELE_MESSAGES_LOCK:
        _PERSISTENT_TELE_MESSAGES.append((sender_name, body))


def _drain_persistent_tele_messages() -> list[tuple[str, str]]:
    with _PERSISTENT_TELE_MESSAGES_LOCK:
        messages = list(_PERSISTENT_TELE_MESSAGES)
        _PERSISTENT_TELE_MESSAGES.clear()
        return messages


def _build_persistent_tele_message_callback(mumble: object) -> Callable[[object], None]:
    def _handle_persistent_tele_message(message: object) -> None:
        message_text = extract_mumble_tele_message(str(getattr(message, "message", "")))
        if message_text is None:
            return

        actor_session = getattr(message, "actor", None)
        actor_name = "tuntematon"
        users = getattr(mumble, "users", {})
        if isinstance(actor_session, int) and hasattr(users, "get"):
            actor = users.get(actor_session)
            if actor is not None:
                actor_name = str(actor.get("name", actor_name))

        _queue_persistent_tele_message(actor_name, message_text)

    return _handle_persistent_tele_message


def _reset_persistent_mumble_connection() -> None:
    global _PERSISTENT_MUMBLE
    with _PERSISTENT_MUMBLE_LOCK:
        mumble = _PERSISTENT_MUMBLE
        _PERSISTENT_MUMBLE = None

    if mumble is None:
        return

    try:
        mumble.stop()
    except Exception:
        pass

    try:
        mumble.join(timeout=1)
    except Exception:
        pass


def _build_snapshot_from_connected_mumble(
    *,
    config: BotConfig,
    mumble: object,
    requested_by: str,
    notify_channels: bool,
    tele_messages: list[tuple[str, str]],
) -> dict[str, object]:
    all_channel_objects = list(getattr(mumble, "channels", {}).values())
    channels = [channel for channel in all_channel_objects if isinstance(channel, dict)]
    if notify_channels:
        channel_notice = format_mumble_channel_notice(requested_by)
        notifiable_channels = [
            channel
            for channel in all_channel_objects
            if callable(getattr(channel, "send_text_message", None))
        ]
        for channel in notifiable_channels:
            try:
                channel.send_text_message(channel_notice)
            except Exception:
                channel_name = "Tuntematon kanava"
                if isinstance(channel, dict):
                    channel_name = str(channel.get("name", channel_name))
                else:
                    channel_name = str(getattr(channel, "name", channel_name))
                LOGGER.warning(
                    "Failed to send mumble command notice to channel %r",
                    channel_name,
                    exc_info=True,
                )

    users = [user for user in getattr(mumble, "users", {}).values() if isinstance(user, dict)]
    own_session = getattr(getattr(mumble, "users", None), "myself_session", None)
    now_monotonic = time.monotonic()
    with _USER_CONNECTED_AT_LOCK:
        update_mumble_connection_tracker(
            users=users,
            own_session=own_session,
            connected_since_by_key=_USER_CONNECTED_AT,
            now_monotonic=now_monotonic,
        )

        output_channels: list[dict[str, object]] = []
        for channel in sorted(channels, key=lambda item: str(item.get("name", "")).casefold()):
            channel_id = channel.get("channel_id")
            channel_name = str(channel.get("name", "Tuntematon kanava"))

            channel_users: list[dict[str, object]] = []
            for user in users:
                if user.get("channel_id") != channel_id:
                    continue
                if own_session is not None and user.get("session") == own_session:
                    continue

                channel_users.append(
                    {
                        "name": user.get("name"),
                        "online_seconds": resolve_online_seconds(
                            user=user,
                            connected_since_by_key=_USER_CONNECTED_AT,
                            now_monotonic=now_monotonic,
                        ),
                        "muted": (
                            bool(user.get("mute"))
                            or bool(user.get("self_mute"))
                            or bool(user.get("suppress"))
                        ),
                        "deafened": bool(user.get("deaf")) or bool(user.get("self_deaf")),
                    }
                )

            output_channels.append({"name": channel_name, "users": channel_users})

    return {
        "server_address": f"{config.mumble_host}:{config.mumble_port}",
        "channels": output_channels,
        "tele_messages": tele_messages,
    }


def _collect_mumble_snapshot(
    config: BotConfig, requested_by: str, *, notify_channels: bool = True
) -> dict[str, object]:
    try:
        import pymumble_py3
        from pymumble_py3.constants import (
            PYMUMBLE_CONN_STATE_CONNECTED,
            PYMUMBLE_CONN_STATE_FAILED,
            PYMUMBLE_CONN_STATE_NOT_CONNECTED,
        )
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        raise RuntimeError("pymumble-kirjasto puuttuu tai sen riippuvuudet eivät ole saatavilla.") from exc

    if not config.mumble_password:
        raise RuntimeError("MUMBLE_PASSWORD puuttuu.")

    retries = max(config.mumble_connect_retries, 1)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        mumble = pymumble_py3.Mumble(
            config.mumble_host,
            config.mumble_username,
            port=config.mumble_port,
            password=config.mumble_password,
            reconnect=False,
        )
        mumble.set_receive_sound(False)
        mumble.start()

        try:
            deadline = time.monotonic() + max(config.mumble_connect_timeout_seconds, 1)
            while True:
                state = getattr(mumble, "connected", None)
                if state == PYMUMBLE_CONN_STATE_CONNECTED:
                    break
                if state == PYMUMBLE_CONN_STATE_FAILED:
                    raise RuntimeError("Yhteys Mumble-palvelimeen epäonnistui.")
                if state not in {None, PYMUMBLE_CONN_STATE_NOT_CONNECTED}:
                    raise RuntimeError(f"Mumble-yhteyden tila on odottamaton: {state!r}.")
                if time.monotonic() >= deadline:
                    raise socket_timeout("Mumble-yhteys aikakatkesi.")
                time.sleep(0.1)

            pending_tele_messages: list[tuple[str, str]] = []
            if config.mumble_tele_chat_id is not None:
                callback_manager = getattr(mumble, "callbacks", None)
                add_callback = getattr(callback_manager, "add_callback", None)
                if callable(add_callback):
                    try:
                        from pymumble_py3.constants import PYMUMBLE_CLBK_TEXTMESSAGERECEIVED

                        def _handle_text_message(message: object) -> None:
                            message_text = extract_mumble_tele_message(
                                str(getattr(message, "message", ""))
                            )
                            if message_text is None:
                                return

                            actor_session = getattr(message, "actor", None)
                            actor_name = "tuntematon"
                            users = getattr(mumble, "users", {})
                            if isinstance(actor_session, int) and hasattr(users, "get"):
                                actor = users.get(actor_session)
                                if actor is not None:
                                    actor_name = str(actor.get("name", actor_name))

                            pending_tele_messages.append((actor_name, message_text))

                        add_callback(PYMUMBLE_CLBK_TEXTMESSAGERECEIVED, _handle_text_message)
                    except Exception:
                        LOGGER.warning("Failed to register mumble text message callback", exc_info=True)

            time.sleep(max(config.mumble_status_wait_seconds, 0))
            return _build_snapshot_from_connected_mumble(
                config=config,
                mumble=mumble,
                requested_by=requested_by,
                notify_channels=notify_channels,
                tele_messages=pending_tele_messages,
            )
        except (RuntimeError, socket_timeout, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            LOGGER.warning(
                "Mumble snapshot attempt %s/%s failed, retrying",
                attempt,
                retries,
                exc_info=True,
            )
        finally:
            try:
                mumble.stop()
            except Exception:
                pass

            try:
                mumble.join(timeout=1)
            except Exception:
                pass

    if last_error is not None:
        raise last_error
    raise RuntimeError("Mumble snapshot collection failed unexpectedly.")


def _collect_monitored_snapshot(config: BotConfig) -> dict[str, object]:
    global _PERSISTENT_MUMBLE

    try:
        import pymumble_py3
        from pymumble_py3.constants import (
            PYMUMBLE_CLBK_TEXTMESSAGERECEIVED,
            PYMUMBLE_CONN_STATE_CONNECTED,
            PYMUMBLE_CONN_STATE_FAILED,
            PYMUMBLE_CONN_STATE_NOT_CONNECTED,
        )
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        raise RuntimeError("pymumble-kirjasto puuttuu tai sen riippuvuudet eivät ole saatavilla.") from exc

    if not config.mumble_password:
        raise RuntimeError("MUMBLE_PASSWORD puuttuu.")

    retries = max(config.mumble_connect_retries, 1)
    last_error: Exception | None = None
    stale_existing: object | None = None

    with _PERSISTENT_MUMBLE_LOCK:
        existing = _PERSISTENT_MUMBLE
        if existing is not None and getattr(existing, "connected", None) == PYMUMBLE_CONN_STATE_CONNECTED:
            try:
                return _build_snapshot_from_connected_mumble(
                    config=config,
                    mumble=existing,
                    requested_by="taustaseuranta",
                    notify_channels=False,
                    tele_messages=_drain_persistent_tele_messages(),
                )
            except (RuntimeError, socket_timeout, TimeoutError, OSError):
                LOGGER.exception("Existing persistent mumble connection became unusable")
                _PERSISTENT_MUMBLE = None
                stale_existing = existing

    if stale_existing is not None:
        try:
            stale_existing.stop()
        except Exception:
            pass
        try:
            stale_existing.join(timeout=1)
        except Exception:
            pass

    for attempt in range(1, retries + 1):
        mumble = pymumble_py3.Mumble(
            config.mumble_host,
            config.mumble_username,
            port=config.mumble_port,
            password=config.mumble_password,
            reconnect=True,
        )
        mumble.set_receive_sound(False)
        mumble.start()
        try:
            deadline = time.monotonic() + max(config.mumble_connect_timeout_seconds, 1)
            while True:
                state = getattr(mumble, "connected", None)
                if state == PYMUMBLE_CONN_STATE_CONNECTED:
                    break
                if state == PYMUMBLE_CONN_STATE_FAILED:
                    raise RuntimeError("Yhteys Mumble-palvelimeen epäonnistui.")
                if state not in {None, PYMUMBLE_CONN_STATE_NOT_CONNECTED}:
                    raise RuntimeError(f"Mumble-yhteyden tila on odottamaton: {state!r}.")
                if time.monotonic() >= deadline:
                    raise socket_timeout("Mumble-yhteys aikakatkesi.")
                time.sleep(0.1)

            tele_chat_id = config.mumble_tele_chat_id
            if tele_chat_id is not None:
                callback_manager = getattr(mumble, "callbacks", None)
                add_callback = getattr(callback_manager, "add_callback", None)
                if callable(add_callback):
                    add_callback(
                        PYMUMBLE_CLBK_TEXTMESSAGERECEIVED,
                        _build_persistent_tele_message_callback(mumble),
                    )

            with _PERSISTENT_MUMBLE_LOCK:
                previous = _PERSISTENT_MUMBLE
                _PERSISTENT_MUMBLE = mumble

            if previous is not None and previous is not mumble:
                try:
                    previous.stop()
                except Exception:
                    pass
                try:
                    previous.join(timeout=1)
                except Exception:
                    pass

            return _build_snapshot_from_connected_mumble(
                config=config,
                mumble=mumble,
                requested_by="taustaseuranta",
                notify_channels=False,
                tele_messages=_drain_persistent_tele_messages(),
            )
        except (RuntimeError, socket_timeout, TimeoutError, OSError) as exc:
            last_error = exc
            try:
                mumble.stop()
            except Exception:
                pass
            try:
                mumble.join(timeout=1)
            except Exception:
                pass
            if attempt >= retries:
                _reset_persistent_mumble_connection()
                raise
            LOGGER.warning(
                "Persistent mumble monitor connect attempt %s/%s failed, retrying",
                attempt,
                retries,
                exc_info=True,
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("Persistent mumble snapshot collection failed unexpectedly.")


def _build_handler(
    config: BotConfig,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def handle_mumble(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context

        message = update.effective_message
        if message is None:
            return

        track_active_chat(update, config.storage_dir)

        if not is_mumble_status_command(message.text):
            return

        requested_by = _resolve_requester_name(update)
        try:
            snapshot = await asyncio.to_thread(_collect_mumble_snapshot, config, requested_by)
            _store_monitored_snapshot(snapshot)
            reply = format_mumble_status_report(
                server_address=str(snapshot["server_address"]),
                channels=list(snapshot["channels"]),
            )
        except (RuntimeError, socket_timeout, TimeoutError, OSError):
            LOGGER.exception("Mumble status check failed")
            cached_snapshot = _get_monitored_snapshot()
            if cached_snapshot is None:
                reply = (
                    "Mumble-tilan haku epäonnistui. Varmista MUMBLE_* asetukset "
                    "(host, käyttäjä ja salasana)."
                )
            else:
                reply = (
                    "⚠️ Mumble-palvelimeen ei juuri nyt saatu yhteyttä. "
                    "Näytetään viimeisin seurannan tallentama tila.\n\n"
                ) + format_mumble_status_report(
                    server_address=str(cached_snapshot["server_address"]),
                    channels=list(cached_snapshot["channels"]),
                )

        await reply_in_chunks(update, reply, config.max_reply_length)

    return handle_mumble


def register(application: Application, config: BotConfig) -> None:
    job_queue = getattr(application, "job_queue", None)
    if job_queue is not None and callable(getattr(job_queue, "run_repeating", None)):
        interval_seconds = config.mumble_monitor_interval_seconds

        async def _refresh_monitored_snapshot(_context: object) -> None:
            try:
                snapshot = await asyncio.to_thread(_collect_monitored_snapshot, config)
                _store_monitored_snapshot(snapshot)
                tele_chat_id = config.mumble_tele_chat_id
                tele_messages = snapshot.get("tele_messages", [])
                if isinstance(tele_chat_id, int) and isinstance(tele_messages, list):
                    for tele_message in tele_messages:
                        if not isinstance(tele_message, tuple) or len(tele_message) != 2:
                            continue
                        sender_name, body = tele_message
                        if not isinstance(sender_name, str) or not isinstance(body, str):
                            continue
                        try:
                            await application.bot.send_message(
                                chat_id=tele_chat_id,
                                text=f"🎧 {sender_name}: {body}",
                            )
                        except Exception:
                            LOGGER.exception("Failed to forward mumble !tele message to Telegram")
            except (RuntimeError, socket_timeout, TimeoutError, OSError):
                LOGGER.exception("Background mumble monitoring check failed")

        job_queue.run_repeating(
            _refresh_monitored_snapshot,
            interval=interval_seconds,
            first=0,
            name="mumble-monitor-snapshot",
        )

    application.add_handler(
        MessageHandler(filters.Regex(r"(?i)^\s*!?mumble\s*$"), _build_handler(config))
    )
