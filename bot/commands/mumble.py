from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from socket import timeout as socket_timeout

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bot.active_chats import track_active_chat
from bot.commands.message_utils import reply_in_chunks
from bot.commands.mumble_logic import format_mumble_status_report, is_mumble_status_command
from bot.config import BotConfig

LOGGER = logging.getLogger(__name__)
COMMAND_USAGE = "!mumble"


def _collect_mumble_snapshot(config: BotConfig) -> dict[str, object]:
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

    target_channels = set(channel.casefold() for channel in config.mumble_target_channels)
    if len(target_channels) < 2:
        raise RuntimeError(
            "MUMBLE_TARGET_CHANNELS vaatii vähintään kaksi kanavaa (ominaisuusvaatimus)."
        )

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
            if state in {PYMUMBLE_CONN_STATE_FAILED, PYMUMBLE_CONN_STATE_NOT_CONNECTED}:
                raise RuntimeError("Yhteys Mumble-palvelimeen epäonnistui.")
            if time.monotonic() >= deadline:
                raise socket_timeout("Mumble-yhteys aikakatkesi.")
            time.sleep(0.1)

        time.sleep(max(config.mumble_status_wait_seconds, 0))

        channels = [
            channel
            for channel in getattr(mumble, "channels", {}).values()
            if isinstance(channel, dict)
        ]
        users = [user for user in getattr(mumble, "users", {}).values() if isinstance(user, dict)]
        own_session = getattr(getattr(mumble, "users", None), "myself_session", None)

        output_channels: list[dict[str, object]] = []
        for channel in sorted(channels, key=lambda item: str(item.get("name", "")).casefold()):
            channel_id = channel.get("channel_id")
            channel_name = str(channel.get("name", "Tuntematon kanava"))
            if channel_name.casefold() not in target_channels:
                continue

            channel_users: list[dict[str, object]] = []
            for user in users:
                if user.get("channel_id") != channel_id:
                    continue
                if own_session is not None and user.get("session") == own_session:
                    continue

                reserved_keys = {
                    "name",
                    "session",
                    "user_id",
                    "channel_id",
                    "mute",
                    "deaf",
                    "self_mute",
                    "self_deaf",
                    "suppress",
                    "recording",
                    "onlinesecs",
                    "idlesecs",
                }
                extras = {
                    key: value
                    for key, value in user.items()
                    if key not in reserved_keys and isinstance(value, (str, int, float, bool))
                }

                channel_users.append(
                    {
                        "name": user.get("name"),
                        "session": user.get("session"),
                        "user_id": user.get("user_id"),
                        "online_seconds": user.get("onlinesecs"),
                        "idle_seconds": user.get("idlesecs"),
                        "mute": user.get("mute"),
                        "deaf": user.get("deaf"),
                        "self_mute": user.get("self_mute"),
                        "self_deaf": user.get("self_deaf"),
                        "suppress": user.get("suppress"),
                        "recording": user.get("recording"),
                        "extras": extras,
                    }
                )

            output_channels.append({"name": channel_name, "users": channel_users})

        return {
            "server_address": f"{config.mumble_host}:{config.mumble_port}",
            "channels": output_channels,
            "expected_channel_count": len(target_channels),
        }
    finally:
        try:
            mumble.stop()
        except Exception:
            pass

        try:
            mumble.join(timeout=1)
        except Exception:
            pass


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

        try:
            snapshot = await asyncio.to_thread(_collect_mumble_snapshot, config)
            reply = format_mumble_status_report(
                server_address=str(snapshot["server_address"]),
                channels=list(snapshot["channels"]),
                expected_channel_count=int(snapshot["expected_channel_count"]),
            )
        except (RuntimeError, socket_timeout, TimeoutError, OSError):
            LOGGER.exception("Mumble status check failed")
            reply = (
                "Mumble-tilan haku epäonnistui. Varmista MUMBLE_* asetukset "
                "(host, käyttäjä, salasana ja kanavat)."
            )

        await reply_in_chunks(update, reply, config.max_reply_length)

    return handle_mumble


def register(application: Application, config: BotConfig) -> None:
    application.add_handler(
        MessageHandler(filters.Regex(r"(?i)^\s*!?mumble\s*$"), _build_handler(config))
    )
