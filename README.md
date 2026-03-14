# P-iv-Botti

Async and modular Telegram bot skeleton.

## Features

- Async bot runtime using `python-telegram-bot`
- Extendable command/reply modules under `bot/commands/`
- Environment-driven configuration via `.env` (template in `example.env`)
- Persistent data directory under `storage/`
- CS2 RSS update watcher that posts new updates to active bot chats

## Current behavior

The bot replies to message text `Päivää` with an AI-generated, deliberately over-the-top cringe uwu greeting.
The latest four generated `Päivää` replies are persisted under `storage/` and used to avoid repeating recent
responses.

The bot also handles `aih: <prompt>` messages by sending the prompt to a local AI backend and replies
with the response. `AI_BACKEND_URL`, `AI_MAX_TOKENS`, `AI_BACKEND_TIMEOUT_SECONDS`, and
`MAX_REPLY_LENGTH` are configured through `.env` (`example.env` has defaults).

The bot polls Steam's CS2 RSS feed in the background and forwards new updates to chats where the bot
has been active. `STEAM_CS2_RSS_URL`, `STEAM_RSS_POLL_INTERVAL_SECONDS`, and
`STEAM_RSS_REQUEST_TIMEOUT_SECONDS` are configurable through `.env`.

The bot supports `mumble` / `!mumble` for local Mumble server status checks on all server channels.
Set `MUMBLE_HOST`, `MUMBLE_PORT`, `MUMBLE_USERNAME`, and `MUMBLE_PASSWORD` in `.env`.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create your runtime config:
   ```bash
   cp example.env .env
   ```
4. Set `TELEGRAM_BOT_TOKEN` in `.env` (and optionally adjust AI backend settings).
5. Run:
   ```bash
   python -m bot.main
   ```

## systemd (Linux)

Example unit:

```ini
[Unit]
Description=P-iv-Botti Telegram Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/P-iv-Botti
EnvironmentFile=/opt/P-iv-Botti/.env
ExecStart=/opt/P-iv-Botti/.venv/bin/python -m bot.main
Restart=always
RestartSec=3
User=botuser
Group=botuser

[Install]
WantedBy=multi-user.target
```

Adjust paths and user/group for your Linux host.
