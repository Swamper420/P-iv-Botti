# P-iv-Botti

Async and modular Telegram bot skeleton.

## Features

- Async bot runtime using `python-telegram-bot`
- Extendable command/reply modules under `bot/commands/`
- Environment-driven configuration via `.env` (template in `example.env`)
- Persistent data directory under `storage/`
- CS2 RSS update watcher that posts new updates to active bot chats

## Current behavior

The bot polls Steam's CS2 RSS feed in the background and forwards new updates to chats where the bot
has been active. `STEAM_CS2_RSS_URL`, `STEAM_RSS_POLL_INTERVAL_SECONDS`, and
`STEAM_RSS_REQUEST_TIMEOUT_SECONDS` are configurable through `.env`.

The bot supports `!naama` on a photo caption or as a reply to a photo. It segments the person from the
source image, applies a random `background*` image, and overlays random `hat*`, `suit*`, `gloves*`,
`cigar*`, and `sun*` PNG/JPG assets from `storage/naama/`.

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
