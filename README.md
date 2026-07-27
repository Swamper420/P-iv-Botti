# P-iv-Botti

Async and modular Telegram bot skeleton.

## Features

- Async bot runtime using `python-telegram-bot`
- Extendable command/reply modules under `bot/commands/`
- Environment-driven configuration via `.env` (template in `example.env`)
- Persistent data directory under `storage/`
- CS2 RSS update watcher that posts new updates to active bot chats

## Current behavior

- **`!help`**: Auto-discovers and lists available bot commands.
- **`!hoi` / `!hoijaa`**: Manages and pings mention lists for chat groups (`!hoi <lista>`, `!hoi @käyttäjä <lista>`, `!hoijaa @käyttäjä <lista>`).
- **`!naama` / `!naamatarra`**: Segments the person from a photo (or replied photo), applies a random background image, and overlays random accessories from `storage/naama/`.
- **`!sääkuva <kaupunki>`**: Fetches current weather information from OpenWeather and weather camera images from Digitraffic for the requested location.
- **`!twitch`**: Shows current status and live links for configured Twitch channels.
- **CS2 RSS Notifier**: Background task polling Steam's CS2 RSS feed and forwarding new updates to active bot chats. `STEAM_CS2_RSS_URL`, `STEAM_RSS_POLL_INTERVAL_SECONDS`, and `STEAM_RSS_REQUEST_TIMEOUT_SECONDS` are configurable through `.env`.
- **Twitch EventSub Notifier**: Real-time WebSocket background task sending instant notifications with stream title, category, preview image, and direct link whenever tracked channels go live on Twitch. Configured via `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`, and `TWITCH_CHANNELS` in `.env`.


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
4. Set `TELEGRAM_BOT_TOKEN` in `.env` (and optionally `OPENWEATHER_API_KEY` for weather reports).
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
