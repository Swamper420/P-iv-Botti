# P-iv-Botti

Async and modular Telegram bot skeleton.

## Features

- Async bot runtime using `python-telegram-bot`
- Extendable command/reply modules under `bot/commands/`
- Environment-driven configuration via `.env` (template in `example.env`)
- Persistent data directory under `storage/`

## Current behavior

The bot replies to message text `Päivää` with:

`Päivää *tips fedora*`

The bot also handles `aih: <prompt>` messages by sending the prompt to a local AI backend at
`http://127.0.0.1:8080/query` with `max_tokens=650` and replies with the response. Long replies are
split into multiple messages (5000 chars per message).

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
4. Set `TELEGRAM_BOT_TOKEN` in `.env`.
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
