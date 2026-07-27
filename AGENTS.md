# Agent Quality Requirements

When extending this project:

1. Keep the bot async (`async def` handlers, non-blocking logic).
2. Keep functionality modular:
   - Add new bot features as separate modules under `bot/commands/`.
   - Keep one command per command module and one command-logic module (do not mix command logic in shared command files).
   - Register new modules through `bot/commands/__init__.py`.
3. Keep configuration environment-based:
   - Add new config keys to `example.env`.
   - Load runtime config only via `bot/config.py`.
   - Do not hardcode runtime tuning values (URLs, timeouts, token limits, message limits) in command modules.
4. Keep persistent data in `storage/` only.
5. Keep failures explicit:
   - Validate configuration at startup.
   - Fail fast with clear logs for unrecoverable setup errors.
6. Add focused tests for new command/reply logic.
7. Preserve Linux + systemd compatibility (`python -m bot.main` as entrypoint).

## Current codebase map (keep this section updated)

- Commands are implemented as module pairs in `bot/commands/`:
  - `aih.py` + `aih_logic.py`
  - `help.py` + `help_logic.py`
  - `hoi.py` + `hoi_logic.py`
  - `naama.py` + `naama_logic.py`
  - `twitch.py` + `twitch_logic.py`
  - `weather.py` + `weather_logic.py`
- Shared command utilities in `bot/commands/`:
  - `common.py`: `@command_handler` decorator for chat tracking and error handling
  - `message_utils.py`: `split_message(...)` and `reply_in_chunks(...)` for long responses
- Command modules are auto-discovered in `bot/commands/__init__.py` (files ending in `_logic.py` are excluded from registration).
- Background tasks are modularized in `bot/tasks/`:
  - `cs2_rss.py`: `Cs2RssNotifier` for background RSS polling
  - `twitch.py`: `TwitchEventSubNotifier` for Twitch EventSub WebSockets live notifications
  - Auto-discovered and registered via `bot/tasks/__init__.py` (`register_tasks`)
- Type protocols live in `bot/protocols.py` (`CommandModule`, `TaskModule`).
- JSON storage abstraction lives in `bot/storage.py` (`load_json_data`, `save_json_data`).
- Message handlers use `filters.Regex` (avoid broad text filters that can block later handlers).
- Runtime configuration is provided by `BotConfig` in `bot/config.py` with domain sub-configs (`WeatherConfig`, `Cs2RssConfig`, `NaamaConfig`, `OllamaConfig`, `TwitchConfig`).


- Active chat persistence is handled in `bot/active_chats.py` under `storage/active_chat_ids.json`.
- Targeted tests: `python -m unittest tests.test_weather_logic` (replace module with the area you changed).
- Full regression: `./venv/bin/python -m unittest`.

## Quick command-extension checklist

1. Add `<name>_logic.py` for pure command logic.
2. Add `<name>.py` with one `register(application, config, ...)` function and `filters.Regex(...)` handler.
3. Add `COMMAND_USAGE` in the command module so `!help` can auto-discover it.
4. Reuse `bot/commands/common.py` and `bot/commands/message_utils.py` for handlers and long replies.
5. Add focused tests in `tests/test_<name>_logic.py` (and handler registration tests only if needed).
