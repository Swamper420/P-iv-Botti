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
  - `paivaa.py` + `paivaa_logic.py`
  - `weather.py` + `weather_logic.py`
- Shared command helpers live in `bot/commands/message_utils.py`:
  - `split_message(...)` for chunking long replies
  - `reply_in_chunks(...)` for sending chunked Telegram responses
- Command modules are auto-discovered in `bot/commands/__init__.py` (files ending in `_logic.py` are excluded from registration).
- Message handlers use `filters.Regex` (avoid broad text filters that can block later handlers).
- Runtime configuration is provided by `BotConfig` in `bot/config.py` and environment defaults in `example.env`.
- Background RSS notifications are managed by `Cs2RssNotifier` in `bot/cs2_rss.py` and started/stopped from `bot/main.py`.
- Active chat persistence is handled in `bot/active_chats.py` under `storage/active_chat_ids.json`.
- Targeted tests: `python -m unittest tests.test_weather_logic` (replace module with the area you changed).
- Full regression: `python -m unittest`.

## Quick command-extension checklist

1. Add `<name>_logic.py` for pure command logic.
2. Add `<name>.py` with one `register(application, config, ...)` function and `filters.Regex(...)` handler.
3. Add `COMMAND_USAGE` in the command module so `!help` can auto-discover it.
4. Reuse `bot/commands/message_utils.py` for long replies.
5. Add focused tests in `tests/test_<name>_logic.py` (and handler registration tests only if needed).
