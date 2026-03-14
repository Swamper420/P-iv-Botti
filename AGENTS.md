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
- Command modules are auto-discovered in `bot/commands/__init__.py` (files ending in `_logic.py` are excluded from registration).
- Message handlers use `filters.Regex` (avoid broad text filters that can block later handlers).
- Runtime configuration is provided by `BotConfig` in `bot/config.py` and environment defaults in `example.env`.
- Targeted tests: `python -m unittest tests.test_weather_logic` (replace module with the area you changed).
- Full regression: `python -m unittest`.
