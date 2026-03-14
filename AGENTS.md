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
