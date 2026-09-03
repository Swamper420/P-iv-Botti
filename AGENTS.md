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
  - `muistuta.py` + `muistuta_logic.py`
  - `naama.py` + `naama_logic.py`
  - `paranna.py` + `paranna_logic.py`
  - `mine.py` + `mine_logic.py`
  - `mumble.py` + `mumble_logic.py`
  - `stt.py` + `stt_logic.py`
  - `telkkari.py` + `telkkari_logic.py`
  - `tiivista.py` + `tiivista_logic.py`
  - `tts.py` + `tts_logic.py`
  - `twitch.py` + `twitch_logic.py`
  - `weather.py` + `weather_logic.py`
- Shared command utilities in `bot/commands/`:
  - `common.py`: `@command_handler` decorator for chat tracking and error handling
  - `message_utils.py`: `split_message(...)`, `reply_in_chunks(...)`, `reply_with_image(...)`, and `reply_with_card(...)` (with automatic fallback to text on upload error)
- Picture rendering system in `bot/rendering/`:
  - `engine.py`: PIL-based card renderer (`render_card`, `render_text_card`, `render_table_card`) with dynamic height sizing and system font resolution
  - `models.py`: `Card`, `Badge`, `BadgeColor`, `Theme`, `DARK_THEME`, and element models (`TextElement`, `KeyValuesElement`, `TableElement`, `DividerElement`, `ProgressBarElement`, `CodeBlockElement`)
- Command modules are auto-discovered in `bot/commands/__init__.py` (files ending in `_logic.py` are excluded from registration).
- Background tasks are modularized in `bot/tasks/`:
  - `cs2_rss.py`: `Cs2RssNotifier` for background RSS polling
  - `mumble.py`: `MumbleTask` for Mumble server presence and stats tracking
  - `reminders.py`: `ReminderNotifier` for scheduled reminder delivery
  - `twitch.py`: `TwitchEventSubNotifier` for Twitch EventSub WebSockets live notifications
  - Auto-discovered and registered via `bot/tasks/__init__.py` (`register_tasks`)
- Type protocols live in `bot/protocols.py` (`CommandModule`, `TaskModule`).
- JSON storage abstraction lives in `bot/storage.py` (`load_json_data`, `save_json_data`).
- Message handlers use `filters.Regex` (avoid broad text filters that can block later handlers).
- Runtime configuration is provided by `BotConfig` in `bot/config.py` with domain sub-configs (`WeatherConfig`, `Cs2RssConfig`, `NaamaConfig`, `OllamaConfig`, `TwitchConfig`, `ParannaConfig`, `TtsConfig`, `TelkkariConfig`, `SttConfig`, `ReminderConfig`, `TiivistaConfig`, `CraftyConfig`, `MumbleConfig`, `RenderingConfig`).

- Active chat persistence is handled in `bot/active_chats.py` under `storage/active_chat_ids.json`.
- Targeted tests: `python -m unittest tests.test_weather_logic` (replace module with the area you changed).
- Full regression: `./venv/bin/python -m unittest`.

## Quick command-extension checklist

1. Add `<name>_logic.py` for pure command logic.
2. Add `<name>.py` with one `register(application, config, ...)` function and `filters.Regex(...)` handler.
3. Add `COMMAND_USAGE` in the command module so `!help` can auto-discover it.
4. Reuse `bot/commands/common.py` and `bot/commands/message_utils.py` for handlers and long replies.
5. Add focused tests in `tests/test_<name>_logic.py` (and handler registration tests only if needed).

## Rendering System (Picture Responses)

Commands can reply with high-contrast, dark-mode picture cards instead of plain text.

### Visual Design Constraint
**NO ROUNDED CORNERS ANYWHERE**: All cards, boxes, badges, progress bars, code blocks, and borders MUST have sharp, 90-degree rectangular corners. The rendering engine strictly adheres to this.

### How to Port or Add Picture Responses to a Command

There are two straightforward approaches:

#### 1. Instant Port (Existing Text Command -> Picture Card)
Convert any existing text reply into a clean picture card with `render_text_card(...)` and `reply_with_image(...)`:

```python
from bot.commands.message_utils import reply_with_image
from bot.rendering import render_text_card

# In command handler:
reply_text = handle_my_command(args)

# Render image asynchronously to avoid blocking the event loop:
image_bytes = await asyncio.to_thread(
    render_text_card,
    title="Otsikko",
    text=reply_text,
    subtitle="Valinnainen alaotsikko",
    badge="STATUS",           # Optional status badge
    badge_color="green",      # "green", "red", "yellow", "blue", "purple", "gray"
    footer="Päivitetty nyt",  # Optional footer note
)

# Send image with automatic fallback to text if photo upload fails:
await reply_with_image(
    update,
    image=image_bytes,
    fallback_text=reply_text,
)
```

#### 2. Structured Card Builder (Dashboards, Tables, Server Status)
Construct rich cards with key-value pairs, progress bars, tables, and badges:

```python
from bot.commands.message_utils import reply_with_card
from bot.rendering import Card, Badge, BadgeColor

card = (
    Card(title="Minecraft Palvelimet", subtitle="crafty.lan", footer="P-iv-Botti")
    .set_badge("ONLINE", BadgeColor.GREEN)
    .add_text("Palvelin on käynnissä ja vastaa kyselyihin.")
    .add_key_value("Versio", "Paper 1.21.1")
    .add_key_value("Pelaajat", "5 / 20")
    .add_progress_bar("Muisti", value=4.2, max_value=8.0, unit="GB", color="#38bdf8")
    .add_divider()
    .add_table(
        headers=["Pelaaja", "Aika"],
        rows=[["Matti", "1h 12m"], ["Teppo", "45m"]],
        alignments=["left", "right"],
    )
    .add_code_block("connect mc.example.fi:25565")
)

# reply_with_card renders card in a worker thread and replies with fallback support:
await reply_with_card(
    update,
    card=card,
    fallback_text="Minecraft Palvelimet: ONLINE, 5/20 pelaajaa.",
)
```

### Best Practices for Picture Responses
- **Always provide `fallback_text`**: If Telegram fails to deliver the photo (e.g. timeout or network glitch), `reply_with_image` and `reply_with_card` will automatically fall back to sending `fallback_text` via `reply_in_chunks`.
- **Non-blocking rendering**: `reply_with_card` automatically renders via `asyncio.to_thread`. When calling `render_card` or `render_text_card` directly in async handlers, wrap with `await asyncio.to_thread(render_card, ...)`.
- **Keep pure logic in `_logic.py`**: Assemble data dicts, tuples, or `Card` objects in `<name>_logic.py` so they are fully unit-testable without Telegram dependencies.
