Implementation Plan: Cohesive, Non-Repeating, Mobile-First UI System
This plan is designed for step-by-step execution by any coding agent. Each phase is self-contained, specifies exact file paths, class/function signatures, strict visual rules, and unit test commands, and adheres to all rules in AGENTS.md (async handlers, <name>.py + <name>_logic.py separation, config via bot/config.py, persistent state in storage/, and strictly sharp 90° rectangular corners — never rounded_rectangle).

1. Diagnosis of Current UI Issues (What Agents Must Fix)
Data Repetition Inside Cards ("Say It Once" Violation)
Weather (bot/commands/weather_logic.py):
Temperature & condition appear 3 times: Header Badge (☀️ 18.5°C Selkeää), Hero Text (☀️ Selkeää — 18.5°C (tuntuu 17.2°C)), and Key-Value grid (Lämpötila: 18.5°C, Tuntuu: 17.2°C).
Humidity and cloudiness appear twice: once as Key-Value pairs (Kosteus: 55%, Pilvisyys: 20%) and immediately again as Progress Bars.
Camera station, angle, and observation time appear 3 times: Card Subtitle, Image Caption, and footer Key-Value (Havainto).
Progress Bars (bot/rendering/engine.py):
When unit == "%" and max_value == 100.0, ProgressBarElement renders 55% / 100% (55.0%) — repeating the percentage twice.
Minecraft & Mumble (bot/commands/mine_logic.py, bot/commands/mumble_logic.py):
Online/offline status is shown in the top-right Badge (ONLINE / PAIKALLA / MYKISTETTY) and repeated in the first Key-Value row (Tila: Päällä / Tila: Paikalla / Tila: Mykistetty).
Server name is repeated in Title/Subtitle and again in Footer.
Copy-Pasted Card Helpers & Inconsistent Footers/Badges
build_mine_info_card, build_telkkari_info_card, build_weather_error_card, and inline Mumble error cards duplicate the same boilerplate with inconsistent badge colors (RED vs YELLOW for argument errors), emoji prefixes, and footer strings.
Text commands (!help, !twitch, !muistuta, !hoi) mix raw text, Markdown (**bold**), and HTML (<b>bold</b>) instead of sharing a unified visual style.
Poor Mobile Readability (800px Desktop Canvas Trap)
Cards render at card_width = 800 with 11px–14px fonts and up to 5 table columns. Inside a mobile Telegram chat bubble (~340px wide), the image scales down by >50%, shrinking body text to ~6px and forcing users to tap and pinch-zoom every reply.
Discoverability vs. Power Features
!help is an ungrouped alphabetical list of raw COMMAND_USAGE strings.
Power features (camera angles, single-channel schedules, server/player stats, mobile mode) require memorizing flags rather than offering quick one-tap Telegram inline actions alongside full CLI syntax.
2. Core Design Rules for All Agents
Every agent implementing any task in this plan MUST obey these invariants:

NO ROUNDED CORNERS ANYWHERE: Never call draw.rounded_rectangle. All cards, badges, tables, progress bars, and photo borders must use sharp 90° draw.rectangle.
Rule of One ("Say It Once"):
Header Title: Primary subject name only (e.g., Helsinki, Bedrock Server, YLE TV1).
Header Subtitle: Context metadata only (e.g., Tie 1 Kehä I • klo 14:20, crafty.lan:19132). Never repeat this in the body or footer.
Header Badge: High-level state only (ONLINE, OFFLINE, LIVE, VIRHE, or primary metric +18.5°C). Never add a Tila: Online key-value row when the badge already states ONLINE.
Footer: Standardized across the entire bot via a single helper: "<Moduuli> • P-iv-Botti".
Progress Bars: If a metric has a ProgressBarElement (CPU, RAM, Humidity, Cloudiness), never also add a KeyValuesElement for the same metric.
Strict Backward Compatibility:
Keep existing function names and default parameter values working so existing tests in tests/ continue to pass while new tests are added.
3. Phased Implementation Plan
Phase 1: Responsive Rendering Engine & Mobile Mode (bot/rendering/)
Goal: Upgrade bot/rendering/models.py and bot/rendering/engine.py so any Card can render in either "desktop" (default, 800px, rich dashboard) or "mobile" (480px width, larger fonts, single-column key-values, compact tables, capped height) without command logic needing to rewrite card construction.

Step 1.1: Update bot/rendering/models.py
Add layout_mode: str = "desktop" ("desktop" | "mobile") and typography scale fields to Theme:
python
Copy
Apply
@dataclass(frozen=True)
class Theme:
    canvas_bg: str = "#111114"
    card_bg: str = "#1b1b20"
    panel_bg: str = "#24242c"
    border_color: str = "#363642"
    accent_color: str = "#38bdf8"
    text_primary: str = "#f4f4f5"
    text_secondary: str = "#a1a1aa"
    text_muted: str = "#71717a"
    code_bg: str = "#151518"
    font_family: str = "auto"
    card_width: int = 800
    padding: int = 18
    element_spacing: int = 12
    layout_mode: str = "desktop"  # "desktop" or "mobile"
    font_scale: float = 1.0
Define preset themes in bot/rendering/models.py:
python
Copy
Apply
DARK_THEME = Theme()
MOBILE_DARK_THEME = Theme(
    card_width=480,
    padding=14,
    element_spacing=10,
    layout_mode="mobile",
    font_scale=1.22,
)
Add mobile-aware metadata fields (with safe defaults) to elements in bot/rendering/models.py:
TableElement: add mobile_columns: list[int] | None = None and mobile_max_rows: int | None = 8.
Behavior: When theme.layout_mode == "mobile" and mobile_columns is set (e.g. [0, 2] to keep only Player and Playtime), the renderer projects headers, rows, alignments, and col_widths to only those column indices. If mobile_columns is None and len(headers) > 3 in mobile mode, automatically keep the first 3 columns ([0, 1, 2]).
CardElement subclasses: add desktop_only: bool = False (default False). When theme.layout_mode == "mobile" and element.desktop_only is True, the renderer skips measuring and drawing that element.
Card: add mobile_subtitle: str | None = None.
Step 1.2: Fix ProgressBar Redundancy & Mobile Layout in bot/rendering/engine.py
Font scaling in CardRenderer.__init__:
Multiply base font sizes (22, 13, 14, 16, 12, 11) by theme.font_scale (rounded to int).
Non-repeating ProgressBarElement label formatting:
In _draw_element for ProgressBarElement:
If element.unit == "%" and element.max_value == 100.0: format val_text = f"{element.value:g}%" (instead of 55% / 100% (55.0%)).
Otherwise: format val_text = f"{element.value:g}{element.unit} / {element.max_value:g}{element.unit} ({pct * 100:.0f}%)".
Mobile adaptations in CardRenderer:
KeyValuesElement: When self.theme.layout_mode == "mobile", force cols = 1 (single-column stacked key-values so long values never clip on narrow screens) and scale row_h with font_scale.
TableElement: When self.theme.layout_mode == "mobile":
Filter columns by element.mobile_columns (or first 3 columns if len(headers) > 3).
Cap rows to element.mobile_max_rows (if smaller than element.max_rows).
ImageGridElement: When self.theme.layout_mode == "mobile", cap max_cell_height to 180 so a 2x2 grid stays compact on phone screens.
Skip desktop_only elements: In _measure_and_layout and _draw_all, filter self.card.elements with [el for el in self.card.elements if not (self.theme.layout_mode == "mobile" and getattr(el, "desktop_only", False))].
Step 1.3: Standardize Shared Card Factories in bot/rendering/helpers.py (New File)
Create bot/rendering/helpers.py and export its functions in bot/rendering/__init__.py to eliminate copy-pasted card builders across commands:

python
Copy
Apply
def format_card_footer(module_name: str) -> str:
    return f"{module_name} • P-iv-Botti"

def build_status_card(
    title: str,
    message: str,
    *,
    module_name: str,
    badge_text: str = "TIETO",
    badge_color: BadgeColor | str = BadgeColor.BLUE,
    subtitle: str | None = None,
    hint: str | None = None,
    usage_code: str | None = None,
) -> Card:
    """Unified builder for errors, empty states, usage hints, and confirmations."""
Standardize badge colors across all commands:

BadgeColor.RED: Network/API failures, offline servers, missing targets (VIRHE, OFFLINE).
BadgeColor.YELLOW: Invalid user arguments or multiple matches (KÄYTTÖ, TARKENNA).
BadgeColor.GRAY: Empty states or unconfigured integrations (TYHJÄ, EI KÄYTÖSSÄ).
BadgeColor.GREEN: Active/online/success states (ONLINE, VALMIS, LIVE).
BadgeColor.BLUE: Informational/help/schedule views (OHJE, TÄNÄÄN).
Verification for Phase 1:

Run ./venv/bin/python -m unittest tests.test_rendering
Add tests in tests/test_rendering.py verifying:
MOBILE_DARK_THEME produces a 480px wide PNG with no rounded corners.
ProgressBarElement("Kosteus", 55, 100, "%") formats label as "55%" without "/ 100% (55.0%)".
TableElement with mobile_columns=[0, 2] renders only 2 columns under MOBILE_DARK_THEME and all columns under DARK_THEME.
desktop_only=True elements are omitted in MOBILE_DARK_THEME.
Phase 2: Per-Chat UI Mode & Inline Toggle Support (bot/ui_prefs.py & bot/commands/message_utils.py)
Goal: Make mobile/simplified mode effortless to use via three mechanisms:

Per-chat persistent default (!ui mobile / !ui desktop stored in storage/ui_prefs.json).
Per-command inline flag (-m or --mobile or --kompakti at the end of any command, e.g. !sääkuva Helsinki -m).
Optional Telegram inline action row beneath cards ([📱 Mobiilinäkymä] / [🖥️ Täysi näkymä]) when configured.
Step 2.1: Add RenderingConfig keys in bot/config.py and example.env
In bot/config.py (RenderingConfig):
Add default_mode: str = "desktop" (RENDER_DEFAULT_MODE, validated to "desktop" or "mobile").
Add mobile_card_width: int = 480 (RENDER_MOBILE_CARD_WIDTH, validated 300..1200).
Add RENDER_DEFAULT_MODE=desktop and RENDER_MOBILE_CARD_WIDTH=480 to example.env.
Step 2.2: Create bot/ui_prefs.py
Store per-chat layout preference in storage/ui_prefs.json using load_json_data and save_json_data from bot/storage.py:
get_chat_layout_mode(storage_dir: Path, chat_id: int | None, default_mode: str = "desktop") -> str
set_chat_layout_mode(storage_dir: Path, chat_id: int, mode: str) -> str
strip_layout_flag(text: str) -> tuple[str, str | None]: strips trailing -m, --mobile, --kompakti (returns "mobile") or --desktop, --full (returns "desktop") from command text so individual commands don't need custom flag parsers.
Step 2.3: Upgrade reply_with_card in bot/commands/message_utils.py
Update reply_with_card(...) to accept layout_mode: str | None = None, config: BotConfig | None = None, and reply_markup: InlineKeyboardMarkup | None = None.
When theme is not explicitly passed:
Resolve mode = layout_mode or get_chat_layout_mode(...).
Select MOBILE_DARK_THEME (with card_width=config.rendering.mobile_card_width) if mode == "mobile", else DARK_THEME (with card_width=config.rendering.card_width).
Verification for Phase 2:

Run ./venv/bin/python -m unittest tests.test_config tests.test_message_utils
Phase 3: Deduplicate & Unify Existing Card Commands (bot/commands/)
Goal: Eliminate all repeated data inside !sääkuva, !mine, !mumble, and !telkkari, and ensure every table and grid defines clean mobile projections.

Step 3.1: Refactor !sääkuva (bot/commands/weather_logic.py)
Remove triple temperature & condition repetition:
Title: <Location>, <Country> (e.g. Helsinki, FI)
Subtitle: <Station Name> • klo <HH:MM> (observation time appears only here, never again in a bottom divider).
Badge: <emoji> <temp>°C (e.g. ☀️ 18.5°C).
Headline Text: <Description> — tuntuu <feels_like>°C (alin <min>°C / ylin <max>°C) (combines description, feels-like, and min/max into one scannable line instead of 3 separate key-value rows).
Deduplicate Key-Values vs. Progress Bars:
Key-Values (2 columns on desktop, 1 on mobile): Only metrics that are not in the headline or progress bars:
Tuuli: <speed> m/s <dir> (puuskat <gust> m/s) (merge wind and gusts into one value!)
Paine: <pressure> hPa (mark or keep concise)
Näkyvyys: <vis>
Aurinko: ↑ <sunrise> ↓ <sunset> (merge sunrise and sunset into one compact key-value instead of a divider + 3 rows!)
Progress Bars: Kosteus (%) and Pilvisyys (%) — remove Kosteus and Pilvisyys from add_key_value(...).
Deduplicate Camera Caption:
Do not repeat the station name in image.caption if it is already in card.subtitle. Use caption only for camera ID / angle hint, and mark the "Vinkki: !sääkuva ..." text element with desktop_only=True.
Use build_status_card for build_weather_error_card.
Step 3.2: Refactor !mine (bot/commands/mine_logic.py)
Single Server Status (build_mine_status_card):
Remove card.add_key_value("Tila", status_disp) because the header badge already displays ONLINE / KÄYNNISTYY / OFFLINE.
Keep Pelaajat, Versio, and CPU / RAM progress bars.
Mark Kuvaus: <motd> as desktop_only=True.
Single Player Stats (build_mine_single_player_stat_card):
Remove card.add_key_value("Tila", ...) when offline (badge already says OFFLINE); when online, rename key to Istunto (<current_session_duration>) so it adds new information instead of repeating "Paikalla".
Do not repeat server_name in both subtitle and footer; use subtitle=f"Palvelin: {server_name} • {stat.role or 'Pelaaja'}" (removing the separate Rooli key-value row) and footer=format_card_footer("Minecraft")!
Mark XUID key-value or text as desktop_only=True.
Server Player Stats Table (build_mine_server_player_stats_card):
Set mobile_columns=[0, 2, 3] (Pelaaja, Peliaika, Kuolemat) and mobile_max_rows=10 on add_table(...) so mobile users get a crisp 3-column leaderboard instead of a cramped 5-column table.
Replace build_mine_info_card internals with build_status_card(..., module_name="Minecraft").
Step 3.3: Refactor !mumble (bot/commands/mumble_logic.py)
Server Summary (build_mumble_summary_card):
On add_table(headers=["Käyttäjä", "Kanava", "Aika", "Viive"], ...), add mobile_columns=[0, 1, 2] (omits Ping column on mobile to give full room to username, channel, and duration).
Footer: format_card_footer("Mumble") (move channel count to subtitle: f"{server_host} • {len(snapshot.channels)} kanavaa").
User Details (build_mumble_user_card):
Remove redundant Tila key-value when it duplicates the header badge (ONLINE / MYKISTETTY / KUULOKKEET POIS).
Combine Paikallaoloaika and Toimettomana (idle) into a single Paikalla key-value: e.g. "1h 20m (aktiivinen)" or "1h 20m (idle 15m)".
Mark technical network rows (Asiakasohjelma, Kaistanleveys, Paketit) as a secondary KeyValuesElement with desktop_only=True so mobile view shows a clean, instant status card while desktop view shows full packet/bandwidth diagnostics.
Step 3.4: Refactor !telkkari (bot/commands/telkkari_logic.py)
Next-Hour & Day Schedule Tables:
Set mobile_max_rows=12 so schedules don't become overly tall vertical scrolls on mobile.
Replace build_telkkari_info_card internals with build_status_card(..., module_name="TV-ohjelmat").
Verification for Phase 3:

Run ./venv/bin/python -m unittest tests.test_weather_logic tests.test_mine_logic tests.test_mumble_logic tests.test_telkkari_logic
Update assertions in those test files where redundant keys (Tila, duplicate Lämpötila / Kosteus key-values) were intentionally consolidated.
Phase 4: Bring Remaining Text Commands into the Cohesive UI (!help, !twitch, !muistuta, !hoi)
Goal: Make !help, !twitch, !muistuta, and !hoi visually consistent with the rest of the bot by adding pure Card builders in their *_logic.py files (with automatic text fallback via reply_with_card).

Step 4.1: Categorized Visual !help + Command Detail View (bot/commands/help_logic.py & bot/commands/help.py)
In bot/commands/help_logic.py:
Keep build_help_reply(command_usages) intact for fallback text and existing tests.
Add build_help_card(command_usages: Sequence[str], topic: str = "") -> Card:
When topic == "": Group commands into logical categories (📊 Tieto & Palvelimet for !sääkuva, !telkkari, !mine, !mumble, !twitch; 🤖 Tekoäly & Media for !aih, !tiivista, !meme, !naama, !paranna, !tts, !stt; 🛠️ Ryhmätyökalut for !hoi, !muistuta, !help) in a clean 2-column table (Komento, Kuvaus). Add footer tip: "Vinkki: !help <komento> tai lisää -m mobiilinäkymään • P-iv-Botti".
When topic is provided (e.g. !help mine or !help sääkuva): Return a focused Card showing all subcommands and copy-pasteable examples for that specific command.
Step 4.2: Port !twitch to Picture Card (bot/commands/twitch_logic.py & bot/commands/twitch.py)
Add build_twitch_status_card(channels_status: list[dict[str, Any]]) -> Card in bot/commands/twitch_logic.py:
Header Badge: f"{live_count} LIVE" (BadgeColor.GREEN if live_count > 0 else BadgeColor.GRAY).
Table columns: ["Kanava", "Tila / Peli", "Otsikko"] (mobile_columns=[0, 1]).
Keep live URLs in the caption or fallback_text so users can still tap links directly in Telegram!
Step 4.3: Port !muistuta List & Confirmations (bot/commands/muistuta_logic.py & bot/commands/muistuta.py)
Keep text replies for quick creations/cancellations or provide build_reminders_list_card(reminders) for !muistutukset:
Renders active reminders in a table: ["ID", "Aika", "Viesti"] (col_widths=[1, 2, 4], mobile_columns=[0, 1, 2]).
Step 4.4: Port !hoi List Overview (bot/commands/hoi_logic.py & bot/commands/hoi.py)
Keep !hoi <lista> ping as a plain text message (because Telegram only triggers user notifications on text @mentions, never inside images!).
Render !hoi (list overview with no args) using a Card table ["Lista", "Jäsenet"] via build_hoi_lists_card(chat_data).
Verification for Phase 4:

Run ./venv/bin/python -m unittest tests.test_help_logic tests.test_twitch_logic tests.test_muistuta_logic
Phase 5: Power-User Ergonomics & Interactive Quick Actions
Goal: Make the bot easy for casual mobile users while keeping fast CLI shortcuts for power users.

Universal -m / --mobile and -d / --desktop Flag:
In bot/commands/common.py (or message_utils.py), inspect incoming command text with strip_layout_flag so appending -m to any card command (!mine -m, !sääkuva Oulu -m, !telkkari -m, !mumble -m) renders that reply in MOBILE_DARK_THEME.
New !ui Command Module (bot/commands/ui.py + bot/commands/ui_logic.py):
COMMAND_USAGE = "!ui [mobile|desktop]":
Allows any chat or DM to switch its persistent default card rendering mode between "mobile" (480px, large text, compact columns) and "desktop" (800px, full dashboard).
Replies with a preview Card rendered in the newly selected mode so the user immediately sees the difference.
4. Agent Execution Checklist & Verification Commands
Low-level agents executing this plan should complete one phase at a time and run the corresponding verification suite before moving to the next phase:

Phase	Files to Modify / Create	Verification Command
Phase 1	bot/rendering/models.py, bot/rendering/engine.py, bot/rendering/helpers.py, bot/rendering/__init__.py, tests/test_rendering.py	./venv/bin/python -m unittest tests.test_rendering
Phase 2	bot/config.py, example.env, bot/ui_prefs.py, bot/commands/message_utils.py, tests/test_config.py, tests/test_message_utils.py	./venv/bin/python -m unittest tests.test_config tests.test_message_utils
Phase 3	bot/commands/weather_logic.py, bot/commands/mine_logic.py, bot/commands/mumble_logic.py, bot/commands/telkkari_logic.py + their tests/test_*_logic.py	./venv/bin/python -m unittest tests.test_weather_logic tests.test_mine_logic tests.test_mumble_logic tests.test_telkkari_logic
Phase 4	bot/commands/help_logic.py, bot/commands/help.py, bot/commands/twitch_logic.py, bot/commands/twitch.py, bot/commands/muistuta_logic.py, bot/commands/hoi_logic.py	./venv/bin/python -m unittest tests.test_help_logic tests.test_twitch_logic tests.test_muistuta_logic
Phase 5	bot/commands/ui.py, bot/commands/ui_logic.py, bot/commands/common.py, AGENTS.md, README.md	./venv/bin/python -m unittest
