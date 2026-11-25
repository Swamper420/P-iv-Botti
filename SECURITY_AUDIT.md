# Security Audit — P-iv-Botti

* **Audit Date:** 2026-11-25
* **Audited Commit:** `57aadff`
* **Scope:** Core infrastructure (`bot/main.py`, `bot/config.py`, `bot/storage.py`, `bot/active_chats.py`, `bot/vision.py`), all 16 command pairs (`bot/commands/`), all 5 background tasks (`bot/tasks/`), PIL rendering engine (`bot/rendering/`), and dependency/environment configuration (`requirements.txt`, `example.env`, `.gitignore`).

---

## Executive Summary

| Severity | Count | Highlights |
| :--- | :---: | :--- |
| **Critical** | 3 | SSRF + broken response-size cap in `!tiivistä`; zero access control (DM allowlist mutation + broadcast hijacking); input-validation bypass in `!mine allowlist add` |
| **High** | 3 | Unsafe `torch.load` deserialization & unverified weight download; `TWITCH_CLIENT_SECRET` in URL query string & Mumble password in debug logs; TLS verification disabled/hardcoded off |
| **Medium** | 4 | Unescaped HTML injection in Twitch/Mine/Mumble Telegram messages; OOM/CPU/disk DoS vectors (`!paranna`, auto-STT, EPG gzip, card rendering); raw exception leakage; external Vision privacy & config mismatch |
| **Low / Hardening** | 4 | Unpinned/incomplete `requirements.txt`; JSON `.tmp` race condition & unsynchronized writes; `!muistuta` integer overflow; `.env` deployment permissions |

---

## Critical Findings

### 1. SSRF and Broken `max_bytes` Limit in `!tiivistä`
* **Location:** `bot/commands/tiivista_logic.py:141-176`
* **Details:**
  1. **Server-Side Request Forgery (SSRF):** `fetch_webpage_text()` passes any user-supplied `http://` or `https://` URL directly to `httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True)` with no hostname/IP validation. Because this bot typically runs alongside unauthenticated local services (`localhost:11434` Ollama, `localhost:8080` TTS, `localhost:8001` STT, `localhost:8443` Crafty, LAN hosts, or cloud metadata `169.254.169.254`), any user can trigger internal GET requests (or redirect an external URL to `127.0.0.1`) and receive the summarized response back as audio.
  2. **Broken `max_bytes` enforcement:** `await client.get(url)` downloads the entire response body into RAM *before* checking `max_bytes`, enabling OOM crashes via large responses or compression bombs. Furthermore, after truncating `content_bytes = content_bytes[:max_bytes]`, lines 168 and 170 ignore `content_bytes` and pass the full untruncated `response.text` to `extract_text_from_html()` / return it directly.
* **Remediation:**
  * Restrict schemes to `http`/`https`, resolve DNS before connecting, and reject any non-global IP (`not ip.is_global`, blocking loopback, RFC1918 private, link-local, CGNAT, multicast, and IPv6 ULA/mapped IPv4).
  * Set `follow_redirects=False` and manually validate every redirect hop (with a max hop cap, e.g., 5).
  * Use `async with client.stream("GET", ...)` and read chunks only up to `max_bytes`, then decode those bounded bytes instead of using `response.text`.

### 2. Missing Access Control: Public DM Access, Unauthenticated Minecraft Whitelisting, and Broadcast Hijacking
* **Location:** `bot/commands/common.py:19-32`, `bot/active_chats.py:38-52`, `bot/commands/mine.py:22-58`
* **Details:**
  * None of the handlers check whether the incoming `chat_id` or `user_id` is authorized. Anyone on Telegram who discovers the bot's username can message it directly (DM) or add it to a group and invoke every command:
    * **Privilege escalation on Minecraft:** `!mine allowlist add <gamertag>` executes `allowlist add` + `allowlist reload` on Crafty Controller's stdin API with no admin or chat check, letting any stranger whitelist themselves.
    * **Internal service recon & quota abuse:** `!mine` / `!mine stats` / `!mine allowlist` expose server names, ports, player XUIDs, and OP roles; `!mumble` exposes server host/port, users, and client OS; `!aih`, `!paranna`, `!naama`, `!meme`, `!tiivistä`, `puhuu:`, and auto-STT let outsiders consume local GPU/CPU and paid/rate-limited APIs.
    * **Broadcast hijacking:** `command_handler` unconditionally calls `track_active_chat()`, persisting any caller's `chat_id` into `storage/active_chat_ids.json`. Any external user who sends `!help` is automatically subscribed to CS2 RSS updates, Twitch notifications, and (when `MARATON_CHAT_IDS` is unset) daily Maraton broadcasts.
* **Remediation:**
  * Add `ALLOWED_CHAT_IDS` (and optionally `ADMIN_USER_IDS` for mutating actions like `!mine allowlist add`) to `BotConfig` and enforce it in `@command_handler` *before* calling `track_active_chat()` or executing the handler.
  * Require explicit opt-in or an allowlist before adding chats to `active_chat_ids.json`.

### 3. Input-Validation Bypass in `!mine allowlist add` Console Command Construction
* **Location:** `bot/commands/mine_logic.py:1341-1388`, `bot/commands/mine_logic.py:528-540`
* **Details:**
  * `add_mine_allowlist_card()` validates `clean_name` (`player_name`) against `_PLAYER_NAME_RE` (`^[a-zA-Z0-9_ ]{1,32}$`) at line 1346, **before** resolving `server_query`.
  * When `len(servers) == 1 and not config.default_server_id` and `server_query` does not match a server name (lines 1384–1387), it rewrites `clean_name = f"{server_query} {clean_name}".strip()` without re-validating `clean_name`.
  * `server_query` is completely unvalidated and can contain quotes (`"`), special characters, Bedrock selectors/arguments, or exceed 32 characters. It is then interpolated into `allowlist add "{player_name}"` and POSTed raw to Crafty's `/api/v2/servers/{id}/stdin`.
* **Remediation:**
  * Move the `_PLAYER_NAME_RE.fullmatch(clean_name)` check *after* the single-server fallback merge (lines 1384–1387), and also enforce `_PLAYER_NAME_RE` defensively inside `CraftyClient.add_to_allowlist()` before sending to `stdin`.

---

## High Findings

### 4. Unsafe PyTorch Deserialization (`torch.load`) and Unverified Model Download
* **Location:** `bot/commands/paranna_logic.py:87-120`
* **Details:**
  * Line 108 calls `torch.load(str(local_path), map_location="cpu")` without `weights_only=True`. On PyTorch versions prior to 2.6, this uses Python `pickle` deserialization, which executes arbitrary code if the `.pth` file is tampered with or replaced.
  * Line 92 uses `urllib.request.urlretrieve(model_url, str(path))` with no SHA-256 checksum verification, no timeout, no scheme restriction (`urlretrieve` supports `file://`), and non-atomic writes directly to the final path.
* **Remediation:**
  * Pass `weights_only=True` to `torch.load()`.
  * Verify downloaded/loaded weights against a pinned SHA-256 digest before loading.
  * Validate `https://` scheme, download via `httpx` or `urlopen` with an explicit timeout into a temporary file, verify the hash, and atomically `os.replace()` into `model_path`.

### 5. Secrets Exposed in URL Query Strings and Debug Logs
* **Location:** `bot/tasks/twitch_logic.py:88-110`, `bot/tasks/mumble_logic.py:168-175`
* **Details:**
  * **Twitch OAuth secret in URL:** `TwitchClient.get_app_token()` sends `client_id` and `client_secret` in the URL query string (`f"{self.config.token_url}?{params}"`) rather than the POST body. Query strings are routinely recorded in HTTP proxy logs, web server logs, and `urllib` `HTTPError`/`URLError` exception messages (which line 109 logs via `LOGGER.error(..., err)`).
  * **Mumble password in debug logs:** Line 174 runs `self.Log.debug("sending: authenticate: %s", authenticate)`, logging the protobuf object containing `self.password` in cleartext if debug logging is ever enabled.
* **Remediation:**
  * Send Twitch OAuth parameters as the `POST` body (`data=params.encode("utf-8")` with `Content-Type: application/x-www-form-urlencoded`) and sanitize error logging.
  * Log only the username (not the `authenticate` protobuf object) in `ModernMumbleClient.connect()`.

### 6. TLS Certificate Verification Disabled by Default / Hardcoded Off
* **Location:** `bot/commands/mine_logic.py:236-242`, `bot/config.py:232`, `bot/tasks/mumble_logic.py:136-145`
* **Details:**
  * `CRAFTY_VERIFY_SSL` defaults to `false` (`ssl._create_unverified_context()`), exposing `Authorization: Bearer <CRAFTY_API_TOKEN>` to MitM if Crafty is accessed over a non-loopback network.
  * `ModernMumbleClient.connect()` hardcodes `context.check_hostname = False` and `context.verify_mode = ssl.CERT_NONE` with no way for operators to enable verification, exposing `MUMBLE_PASSWORD` and control traffic to MitM.
* **Remediation:**
  * Default `CRAFTY_VERIFY_SSL=true` (and optionally add `CRAFTY_CA_BUNDLE` for Crafty's self-signed certificate).
  * Add `MUMBLE_VERIFY_SSL` / `MUMBLE_CA_BUNDLE` (or SHA-256 server cert fingerprint pinning) to `MumbleConfig`.

---

## Medium Findings

### 7. HTML Injection in Telegram `parse_mode="HTML"` Messages
* **Location:** `bot/tasks/twitch_logic.py:28-69`, `bot/commands/mine_logic.py:714`, `bot/commands/mumble_logic.py:82-86`
* **Details:**
  * `TwitchStreamNotification.format_telegram_message()` and `TwitchStreamSummaryNotification.format_telegram_message()` interpolate `broadcaster_name`, `title`, `game_name`, `stream_url`, and `vod_url` **without `html.escape()`**, and `bot/tasks/twitch.py:47-83` sends them with `parse_mode="HTML"`. A Twitch stream title containing `<a href="...">` can inject arbitrary clickable phishing links into group notifications, or malformed `<`/`&` characters can break Telegram HTML parsing and drop notifications. (Notably, `bot/commands/twitch_logic.py:50-54` properly escapes these, while `bot/tasks/twitch_logic.py` does not.)
  * `_resolve_servers()` in `bot/commands/mine_logic.py` returns unescaped `server_query` (and raw `{exc}` strings elsewhere) into messages sent with `parse_mode="HTML"` by `bot/commands/mine.py:51-55`.
  * `bot/commands/mumble_logic.py` interpolates `config.host` and `snapshot.error_message` without `html.escape()`.
* **Remediation:**
  * Wrap all dynamic/external fields in `html.escape(str(val), quote=True)` wherever `parse_mode="HTML"` is used.

### 8. Resource Exhaustion / Denial-of-Service Vectors (RAM, CPU, Disk, Downstream)
* **Locations:**
  * **~1.26 GB RAM per `!paranna` image:** `bot/commands/paranna_logic.py:131-136` — With `max_input_dimension=2560` and `scale=4`, `_tiled_upscale` allocates a `float32` tensor of shape `(1, 3, 10240, 10240)` (~1.26 GB) *before* post-scaling down to `4096`. Combined with no concurrency semaphore, 2–3 concurrent `!paranna` calls can OOM-kill the host.
  * **Download-before-size-check & unbounded auto-STT:** `bot/commands/stt.py:36-56`, `bot/commands/paranna.py:59-64`, `bot/commands/naama.py:87-92`, `bot/commands/meme.py:78-88`, `bot/commands/tiivista.py:101-115` — Image commands download full files before checking `max_image_bytes` instead of pre-checking `target_photo.file_size`. `stt.py` automatically downloads and uploads *every* video/voice/video_note with **no size limit** and forwards the attacker-controlled `message.video.file_name` verbatim to the local STT server (risking path traversal or argument injection if the STT service uses `filename` on disk/ffmpeg).
  * **Unbounded gzip decompression & XML parsing:** `bot/commands/telkkari_logic.py:88-94`, `bot/commands/weather_logic.py:75-93` — `resp.read()` and `gzip.decompress()` have no byte caps (vulnerable to gzip bombs if upstream is compromised), and `_EPG_CACHE` in `telkkari_logic.py` has no lock across worker threads.
  * **Unbounded card height & $O(N^2)$ word splitting:** `bot/rendering/engine.py:227-235`, `bot/rendering/engine.py:502-515` — `Card.max_height` defaults to `None`, and user arguments in `!telkkari`, `!mumble`, `!mine`, and `!sääkuva` are rendered into cards without length truncation. `_wrap_text()` breaks overlong spaceless strings char-by-char while re-measuring width each iteration ($O(N^2)$).
  * **Unbounded `!hoi` storage:** `bot/commands/hoi_logic.py:58-87` — Unlike `!muistuta` (`max_per_chat=50`), `!hoi` has no caps on list count, list name length, or users per list.
* **Remediation:**
  * Add `asyncio.Semaphore` concurrency guards (e.g., 1–2 concurrent ML/upscale jobs) and per-chat/per-user rate limiting.
  * Cap `paranna` pre-scale to `max_output_dimension // 4` (e.g., 1024px) or accumulate tiles in `uint8`.
  * Check `media.file_size` before `download_as_bytearray()`; enforce a max size in `stt.py` and replace `message.video.file_name` with a fixed safe filename (`"video.mp4"`).
  * Enforce streaming size caps on HTTP reads and `zlib`/`gzip` decompression (`decompressobj.decompress(data, max_length=...)`), and consider `defusedxml` for external XML.
  * Set a default safety cap on `Card.max_height` (e.g., 4000px), truncate user-supplied query strings placed on cards, and add caps to `hoi_logic.py`.

### 9. Internal Error Information Disclosure
* **Location:** `bot/commands/aih.py:88-93`, `bot/commands/mine_logic.py:1166-1175`, `bot/commands/mumble_logic.py:81-87`
* **Details:**
  * Raw Python/HTTP exception strings (`{exc}`, `{exc.reason}`) and internal hostnames/ports (`config.host:config.port`) are sent directly to Telegram users. On network or HTTP failures, `httpx` and `urllib` exceptions routinely include the target internal URL (`http://localhost:11434/api/generate`, `https://localhost:8443/...`), connection details, or stack error messages.
* **Remediation:**
  * Log full exception details server-side with `LOGGER.exception(...)` and return generic, localized error messages to Telegram users.

### 10. Third-Party Image Privacy Exposure, Config Mismatch, & Indirect Prompt Injection
* **Location:** `bot/vision.py:73-137`, `bot/config.py:54-60`, `example.env:24-29`, `bot/commands/tiivista.py:119-122`, `bot/commands/meme.py:98-100`
* **Details:**
  * **External unauthenticated photo upload:** `!meme` and `!tiivistä` send user-uploaded Telegram photos by default to an external third-party vision endpoint (`https://eyes.tsih.fi/v1` in `bot/config.py`, while `example.env` specifies a different domain `https://baalls.tsah.se/v1`) with no API key / auth header support. Users in private group chats may not expect photos to leave the self-hosted server, and a typo/expired third-party domain could silently route photos to an attacker-controlled server.
  * **Indirect prompt injection:** User text (`raw_text`, `hint`) and scraped webpage / OCR text are concatenated directly into the vision/Ollama prompt instructions. While the bot has no tool-calling today, crafted webpage/image text can hijack summaries (spoken aloud via TTS) or override meme caption formatting.
* **Remediation:**
  * Reconcile `EYES_BASE_URL` between `bot/config.py` and `example.env`, consider making external Eyes opt-in (empty default -> local fallback), and add optional `EYES_API_KEY` (`Authorization: Bearer ...`) support.
  * Delimit untrusted user/web/OCR content clearly (e.g., XML tags + system instruction to treat enclosed text strictly as data) and keep LLM outputs untrusted (never pass to HTML/shell/eval).

---

## Low & Hardening Findings

### 11. Unpinned and Incomplete `requirements.txt`
* **Location:** `requirements.txt:1-6`
* **Details:** All 6 listed dependencies lack version pins (and direct runtime imports `httpx` and `torch` are missing from `requirements.txt`). Unpinned installs risk pulling vulnerable or incompatible versions (e.g., Pillow image-parsing CVEs).
* **Remediation:** Add `httpx` and `torch`, pin tested minimum/exact versions (or maintain a lockfile with hashes), and periodically run `pip-audit`.

### 12. JSON Storage Temp-File Race & Unsynchronized Read-Modify-Write
* **Location:** `bot/storage.py:28-41`, `bot/active_chats.py:43-52`, `bot/commands/hoi_logic.py:58-115`, `bot/commands/muistuta_logic.py:183-310`
* **Details:** `save_json_data()` always writes to `file_path.with_suffix(".tmp")`. Concurrent writes (e.g., `ReminderNotifier` deleting delivered reminders while a user creates a new reminder) can race on the shared `.tmp` path or overwrite each other's updates due to unlocked read-modify-write cycles.
* **Remediation:** Write to a unique temp file via `tempfile.NamedTemporaryFile(dir=file_path.parent, delete=False)` before `os.replace()`, and guard read-modify-write storage operations with per-file locks.

### 13. Unhandled `OverflowError` in `!muistuta` Relative Time Parsing
* **Location:** `bot/commands/muistuta_logic.py:38-55`, `bot/tasks/reminders.py:72-83`
* **Details:** `parse_relative_time()` accepts arbitrary digit lengths (e.g., `!muistuta 999999999999d`), raising an uncaught `OverflowError` in `timedelta(...)` instead of returning a validation error. Also in `bot/tasks/reminders.py:72-83`, if a media reminder's caption exceeds Telegram's 1024-character limit, sending fails and the reminder is permanently dropped in `finally:`.
* **Remediation:** Cap relative offsets (e.g., max 365 days / reasonable upper bound) and catch `OverflowError`; truncate or split captions exceeding 1024 chars on media reminders.

### 14. Operational / Secrets Hygiene
* **Location:** `bot/config.py:11-25`, `.gitignore:112`
* **Details:** `.env` is properly gitignored and no live secrets were found in git history or the repository. On the Linux/systemd host, verify `.env` is owned by the bot service user with `chmod 600`, `storage/` has `chmod 700`, and the systemd unit runs as a dedicated unprivileged user with sandboxing (`NoNewPrivileges=true`, `ProtectSystem=strict`, `ReadWritePaths=<storage>`).

---

## Positive Controls Verified

* **No OS command injection in subprocesses:** Both Tesseract OCR (`bot/commands/tiivista_logic.py:403-425`) and FFmpeg (`bot/commands/tts_logic.py:303-327`) use list-form `subprocess.run` / `create_subprocess_exec` with `shell=False`, explicit timeouts, and stdin/stdout pipes rather than temp files or shell interpolation.
* **No committed secrets:** Git history check for `.env` and secret pattern scan found only placeholder values in `example.env`.
* **No `eval` / `exec` / unsafe `yaml.load`:** None found in the codebase.
* **Good HTML escaping in newer modules:** `bot/commands/maraton_logic.py`, `bot/tasks/maraton_logic.py`, and `bot/commands/twitch_logic.py` consistently apply `html.escape()`.

---

## Prioritized Remediation Order

1. **Immediate (Critical):**
   * Add SSRF IP/redirect validation and true streaming `max_bytes` enforcement to `bot/commands/tiivista_logic.py:141-176`.
   * Add `ALLOWED_CHAT_IDS` (and `ADMIN_USER_IDS` for `!mine allowlist add`) enforcement in `bot/commands/common.py:19-32`.
   * Fix the `_PLAYER_NAME_RE` validation bypass after `server_query` merge in `bot/commands/mine_logic.py:1341-1388`.
2. **Short-term (High):**
   * Add `weights_only=True` + SHA-256 verification to `bot/commands/paranna_logic.py:87-120`.
   * Move Twitch OAuth credentials from URL query string to POST body in `bot/tasks/twitch_logic.py:88-110` and redact Mumble password from debug logs in `bot/tasks/mumble_logic.py:168-175`.
   * Enable TLS verification (or CA bundle / cert pinning) for Crafty and Mumble.
3. **Follow-up (Medium/Low):**
   * Escape HTML in `bot/tasks/twitch_logic.py`, `bot/commands/mine_logic.py`, and `bot/commands/mumble_logic.py`, and stop returning raw `{exc}` strings to Telegram.
   * Add concurrency semaphores, pre-download file-size limits, safe STT filenames, bounded gzip decompression, and `paranna` memory cap.
   * Reconcile `EYES_BASE_URL` defaults, pin `requirements.txt`, and fix the `bot/storage.py` `.tmp` race condition.
