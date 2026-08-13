from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        os.environ.setdefault(key, value.strip())


@dataclass(frozen=True)
class WeatherConfig:
    openweather_api_key: str
    weathercam_stations_url: str
    weathercam_image_base_url: str
    openweather_current_url: str
    timeout_seconds: int
    digitraffic_user: str


@dataclass(frozen=True)
class Cs2RssConfig:
    url: str
    poll_interval_seconds: int
    request_timeout_seconds: int


@dataclass(frozen=True)
class NaamaConfig:
    model_name: str = "yolo26n-seg.pt"
    confidence_threshold: float = 0.15
    mask_threshold: float = 0.35
    max_image_bytes: int = 10_000_000


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "gemma3"
    default_num_predict: int = 100
    max_num_predict: int = 2000
    num_ctx: int = 2048
    timeout_seconds: int = 120
    system_prompt: str = "Vastaa aina suomeksi suoraan ja tiiviisti."




@dataclass(frozen=True)
class TwitchConfig:
    client_id: str = ""
    client_secret: str = ""
    user_access_token: str = ""
    channels: tuple[str, ...] = ()
    websocket_url: str = "wss://eventsub.wss.twitch.tv/ws"
    token_url: str = "https://id.twitch.tv/oauth2/token"
    helix_base_url: str = "https://api.twitch.tv/helix"
    reconnect_delay_seconds: int = 5
    poll_interval_seconds: int = 60

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.channels)



@dataclass(frozen=True)
class ParannaConfig:
    model_path: str = "storage/models/RealESRGAN_x4plus_anime_6B.pth"
    model_url: str = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    )
    tile_size: int = 256
    tile_pad: int = 10
    max_input_dimension: int = 2560
    max_output_dimension: int = 4096
    jpeg_quality: int = 95
    max_image_bytes: int = 15_000_000

@dataclass(frozen=True)
class TtsConfig:
    base_url: str = "http://localhost:8080"
    timeout_seconds: int = 60
    format: str = "ogg"
    error_message: str = "Virhe puheen synteesissä."
    language: str = "fi"
    speed: float = 1.0
    num_step: int = 32
    guidance_scale: float = 2.0
    reencode_audio: bool = True
    ffmpeg_path: str = "ffmpeg"
    audio_sample_rate: int = 48000
    audio_channels: int = 1
    audio_bitrate: str = "32k"




@dataclass(frozen=True)
class TelkkariConfig:
    epg_url: str = "https://epgshare01.online/epgshare01/epg_ripper_FI1.xml.gz"
    default_channels: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    cache_timeout_seconds: int = 1800
    timeout_seconds: int = 30


@dataclass(frozen=True)
class SttConfig:
    base_url: str = "http://localhost:8001"
    timeout_seconds: int = 60
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False
    initial_prompt: str = ""


@dataclass(frozen=True)
class ReminderConfig:
    check_interval_seconds: int = 5
    max_per_chat: int = 50


@dataclass(frozen=True)
class TiivistaConfig:
    timeout_seconds: int = 60
    max_web_page_bytes: int = 2_000_000
    max_text_length: int = 12_000
    summary_num_predict: int = 300
    user_agent: str = "Mozilla/5.0 (compatible; P-iv-Botti/1.0)"
    system_prompt: str = (
        "Tiivistä annettu teksti tiiviiksi, selkeäksi ja sujuvaksi suomenkieliseksi "
        "yhteenvedoksi. Vastaa pelkällä tiivistelmällä ilman alustuksia, sillä se luetaan ääneen."
    )
    caption_num_words: int = 3


@dataclass(frozen=True)
class BotConfig:
    telegram_bot_token: str
    storage_dir: Path
    max_reply_length: int
    weather: WeatherConfig
    cs2_rss: Cs2RssConfig
    naama: NaamaConfig
    ollama: OllamaConfig = OllamaConfig()
    twitch: TwitchConfig = TwitchConfig()
    paranna: ParannaConfig = ParannaConfig()
    tts: TtsConfig = TtsConfig()
    telkkari: TelkkariConfig = TelkkariConfig()
    stt: SttConfig = SttConfig()
    reminder: ReminderConfig = ReminderConfig()
    tiivista: TiivistaConfig = TiivistaConfig()






    # Backward compatibility properties
    @property
    def openweather_api_key(self) -> str:
        return self.weather.openweather_api_key

    @property
    def weathercam_stations_url(self) -> str:
        return self.weather.weathercam_stations_url

    @property
    def weathercam_image_base_url(self) -> str:
        return self.weather.weathercam_image_base_url

    @property
    def openweather_current_url(self) -> str:
        return self.weather.openweather_current_url

    @property
    def weather_api_timeout_seconds(self) -> int:
        return self.weather.timeout_seconds

    @property
    def digitraffic_user(self) -> str:
        return self.weather.digitraffic_user

    @property
    def steam_cs2_rss_url(self) -> str:
        return self.cs2_rss.url

    @property
    def steam_rss_poll_interval_seconds(self) -> int:
        return self.cs2_rss.poll_interval_seconds

    @property
    def steam_rss_request_timeout_seconds(self) -> int:
        return self.cs2_rss.request_timeout_seconds

    @property
    def naama_model_name(self) -> str:
        return self.naama.model_name

    @property
    def naama_confidence_threshold(self) -> float:
        return self.naama.confidence_threshold

    @property
    def naama_mask_threshold(self) -> float:
        return self.naama.mask_threshold

    @property
    def naama_max_image_bytes(self) -> int:
        return self.naama.max_image_bytes

    @property
    def ollama_base_url(self) -> str:
        return self.ollama.base_url

    @property
    def ollama_model(self) -> str:
        return self.ollama.model

    @property
    def ollama_default_num_predict(self) -> int:
        return self.ollama.default_num_predict

    @property
    def ollama_max_num_predict(self) -> int:
        return self.ollama.max_num_predict

    @property
    def ollama_num_ctx(self) -> int:
        return self.ollama.num_ctx

    @property
    def ollama_timeout_seconds(self) -> int:
        return self.ollama.timeout_seconds

    @property
    def ollama_system_prompt(self) -> str:
        return self.ollama.system_prompt

    @property
    def twitch_client_id(self) -> str:
        return self.twitch.client_id

    @property
    def twitch_client_secret(self) -> str:
        return self.twitch.client_secret

    @property
    def twitch_channels(self) -> tuple[str, ...]:
        return self.twitch.channels

    @property
    def tts_base_url(self) -> str:
        return self.tts.base_url

    @property
    def tts_timeout_seconds(self) -> int:
        return self.tts.timeout_seconds

    @property
    def tts_format(self) -> str:
        return self.tts.format

    @property
    def tts_error_message(self) -> str:
        return self.tts.error_message

    @property
    def tts_language(self) -> str:
        return self.tts.language

    @property
    def tts_speed(self) -> float:
        return self.tts.speed

    @property
    def tts_num_step(self) -> int:
        return self.tts.num_step

    @property
    def tts_guidance_scale(self) -> float:
        return self.tts.guidance_scale

    @property
    def tts_reencode_audio(self) -> bool:
        return self.tts.reencode_audio

    @property
    def tts_ffmpeg_path(self) -> str:
        return self.tts.ffmpeg_path

    @property
    def tts_audio_sample_rate(self) -> int:
        return self.tts.audio_sample_rate

    @property
    def tts_audio_channels(self) -> int:
        return self.tts.audio_channels

    @property
    def tts_audio_bitrate(self) -> str:
        return self.tts.audio_bitrate

    @property
    def stt_base_url(self) -> str:
        return self.stt.base_url

    @property
    def stt_timeout_seconds(self) -> int:
        return self.stt.timeout_seconds

    @property
    def stt_beam_size(self) -> int:
        return self.stt.beam_size

    @property
    def stt_vad_filter(self) -> bool:
        return self.stt.vad_filter

    @property
    def stt_word_timestamps(self) -> bool:
        return self.stt.word_timestamps

    @property
    def stt_initial_prompt(self) -> str:
        return self.stt.initial_prompt


    @classmethod

    def from_environment(cls) -> "BotConfig":
        project_root = Path(__file__).resolve().parent.parent
        _load_env_file(project_root / ".env")

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        storage_dir = Path(
            os.getenv("STORAGE_DIR", str(project_root / "storage"))
        ).resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)

        naama_confidence_threshold = float(
            os.getenv("NAAMA_CONFIDENCE_THRESHOLD", "0.15")
        )
        if not 0 <= naama_confidence_threshold <= 1:
            raise ValueError("NAAMA_CONFIDENCE_THRESHOLD must be between 0 and 1")
        naama_mask_threshold = float(
            os.getenv("NAAMA_MASK_THRESHOLD", "0.35")
        )
        if not 0 <= naama_mask_threshold <= 1:
            raise ValueError("NAAMA_MASK_THRESHOLD must be between 0 and 1")
        naama_max_image_bytes = int(
            os.getenv("NAAMA_MAX_IMAGE_BYTES", "10000000")
        )
        if naama_max_image_bytes < 1:
            raise ValueError("NAAMA_MAX_IMAGE_BYTES must be >= 1")

        ollama_default_num_predict = int(
            os.getenv("OLLAMA_DEFAULT_NUM_PREDICT", "100")
        )
        if ollama_default_num_predict < 1:
            raise ValueError("OLLAMA_DEFAULT_NUM_PREDICT must be >= 1")
        ollama_max_num_predict = int(
            os.getenv("OLLAMA_MAX_NUM_PREDICT", "2000")
        )
        if ollama_max_num_predict < 1:
            raise ValueError("OLLAMA_MAX_NUM_PREDICT must be >= 1")
        ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
        if ollama_num_ctx < 1:
            raise ValueError("OLLAMA_NUM_CTX must be >= 1")
        ollama_timeout_seconds = int(
            os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")
        )
        if ollama_timeout_seconds < 1:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be >= 1")
        default_system = "Vastaa aina suomeksi suoraan ja tiiviisti."
        ollama_system_prompt = os.getenv("OLLAMA_SYSTEM_PROMPT", default_system).strip()

        weather_config = WeatherConfig(
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY", "").strip(),
            weathercam_stations_url=os.getenv(
                "WEATHERCAM_STATIONS_URL",
                "https://tie.digitraffic.fi/api/weathercam/v1/stations",
            ).strip(),
            weathercam_image_base_url=os.getenv(
                "WEATHERCAM_IMAGE_BASE_URL", "https://weathercam.digitraffic.fi"
            ).strip(),
            openweather_current_url=os.getenv(
                "OPENWEATHER_CURRENT_URL",
                "https://api.openweathermap.org/data/2.5/weather",
            ).strip(),
            timeout_seconds=int(
                os.getenv("WEATHER_API_TIMEOUT_SECONDS", "30")
            ),
            digitraffic_user=os.getenv("DIGITRAFFIC_USER", "telegram-bot-1.0").strip(),
        )

        cs2_rss_config = Cs2RssConfig(
            url=os.getenv(
                "STEAM_CS2_RSS_URL", "https://steamcommunity.com/games/csgo/rss/"
            ).strip(),
            poll_interval_seconds=int(
                os.getenv("STEAM_RSS_POLL_INTERVAL_SECONDS", "300")
            ),
            request_timeout_seconds=int(
                os.getenv("STEAM_RSS_REQUEST_TIMEOUT_SECONDS", "30")
            ),
        )

        naama_config = NaamaConfig(
            model_name=os.getenv("NAAMA_MODEL_NAME", "yolo26n-seg.pt").strip(),
            confidence_threshold=naama_confidence_threshold,
            mask_threshold=naama_mask_threshold,
            max_image_bytes=naama_max_image_bytes,
        )

        ollama_config = OllamaConfig(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
            model=os.getenv("OLLAMA_MODEL", "gemma3").strip(),
            default_num_predict=ollama_default_num_predict,
            max_num_predict=ollama_max_num_predict,
            num_ctx=ollama_num_ctx,
            timeout_seconds=ollama_timeout_seconds,
            system_prompt=ollama_system_prompt,
        )

        raw_channels = os.getenv("TWITCH_CHANNELS", "").strip()
        twitch_channels = tuple(
            c.strip().lower() for c in raw_channels.split(",") if c.strip()
        )
        twitch_config = TwitchConfig(
            client_id=os.getenv("TWITCH_CLIENT_ID", "").strip(),
            client_secret=os.getenv("TWITCH_CLIENT_SECRET", "").strip(),
            user_access_token=os.getenv("TWITCH_USER_ACCESS_TOKEN", "").strip(),
            channels=twitch_channels,
            websocket_url=os.getenv(
                "TWITCH_WEBSOCKET_URL", "wss://eventsub.wss.twitch.tv/ws"
            ).strip(),
            token_url=os.getenv(
                "TWITCH_TOKEN_URL", "https://id.twitch.tv/oauth2/token"
            ).strip(),
            helix_base_url=os.getenv(
                "TWITCH_HELIX_BASE_URL", "https://api.twitch.tv/helix"
            ).strip(),
            reconnect_delay_seconds=int(
                os.getenv("TWITCH_RECONNECT_DELAY_SECONDS", "5")
            ),
            poll_interval_seconds=int(
                os.getenv("TWITCH_POLL_INTERVAL_SECONDS", "60")
            ),
        )


        paranna_config = ParannaConfig(
            model_path=os.getenv(
                "PARANNA_MODEL_PATH", "storage/models/RealESRGAN_x4plus_anime_6B.pth"
            ).strip(),
            model_url=os.getenv(
                "PARANNA_MODEL_URL",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            ).strip(),
            tile_size=int(os.getenv("PARANNA_TILE_SIZE", "256")),
            tile_pad=int(os.getenv("PARANNA_TILE_PAD", "10")),
            max_input_dimension=int(os.getenv("PARANNA_MAX_INPUT_DIMENSION", "2560")),
            max_output_dimension=int(os.getenv("PARANNA_MAX_OUTPUT_DIMENSION", "4096")),
            jpeg_quality=int(os.getenv("PARANNA_JPEG_QUALITY", "95")),
            max_image_bytes=int(os.getenv("PARANNA_MAX_IMAGE_BYTES", "15000000")),
        )

        tts_config = TtsConfig(
            base_url=os.getenv("TTS_BASE_URL", "http://localhost:8080").strip(),
            timeout_seconds=int(os.getenv("TTS_TIMEOUT_SECONDS", "60")),
            format=os.getenv("TTS_FORMAT", "ogg").strip(),
            error_message=os.getenv(
                "TTS_ERROR_MESSAGE", "Virhe puheen synteesissä."
            ).strip(),
            language=os.getenv("TTS_LANGUAGE", "fi").strip(),
            speed=float(os.getenv("TTS_SPEED", "1.0")),
            num_step=int(os.getenv("TTS_NUM_STEP", "32")),
            guidance_scale=float(os.getenv("TTS_GUIDANCE_SCALE", "2.0")),
            reencode_audio=(
                os.getenv("TTS_REENCODE_AUDIO", "true").strip().lower() == "true"
            ),
            ffmpeg_path=os.getenv("TTS_FFMPEG_PATH", "ffmpeg").strip(),
            audio_sample_rate=int(os.getenv("TTS_AUDIO_SAMPLE_RATE", "48000")),
            audio_channels=int(os.getenv("TTS_AUDIO_CHANNELS", "1")),
            audio_bitrate=os.getenv("TTS_AUDIO_BITRATE", "32k").strip(),
        )

        raw_telkkari_channels = os.getenv(
            "TELKKARI_DEFAULT_CHANNELS", "1,2,3,4,5,6"
        ).strip()
        telkkari_default_channels = tuple(
            int(c.strip())
            for c in raw_telkkari_channels.split(",")
            if c.strip().isdigit()
        )
        if not telkkari_default_channels:
            telkkari_default_channels = (1, 2, 3, 4, 5, 6)

        telkkari_config = TelkkariConfig(
            epg_url=os.getenv(
                "TELKKARI_EPG_URL",
                "https://epgshare01.online/epgshare01/epg_ripper_FI1.xml.gz",
            ).strip(),
            default_channels=telkkari_default_channels,
            cache_timeout_seconds=int(
                os.getenv("TELKKARI_CACHE_TIMEOUT_SECONDS", "1800")
            ),
            timeout_seconds=int(os.getenv("TELKKARI_TIMEOUT_SECONDS", "30")),
        )

        stt_beam_size = int(os.getenv("STT_BEAM_SIZE", "5"))
        if stt_beam_size < 1:
            raise ValueError("STT_BEAM_SIZE must be >= 1")

        stt_timeout_seconds = int(os.getenv("STT_TIMEOUT_SECONDS", "60"))
        if stt_timeout_seconds < 1:
            raise ValueError("STT_TIMEOUT_SECONDS must be >= 1")

        stt_config = SttConfig(
            base_url=os.getenv("STT_BASE_URL", "http://localhost:8001").strip(),
            timeout_seconds=stt_timeout_seconds,
            beam_size=stt_beam_size,
            vad_filter=(
                os.getenv("STT_VAD_FILTER", "true").strip().lower() == "true"
            ),
            word_timestamps=(
                os.getenv("STT_WORD_TIMESTAMPS", "false").strip().lower() == "true"
            ),
            initial_prompt=os.getenv("STT_INITIAL_PROMPT", "").strip(),
        )

        reminder_check_interval_seconds = int(
            os.getenv("REMINDER_CHECK_INTERVAL_SECONDS", "5")
        )
        if reminder_check_interval_seconds < 1:
            raise ValueError("REMINDER_CHECK_INTERVAL_SECONDS must be >= 1")

        reminder_max_per_chat = int(
            os.getenv("REMINDER_MAX_PER_CHAT", "50")
        )
        if reminder_max_per_chat < 1:
            raise ValueError("REMINDER_MAX_PER_CHAT must be >= 1")

        reminder_config = ReminderConfig(
            check_interval_seconds=reminder_check_interval_seconds,
            max_per_chat=reminder_max_per_chat,
        )

        tiivista_timeout_seconds = int(os.getenv("TIIVISTA_TIMEOUT_SECONDS", "60"))
        if tiivista_timeout_seconds < 1:
            raise ValueError("TIIVISTA_TIMEOUT_SECONDS must be >= 1")

        tiivista_max_web_page_bytes = int(
            os.getenv("TIIVISTA_MAX_WEB_PAGE_BYTES", "2000000")
        )
        if tiivista_max_web_page_bytes < 1:
            raise ValueError("TIIVISTA_MAX_WEB_PAGE_BYTES must be >= 1")

        tiivista_max_text_length = int(
            os.getenv("TIIVISTA_MAX_TEXT_LENGTH", "12000")
        )
        if tiivista_max_text_length < 1:
            raise ValueError("TIIVISTA_MAX_TEXT_LENGTH must be >= 1")

        tiivista_summary_num_predict = int(
            os.getenv("TIIVISTA_SUMMARY_NUM_PREDICT", "300")
        )
        if tiivista_summary_num_predict < 1:
            raise ValueError("TIIVISTA_SUMMARY_NUM_PREDICT must be >= 1")

        tiivista_caption_num_words = int(
            os.getenv("TIIVISTA_CAPTION_NUM_WORDS", "3")
        )
        if tiivista_caption_num_words < 1:
            raise ValueError("TIIVISTA_CAPTION_NUM_WORDS must be >= 1")

        default_tiivista_prompt = (
            "Tiivistä annettu teksti tiiviiksi, selkeäksi ja sujuvaksi suomenkieliseksi "
            "yhteenvedoksi. Vastaa pelkällä tiivistelmällä ilman alustuksia, sillä se luetaan ääneen."
        )

        tiivista_config = TiivistaConfig(
            timeout_seconds=tiivista_timeout_seconds,
            max_web_page_bytes=tiivista_max_web_page_bytes,
            max_text_length=tiivista_max_text_length,
            summary_num_predict=tiivista_summary_num_predict,
            user_agent=os.getenv(
                "TIIVISTA_USER_AGENT", "Mozilla/5.0 (compatible; P-iv-Botti/1.0)"
            ).strip(),
            system_prompt=os.getenv(
                "TIIVISTA_SYSTEM_PROMPT", default_tiivista_prompt
            ).strip(),
            caption_num_words=tiivista_caption_num_words,
        )

        return cls(
            telegram_bot_token=token,
            storage_dir=storage_dir,
            max_reply_length=int(os.getenv("MAX_REPLY_LENGTH", "5000")),
            weather=weather_config,
            cs2_rss=cs2_rss_config,
            naama=naama_config,
            ollama=ollama_config,
            twitch=twitch_config,
            paranna=paranna_config,
            tts=tts_config,
            telkkari=telkkari_config,
            stt=stt_config,
            reminder=reminder_config,
            tiivista=tiivista_config,
        )



