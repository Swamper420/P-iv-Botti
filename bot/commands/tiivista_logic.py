from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from html.parser import HTMLParser
from io import BytesIO
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageOps

from bot.commands.aih_logic import stream_ollama_completion

LOGGER = logging.getLogger(__name__)


def parse_tiivista_command(text: str) -> tuple[bool, str | None, str]:
    """Parse a !tiivistä command string into (is_match, voice, content_or_url).

    Examples:
        !tiivistä https://example.com -> (True, None, "https://example.com")
        !tiivisto Matti https://example.com -> (True, "Matti", "https://example.com")
        !tiivistä -> (True, None, "")
    """
    if not text:
        return False, None, ""

    match = re.match(
        r"(?i)^\s*!?tiivist(?:ä|a)(?::?\s*|\s+)?(.*)$",
        text,
        re.DOTALL,
    )
    if not match:
        return False, None, ""

    rest = (match.group(1) or "").strip()
    if not rest:
        return True, None, ""

    # Check if rest starts with a URL
    if re.match(r"(?i)^https?://", rest) or re.match(r"(?i)^www\.", rest):
        return True, None, rest

    # Split into first word and remaining text
    parts = rest.split(None, 1)
    if len(parts) == 1:
        first_token = parts[0]
        # If single token looks like a URL or contains URL characters/slashes, treat as content
        if "://" in first_token or "." in first_token or "/" in first_token:
            return True, None, first_token
        # Otherwise, treat as content if no voice separation is needed
        return True, None, first_token

    first_word, remaining = parts[0], parts[1]
    # If first word doesn't look like a URL and remaining contains a URL or text, first word is voice
    if not re.match(r"(?i)^https?://", first_word) and not ("/" in first_word and "." in first_word):
        return True, first_word.strip(), remaining.strip()

    return True, None, rest


def extract_urls(text: str) -> list[str]:
    """Extract all HTTP/HTTPS URLs from the input text."""
    if not text:
        return []
    raw_urls = re.findall(r"https?://[^\s<>\"']+", text)
    cleaned_urls: list[str] = []
    for url in raw_urls:
        cleaned = url.rstrip("!.,;:?)]}>")
        if cleaned:
            cleaned_urls.append(cleaned)
    return cleaned_urls



class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []
        self._ignored_tags = {
            "script",
            "style",
            "noscript",
            "header",
            "footer",
            "nav",
            "svg",
            "head",
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._ignored_tags:
            self._skip_depth += 1
        elif tag.lower() in {"p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._ignored_tags:
            if self._skip_depth > 0:
                self._skip_depth -= 1
        elif tag.lower() in {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            cleaned = data.strip()
            if cleaned:
                self._chunks.append(f" {cleaned} ")

    def get_text(self) -> str:
        raw_text = "".join(self._chunks)
        # Normalize whitespace and blank lines
        lines = [line.strip() for line in raw_text.splitlines()]
        cleaned_lines = [line for line in lines if line]
        return "\n".join(cleaned_lines)


def extract_text_from_html(html_str: str) -> str:
    """Clean HTML and return extracted readable body text."""
    if not html_str or not html_str.strip():
        return ""

    parser = _HTMLTextExtractor()
    try:
        parser.feed(html_str)
        parser.close()
        return parser.get_text()
    except Exception as err:
        LOGGER.warning("HTML parsing failed: %s", err)
        # Fallback to basic tag removal via regex
        cleaned = re.sub(r"<[^>]+>", " ", html_str)
        return " ".join(cleaned.split())


async def fetch_webpage_text(
    url: str,
    user_agent: str = "Mozilla/5.0 (compatible; P-iv-Botti/1.0)",
    timeout_seconds: int = 60,
    max_bytes: int = 2_000_000,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Fetch webpage content over HTTP and extract clean plain text."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {"User-Agent": user_agent}
    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True)
        close_client = True

    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        content_bytes = response.content
        if len(content_bytes) > max_bytes:
            content_bytes = content_bytes[:max_bytes]

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type or "<html" in content_bytes[:500].decode("utf-8", errors="ignore").lower():
            text = extract_text_from_html(response.text)
        else:
            text = response.text

        return text
    finally:
        if close_client:
            await client.aclose()


async def summarize_text_with_ollama(
    base_url: str,
    model: str,
    text: str,
    num_predict: int = 300,
    num_ctx: int = 2048,
    timeout_seconds: int = 120,
    system_prompt: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Summarize text using Ollama completion stream."""
    if not text or not text.strip():
        return ""

    chunks: list[str] = []
    async for chunk in stream_ollama_completion(
        base_url=base_url,
        model=model,
        prompt=text,
        num_predict=num_predict,
        num_ctx=num_ctx,
        timeout_seconds=timeout_seconds,
        system_prompt=system_prompt,
        client=client,
    ):
        chunks.append(chunk)

    return "".join(chunks).strip()


def create_3_word_caption(summary_text: str, max_words: int = 3) -> str:
    """Extract a concise 3-word summary caption from the summary text."""
    if not summary_text:
        return ""

    # Clean punctuation and split into words
    words = re.findall(r"\w+", summary_text)
    if not words:
        return ""

    selected = words[:max_words]
    caption = " ".join(selected)
    return caption.capitalize()


COCO_FINNISH_NAMES: dict[str, tuple[str, str]] = {
    "person": ("henkilö", "henkilöä"),
    "bicycle": ("polkupyörä", "polkupyörää"),
    "car": ("auto", "autoa"),
    "motorcycle": ("moottoripyörä", "moottoripyörää"),
    "airplane": ("lentokone", "lentokonetta"),
    "bus": ("bussi", "bussia"),
    "train": ("juna", "junaa"),
    "truck": ("kuorma-auto", "kuorma-autoa"),
    "boat": ("vene", "venettä"),
    "traffic light": ("liikennevalo", "liikennevaloa"),
    "fire hydrant": ("paloposti", "palopostia"),
    "stop sign": ("pysähtymismerkki", "pysähtymismerkkiä"),
    "bench": ("penkki", "penkkiä"),
    "bird": ("lintu", "lintua"),
    "cat": ("kissa", "kissaa"),
    "dog": ("koira", "koiraa"),
    "horse": ("hevonen", "hevosta"),
    "sheep": ("lammas", "lammasta"),
    "cow": ("lehmä", "lehmää"),
    "elephant": ("norsu", "norsua"),
    "bear": ("karhu", "karhua"),
    "zebra": ("sebra", "sebraa"),
    "giraffe": ("kirahvi", "kirahvia"),
    "backpack": ("reppu", "reppua"),
    "umbrella": ("sateenvarjo", "sateenvarjoa"),
    "handbag": ("käsilaukku", "käsilaukkua"),
    "tie": ("solmio", "solmiota"),
    "suitcase": ("matkalaukku", "matkalaukkua"),
    "frisbee": ("frisbee", "frisbeetä"),
    "skis": ("sukset", "suksia"),
    "snowboard": ("lumilauta", "lumilautaa"),
    "sports ball": ("pallo", "palloa"),
    "kite": ("leija", "leijaa"),
    "baseball bat": ("pesäpallomaila", "pesäpallomailaa"),
    "baseball glove": ("pesäpallohanska", "pesäpallohanskaa"),
    "skateboard": ("rullalauta", "rullalautaa"),
    "surfboard": ("surffilauta", "surffilautaa"),
    "tennis racket": ("tennismaila", "tennismailaa"),
    "bottle": ("pullo", "pulloa"),
    "wine glass": ("viinilasi", "viinilasia"),
    "cup": ("muki", "mukia"),
    "fork": ("haarukka", "haarukkaa"),
    "knife": ("veitsi", "veistä"),
    "spoon": ("lusikka", "lusikkaa"),
    "bowl": ("kulho", "kulhoa"),
    "banana": ("banaani", "banaania"),
    "apple": ("omena", "omenaa"),
    "sandwich": ("voileipä", "voileipää"),
    "orange": ("appelsiini", "appelsiinia"),
    "broccoli": ("parsakaali", "parsakaalia"),
    "carrot": ("porkkana", "porkkanaa"),
    "hot dog": ("nakki", "nakkia"),
    "pizza": ("pitsa", "pitsaa"),
    "donut": ("donitsi", "donitsia"),
    "cake": ("kakku", "kakkua"),
    "chair": ("tuoli", "tuolia"),
    "couch": ("sohva", "sohvaa"),
    "potted plant": ("ruukkukasvi", "ruukkukasvia"),
    "bed": ("sänky", "sänkyä"),
    "dining table": ("ruokapöytä", "ruokapöytää"),
    "toilet": ("vessa", "vessaa"),
    "tv": ("televisio", "televisiota"),
    "laptop": ("kannettava tietokone", "kannettavaa tietokonetta"),
    "mouse": ("hiiri", "hiirtä"),
    "remote": ("kaukosäädin", "kaukosäädintä"),
    "keyboard": ("näppäimistö", "näppäimistöä"),
    "cell phone": ("matkapuhelin", "matkapuhelinta"),
    "microwave": ("mikroaaltouuni", "mikroaaltouunia"),
    "oven": ("uuni", "uunia"),
    "toaster": ("paahdin", "paahdinta"),
    "sink": ("pesuallas", "pesuallasta"),
    "refrigerator": ("jääkaappi", "jääkaappia"),
    "book": ("kirja", "kirjaa"),
    "clock": ("kello", "kelloa"),
    "vase": ("maljakko", "maljakkoa"),
    "scissors": ("sakset", "saksia"),
    "teddy bear": ("nalle", "nallea"),
    "hair drier": ("hiustenkuivain", "hiustenkuivainta"),
    "toothbrush": ("hammasharja", "hammasharjaa"),
}

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_yolo_model(model_name: str, model_loader: Any = None) -> Any:
    if model_loader is not None:
        return model_loader(model_name)
    with _MODEL_CACHE_LOCK:
        if model_name not in _MODEL_CACHE:
            from ultralytics import YOLO

            _MODEL_CACHE[model_name] = YOLO(model_name)
        return _MODEL_CACHE[model_name]


def recognize_objects_with_yolo(
    image_bytes: bytes,
    *,
    model_name: str = "yolo26n.pt",
    confidence_threshold: float = 0.25,
    model_loader: Any = None,
) -> str:
    """Recognize objects in an image using YOLO model and return a Finnish text description."""
    if not image_bytes:
        return ""

    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            image_rgb = np.asarray(
                ImageOps.exif_transpose(source_image).convert("RGB")
            )
    except Exception as err:
        LOGGER.warning("Failed to open image for YOLO object recognition: %s", err)
        return ""

    try:
        model = _get_yolo_model(model_name, model_loader)
        results = model.predict(
            source=image_rgb, conf=confidence_threshold, device="cpu", verbose=False
        )
    except Exception as err:
        LOGGER.exception("YOLO object recognition inference failed: %s", err)
        return ""

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return "Kuvassa ei tunnistettu kohteita."

    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy()
    names = getattr(results[0], "names", {})

    counts: Counter[str] = Counter()
    for cls_id in classes:
        cls_int = int(cls_id)
        if isinstance(names, dict):
            raw_name = names.get(cls_int, f"class_{cls_int}")
        elif isinstance(names, (list, tuple)) and 0 <= cls_int < len(names):
            raw_name = names[cls_int]
        else:
            raw_name = f"class_{cls_int}"

        counts[str(raw_name).lower()] += 1

    formatted_items: list[str] = []
    for raw_name, count in counts.items():
        if raw_name in COCO_FINNISH_NAMES:
            sing, plur = COCO_FINNISH_NAMES[raw_name]
            noun = sing if count == 1 else plur
        else:
            noun = raw_name
        formatted_items.append(f"{count} {noun}")

    return "Kuvasta tunnistettiin: " + ", ".join(formatted_items) + "."
