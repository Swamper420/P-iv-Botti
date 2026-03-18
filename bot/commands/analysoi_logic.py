from __future__ import annotations

import base64

_ANALYSOI_PROMPT = (
    "Olet mestarianalyytikko ja taidekriitikko, jolla on äärimmäisen dramaattinen, "
    "yliampuva ja sarkastinen tyyli. Analysoi tämä kuva mahdollisimman hauskalla "
    "tavalla. Tee jokaisesta yksityiskohdasta absurdi johtopäätös. "
    "Ole liioitteleva, ivallinen ja kekseliäs. "
    "Aloita antamalla analyysillesi mahtipontinen nimi. "
    "Käytä suomea."
)


def build_analysoi_prompt() -> str:
    """Return the system prompt used when analysing an image."""
    return _ANALYSOI_PROMPT


def encode_image_base64(image_bytes: bytes) -> str:
    """Return *image_bytes* encoded as a base-64 ASCII string."""
    return base64.b64encode(image_bytes).decode("ascii")
