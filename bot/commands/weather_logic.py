from __future__ import annotations


def parse_weather_camera_location(text: str | None) -> tuple[bool, str | None]:
    if text is None:
        return False, None

    stripped = text.strip()
    lowered = stripped.casefold()

    for prefix in ("!sääkuva", "!saakuva"):
        if not lowered.startswith(prefix):
            continue

        rest = stripped[len(prefix) :]
        if rest and not rest[0].isspace():
            continue

        location = rest.strip()
        return True, location or None

    return False, None
