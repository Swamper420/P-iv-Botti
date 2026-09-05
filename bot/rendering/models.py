from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BadgeColor(str, Enum):
    GREEN = "green"
    RED = "red"
    YELLOW = "yellow"
    BLUE = "blue"
    PURPLE = "purple"
    GRAY = "gray"


@dataclass(frozen=True)
class BadgeStyle:
    bg_color: str
    fg_color: str
    border_color: str


BADGE_STYLES: dict[str, BadgeStyle] = {
    BadgeColor.GREEN.value: BadgeStyle(bg_color="#14532d", fg_color="#86efac", border_color="#22c55e"),
    BadgeColor.RED.value: BadgeStyle(bg_color="#7f1d1d", fg_color="#fca5a5", border_color="#ef4444"),
    BadgeColor.YELLOW.value: BadgeStyle(bg_color="#713f12", fg_color="#fde047", border_color="#eab308"),
    BadgeColor.BLUE.value: BadgeStyle(bg_color="#1e3a8a", fg_color="#93c5fd", border_color="#3b82f6"),
    BadgeColor.PURPLE.value: BadgeStyle(bg_color="#581c87", fg_color="#d8b4fe", border_color="#a855f7"),
    BadgeColor.GRAY.value: BadgeStyle(bg_color="#27272a", fg_color="#d4d4d8", border_color="#52525b"),
}


@dataclass
class Badge:
    text: str
    color: BadgeColor | str = BadgeColor.GREEN
    bg_color: str | None = None
    fg_color: str | None = None
    border_color: str | None = None

    def get_style(self) -> BadgeStyle:
        color_key = self.color.value if isinstance(self.color, BadgeColor) else str(self.color).lower()
        base = BADGE_STYLES.get(color_key, BADGE_STYLES[BadgeColor.GRAY.value])
        return BadgeStyle(
            bg_color=self.bg_color or base.bg_color,
            fg_color=self.fg_color or base.fg_color,
            border_color=self.border_color or base.border_color,
        )


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


DARK_THEME = Theme()


class CardElement:
    """Base class for renderable card elements."""
    pass


@dataclass
class TextElement(CardElement):
    text: str
    is_code: bool = False
    muted: bool = False
    bold: bool = False
    color: str | None = None
    max_lines: int | None = None
    overflow: str = "wrap"  # "wrap", "ellipsis", "clip"


@dataclass
class KeyValuesElement(CardElement):
    items: list[tuple[str, str | Badge]] = field(default_factory=list)
    columns: int = 2
    overflow: str = "ellipsis"  # "ellipsis", "clip", "none"


@dataclass
class TableElement(CardElement):
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    alignments: list[str] | None = None  # "left", "center", "right"
    col_widths: list[int | float] | None = None  # explicit pixels or proportional weights
    max_rows: int | None = None
    overflow: str = "ellipsis"  # "ellipsis", "clip", "none"
    primary_col: int | None = 0  # Column to highlight/make bigger and bold (default 0)
    bold_cols: list[int] | None = None  # Specific columns to render in bold


@dataclass
class DividerElement(CardElement):
    line: bool = True
    margin_top: int = 4
    margin_bottom: int = 4


@dataclass
class ProgressBarElement(CardElement):
    label: str
    value: float
    max_value: float = 100.0
    unit: str = "%"
    color: str | None = None


@dataclass
class CodeBlockElement(CardElement):
    code: str
    language: str | None = None
    max_lines: int | None = None
    overflow: str = "ellipsis"  # "ellipsis", "wrap", "clip"


@dataclass
class ImageElement(CardElement):
    image_bytes: bytes = b""
    caption: str | None = None
    max_height: int | None = 420


@dataclass
class Card:
    title: str
    subtitle: str | None = None
    badge: Badge | None = None
    accent_color: str | None = None
    footer: str | None = None
    elements: list[CardElement] = field(default_factory=list)
    max_height: int | None = None

    def set_badge(
        self,
        text: str,
        color: BadgeColor | str = BadgeColor.GREEN,
        bg_color: str | None = None,
        fg_color: str | None = None,
        border_color: str | None = None,
    ) -> Card:
        self.badge = Badge(
            text=text,
            color=color,
            bg_color=bg_color,
            fg_color=fg_color,
            border_color=border_color,
        )
        return self

    def set_footer(self, footer: str) -> Card:
        self.footer = footer
        return self

    def add_text(
        self,
        text: str,
        is_code: bool = False,
        muted: bool = False,
        bold: bool = False,
        color: str | None = None,
        max_lines: int | None = None,
        overflow: str = "wrap",
    ) -> Card:
        self.elements.append(
            TextElement(
                text=text,
                is_code=is_code,
                muted=muted,
                bold=bold,
                color=color,
                max_lines=max_lines,
                overflow=overflow,
            )
        )
        return self

    def add_key_value(self, key: str, value: str | Badge) -> Card:
        if self.elements and isinstance(self.elements[-1], KeyValuesElement):
            self.elements[-1].items.append((key, value))
        else:
            self.elements.append(KeyValuesElement(items=[(key, value)], columns=2))
        return self

    def add_key_values(
        self,
        items: list[tuple[str, str | Badge]],
        columns: int = 2,
        overflow: str = "ellipsis",
    ) -> Card:
        self.elements.append(
            KeyValuesElement(items=list(items), columns=columns, overflow=overflow)
        )
        return self

    def add_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        alignments: list[str] | None = None,
        col_widths: list[int | float] | None = None,
        max_rows: int | None = None,
        overflow: str = "ellipsis",
        primary_col: int | None = 0,
        bold_cols: list[int] | None = None,
    ) -> Card:
        self.elements.append(
            TableElement(
                headers=headers,
                rows=rows,
                alignments=alignments,
                col_widths=col_widths,
                max_rows=max_rows,
                overflow=overflow,
                primary_col=primary_col,
                bold_cols=bold_cols,
            )
        )
        return self

    def add_divider(
        self, line: bool = True, margin_top: int = 4, margin_bottom: int = 4
    ) -> Card:
        self.elements.append(
            DividerElement(line=line, margin_top=margin_top, margin_bottom=margin_bottom)
        )
        return self

    def add_progress_bar(
        self,
        label: str,
        value: float,
        max_value: float = 100.0,
        unit: str = "%",
        color: str | None = None,
    ) -> Card:
        self.elements.append(
            ProgressBarElement(
                label=label,
                value=value,
                max_value=max_value,
                unit=unit,
                color=color,
            )
        )
        return self

    def add_code_block(
        self,
        code: str,
        language: str | None = None,
        max_lines: int | None = None,
        overflow: str = "ellipsis",
    ) -> Card:
        self.elements.append(
            CodeBlockElement(
                code=code,
                language=language,
                max_lines=max_lines,
                overflow=overflow,
            )
        )
        return self

    def add_image(
        self,
        image_bytes: bytes,
        caption: str | None = None,
        max_height: int | None = 420,
    ) -> Card:
        self.elements.append(
            ImageElement(
                image_bytes=image_bytes,
                caption=caption,
                max_height=max_height,
            )
        )
        return self
