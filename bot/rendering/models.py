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
    BadgeColor.GREEN.value: BadgeStyle(bg_color="#0f3d24", fg_color="#6ee7b7", border_color="#22c55e"),
    BadgeColor.RED.value: BadgeStyle(bg_color="#4c1319", fg_color="#fda4af", border_color="#ef4444"),
    BadgeColor.YELLOW.value: BadgeStyle(bg_color="#452b0b", fg_color="#fde047", border_color="#eab308"),
    BadgeColor.BLUE.value: BadgeStyle(bg_color="#132c57", fg_color="#7dd3fc", border_color="#38bdf8"),
    BadgeColor.PURPLE.value: BadgeStyle(bg_color="#38165a", fg_color="#d8b4fe", border_color="#a855f7"),
    BadgeColor.GRAY.value: BadgeStyle(bg_color="#1f2430", fg_color="#cbd5e1", border_color="#475569"),
    "cyan": BadgeStyle(bg_color="#0d3545", fg_color="#67e8f9", border_color="#06b6d4"),
    "orange": BadgeStyle(bg_color="#4a200d", fg_color="#fdba74", border_color="#f97316"),
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
    canvas_bg: str = "#0e1015"
    card_bg: str = "#151821"
    panel_bg: str = "#1c202c"
    panel_recessed: str = "#10131a"
    border_color: str = "#2a3040"
    border_bright: str = "#3b4358"
    accent_color: str = "#38bdf8"
    text_primary: str = "#f4f6fb"
    text_secondary: str = "#a5adc0"
    text_muted: str = "#6b7388"
    code_bg: str = "#0f1218"
    font_family: str = "auto"
    card_width: int = 800
    padding: int = 22
    element_spacing: int = 14


DARK_THEME = Theme()


def theme_from_config(rendering_config: Any) -> Theme:
    """Builds a Theme from a RenderingConfig-like object."""
    if rendering_config is None:
        return DARK_THEME
    card_width = getattr(rendering_config, "card_width", DARK_THEME.card_width)
    font_family = getattr(rendering_config, "font_family", DARK_THEME.font_family)
    try:
        card_width = int(card_width)
    except (TypeError, ValueError):
        card_width = DARK_THEME.card_width
    card_width = max(300, min(4000, card_width))
    return Theme(
        card_width=card_width,
        font_family=str(font_family or "auto"),
    )


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
    rows: list[list[str | Badge]] = field(default_factory=list)
    alignments: list[str] | None = None  # "left", "center", "right"
    col_widths: list[int | float] | None = None  # explicit pixels or proportional weights
    max_rows: int | None = None
    overflow: str = "ellipsis"  # "ellipsis", "clip", "none"
    primary_col: int | None = 0  # Column to highlight/make bigger and bold (default 0)
    bold_cols: list[int] | None = None  # Specific columns to render in bold
    mono_cols: list[int] | None = None  # Specific columns to render in monospace


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
    sublabel: str | None = None
    show_percentage: bool = True


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
class ImageGridElement(CardElement):
    images: list[bytes] = field(default_factory=list)
    labels: list[str] | None = None
    caption: str | None = None
    max_cell_height: int | None = 280
    gap: int = 8


@dataclass
class SectionHeaderElement(CardElement):
    title: str
    right_text: str | None = None
    accent_color: str | None = None


@dataclass
class HeroElement(CardElement):
    value: str
    label: str
    subtext: str | None = None
    accent_color: str | None = None
    icon: str | None = None
    right_label: str | None = None
    right_value: str | None = None


@dataclass(frozen=True)
class StatTile:
    label: str
    value: str
    subtext: str | None = None
    accent_color: str | None = None


@dataclass
class StatGridElement(CardElement):
    items: list[StatTile] = field(default_factory=list)
    columns: int = 3


@dataclass(frozen=True)
class TimelineItem:
    label: str
    title: str
    start_frac: float
    end_frac: float
    time_text: str
    is_live: bool = False
    color: str | None = None
    progress: float | None = None


@dataclass
class TimelineElement(CardElement):
    items: list[TimelineItem] = field(default_factory=list)
    start_label: str = ""
    end_label: str = ""
    now_fraction: float | None = None
    now_label: str = "NYT"


@dataclass(frozen=True)
class BarItem:
    label: str
    value: float
    value_text: str
    subtext: str | None = None
    color: str | None = None
    badge: Badge | None = None


@dataclass
class BarListElement(CardElement):
    items: list[BarItem] = field(default_factory=list)
    max_value: float | None = None
    show_rank: bool = False


@dataclass
class RangeBarElement(CardElement):
    label: str
    value_frac: float
    start_label: str
    end_label: str
    current_label: str | None = None
    color: str | None = None
    track_color: str | None = None


@dataclass
class CalloutElement(CardElement):
    text: str
    title: str | None = None
    color: str | None = None
    tone: str = "info"  # "info", "live", "warning", "error", "success"


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
        rows: list[list[str | Badge]],
        alignments: list[str] | None = None,
        col_widths: list[int | float] | None = None,
        max_rows: int | None = None,
        overflow: str = "ellipsis",
        primary_col: int | None = 0,
        bold_cols: list[int] | None = None,
        mono_cols: list[int] | None = None,
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
                mono_cols=mono_cols,
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
        sublabel: str | None = None,
        show_percentage: bool = True,
    ) -> Card:
        self.elements.append(
            ProgressBarElement(
                label=label,
                value=value,
                max_value=max_value,
                unit=unit,
                color=color,
                sublabel=sublabel,
                show_percentage=show_percentage,
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

    def add_image_grid(
        self,
        images: list[bytes],
        labels: list[str] | None = None,
        caption: str | None = None,
        max_cell_height: int | None = 280,
        gap: int = 8,
    ) -> Card:
        self.elements.append(
            ImageGridElement(
                images=list(images),
                labels=list(labels) if labels is not None else None,
                caption=caption,
                max_cell_height=max_cell_height,
                gap=gap,
            )
        )
        return self

    def add_section(
        self,
        title: str,
        right_text: str | None = None,
        accent_color: str | None = None,
    ) -> Card:
        self.elements.append(
            SectionHeaderElement(
                title=title,
                right_text=right_text,
                accent_color=accent_color,
            )
        )
        return self

    def add_hero(
        self,
        value: str,
        label: str,
        subtext: str | None = None,
        accent_color: str | None = None,
        icon: str | None = None,
        right_label: str | None = None,
        right_value: str | None = None,
    ) -> Card:
        self.elements.append(
            HeroElement(
                value=value,
                label=label,
                subtext=subtext,
                accent_color=accent_color,
                icon=icon,
                right_label=right_label,
                right_value=right_value,
            )
        )
        return self

    def add_stat_grid(
        self,
        items: list[StatTile],
        columns: int = 3,
    ) -> Card:
        self.elements.append(
            StatGridElement(items=list(items), columns=columns)
        )
        return self

    def add_timeline(
        self,
        items: list[TimelineItem],
        start_label: str = "",
        end_label: str = "",
        now_fraction: float | None = None,
        now_label: str = "NYT",
    ) -> Card:
        self.elements.append(
            TimelineElement(
                items=list(items),
                start_label=start_label,
                end_label=end_label,
                now_fraction=now_fraction,
                now_label=now_label,
            )
        )
        return self

    def add_bar_list(
        self,
        items: list[BarItem],
        max_value: float | None = None,
        show_rank: bool = False,
    ) -> Card:
        self.elements.append(
            BarListElement(
                items=list(items),
                max_value=max_value,
                show_rank=show_rank,
            )
        )
        return self

    def add_range_bar(
        self,
        label: str,
        value_frac: float,
        start_label: str,
        end_label: str,
        current_label: str | None = None,
        color: str | None = None,
        track_color: str | None = None,
    ) -> Card:
        self.elements.append(
            RangeBarElement(
                label=label,
                value_frac=value_frac,
                start_label=start_label,
                end_label=end_label,
                current_label=current_label,
                color=color,
                track_color=track_color,
            )
        )
        return self

    def add_callout(
        self,
        text: str,
        title: str | None = None,
        color: str | None = None,
        tone: str = "info",
    ) -> Card:
        self.elements.append(
            CalloutElement(
                text=text,
                title=title,
                color=color,
                tone=tone,
            )
        )
        return self
