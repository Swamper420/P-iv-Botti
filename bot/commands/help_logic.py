from __future__ import annotations

from collections.abc import Sequence


def build_help_reply(command_usages: Sequence[str]) -> str:
    commands = sorted(set(command_usages), key=str.casefold)
    lines = ["Käytettävissä olevat komennot:"]
    lines.extend(f"- `{command}`" for command in commands)
    return "\n".join(lines)
