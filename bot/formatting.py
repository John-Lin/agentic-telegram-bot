"""Convert standard Markdown to Telegram-compatible HTML."""

from __future__ import annotations

from typing import Any

import mistune
from mistune_telegram import TelegramHTMLRenderer


class _OrderedListRenderer(TelegramHTMLRenderer):
    """Preserve ordered list numbering instead of converting to dashes."""

    def list(self, text: str, ordered: bool, **attrs: Any) -> str:
        if ordered:
            lines = text.split("\n")
            result: list[str] = []
            counter = 0
            for line in lines:
                if line.startswith("- "):
                    counter += 1
                    result.append(f"{counter}. {line[2:]}")
                else:
                    result.append(line)
            text = "\n".join(result)
        return text + "\n"


_md = mistune.create_markdown(renderer=_OrderedListRenderer())


def markdown_to_telegram_html(text: str) -> str:
    """Convert standard Markdown to Telegram-supported HTML.

    Returns the converted text with trailing whitespace stripped.
    """
    return _md(text).strip()
