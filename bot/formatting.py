"""Convert standard Markdown to Telegram-compatible HTML."""

from __future__ import annotations

from typing import Any

import mistune
from mistune_telegram import TelegramHTMLRenderer


class _OrderedListRenderer(TelegramHTMLRenderer):
    """Render ordered lists as bullets for Telegram readability."""

    def list(self, text: str, ordered: bool, **attrs: Any) -> str:
        return text + "\n"


_md = mistune.create_markdown(renderer=_OrderedListRenderer())


def markdown_to_telegram_html(text: str) -> str:
    """Convert standard Markdown to Telegram-supported HTML.

    Returns the converted text with trailing whitespace stripped.
    """
    result = _md(text)
    assert isinstance(result, str)
    return result.strip()
