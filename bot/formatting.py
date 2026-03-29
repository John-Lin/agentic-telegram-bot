"""Convert standard Markdown to Telegram-compatible HTML."""

from __future__ import annotations

import mistune
from mistune_telegram import TelegramHTMLRenderer

_md = mistune.create_markdown(renderer=TelegramHTMLRenderer())


def markdown_to_telegram_html(text: str) -> str:
    """Convert standard Markdown to Telegram-supported HTML.

    Returns the converted text with trailing whitespace stripped.
    """
    return _md(text).strip()
