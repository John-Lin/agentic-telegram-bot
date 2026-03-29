"""Tests for Telegram bot command handlers."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from bot.telegram import TelegramMCPBot


@pytest.fixture
def bot():
    agent = MagicMock()
    return TelegramMCPBot(token="fake:token", bot_username="@testbot", openai_agent=agent)


def _make_update(chat_id: int) -> MagicMock:
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock()
    return update


@pytest.mark.anyio
async def test_chatid_command_replies_with_chat_id(bot):
    update = _make_update(chat_id=-100123456789)

    await bot.chatid_command(update, MagicMock())

    update.message.reply_text.assert_called_once_with("-100123456789")


@pytest.mark.anyio
async def test_chatid_command_works_in_private_chat(bot):
    update = _make_update(chat_id=42)

    await bot.chatid_command(update, MagicMock())

    update.message.reply_text.assert_called_once_with("42")


@pytest.mark.anyio
async def test_chatid_command_ignores_missing_message(bot):
    update = MagicMock()
    update.message = None

    await bot.chatid_command(update, MagicMock())
    # Should not raise
