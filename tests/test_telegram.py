"""Tests for Telegram bot command handlers."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from bot.telegram import TelegramMCPBot


@pytest.fixture
def bot():
    agent = MagicMock()
    return TelegramMCPBot(token="fake:token", bot_username="@testbot", openai_agent=agent)


@pytest.fixture
def respond_bot():
    """Bot with a mock agent that has a controllable run method."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value="hello")
    bot = TelegramMCPBot(token="fake:token", bot_username="@testbot", openai_agent=agent)
    bot.rate_limiter = MagicMock()
    bot.rate_limiter.is_allowed.return_value = True
    return bot


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


@pytest.mark.anyio
async def test_respond_sends_typing_action(respond_bot):
    update = _make_update(chat_id=42)
    update.message.text = "hi"
    update.message.from_user.id = 1
    update.message.chat.send_action = AsyncMock()

    await respond_bot._respond(update)

    update.message.chat.send_action.assert_called()
    from telegram.constants import ChatAction

    call_args = update.message.chat.send_action.call_args_list[0]
    assert call_args[0][0] == ChatAction.TYPING


@pytest.mark.anyio
async def test_respond_typing_continues_during_slow_agent(respond_bot):
    """Typing action should be sent multiple times for slow responses."""
    respond_bot.TYPING_INTERVAL_SECONDS = 0.05
    call_count = 0

    async def slow_run(chat_id, message):
        nonlocal call_count
        # Wait long enough for at least one more typing action from the loop
        while call_count < 2:
            await asyncio.sleep(0.02)
        return "done"

    respond_bot.agent.run = slow_run

    update = _make_update(chat_id=42)
    update.message.text = "hi"
    update.message.from_user.id = 1

    original_send_action = AsyncMock()

    async def counting_send_action(action):
        nonlocal call_count
        call_count += 1
        return await original_send_action(action)

    update.message.chat.send_action = counting_send_action

    await respond_bot._respond(update)

    # 1 immediate + at least 1 from the loop
    assert call_count >= 2


@pytest.mark.anyio
async def test_respond_typing_stops_after_reply(respond_bot):
    """Typing task should be cancelled after agent responds."""
    update = _make_update(chat_id=42)
    update.message.text = "hi"
    update.message.from_user.id = 1
    update.message.chat.send_action = AsyncMock()

    await respond_bot._respond(update)

    # After _respond returns, no more typing actions should fire
    send_count = update.message.chat.send_action.call_count
    await asyncio.sleep(0.15)
    assert update.message.chat.send_action.call_count == send_count
