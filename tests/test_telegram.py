"""Tests for Telegram bot command handlers."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

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


def _make_group_update(chat_id: int, user_id: int, text: str, mention: str | None = None) -> MagicMock:
    """Create a mock group message update, optionally with a @mention entity."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.text = text
    update.message.from_user.id = user_id
    update.message.reply_text = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    if mention:
        entity = MagicMock()
        entity.type = "mention"
        entity.offset = text.index(mention)
        entity.length = len(mention)
        update.message.entities = [entity]
    else:
        update.message.entities = []
    return update


class TestHandleGroupAllowFrom:
    """Test that allowFrom bypasses requireMention."""

    @pytest.fixture
    def group_bot(self):
        agent = MagicMock()
        agent.run = AsyncMock(return_value="response")
        bot = TelegramMCPBot(token="fake:token", bot_username="@testbot", openai_agent=agent)
        bot.rate_limiter = MagicMock()
        bot.rate_limiter.is_allowed.return_value = True
        bot.TYPING_INTERVAL_SECONDS = 100  # prevent loop from firing
        return bot

    @pytest.mark.anyio
    async def test_allow_from_user_triggers_without_mention(self, group_bot):
        """User in allowFrom should trigger bot even without @mention."""
        config = {"requireMention": True, "allowFrom": ["111"]}
        update = _make_group_update(chat_id=-1001234, user_id=111, text="hello")

        with (
            patch("bot.telegram.get_group_config", return_value=config),
            patch("bot.telegram.get_dm_policy", return_value="pairing"),
        ):
            await group_bot.handle_group(update, MagicMock())

        group_bot.agent.run.assert_called_once()

    @pytest.mark.anyio
    async def test_non_allow_from_user_needs_mention(self, group_bot):
        """User NOT in allowFrom should be ignored without @mention when requireMention=True."""
        config = {"requireMention": True, "allowFrom": ["111"]}
        update = _make_group_update(chat_id=-1001234, user_id=222, text="hello")

        with (
            patch("bot.telegram.get_group_config", return_value=config),
            patch("bot.telegram.get_dm_policy", return_value="pairing"),
        ):
            await group_bot.handle_group(update, MagicMock())

        group_bot.agent.run.assert_not_called()

    @pytest.mark.anyio
    async def test_empty_allow_from_requires_mention(self, group_bot):
        """Empty allowFrom with requireMention=True should require @mention."""
        config = {"requireMention": True, "allowFrom": []}
        update = _make_group_update(chat_id=-1001234, user_id=111, text="hello")

        with (
            patch("bot.telegram.get_group_config", return_value=config),
            patch("bot.telegram.get_dm_policy", return_value="pairing"),
        ):
            await group_bot.handle_group(update, MagicMock())

        group_bot.agent.run.assert_not_called()

    @pytest.mark.anyio
    async def test_empty_allow_from_with_mention_triggers(self, group_bot):
        """Empty allowFrom with requireMention=True + mention should trigger."""
        config = {"requireMention": True, "allowFrom": []}
        update = _make_group_update(chat_id=-1001234, user_id=111, text="hey @testbot", mention="@testbot")

        with (
            patch("bot.telegram.get_group_config", return_value=config),
            patch("bot.telegram.get_dm_policy", return_value="pairing"),
        ):
            await group_bot.handle_group(update, MagicMock())

        group_bot.agent.run.assert_called_once()

    @pytest.mark.anyio
    async def test_empty_allow_from_no_mention_required(self, group_bot):
        """Empty allowFrom with requireMention=False should trigger for anyone."""
        config = {"requireMention": False, "allowFrom": []}
        update = _make_group_update(chat_id=-1001234, user_id=999, text="hello")

        with (
            patch("bot.telegram.get_group_config", return_value=config),
            patch("bot.telegram.get_dm_policy", return_value="pairing"),
        ):
            await group_bot.handle_group(update, MagicMock())

        group_bot.agent.run.assert_called_once()


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
