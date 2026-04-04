"""Tests for Telegram bot command handlers."""

import asyncio
from collections import deque
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from bot.telegram import MAX_REPLY_CHAIN_IDS
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


def _make_group_update(
    chat_id: int,
    user_id: int,
    text: str,
    mention: str | None = None,
    message_id: int = 1,
    reply_to_message_id: int | None = None,
) -> MagicMock:
    """Create a mock group message update, optionally with a @mention entity or reply."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.message_id = message_id
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
    if reply_to_message_id is not None:
        update.message.reply_to_message = MagicMock()
        update.message.reply_to_message.message_id = reply_to_message_id
    else:
        update.message.reply_to_message = None
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


class TestReplyChain:
    """Test that reply chains allow conversation without @mention."""

    @pytest.fixture
    def group_bot(self):
        agent = MagicMock()
        agent.run = AsyncMock(return_value="response")
        bot = TelegramMCPBot(token="fake:token", bot_username="@testbot", openai_agent=agent)
        bot.rate_limiter = MagicMock()
        bot.rate_limiter.is_allowed.return_value = True
        bot.TYPING_INTERVAL_SECONDS = 100
        return bot

    @pytest.mark.anyio
    async def test_mention_seeds_reply_chain(self, group_bot):
        """After @mention triggers bot, both the user message and bot reply are tracked."""
        config = {"requireMention": True, "allowFrom": []}
        bot_reply = MagicMock()
        bot_reply.message_id = 99
        update = _make_group_update(
            chat_id=-1001234, user_id=111, text="hey @testbot", mention="@testbot", message_id=10
        )
        update.message.reply_text = AsyncMock(return_value=bot_reply)

        with patch("bot.telegram.get_group_config", return_value=config):
            await group_bot.handle_group(update, MagicMock())

        chain = group_bot._reply_chains.get(-1001234, set())
        assert 10 in chain  # user's message
        assert 99 in chain  # bot's reply

    @pytest.mark.anyio
    async def test_reply_to_tracked_message_triggers_without_mention(self, group_bot):
        """Reply to a tracked message should trigger bot without @mention."""
        config = {"requireMention": True, "allowFrom": []}
        group_bot._reply_chains[-1001234] = deque([50], maxlen=MAX_REPLY_CHAIN_IDS)  # seed: message 50 is tracked

        update = _make_group_update(
            chat_id=-1001234, user_id=222, text="continue", message_id=51, reply_to_message_id=50
        )

        with patch("bot.telegram.get_group_config", return_value=config):
            await group_bot.handle_group(update, MagicMock())

        group_bot.agent.run.assert_called_once()

    @pytest.mark.anyio
    async def test_reply_to_untracked_message_requires_mention(self, group_bot):
        """Reply to an untracked message should still require @mention."""
        config = {"requireMention": True, "allowFrom": []}
        group_bot._reply_chains[-1001234] = deque([50], maxlen=MAX_REPLY_CHAIN_IDS)  # only message 50 is tracked

        update = _make_group_update(chat_id=-1001234, user_id=222, text="hello", message_id=51, reply_to_message_id=999)

        with patch("bot.telegram.get_group_config", return_value=config):
            await group_bot.handle_group(update, MagicMock())

        group_bot.agent.run.assert_not_called()

    @pytest.mark.anyio
    async def test_reply_chain_grows_on_each_turn(self, group_bot):
        """Each new reply in the chain adds both messages to the tracked set."""
        config = {"requireMention": True, "allowFrom": []}
        group_bot._reply_chains[-1001234] = deque([10, 11], maxlen=MAX_REPLY_CHAIN_IDS)  # existing chain

        bot_reply = MagicMock()
        bot_reply.message_id = 13
        update = _make_group_update(
            chat_id=-1001234, user_id=222, text="next question", message_id=12, reply_to_message_id=11
        )
        update.message.reply_text = AsyncMock(return_value=bot_reply)

        with patch("bot.telegram.get_group_config", return_value=config):
            await group_bot.handle_group(update, MagicMock())

        chain = group_bot._reply_chains[-1001234]
        assert 12 in chain  # new user message
        assert 13 in chain  # new bot reply

    @pytest.mark.anyio
    async def test_reply_chain_evicts_oldest_when_full(self, group_bot):
        """When chain exceeds MAX_REPLY_CHAIN_IDS, oldest IDs are evicted."""
        config = {"requireMention": True, "allowFrom": []}
        chat_id = -1001234

        # Seed chain to capacity with IDs 1..MAX_REPLY_CHAIN_IDS
        group_bot._reply_chains[chat_id] = deque(range(1, MAX_REPLY_CHAIN_IDS + 1), maxlen=MAX_REPLY_CHAIN_IDS)

        # Trigger one more turn: reply to the last tracked message
        bot_reply = MagicMock()
        bot_reply.message_id = MAX_REPLY_CHAIN_IDS + 2
        update = _make_group_update(
            chat_id=chat_id,
            user_id=111,
            text="overflow",
            message_id=MAX_REPLY_CHAIN_IDS + 1,
            reply_to_message_id=MAX_REPLY_CHAIN_IDS,  # reply to last tracked
        )
        update.message.reply_text = AsyncMock(return_value=bot_reply)

        with patch("bot.telegram.get_group_config", return_value=config):
            await group_bot.handle_group(update, MagicMock())

        chain = group_bot._reply_chains[chat_id]
        # Oldest IDs should have been evicted
        assert 1 not in chain
        assert 2 not in chain
        # New IDs should be present
        assert MAX_REPLY_CHAIN_IDS + 1 in chain
        assert MAX_REPLY_CHAIN_IDS + 2 in chain
        assert len(chain) == MAX_REPLY_CHAIN_IDS

    @pytest.mark.anyio
    async def test_reply_chain_isolated_per_group(self, group_bot):
        """Reply chain from one group does not affect another group."""
        config = {"requireMention": True, "allowFrom": []}
        group_bot._reply_chains[-1001111] = deque([50], maxlen=MAX_REPLY_CHAIN_IDS)  # only in group 1111

        update = _make_group_update(chat_id=-1002222, user_id=222, text="hello", message_id=51, reply_to_message_id=50)

        with patch("bot.telegram.get_group_config", return_value=config):
            await group_bot.handle_group(update, MagicMock())

        group_bot.agent.run.assert_not_called()


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
