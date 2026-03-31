from __future__ import annotations

import asyncio
from unittest.mock import create_autospec

import pytest
from agents.models.interface import Model

from bot.agents import DEFAULT_INSTRUCTIONS
from bot.agents import MAX_CHATS
from bot.agents import MAX_TURNS
from bot.agents import OpenAIAgent


@pytest.fixture(autouse=True)
def _mock_model(monkeypatch):
    """Prevent tests from constructing a real OpenAI client."""
    monkeypatch.setattr("bot.agents._get_model", lambda: create_autospec(Model))


class TestPerChatConversations:
    def test_separate_chats_have_independent_history(self):
        """Different chat_ids should maintain separate message histories."""
        agent = OpenAIAgent(name="test")
        agent.append_user_message(chat_id=100, message="hello from chat 100")
        agent.append_user_message(chat_id=200, message="hello from chat 200")

        msgs_100 = agent.get_messages(chat_id=100)
        msgs_200 = agent.get_messages(chat_id=200)

        assert len(msgs_100) == 1
        assert len(msgs_200) == 1
        assert msgs_100[0]["content"] == "hello from chat 100"
        assert msgs_200[0]["content"] == "hello from chat 200"

    def test_same_chat_accumulates_messages(self):
        agent = OpenAIAgent(name="test")
        agent.append_user_message(chat_id=100, message="first")
        agent.append_user_message(chat_id=100, message="second")

        msgs = agent.get_messages(chat_id=100)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"

    def test_unknown_chat_returns_empty(self):
        agent = OpenAIAgent(name="test")
        assert agent.get_messages(chat_id=999) == []

    def test_set_messages_replaces_history(self):
        agent = OpenAIAgent(name="test")
        agent.append_user_message(chat_id=100, message="old")
        new_msgs = [{"role": "user", "content": "replaced"}]
        agent.set_messages(chat_id=100, messages=new_msgs)
        assert agent.get_messages(chat_id=100) == new_msgs

    def test_set_messages_does_not_affect_other_chats(self):
        agent = OpenAIAgent(name="test")
        agent.append_user_message(chat_id=100, message="chat 100")
        agent.append_user_message(chat_id=200, message="chat 200")
        agent.set_messages(chat_id=100, messages=[])
        assert agent.get_messages(chat_id=100) == []
        assert len(agent.get_messages(chat_id=200)) == 1


class TestInstructions:
    def test_default_instructions_when_none_provided(self):
        agent = OpenAIAgent(name="test")
        assert agent.agent.instructions == DEFAULT_INSTRUCTIONS

    def test_custom_instructions(self):
        agent = OpenAIAgent(name="test", instructions="Be a HN bot.")
        assert agent.agent.instructions == "Be a HN bot."

    def test_from_dict_reads_instructions(self):
        config = {
            "instructions": "Custom prompt here.",
            "mcpServers": {},
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert agent.agent.instructions == "Custom prompt here."

    def test_from_dict_uses_default_without_instructions(self):
        config = {
            "mcpServers": {},
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert agent.agent.instructions == DEFAULT_INSTRUCTIONS


class TestHistoryTruncation:
    def test_default_max_turns(self):
        assert MAX_TURNS == 25

    def test_truncate_keeps_recent_turns(self):
        agent = OpenAIAgent(name="test")
        # Simulate 30 turns: each turn is a user msg + assistant msg
        for i in range(30):
            agent.set_messages(
                chat_id=100,
                messages=agent.get_messages(chat_id=100)
                + [
                    {"role": "user", "content": f"user-{i}"},
                    {"role": "assistant", "content": f"assistant-{i}"},
                ],
            )

        agent.truncate_history(chat_id=100)
        msgs = agent.get_messages(chat_id=100)

        # Should keep last 25 turns = 50 messages (user+assistant each)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == MAX_TURNS
        # Oldest kept should be turn 5 (0-4 dropped)
        assert user_msgs[0]["content"] == "user-5"
        # Most recent should be turn 29
        assert user_msgs[-1]["content"] == "user-29"

    def test_truncate_preserves_tool_messages_within_turn(self):
        agent = OpenAIAgent(name="test")
        # Build history with tool calls in a turn
        history = []
        for i in range(MAX_TURNS + 2):
            history.append({"role": "user", "content": f"user-{i}"})
            if i == MAX_TURNS + 1:
                # Last turn has tool calls
                history.append({"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]})
                history.append({"role": "tool", "content": "tool-result", "tool_call_id": "tc1"})
            history.append({"role": "assistant", "content": f"assistant-{i}"})

        agent.set_messages(chat_id=100, messages=history)
        agent.truncate_history(chat_id=100)
        msgs = agent.get_messages(chat_id=100)

        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == MAX_TURNS
        # Tool messages from the last turn should be preserved
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

    def test_no_truncation_when_under_limit(self):
        agent = OpenAIAgent(name="test")
        for i in range(3):
            agent.set_messages(
                chat_id=100,
                messages=agent.get_messages(chat_id=100)
                + [
                    {"role": "user", "content": f"user-{i}"},
                    {"role": "assistant", "content": f"assistant-{i}"},
                ],
            )

        agent.truncate_history(chat_id=100)
        msgs = agent.get_messages(chat_id=100)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 3


class TestChatEviction:
    def test_default_max_chats(self):
        assert MAX_CHATS == 200

    def test_evicts_oldest_chat_when_limit_exceeded(self, monkeypatch):
        monkeypatch.setattr("bot.agents.MAX_CHATS", 3)
        agent = OpenAIAgent(name="test")

        agent.set_messages(100, [{"role": "user", "content": "a"}])
        agent.set_messages(200, [{"role": "user", "content": "b"}])
        agent.set_messages(300, [{"role": "user", "content": "c"}])
        # At limit — all present
        assert len(agent._conversations) == 3

        # Adding a 4th should evict the oldest (100)
        agent.set_messages(400, [{"role": "user", "content": "d"}])
        assert 100 not in agent._conversations
        assert len(agent._conversations) == 3
        assert set(agent._conversations.keys()) == {200, 300, 400}

    def test_updating_existing_chat_does_not_evict(self, monkeypatch):
        monkeypatch.setattr("bot.agents.MAX_CHATS", 2)
        agent = OpenAIAgent(name="test")

        agent.set_messages(100, [{"role": "user", "content": "a"}])
        agent.set_messages(200, [{"role": "user", "content": "b"}])
        # Updating chat 100 should not trigger eviction
        agent.set_messages(100, [{"role": "user", "content": "updated"}])
        assert len(agent._conversations) == 2
        assert agent.get_messages(100)[0]["content"] == "updated"

    def test_append_to_new_chat_triggers_eviction(self, monkeypatch):
        monkeypatch.setattr("bot.agents.MAX_CHATS", 2)
        agent = OpenAIAgent(name="test")

        agent.set_messages(100, [{"role": "user", "content": "a"}])
        agent.set_messages(200, [{"role": "user", "content": "b"}])
        agent.append_user_message(300, "c")

        assert 100 not in agent._conversations
        assert len(agent._conversations) == 2

    def test_accessing_chat_refreshes_its_position(self, monkeypatch):
        monkeypatch.setattr("bot.agents.MAX_CHATS", 3)
        agent = OpenAIAgent(name="test")

        agent.set_messages(100, [{"role": "user", "content": "a"}])
        agent.set_messages(200, [{"role": "user", "content": "b"}])
        agent.set_messages(300, [{"role": "user", "content": "c"}])

        # Access chat 100 to refresh it (move to end)
        agent.set_messages(100, [{"role": "user", "content": "refreshed"}])

        # Now 200 is oldest — adding 400 should evict 200, not 100
        agent.set_messages(400, [{"role": "user", "content": "d"}])
        assert 200 not in agent._conversations
        assert 100 in agent._conversations

    def test_eviction_also_cleans_up_lock(self, monkeypatch):
        monkeypatch.setattr("bot.agents.MAX_CHATS", 2)
        agent = OpenAIAgent(name="test")

        agent._locks[100] = asyncio.Lock()
        agent.set_messages(100, [{"role": "user", "content": "a"}])
        agent._locks[200] = asyncio.Lock()
        agent.set_messages(200, [{"role": "user", "content": "b"}])

        agent.set_messages(300, [{"role": "user", "content": "c"}])
        assert 100 not in agent._locks
