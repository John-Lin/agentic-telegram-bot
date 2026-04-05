from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import create_autospec
from unittest.mock import patch

import pytest
from agents import WebSearchTool
from agents.models.interface import Model
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel

from bot.agents import DEFAULT_INSTRUCTIONS
from bot.agents import MAX_TURNS
from bot.agents import OpenAIAgent
from bot.agents import _get_model


@pytest.fixture(autouse=True)
def _mock_model(monkeypatch):
    """Prevent tests from constructing a real OpenAI client."""
    monkeypatch.setattr("bot.agents._get_model", lambda: create_autospec(Model))


class TestGetModel:
    def test_uses_azure_when_both_azure_env_vars_present(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my.azure.com")
        monkeypatch.delenv("OPENAI_API_VERSION", raising=False)

        with patch("bot.agents.AsyncAzureOpenAI") as mock_azure, patch("bot.agents.AsyncOpenAI") as mock_openai:
            mock_azure.return_value = MagicMock()
            _get_model()
            mock_azure.assert_called_once()
            mock_openai.assert_not_called()

    def test_uses_standard_openai_when_only_api_key_set(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

        with patch("bot.agents.AsyncAzureOpenAI") as mock_azure, patch("bot.agents.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            _get_model()
            mock_openai.assert_called_once()
            mock_azure.assert_not_called()

    def test_uses_standard_openai_when_no_azure_vars(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

        with patch("bot.agents.AsyncAzureOpenAI") as mock_azure, patch("bot.agents.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            _get_model()
            mock_openai.assert_called_once()
            mock_azure.assert_not_called()

    def test_returns_responses_model_by_default(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_TYPE", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

        with patch("bot.agents.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model()
        assert isinstance(model, OpenAIResponsesModel)

    def test_returns_responses_model_when_api_type_is_responses(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_TYPE", "responses")
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

        with patch("bot.agents.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model()
        assert isinstance(model, OpenAIResponsesModel)

    def test_returns_chat_completions_model_when_api_type_is_chat_completions(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_TYPE", "chat_completions")
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

        with patch("bot.agents.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model()
        assert isinstance(model, OpenAIChatCompletionsModel)


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
        assert MAX_TURNS == 10

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

        # Should keep last MAX_TURNS turns (user+assistant each)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == MAX_TURNS
        # Oldest kept should be turn (30 - MAX_TURNS)
        assert user_msgs[0]["content"] == f"user-{30 - MAX_TURNS}"
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


class TestWebSearchTool:
    def test_web_search_tool_included_with_standard_openai_responses(self, monkeypatch):
        """WebSearchTool should be added when using Responses API with standard OpenAI."""
        mock_model = create_autospec(OpenAIResponsesModel)
        monkeypatch.setattr("bot.agents._get_model", lambda: mock_model)
        monkeypatch.setattr("bot.agents._is_azure", lambda: False)

        agent = OpenAIAgent(name="test")
        web_tools = [t for t in agent.agent.tools if isinstance(t, WebSearchTool)]
        assert len(web_tools) == 1

    def test_web_search_tool_excluded_with_chat_completions(self, monkeypatch):
        """WebSearchTool should NOT be added when using Chat Completions API."""
        mock_model = create_autospec(OpenAIChatCompletionsModel)
        monkeypatch.setattr("bot.agents._get_model", lambda: mock_model)

        agent = OpenAIAgent(name="test")
        web_tools = [t for t in agent.agent.tools if isinstance(t, WebSearchTool)]
        assert len(web_tools) == 0

    def test_web_search_tool_excluded_with_azure(self, monkeypatch):
        """WebSearchTool should NOT be added when using Azure, even with Responses API."""
        mock_model = create_autospec(OpenAIResponsesModel)
        monkeypatch.setattr("bot.agents._get_model", lambda: mock_model)
        monkeypatch.setattr("bot.agents._is_azure", lambda: True)

        agent = OpenAIAgent(name="test")
        web_tools = [t for t in agent.agent.tools if isinstance(t, WebSearchTool)]
        assert len(web_tools) == 0
