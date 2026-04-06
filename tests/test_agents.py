from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import create_autospec
from unittest.mock import patch

import pytest
from agents import ShellTool
from agents.mcp import MCPServerStdio
from agents.mcp import MCPServerStreamableHttp
from agents.models.interface import Model
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel

from bot.agents import MAX_TURNS
from bot.agents import OpenAIAgent
from bot.agents import _get_model


@pytest.fixture
def _stub_instructions(monkeypatch):
    """Stub out instructions.md loading for from_dict tests."""
    monkeypatch.setattr("bot.agents._load_instructions", lambda: "stub instructions")


@pytest.fixture(autouse=True)
def _mock_model(monkeypatch, tmp_path_factory):
    """Prevent tests from constructing a real OpenAI client and isolate skills."""
    monkeypatch.setattr("bot.agents._get_model", lambda: create_autospec(Model))
    # Point SKILLS_DIR at an empty directory so tests do not auto-load the
    # real ./skills/ on disk. Tests that need skills can override this.
    monkeypatch.setattr("bot.agents.SKILLS_DIR", tmp_path_factory.mktemp("empty_skills"))


class TestGetModel:
    def test_returns_responses_model_by_default(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_TYPE", raising=False)

        with patch("bot.agents.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model()
        assert isinstance(model, OpenAIResponsesModel)

    def test_returns_responses_model_when_api_type_is_responses(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_TYPE", "responses")

        with patch("bot.agents.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model()
        assert isinstance(model, OpenAIResponsesModel)

    def test_returns_chat_completions_model_when_api_type_is_chat_completions(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_TYPE", "chat_completions")

        with patch("bot.agents.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model()
        assert isinstance(model, OpenAIChatCompletionsModel)


class TestPerChatConversations:
    def test_separate_chats_have_independent_history(self):
        """Different chat_ids should maintain separate message histories."""
        agent = OpenAIAgent(name="test", instructions="test-prompt")
        agent.append_user_message(chat_id=100, message="hello from chat 100")
        agent.append_user_message(chat_id=200, message="hello from chat 200")

        msgs_100 = agent.get_messages(chat_id=100)
        msgs_200 = agent.get_messages(chat_id=200)

        assert len(msgs_100) == 1
        assert len(msgs_200) == 1
        assert msgs_100[0]["content"] == "hello from chat 100"
        assert msgs_200[0]["content"] == "hello from chat 200"

    def test_same_chat_accumulates_messages(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")
        agent.append_user_message(chat_id=100, message="first")
        agent.append_user_message(chat_id=100, message="second")

        msgs = agent.get_messages(chat_id=100)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"

    def test_unknown_chat_returns_empty(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")
        assert agent.get_messages(chat_id=999) == []

    def test_set_messages_replaces_history(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")
        agent.append_user_message(chat_id=100, message="old")
        new_msgs = [{"role": "user", "content": "replaced"}]
        agent.set_messages(chat_id=100, messages=new_msgs)
        assert agent.get_messages(chat_id=100) == new_msgs

    def test_set_messages_does_not_affect_other_chats(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")
        agent.append_user_message(chat_id=100, message="chat 100")
        agent.append_user_message(chat_id=200, message="chat 200")
        agent.set_messages(chat_id=100, messages=[])
        assert agent.get_messages(chat_id=100) == []
        assert len(agent.get_messages(chat_id=200)) == 1


class TestInstructions:
    def test_custom_instructions(self):
        agent = OpenAIAgent(name="test", instructions="Be a HN bot.")
        assert agent.agent.instructions == "Be a HN bot."

    def test_from_dict_loads_instructions_from_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "instructions.md").write_text("From file prompt.", encoding="utf-8")
        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})
        assert agent.agent.instructions == "From file prompt."

    def test_from_dict_fails_fast_when_instructions_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="Instructions file not found"):
            OpenAIAgent.from_dict("test", {"mcpServers": {}})


@pytest.mark.usefixtures("_stub_instructions")
class TestFromDictMcpServers:
    def test_url_creates_streamable_http_server(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:8000/mcp",
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1
        assert isinstance(agent.agent.mcp_servers[0], MCPServerStreamableHttp)

    def test_url_passes_headers(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:8000/mcp",
                    "headers": {"Authorization": "Bearer token"},
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        server = agent.agent.mcp_servers[0]
        assert isinstance(server, MCPServerStreamableHttp)

    def test_command_creates_stdio_server(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "command": "npx",
                    "args": ["-y", "some-mcp-server"],
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1
        assert isinstance(agent.agent.mcp_servers[0], MCPServerStdio)

    def test_mixed_servers(self):
        config = {
            "mcpServers": {
                "remote": {"url": "http://localhost:8000/mcp"},
                "local": {"command": "npx", "args": ["-y", "server"]},
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        types = {type(s) for s in agent.agent.mcp_servers}
        assert types == {MCPServerStreamableHttp, MCPServerStdio}


class TestHistoryTruncation:
    def test_default_max_turns(self):
        assert MAX_TURNS == 10

    def test_truncate_keeps_recent_turns(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")
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
        agent = OpenAIAgent(name="test", instructions="test-prompt")
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
        agent = OpenAIAgent(name="test", instructions="test-prompt")
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


class TestLoadShellSkills:
    def test_no_shell_tool_when_skills_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bot.agents.SKILLS_DIR", tmp_path / "nonexistent")
        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})
        shell_tools = [t for t in agent.agent.tools if isinstance(t, ShellTool)]
        assert len(shell_tools) == 0

    def test_shell_tool_added_when_skill_found(self, tmp_path, monkeypatch):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: A test skill\n---\n")
        monkeypatch.setattr("bot.agents.SKILLS_DIR", tmp_path)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})
        shell_tool = next(t for t in agent.agent.tools if isinstance(t, ShellTool))
        skill = shell_tool.environment["skills"][0]
        assert skill["name"] == "my-skill"
        assert skill["description"] == "A test skill"
        assert skill["path"] == str(skill_dir)

    def test_multiple_skills_all_mounted(self, tmp_path, monkeypatch):
        for name in ["skill-a", "skill-b"]:
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: desc {name}\n---\n")
        monkeypatch.setattr("bot.agents.SKILLS_DIR", tmp_path)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})
        shell_tool = next(t for t in agent.agent.tools if isinstance(t, ShellTool))
        assert len(shell_tool.environment["skills"]) == 2

    def test_directory_without_skill_md_is_skipped(self, tmp_path, monkeypatch):
        (tmp_path / "not-a-skill").mkdir()
        good = tmp_path / "real-skill"
        good.mkdir()
        (good / "SKILL.md").write_text("---\nname: real-skill\ndescription: d\n---\n")
        monkeypatch.setattr("bot.agents.SKILLS_DIR", tmp_path)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})
        shell_tool = next(t for t in agent.agent.tools if isinstance(t, ShellTool))
        skills = shell_tool.environment["skills"]
        assert len(skills) == 1
        assert skills[0]["name"] == "real-skill"

    def test_mcp_servers_and_shell_skills_coexist(self, tmp_path, monkeypatch):
        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n")
        monkeypatch.setattr("bot.agents.SKILLS_DIR", tmp_path)

        config = {"mcpServers": {"my-mcp": {"command": "uvx", "args": ["something"]}}}
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1
        shell_tools = [t for t in agent.agent.tools if isinstance(t, ShellTool)]
        assert len(shell_tools) == 1
