from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from agents import Agent
from agents import Runner
from agents import TResponseInputItem
from agents.mcp import MCPServerStdio
from agents.mcp import MCPServerStreamableHttp
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from agents.tracing import set_tracing_disabled
from openai import AsyncOpenAI

INSTRUCTIONS_FILE = Path("instructions.md")

MAX_TURNS = 10
MCP_SESSION_TIMEOUT_SECONDS = 30.0

set_tracing_disabled(True)


def _load_instructions() -> str:
    """Load agent instructions from ``instructions.md`` in the working directory.

    Fails fast with a clear error if the file is missing, so the bot never
    silently starts up with an empty or stale prompt.
    """
    try:
        return INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Instructions file not found: {INSTRUCTIONS_FILE.resolve()}. "
            "Create an instructions.md file with the agent's system prompt."
        ) from e


def _get_model() -> OpenAIResponsesModel | OpenAIChatCompletionsModel:
    """Create an OpenAI model from environment variables.

    Uses the standard OpenAI client, which works with both OpenAI and
    Azure OpenAI v1 API (via OPENAI_BASE_URL + OPENAI_API_KEY).

    OPENAI_API_TYPE controls which API the model uses:
      - "responses" (default): OpenAI Responses API — recommended by the SDK
      - "chat_completions": Chat Completions API
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.4")
    api_type = os.getenv("OPENAI_API_TYPE", "responses")
    client = AsyncOpenAI()

    if api_type == "chat_completions":
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)
    return OpenAIResponsesModel(model=model_name, openai_client=client)


class OpenAIAgent:
    """A wrapper for OpenAI Agent with MCP server support."""

    def __init__(
        self,
        name: str,
        instructions: str,
        mcp_servers: list | None = None,
    ) -> None:
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=_get_model(),
            mcp_servers=(mcp_servers if mcp_servers is not None else []),
        )
        self.name = name
        self._conversations: dict[int, list[TResponseInputItem]] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    def get_messages(self, chat_id: int) -> list[TResponseInputItem]:
        return self._conversations.get(chat_id, [])

    def set_messages(self, chat_id: int, messages: list[TResponseInputItem]) -> None:
        self._conversations[chat_id] = messages

    def append_user_message(self, chat_id: int, message: str) -> None:
        if chat_id not in self._conversations:
            self._conversations[chat_id] = []
        self._conversations[chat_id].append({"role": "user", "content": message})

    def truncate_history(self, chat_id: int) -> None:
        """Keep only the last MAX_TURNS turns of conversation history.

        A turn starts at each user message. All messages between two user
        messages (assistant replies, tool calls, tool results) belong to
        the preceding turn.
        """
        msgs = self.get_messages(chat_id)
        user_indices = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
        if len(user_indices) <= MAX_TURNS:
            return
        cut = user_indices[-MAX_TURNS]
        self._conversations[chat_id] = msgs[cut:]

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> OpenAIAgent:
        mcp_servers: list[MCPServerStreamableHttp | MCPServerStdio] = []
        for mcp_srv in config.get("mcpServers", {}).values():
            if "url" in mcp_srv:
                mcp_servers.append(
                    MCPServerStreamableHttp(
                        client_session_timeout_seconds=MCP_SESSION_TIMEOUT_SECONDS,
                        params={
                            "url": mcp_srv["url"],
                            "headers": mcp_srv.get("headers", {}),
                        },
                    )
                )
            else:
                mcp_servers.append(
                    MCPServerStdio(
                        client_session_timeout_seconds=MCP_SESSION_TIMEOUT_SECONDS,
                        params={
                            "command": mcp_srv["command"],
                            "args": mcp_srv.get("args", []),
                            "env": mcp_srv.get("env"),
                        },
                    )
                )
        instructions = _load_instructions()
        return cls(name, instructions=instructions, mcp_servers=mcp_servers)

    async def connect(self) -> None:
        for mcp_server in self.agent.mcp_servers:
            try:
                await mcp_server.connect()
                logging.info(f"Server {mcp_server.name} connected")
            except Exception:
                logging.warning(
                    f"MCP server {mcp_server.name} failed to connect — bot will run without its tools", exc_info=True
                )

    async def run(self, chat_id: int, message: str) -> str:
        """Run a workflow starting at the given agent.

        A per-chat async lock ensures that when two messages arrive for the
        same chat in quick succession, they are processed sequentially.
        Without the lock, the second ``await Runner.run`` could start before
        the first one finishes, and whichever completes last would overwrite
        the conversation history — silently dropping the other message and
        its reply.  Different chats are unaffected and still run in parallel.
        """
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            history = self.get_messages(chat_id) + [{"role": "user", "content": message}]
            result = await Runner.run(self.agent, input=history)
            self.set_messages(chat_id, result.to_input_list())
            self.truncate_history(chat_id)
            return str(result.final_output)

    async def cleanup(self) -> None:
        """Clean up resources."""
        for mcp_server in self.agent.mcp_servers:
            try:
                await mcp_server.cleanup()
                logging.info(f"Server {mcp_server.name} cleaned up")
            except Exception as e:
                logging.error(f"Error during cleanup of server {mcp_server.name}: {e}")
