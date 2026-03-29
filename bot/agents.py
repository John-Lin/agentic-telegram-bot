from __future__ import annotations

import logging
import os
from typing import Any

from agents import Agent
from agents import Runner
from agents import TResponseInputItem
from agents.mcp import MCPServerStdio
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncAzureOpenAI
from openai import AsyncOpenAI

DEFAULT_INSTRUCTIONS = (
    "You are a helpful assistant in a Telegram chat. "
    "When referencing articles, websites, or resources, always include "
    "the URL as a markdown hyperlink, e.g. [title](https://example.com). "
    "Keep responses concise and well-structured for mobile reading."
)


def _get_model() -> OpenAIChatCompletionsModel:
    """Create an OpenAI model from environment variables."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")

    client: AsyncOpenAI
    if os.getenv("AZURE_OPENAI_API_KEY"):
        client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("OPENAI_API_VERSION", "2025-04-01-preview"),
        )
    else:
        client = AsyncOpenAI()

    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


class OpenAIAgent:
    """A wrapper for OpenAI Agent with MCP server support."""

    def __init__(
        self,
        name: str,
        mcp_servers: list | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
    ) -> None:
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=_get_model(),
            mcp_servers=(mcp_servers if mcp_servers is not None else []),
        )
        self.name = name
        self._conversations: dict[int, list[TResponseInputItem]] = {}

    def get_messages(self, chat_id: int) -> list[TResponseInputItem]:
        return self._conversations.get(chat_id, [])

    def set_messages(self, chat_id: int, messages: list[TResponseInputItem]) -> None:
        self._conversations[chat_id] = messages

    def append_user_message(self, chat_id: int, message: str) -> None:
        if chat_id not in self._conversations:
            self._conversations[chat_id] = []
        self._conversations[chat_id].append({"role": "user", "content": message})

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> OpenAIAgent:
        mcp_servers = [
            MCPServerStdio(
                client_session_timeout_seconds=30.0,
                params={
                    "command": mcp_srv["command"],
                    "args": mcp_srv["args"],
                    "env": mcp_srv.get("env", {}),
                },
            )
            for mcp_srv in config.get("mcpServers", {}).values()
        ]
        instructions = config.get("instructions", DEFAULT_INSTRUCTIONS)
        return cls(name, mcp_servers, instructions=instructions)

    async def connect(self) -> None:
        for mcp_server in self.agent.mcp_servers:
            try:
                await mcp_server.connect()
                logging.info(f"Server {mcp_server.name} connected")
            except Exception as e:
                logging.error(f"Error during connecting of server {mcp_server.name}: {e}")

    async def run(self, chat_id: int, message: str) -> str:
        """Run a workflow starting at the given agent."""
        self.append_user_message(chat_id, message)
        result = await Runner.run(self.agent, input=self.get_messages(chat_id))
        self.set_messages(chat_id, result.to_input_list())
        return str(result.final_output)

    async def cleanup(self) -> None:
        """Clean up resources."""
        for mcp_server in self.agent.mcp_servers:
            try:
                await mcp_server.cleanup()
                logging.info(f"Server {mcp_server.name} cleaned up")
            except Exception as e:
                logging.error(f"Error during cleanup of server {mcp_server.name}: {e}")
