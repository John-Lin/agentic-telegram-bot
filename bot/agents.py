from __future__ import annotations

import logging
from typing import Any

from agentize.model import get_openai_model
from agentize.prompts.summary import INSTRUCTIONS
from agentize.prompts.summary import Summary
from agentize.tools.duckduckgo import duckduckgo_search
from agentize.tools.firecrawl import map_tool
from agentize.tools.markitdown import markitdown_scrape_tool
from agentize.tools.telegraph import publish_page
from agentize.utils import configure_langfuse
from agents import Agent
from agents import Runner
from agents import TResponseInputItem
from agents.mcp import MCPServerStdio


class OpenAIAgent:
    """A wrapper for OpenAI Agent"""

    def __init__(self, name: str, mcp_servers: list | None = None) -> None:
        configure_langfuse("Telegram Bot")
        self.summary_agent = Agent(
            name="summary_agent",
            instructions=INSTRUCTIONS.format(lang="台灣中文", length=1_000),
            model=get_openai_model(model="o3-mini", api_type="chat_completions"),
            output_type=Summary,
        )

        self.main_agent = Agent(
            name=name,
            instructions="You are a helpful assistant. Handoff to the summary agent when you need to summarize.",
            model=get_openai_model(model="gpt-4.1"),
            tools=[markitdown_scrape_tool, map_tool, duckduckgo_search, publish_page],
            handoffs=[self.summary_agent],
            mcp_servers=(mcp_servers if mcp_servers is not None else []),
        )
        self.name = name
        self.messages: list[TResponseInputItem] = []

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
            for mcp_srv in config.values()
        ]
        return cls(name, mcp_servers)

    async def connect(self) -> None:
        for mcp_server in self.main_agent.mcp_servers:
            try:
                await mcp_server.connect()
                logging.info(f"Server {mcp_server.name} connecting")
            except Exception as e:
                logging.error(f"Error during connecting of server {mcp_server.name}: {e}")

    async def run(self, message: str) -> str:
        """Run a workflow starting at the given agent."""
        self.messages.append(
            {
                "role": "user",
                "content": message,
            }
        )
        result = await Runner.run(self.main_agent, input=self.messages)
        self.messages = result.to_input_list()
        # Add conversation history (last 5 messages)
        self.messages = self.messages[-5:]
        return str(result.final_output)

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up servers
        for mcp_server in self.main_agent.mcp_servers:
            try:
                await mcp_server.cleanup()
                logging.info(f"Server {mcp_server.name} cleaned up")
            except Exception as e:
                logging.error(f"Error during cleanup of server {mcp_server.name}: {e}")
