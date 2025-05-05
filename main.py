from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from agentize.agents.summary import get_summary_agent
from agentize.crawler.firecrawl import map_tool
from agentize.crawler.firecrawl import scrape_tool
from agentize.crawler.firecrawl import search_tool
from agentize.model import get_openai_model
from agentize.model import get_openai_model_settings
from agentize.utils import configure_langfuse
from agents import Agent
from agents import Runner
from agents.mcp import MCPServerStdio
from dotenv import find_dotenv
from dotenv import load_dotenv
from telegram import ForceReply
from telegram import Update
from telegram.ext import Application
from telegram.ext import CommandHandler
from telegram.ext import ContextTypes
from telegram.ext import MessageHandler
from telegram.ext import filters


class Configuration:
    """Manages configuration and environment variables for the MCP Telegram bot."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv(find_dotenv())

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path) as f:
            return json.load(f)


class OpenAIAgent:
    """A wrapper for OpenAI Agent"""

    def __init__(self, name: str, mcp_servers: list | None = None) -> None:
        configure_langfuse("Telegram Bot")
        self.summary_agent = get_summary_agent(lang="台灣中文", length=1_000)
        self.main_agent = Agent(
            name=name,
            instructions="You are a helpful assistant. Handoff to the summary agent when you need to summarize.",
            model=get_openai_model(),
            model_settings=get_openai_model_settings(),
            tools=[scrape_tool, map_tool, search_tool],
            handoffs=[self.summary_agent],
            mcp_servers=(mcp_servers if mcp_servers is not None else []),
        )
        self.name = name

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

    async def run(self, messages: list) -> str:
        """Run a workflow starting at the given agent."""
        result = await Runner.run(self.main_agent, input=messages)
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


class TelegramMCPBot:
    def __init__(self, token: str | None, openai_agent: OpenAIAgent) -> None:
        self.agent = openai_agent
        self.conversations: dict[
            int, dict[str, list[dict[str, str | Any | None]]]
        ] = {}  # Store conversation context per channel
        if token is None:
            raise ValueError("TELEGRAM_TOKEN is not set")
        self.application = Application.builder().token(token).build()

    async def run(self) -> None:
        # https://github.com/python-telegram-bot/python-telegram-bot/discussions/3310
        # https://docs.python-telegram-bot.org/en/stable/telegram.ext.application.html#telegram.ext.Application.run_polling
        # inits bot, update, persistence
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        await self.initialize_agent()

        # on different commands - answer in Telegram
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))

        # on non command i.e message - handle the message on Telegram
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logging.info("Stopping bot application")
        except Exception as e:
            logging.error(f"Error steop bot application: {e}")

    async def initialize_agent(self) -> None:
        """Initialize all MCP servers and discover tools."""
        try:
            await self.agent.connect()
            logging.info(f"Initialized agent {self.agent.name} with tools")
        except Exception as e:
            logging.error(f"Failed to initialize agent {self.agent.name}: {e}")

    # Define a few command handlers. These usually take the two arguments update and
    # context.
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        user = update.effective_user
        await update.message.reply_html(
            rf"Hi {user.mention_html()}!",
            reply_markup=ForceReply(selective=True),
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        await update.message.reply_text("Help!")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._procress_message(update, context)

    async def _procress_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process incoming messages and generate responses."""
        # Get or create conversation context
        chat_id = update.message.chat_id

        if chat_id not in self.conversations:
            self.conversations[chat_id] = {"messages": []}

        try:
            messages = []

            # Add user message to history
            self.conversations[chat_id]["messages"].append({"role": "user", "content": update.message.text})

            # Add conversation history (last 5 messages)
            if "messages" in self.conversations[chat_id]:
                messages.extend(self.conversations[chat_id]["messages"][-5:])

            logging.info("---------------------- History ---------------------------")
            logging.info(self.conversations)
            logging.info("----------------------------------------------------------")

            # Get LLM response
            agent_resp = await self.agent.run(messages)
            logging.info("---------------- agent response --------------------------")
            logging.info(agent_resp)
            logging.info("----------------------------------------------------------")

            # Add assistant response to conversation history
            self.conversations[chat_id]["messages"].append({"role": "assistant", "content": agent_resp})
            await update.message.reply_text(agent_resp)
        except Exception as e:
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            logging.error(f"Error processing message: {e}", exc_info=True)
            await update.message.reply_text(error_message)


async def main() -> None:
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.getLogger(__name__)
    """Initialize and run the Telegram bot."""
    config = Configuration()

    server_config = config.load_config("servers_config.json")
    openai_agent = OpenAIAgent.from_dict("Telegram Bot Agent", server_config["mcpServers"])

    # Initialize the OpenAI agents with mcp servers
    # openai_agent = OpenAIAgent("Telegram Bot Agent")

    tg_bot = TelegramMCPBot(
        config.telegram_bot_token,
        openai_agent,
    )

    try:
        await tg_bot.run()
        # Keep the main task alive until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        await tg_bot.cleanup()
        await openai_agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
