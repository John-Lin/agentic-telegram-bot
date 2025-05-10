from __future__ import annotations

import asyncio
import logging

from bot.agents import OpenAIAgent
from bot.config import Configuration
from bot.telegram import TelegramMCPBot


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
        config.bot_username,
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


def run():
    asyncio.run(main())
