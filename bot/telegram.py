from __future__ import annotations

import logging
from typing import Any

from telegram import Update
from telegram.ext import Application
from telegram.ext import CommandHandler
from telegram.ext import ContextTypes
from telegram.ext import MessageHandler
from telegram.ext import filters

from .agents import OpenAIAgent


class TelegramMCPBot:
    def __init__(self, token: str | None, bot_username: str | None, openai_agent: OpenAIAgent) -> None:
        if bot_username is None:
            raise ValueError("BOT_USERNAME is not set")

        if token is None:
            raise ValueError("TELEGRAM_TOKEN is not set")

        self.bot_username = bot_username
        self.agent = openai_agent
        self.conversations: dict[
            int, dict[str, list[dict[str, str | Any | None]]]
        ] = {}  # Store conversation context per channel
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
        self.application.add_handler(CommandHandler("help", self.help_command))

        # on non command i.e message - handle the message on Telegram
        # Add a message handler to handle replies
        self.application.add_handler(MessageHandler(filters.REPLY & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.Mention(self.bot_username), self.handle_message))

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
