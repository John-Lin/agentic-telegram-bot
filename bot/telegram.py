from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import Application
from telegram.ext import CommandHandler
from telegram.ext import ContextTypes
from telegram.ext import MessageHandler
from telegram.ext import filters

from .agents import OpenAIAgent
from .auth import create_pairing_code
from .auth import get_group_config
from .auth import is_allowed


class TelegramMCPBot:
    def __init__(self, token: str | None, bot_username: str | None, openai_agent: OpenAIAgent) -> None:
        if bot_username is None:
            raise ValueError("BOT_USERNAME is not set")

        if token is None:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set")

        self.bot_username = bot_username
        self.agent = openai_agent
        self.application = Application.builder().token(token).build()

    async def run(self) -> None:
        # https://github.com/python-telegram-bot/python-telegram-bot/discussions/3310
        await self.application.initialize()
        await self.application.start()
        assert self.application.updater is not None, "Updater is None"
        await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        await self.initialize_agent()

        self.application.add_handler(CommandHandler("help", self.help_command))

        # Private chat: respond to all text messages
        self.application.add_handler(
            MessageHandler(filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND, self.handle_private)
        )
        # Group chat: handle via group access control
        self.application.add_handler(
            MessageHandler(filters.ChatType.GROUPS & filters.TEXT & ~filters.COMMAND, self.handle_group)
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            logging.info("Stopping bot application")
            assert self.application.updater is not None
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
        except Exception as e:
            logging.error(f"Error stop bot application: {e}")

    async def initialize_agent(self) -> None:
        """Initialize all MCP servers and discover tools."""
        try:
            await self.agent.connect()
            logging.info(f"Initialized agent {self.agent.name} with tools")
        except Exception as e:
            logging.error(f"Failed to initialize agent {self.agent.name}: {e}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        if update.message is None:
            return
        await update.message.reply_text("Help!")

    async def handle_private(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle private chat messages with pairing support."""
        if update.message is None or update.message.text is None:
            return

        user = update.message.from_user
        if user is None:
            return

        if not is_allowed(user.id):
            code = create_pairing_code(user.id, user.username or "")
            await update.message.reply_text(
                f"Your pairing code: {code}\n\nRun this in your terminal to complete pairing:\n  uv run bot pair {code}"
            )
            return

        await self._respond(update)

    async def handle_group(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle group messages with group access control."""
        if update.message is None or update.message.text is None or update.effective_chat is None:
            return

        user = update.message.from_user
        if user is None:
            return

        group_config = get_group_config(update.effective_chat.id)
        if group_config is None:
            return  # Group not allowed, silently ignore

        # Check mention requirement
        if group_config["require_mention"] and not (
            update.message.entities
            and any(
                e.type == "mention" and update.message.text[e.offset : e.offset + e.length] == self.bot_username
                for e in update.message.entities
            )
        ):
            return

        # Check allowed_members if configured
        allowed_members = group_config["allowed_members"]
        if allowed_members and user.id not in allowed_members:
            return

        await self._respond(update)

    async def _respond(self, update: Update) -> None:
        """Run agent and reply."""
        assert update.message is not None and update.message.text is not None
        try:
            asst_text = await self.agent.run(update.message.text)
            await update.message.reply_text(text=asst_text)
        except Exception as e:
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            logging.error(f"Error processing message: {e}", exc_info=True)
            await update.message.reply_text(error_message)
