from __future__ import annotations

import asyncio
import logging
from collections import deque

from telegram import Message
from telegram import Update
from telegram.constants import ChatAction
from telegram.constants import ParseMode
from telegram.ext import Application
from telegram.ext import CommandHandler
from telegram.ext import ContextTypes
from telegram.ext import MessageHandler
from telegram.ext import filters

from .agents import OpenAIAgent
from .auth import create_pairing_code
from .auth import get_dm_policy
from .auth import get_group_config
from .auth import is_allowed
from .formatting import markdown_to_telegram_html
from .ratelimit import RateLimiter

MAX_REPLY_CHAIN_IDS = 20


class TelegramMCPBot:
    def __init__(self, token: str | None, bot_username: str | None, openai_agent: OpenAIAgent) -> None:
        if bot_username is None:
            raise ValueError("BOT_USERNAME is not set")

        if token is None:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set")

        self.bot_username = bot_username
        self.agent = openai_agent
        self.application = Application.builder().token(token).build()
        self.rate_limiter = RateLimiter()
        self._reply_chains: dict[int, deque[int]] = {}

    async def run(self) -> None:
        # https://github.com/python-telegram-bot/python-telegram-bot/discussions/3310
        await self.application.initialize()
        await self.application.start()
        assert self.application.updater is not None, "Updater is None"
        await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        await self.initialize_agent()

        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("chatid", self.chatid_command))

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
        await self.agent.connect()
        logging.info(f"Initialized agent {self.agent.name}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        if update.message is None:
            return
        await update.message.reply_text("Help!")

    async def chatid_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reply with the current chat ID. Bypasses auth for group discovery."""
        if update.message is None or update.effective_chat is None:
            return
        await update.message.reply_text(str(update.effective_chat.id))

    async def handle_private(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle private chat messages with dmPolicy support."""
        if update.message is None or update.message.text is None:
            return

        user = update.message.from_user
        if user is None:
            return

        policy = await asyncio.to_thread(get_dm_policy)

        if policy == "disabled":
            return

        if await asyncio.to_thread(is_allowed, user.id):
            await self._respond(update)
            return

        # User not in allowFrom
        if policy == "allowlist":
            return  # Silent drop

        # policy == "pairing"
        code = await asyncio.to_thread(create_pairing_code, user.id, user.username or "")
        await update.message.reply_text(
            f"Your pairing code: {code}\n\n"
            f"Run this in your terminal to complete pairing:\n  uv run bot access pair {code}"
        )

    async def handle_group(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle group messages with group access control."""
        if update.message is None or update.message.text is None or update.effective_chat is None:
            return

        user = update.message.from_user
        if user is None:
            return

        group_config = await asyncio.to_thread(get_group_config, update.effective_chat.id)
        if group_config is None:
            return  # Group not configured, silently ignore

        allow_from = group_config["allowFrom"]
        if allow_from:
            # Only listed users can trigger, no mention needed
            if str(user.id) not in allow_from:
                return
        elif group_config["requireMention"]:
            # Allow if this message is a reply to a tracked message in the chain
            reply_to = update.message.reply_to_message
            in_reply_chain = reply_to is not None and reply_to.message_id in self._reply_chains.get(
                update.effective_chat.id, set()
            )
            if not in_reply_chain:
                # Any member can trigger, but must mention
                has_mention = update.message.entities and any(
                    e.type == "mention"
                    and update.message.text[e.offset : e.offset + e.length].lstrip("@") == self.bot_username.lstrip("@")
                    for e in update.message.entities
                )
                if not has_mention:
                    return

        sent = await self._respond(update)
        if sent is not None:
            chat_id = update.effective_chat.id
            chain = self._reply_chains.setdefault(chat_id, deque(maxlen=MAX_REPLY_CHAIN_IDS))
            chain.append(update.message.message_id)
            chain.append(sent.message_id)

    TYPING_INTERVAL_SECONDS = 4

    async def _send_typing_loop(self, update: Update) -> None:
        """Send typing action repeatedly until cancelled."""
        assert update.message is not None
        try:
            while True:
                await update.message.chat.send_action(ChatAction.TYPING)
                await asyncio.sleep(self.TYPING_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass

    async def _respond(self, update: Update) -> Message | None:
        """Run agent and reply. Returns the sent Message object, or None on failure."""
        assert update.message is not None and update.message.text is not None
        assert update.effective_chat is not None
        user = update.message.from_user
        if user is not None and not self.rate_limiter.is_allowed(user.id):
            await update.message.reply_text("Rate limit exceeded. Please try again later.")
            return None
        await update.message.chat.send_action(ChatAction.TYPING)
        typing_task = asyncio.create_task(self._send_typing_loop(update))
        try:
            asst_text = await self.agent.run(update.effective_chat.id, update.message.text)
            html_text = markdown_to_telegram_html(asst_text)
            try:
                return await update.message.reply_text(text=html_text, parse_mode=ParseMode.HTML)
            except Exception:
                logging.warning("Failed to send HTML-formatted message, falling back to plain text")
                return await update.message.reply_text(text=asst_text)
        except Exception as e:
            logging.error(f"Error processing message: {e}", exc_info=True)
            await update.message.reply_text("I'm sorry, I encountered an error processing your request.")
            return None
        finally:
            typing_task.cancel()
