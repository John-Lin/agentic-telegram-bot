import json
import os
from typing import Any

from dotenv import find_dotenv
from dotenv import load_dotenv


class Configuration:
    """Manages configuration and environment variables for the MCP Telegram bot."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.bot_username = os.getenv("BOT_USERNAME")

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
