import json
import logging
import os
from typing import Any

from dotenv import find_dotenv
from dotenv import load_dotenv

__all__ = ["Configuration", "env_flag"]

logger = logging.getLogger(__name__)

_FALSY = {"", "0", "false", "no", "off"}


def env_flag(name: str) -> bool:
    """Return True when env var ``name`` is set to a non-falsy value.

    Matches the semantics previously provided by ``agent_core.env.env_flag``
    so local callers (e.g. ``AGENT_VERBOSE_LOG``) keep working after the
    upstream module was removed.
    """
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw.strip().lower() not in _FALSY


class Configuration:
    """Manages configuration and environment variables for the Telegram bot."""

    def __init__(self) -> None:
        self.load_env()
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.bot_username = os.getenv("BOT_USERNAME")

    @staticmethod
    def load_env() -> None:
        load_dotenv(find_dotenv())

    _DEFAULT_CONFIG: dict[str, Any] = {"mcp": {}}

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load agent configuration from JSON file.

        Returns a default empty config when the file does not exist.

        Raises:
            JSONDecodeError: If configuration file is invalid JSON.
        """
        try:
            with open(file_path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config file %s not found, using default empty config", file_path)
            return dict(Configuration._DEFAULT_CONFIG)
