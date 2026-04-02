from __future__ import annotations

import json

import pytest

from bot.config import Configuration


@pytest.fixture()
def config_env(monkeypatch):
    """Set required environment variables for Configuration."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("BOT_USERNAME", "@testbot")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Prevent find_dotenv from loading a real .envrc
    monkeypatch.setattr("bot.config.find_dotenv", lambda: "")


class TestConfiguration:
    def test_loads_telegram_token(self, config_env):
        cfg = Configuration()
        assert cfg.telegram_bot_token == "test-token"

    def test_loads_bot_username(self, config_env):
        cfg = Configuration()
        assert cfg.bot_username == "@testbot"

    def test_loads_openai_key(self, config_env):
        cfg = Configuration()
        assert cfg.openai_api_key == "sk-test"

    def test_missing_values_are_none(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("BOT_USERNAME", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("bot.config.find_dotenv", lambda: "")
        cfg = Configuration()
        assert cfg.telegram_bot_token is None
        assert cfg.bot_username is None
        assert cfg.openai_api_key is None


class TestLoadConfig:
    def test_loads_valid_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_data = {"mcpServers": {"test": {"command": "echo", "args": []}}}
        config_file.write_text(json.dumps(config_data))
        result = Configuration.load_config(str(config_file))
        assert result == config_data

    def test_returns_default_on_missing_file(self):
        result = Configuration.load_config("/nonexistent/config.json")
        assert result == {"mcpServers": {}}

    def test_raises_on_invalid_json(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            Configuration.load_config(str(config_file))
