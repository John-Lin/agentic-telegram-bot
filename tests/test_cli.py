from __future__ import annotations

import argparse

import pytest

from app import cmd_allow
from app import cmd_pair
from app import cmd_policy
from app import cmd_remove
from bot.auth import create_pairing_code
from bot.auth import get_dm_policy
from bot.auth import is_allowed
from bot.auth import save_auth


@pytest.fixture(autouse=True)
def auth_file(tmp_path, monkeypatch):
    """Use a temporary auth file for every test."""
    path = tmp_path / "access.json"
    monkeypatch.setattr("bot.auth.AUTH_FILE", path)
    return path


class TestCmdAllow:
    def test_allow_adds_user(self, capsys):
        args = argparse.Namespace(user_id=123)
        cmd_allow(args)
        assert is_allowed(123) is True
        captured = capsys.readouterr()
        assert "123" in captured.out

    def test_allow_duplicate_user(self, capsys):
        save_auth({"dmPolicy": "pairing", "allowFrom": ["123"], "groups": {}, "pending": {}})
        args = argparse.Namespace(user_id=123)
        cmd_allow(args)
        assert is_allowed(123) is True
        captured = capsys.readouterr()
        assert "already" in captured.out.lower()


class TestCmdRemove:
    def test_removes_existing_user(self, capsys):
        save_auth({"dmPolicy": "pairing", "allowFrom": ["123", "456"], "groups": {}, "pending": {}})
        args = argparse.Namespace(user_id=123)
        cmd_remove(args)
        assert is_allowed(123) is False
        assert is_allowed(456) is True
        captured = capsys.readouterr()
        assert "123" in captured.out
        assert "removed" in captured.out.lower()

    def test_nonexistent_user_exits_1(self, capsys):
        args = argparse.Namespace(user_id=999)
        with pytest.raises(SystemExit) as exc_info:
            cmd_remove(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "999" in captured.out
        assert "not found" in captured.out.lower()


class TestCmdPair:
    def test_pair_success(self, capsys):
        code = create_pairing_code(123, "john")
        args = argparse.Namespace(code=code)
        cmd_pair(args)
        assert is_allowed(123) is True
        captured = capsys.readouterr()
        assert "123" in captured.out

    def test_pair_invalid_code_exits_1(self, capsys):
        args = argparse.Namespace(code="BADCODE")
        with pytest.raises(SystemExit) as exc_info:
            cmd_pair(args)
        assert exc_info.value.code == 1


class TestCmdPolicy:
    def test_set_policy(self, capsys):
        args = argparse.Namespace(policy="allowlist")
        cmd_policy(args)
        assert get_dm_policy() == "allowlist"
        captured = capsys.readouterr()
        assert "allowlist" in captured.out

    def test_set_invalid_policy_exits_1(self, capsys):
        args = argparse.Namespace(policy="bogus")
        with pytest.raises(SystemExit) as exc_info:
            cmd_policy(args)
        assert exc_info.value.code == 1

    def test_show_current_policy(self, capsys):
        args = argparse.Namespace(policy=None)
        cmd_policy(args)
        captured = capsys.readouterr()
        assert "pairing" in captured.out
