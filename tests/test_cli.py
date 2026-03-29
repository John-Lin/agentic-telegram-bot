from __future__ import annotations

import argparse

import pytest

from app import cmd_user_list
from app import cmd_user_remove
from bot.auth import is_allowed
from bot.auth import save_auth


@pytest.fixture(autouse=True)
def auth_file(tmp_path, monkeypatch):
    """Use a temporary auth file for every test."""
    path = tmp_path / "allowlist.json"
    monkeypatch.setattr("bot.auth.AUTH_FILE", path)
    return path


class TestCmdUserRemove:
    def test_removes_existing_user(self, capsys):
        save_auth({"allowed_users": [123, 456], "allowed_groups": {}, "pending": {}})
        args = argparse.Namespace(user_id=123)
        cmd_user_remove(args)
        assert is_allowed(123) is False
        assert is_allowed(456) is True
        captured = capsys.readouterr()
        assert "123" in captured.out
        assert "removed" in captured.out.lower()

    def test_nonexistent_user_exits_1(self, capsys):
        args = argparse.Namespace(user_id=999)
        with pytest.raises(SystemExit) as exc_info:
            cmd_user_remove(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "999" in captured.out
        assert "not found" in captured.out.lower()


class TestCmdUserList:
    def test_lists_users(self, capsys):
        save_auth({"allowed_users": [111, 222], "allowed_groups": {}, "pending": {}})
        args = argparse.Namespace()
        cmd_user_list(args)
        captured = capsys.readouterr()
        assert "111" in captured.out
        assert "222" in captured.out

    def test_empty_list(self, capsys):
        args = argparse.Namespace()
        cmd_user_list(args)
        captured = capsys.readouterr()
        assert "no" in captured.out.lower()
