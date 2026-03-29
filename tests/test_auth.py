from __future__ import annotations

import pytest

from bot.auth import add_group
from bot.auth import confirm_pairing
from bot.auth import create_pairing_code
from bot.auth import get_group_config
from bot.auth import is_allowed
from bot.auth import list_groups
from bot.auth import load_auth
from bot.auth import remove_group
from bot.auth import save_auth


@pytest.fixture(autouse=True)
def auth_file(tmp_path, monkeypatch):
    """Use a temporary auth file for every test."""
    path = tmp_path / "allowlist.json"
    monkeypatch.setattr("bot.auth.AUTH_FILE", path)
    return path


class TestLoadSave:
    def test_load_returns_default_when_file_missing(self):
        data = load_auth()
        assert data == {"allowed_users": [], "allowed_groups": {}, "pending": {}}

    def test_save_and_load_roundtrip(self):
        data = {"allowed_users": [123], "allowed_groups": {}, "pending": {}}
        save_auth(data)
        assert load_auth() == data


class TestIsAllowed:
    def test_unknown_user_not_allowed(self):
        assert is_allowed(999) is False

    def test_allowed_user(self):
        save_auth({"allowed_users": [123], "allowed_groups": {}, "pending": {}})
        assert is_allowed(123) is True


class TestPairing:
    def test_create_pairing_code_returns_6_chars(self):
        code = create_pairing_code(123, "john")
        assert len(code) == 6
        assert code.isalnum()

    def test_create_pairing_code_stores_pending(self):
        code = create_pairing_code(123, "john")
        data = load_auth()
        assert code in data["pending"]
        assert data["pending"][code]["user_id"] == 123
        assert data["pending"][code]["username"] == "john"

    def test_confirm_pairing_adds_to_allowlist(self):
        code = create_pairing_code(123, "john")
        user_id = confirm_pairing(code)
        assert user_id == 123
        assert is_allowed(123) is True

    def test_confirm_pairing_removes_pending(self):
        code = create_pairing_code(123, "john")
        confirm_pairing(code)
        data = load_auth()
        assert code not in data["pending"]

    def test_confirm_pairing_invalid_code_returns_none(self):
        assert confirm_pairing("BADCODE") is None

    def test_confirm_pairing_is_one_time(self):
        code = create_pairing_code(123, "john")
        confirm_pairing(code)
        assert confirm_pairing(code) is None

    def test_multiple_users_can_pair(self):
        code1 = create_pairing_code(111, "alice")
        code2 = create_pairing_code(222, "bob")
        confirm_pairing(code1)
        confirm_pairing(code2)
        assert is_allowed(111) is True
        assert is_allowed(222) is True

    def test_duplicate_user_not_added_twice(self):
        code1 = create_pairing_code(123, "john")
        confirm_pairing(code1)
        code2 = create_pairing_code(123, "john")
        confirm_pairing(code2)
        data = load_auth()
        assert data["allowed_users"].count(123) == 1


class TestGroupAccess:
    def test_add_group_default_config(self):
        add_group(-1001234)
        config = get_group_config(-1001234)
        assert config is not None
        assert config["require_mention"] is True
        assert config["allowed_members"] == []

    def test_add_group_no_mention(self):
        add_group(-1001234, require_mention=False)
        config = get_group_config(-1001234)
        assert config["require_mention"] is False

    def test_add_group_with_allowed_members(self):
        add_group(-1001234, allowed_members=[111, 222])
        config = get_group_config(-1001234)
        assert config["allowed_members"] == [111, 222]

    def test_get_group_config_unknown_group(self):
        assert get_group_config(-9999) is None

    def test_remove_group(self):
        add_group(-1001234)
        removed = remove_group(-1001234)
        assert removed is True
        assert get_group_config(-1001234) is None

    def test_remove_group_unknown(self):
        assert remove_group(-9999) is False

    def test_list_groups_empty(self):
        assert list_groups() == {}

    def test_list_groups(self):
        add_group(-1001234)
        add_group(-1005678, require_mention=False)
        groups = list_groups()
        assert len(groups) == 2
        assert str(-1001234) in groups
        assert str(-1005678) in groups

    def test_add_group_updates_existing(self):
        add_group(-1001234)
        add_group(-1001234, require_mention=False, allowed_members=[111])
        config = get_group_config(-1001234)
        assert config["require_mention"] is False
        assert config["allowed_members"] == [111]
