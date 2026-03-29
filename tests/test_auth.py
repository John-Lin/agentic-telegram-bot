from __future__ import annotations

import time

import pytest

from bot.auth import add_group
from bot.auth import confirm_pairing
from bot.auth import create_pairing_code
from bot.auth import get_dm_policy
from bot.auth import get_group_config
from bot.auth import is_allowed
from bot.auth import list_groups
from bot.auth import load_auth
from bot.auth import locked_auth
from bot.auth import remove_group
from bot.auth import remove_user
from bot.auth import save_auth
from bot.auth import set_dm_policy


@pytest.fixture(autouse=True)
def auth_file(tmp_path, monkeypatch):
    """Use a temporary auth file for every test."""
    path = tmp_path / "access.json"
    monkeypatch.setattr("bot.auth.AUTH_FILE", path)
    return path


class TestLoadSave:
    def test_load_returns_default_when_file_missing(self):
        data = load_auth()
        assert data == {
            "dmPolicy": "pairing",
            "allowFrom": [],
            "groups": {},
            "pending": {},
        }

    def test_save_and_load_roundtrip(self):
        data = {"dmPolicy": "pairing", "allowFrom": ["123"], "groups": {}, "pending": {}}
        save_auth(data)
        assert load_auth() == data

    def test_load_backfills_missing_dm_policy(self):
        """Old files without dmPolicy get the default."""
        save_auth({"allowFrom": ["123"], "groups": {}, "pending": {}})
        data = load_auth()
        assert data["dmPolicy"] == "pairing"


class TestIsAllowed:
    def test_unknown_user_not_allowed(self):
        assert is_allowed(999) is False

    def test_allowed_user(self):
        save_auth({"dmPolicy": "pairing", "allowFrom": ["123"], "groups": {}, "pending": {}})
        assert is_allowed(123) is True

    def test_allowed_user_compared_as_string(self):
        """User IDs stored as strings, is_allowed accepts int."""
        save_auth({"dmPolicy": "pairing", "allowFrom": ["123"], "groups": {}, "pending": {}})
        assert is_allowed(123) is True
        assert is_allowed(456) is False


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

    def test_confirm_pairing_adds_to_allow_from(self):
        code = create_pairing_code(123, "john")
        user_id = confirm_pairing(code)
        assert user_id == 123
        assert is_allowed(123) is True

    def test_confirm_pairing_stores_as_string(self):
        code = create_pairing_code(123, "john")
        confirm_pairing(code)
        data = load_auth()
        assert "123" in data["allowFrom"]

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
        assert data["allowFrom"].count("123") == 1


class TestGroupAccess:
    def test_add_group_default_config(self):
        add_group(-1001234)
        config = get_group_config(-1001234)
        assert config is not None
        assert config["requireMention"] is True
        assert config["allowFrom"] == []

    def test_add_group_no_mention(self):
        add_group(-1001234, require_mention=False)
        config = get_group_config(-1001234)
        assert config["requireMention"] is False

    def test_add_group_with_allowed_members(self):
        add_group(-1001234, allowed_members=[111, 222])
        config = get_group_config(-1001234)
        assert config["allowFrom"] == ["111", "222"]

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
        assert config["requireMention"] is False
        assert config["allowFrom"] == ["111"]


class TestLockedAuth:
    def test_locked_auth_loads_data(self):
        save_auth({"dmPolicy": "pairing", "allowFrom": ["123"], "groups": {}, "pending": {}})
        with locked_auth() as data:
            assert data["allowFrom"] == ["123"]

    def test_locked_auth_saves_on_exit(self):
        with locked_auth() as data:
            data["allowFrom"].append("456")
        assert load_auth()["allowFrom"] == ["456"]

    def test_locked_auth_does_not_save_on_exception(self):
        save_auth({"dmPolicy": "pairing", "allowFrom": ["123"], "groups": {}, "pending": {}})
        with pytest.raises(RuntimeError), locked_auth() as data:
            data["allowFrom"].append("999")
            raise RuntimeError("boom")
        # Original data should be preserved
        assert load_auth()["allowFrom"] == ["123"]


class TestPairingExpiration:
    def test_expired_code_returns_none(self, monkeypatch):
        code = create_pairing_code(123, "john")
        # Simulate 11 minutes passing
        data = load_auth()
        data["pending"][code]["created_at"] = time.time() - 660
        save_auth(data)
        assert confirm_pairing(code) is None

    def test_fresh_code_works(self):
        code = create_pairing_code(123, "john")
        user_id = confirm_pairing(code)
        assert user_id == 123

    def test_expired_code_is_cleaned_up(self):
        code = create_pairing_code(123, "john")
        data = load_auth()
        data["pending"][code]["created_at"] = time.time() - 660
        save_auth(data)
        confirm_pairing(code)
        # Expired code should be removed from pending
        data = load_auth()
        assert code not in data["pending"]

    def test_create_pairing_code_stores_created_at(self):
        before = time.time()
        code = create_pairing_code(123, "john")
        after = time.time()
        data = load_auth()
        assert "created_at" in data["pending"][code]
        assert before <= data["pending"][code]["created_at"] <= after


class TestRemoveUser:
    def test_remove_existing_user(self):
        save_auth({"dmPolicy": "pairing", "allowFrom": ["123", "456"], "groups": {}, "pending": {}})
        assert remove_user(123) is True
        assert is_allowed(123) is False
        assert is_allowed(456) is True

    def test_remove_nonexistent_user(self):
        assert remove_user(999) is False

    def test_remove_user_idempotent(self):
        save_auth({"dmPolicy": "pairing", "allowFrom": ["123"], "groups": {}, "pending": {}})
        assert remove_user(123) is True
        assert remove_user(123) is False


class TestDmPolicy:
    def test_default_policy_is_pairing(self):
        assert get_dm_policy() == "pairing"

    def test_set_policy_allowlist(self):
        set_dm_policy("allowlist")
        assert get_dm_policy() == "allowlist"

    def test_set_policy_disabled(self):
        set_dm_policy("disabled")
        assert get_dm_policy() == "disabled"

    def test_set_policy_pairing(self):
        set_dm_policy("disabled")
        set_dm_policy("pairing")
        assert get_dm_policy() == "pairing"

    def test_set_invalid_policy_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            set_dm_policy("bogus")

    def test_policy_persisted(self):
        set_dm_policy("allowlist")
        data = load_auth()
        assert data["dmPolicy"] == "allowlist"
