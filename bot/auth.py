from __future__ import annotations

import fcntl
import json
import secrets
import string
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

AUTH_FILE: str | Path = "access.json"

PAIRING_CODE_TTL_SECONDS = 600  # 10 minutes

VALID_DM_POLICIES = ("pairing", "allowlist", "disabled")


def _default_auth() -> dict[str, Any]:
    return {"dmPolicy": "pairing", "allowFrom": [], "groups": {}, "pending": {}}


def load_auth() -> dict[str, Any]:
    path = Path(AUTH_FILE)
    if not path.exists():
        return _default_auth()
    with open(path) as f:
        data = json.load(f)
    # Backfill missing keys for older files
    for key, default in _default_auth().items():
        data.setdefault(key, type(default)() if not isinstance(default, str) else default)
    return data


def save_auth(data: dict[str, Any]) -> None:
    with open(Path(AUTH_FILE), "w") as f:
        json.dump(data, f, indent=2)


@contextmanager
def locked_auth() -> Generator[dict[str, Any]]:
    """Load auth data under an exclusive file lock; save only on clean exit."""
    lock_path = Path(str(AUTH_FILE) + ".lock")
    lock_path.touch(exist_ok=True)
    with open(lock_path) as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            data = load_auth()
            yield data
            save_auth(data)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


# --- DM policy ---


def get_dm_policy() -> str:
    return load_auth()["dmPolicy"]


def set_dm_policy(policy: str) -> None:
    if policy not in VALID_DM_POLICIES:
        raise ValueError(f"invalid dmPolicy: {policy!r} (must be one of {VALID_DM_POLICIES})")
    with locked_auth() as data:
        data["dmPolicy"] = policy


# --- User pairing ---


def is_allowed(user_id: int) -> bool:
    return str(user_id) in load_auth()["allowFrom"]


def create_pairing_code(user_id: int, username: str) -> str:
    alphabet = string.ascii_uppercase + string.digits
    code = "".join(secrets.choice(alphabet) for _ in range(6))
    with locked_auth() as data:
        data["pending"][code] = {
            "user_id": user_id,
            "username": username,
            "created_at": time.time(),
        }
    return code


def confirm_pairing(code: str) -> int | None:
    with locked_auth() as data:
        if code not in data["pending"]:
            return None
        entry = data["pending"][code]
        # Check expiration
        created_at = entry.get("created_at", 0)
        if time.time() - created_at > PAIRING_CODE_TTL_SECONDS:
            del data["pending"][code]
            return None
        user_id = data["pending"].pop(code)["user_id"]
        uid_str = str(user_id)
        if uid_str not in data["allowFrom"]:
            data["allowFrom"].append(uid_str)
    return user_id


def allow_user(user_id: int) -> bool:
    """Add a user to allowFrom. Returns False if already present."""
    uid_str = str(user_id)
    with locked_auth() as data:
        if uid_str in data["allowFrom"]:
            return False
        data["allowFrom"].append(uid_str)
    return True


def remove_user(user_id: int) -> bool:
    uid_str = str(user_id)
    with locked_auth() as data:
        if uid_str not in data["allowFrom"]:
            return False
        data["allowFrom"].remove(uid_str)
    return True


# --- Group access ---


def get_group_config(group_id: int) -> dict[str, Any] | None:
    groups = load_auth()["groups"]
    return groups.get(str(group_id))


def add_group(
    group_id: int,
    require_mention: bool = True,
    allowed_members: list[int] | None = None,
) -> None:
    with locked_auth() as data:
        data["groups"][str(group_id)] = {
            "requireMention": require_mention,
            "allowFrom": [str(m) for m in (allowed_members or [])],
        }


def remove_group(group_id: int) -> bool:
    with locked_auth() as data:
        if str(group_id) not in data["groups"]:
            return False
        del data["groups"][str(group_id)]
    return True


def list_groups() -> dict[str, Any]:
    return load_auth()["groups"]
