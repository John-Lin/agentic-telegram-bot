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

AUTH_FILE: str | Path = "allowlist.json"

PAIRING_CODE_TTL_SECONDS = 600  # 10 minutes


def _default_auth() -> dict[str, Any]:
    return {"allowed_users": [], "allowed_groups": {}, "pending": {}}


def load_auth() -> dict[str, Any]:
    path = Path(AUTH_FILE)
    if not path.exists():
        return _default_auth()
    with open(path) as f:
        data = json.load(f)
    # Backfill missing keys for older files
    for key, default in _default_auth().items():
        data.setdefault(key, type(default)())
    return data


def save_auth(data: dict[str, Any]) -> None:
    with open(Path(AUTH_FILE), "w") as f:
        json.dump(data, f, indent=2)


@contextmanager
def locked_auth() -> Generator[dict[str, Any], None, None]:
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


# --- User pairing ---


def is_allowed(user_id: int) -> bool:
    return user_id in load_auth()["allowed_users"]


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
        if user_id not in data["allowed_users"]:
            data["allowed_users"].append(user_id)
    return user_id


def remove_user(user_id: int) -> bool:
    with locked_auth() as data:
        if user_id not in data["allowed_users"]:
            return False
        data["allowed_users"].remove(user_id)
    return True


# --- Group access ---


def get_group_config(group_id: int) -> dict[str, Any] | None:
    groups = load_auth()["allowed_groups"]
    return groups.get(str(group_id))


def add_group(
    group_id: int,
    require_mention: bool = True,
    allowed_members: list[int] | None = None,
) -> None:
    with locked_auth() as data:
        data["allowed_groups"][str(group_id)] = {
            "require_mention": require_mention,
            "allowed_members": allowed_members or [],
        }


def remove_group(group_id: int) -> bool:
    with locked_auth() as data:
        if str(group_id) not in data["allowed_groups"]:
            return False
        del data["allowed_groups"][str(group_id)]
    return True


def list_groups() -> dict[str, Any]:
    return load_auth()["allowed_groups"]
