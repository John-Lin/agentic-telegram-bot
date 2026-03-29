from __future__ import annotations

import json
import secrets
import string
from pathlib import Path
from typing import Any

AUTH_FILE: str | Path = "allowlist.json"


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


# --- User pairing ---


def is_allowed(user_id: int) -> bool:
    return user_id in load_auth()["allowed_users"]


def create_pairing_code(user_id: int, username: str) -> str:
    alphabet = string.ascii_uppercase + string.digits
    code = "".join(secrets.choice(alphabet) for _ in range(6))
    data = load_auth()
    data["pending"][code] = {"user_id": user_id, "username": username}
    save_auth(data)
    return code


def confirm_pairing(code: str) -> int | None:
    data = load_auth()
    if code not in data["pending"]:
        return None
    user_id = data["pending"].pop(code)["user_id"]
    if user_id not in data["allowed_users"]:
        data["allowed_users"].append(user_id)
    save_auth(data)
    return user_id


# --- Group access ---


def get_group_config(group_id: int) -> dict[str, Any] | None:
    groups = load_auth()["allowed_groups"]
    return groups.get(str(group_id))


def add_group(
    group_id: int,
    require_mention: bool = True,
    allowed_members: list[int] | None = None,
) -> None:
    data = load_auth()
    data["allowed_groups"][str(group_id)] = {
        "require_mention": require_mention,
        "allowed_members": allowed_members or [],
    }
    save_auth(data)


def remove_group(group_id: int) -> bool:
    data = load_auth()
    if str(group_id) not in data["allowed_groups"]:
        return False
    del data["allowed_groups"][str(group_id)]
    save_auth(data)
    return True


def list_groups() -> dict[str, Any]:
    return load_auth()["allowed_groups"]
