from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys

from bot.agents import OpenAIAgent
from bot.auth import add_group
from bot.auth import confirm_pairing
from bot.auth import list_groups
from bot.auth import load_auth
from bot.auth import remove_group
from bot.auth import remove_user
from bot.config import Configuration
from bot.telegram import TelegramMCPBot


async def start_bot() -> None:
    """Initialize and run the Telegram bot."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    config = Configuration()

    server_config = config.load_config("servers_config.json")
    openai_agent = OpenAIAgent.from_dict("Telegram Bot Agent", server_config)

    tg_bot = TelegramMCPBot(
        config.telegram_bot_token,
        config.bot_username,
        openai_agent,
    )

    try:
        await tg_bot.run()
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        await tg_bot.cleanup()
        await openai_agent.cleanup()


def cmd_pair(args: argparse.Namespace) -> None:
    """Confirm a pairing code from the terminal."""
    user_id = confirm_pairing(args.code)
    if user_id is None:
        print(f"Invalid or expired pairing code: {args.code}")
        sys.exit(1)
    print(f"Paired successfully! User ID {user_id} has been added to the allowlist.")


def cmd_group_add(args: argparse.Namespace) -> None:
    allowed_members = []
    if args.allow:
        allowed_members = [int(x) for x in args.allow.split(",")]
    add_group(args.group_id, require_mention=not args.no_mention, allowed_members=allowed_members)
    print(f"Group {args.group_id} added.")


def cmd_group_remove(args: argparse.Namespace) -> None:
    if remove_group(args.group_id):
        print(f"Group {args.group_id} removed.")
    else:
        print(f"Group {args.group_id} not found.")
        sys.exit(1)


def cmd_group_list(args: argparse.Namespace) -> None:
    groups = list_groups()
    if not groups:
        print("No groups configured.")
        return
    for gid, config in groups.items():
        mention = "mention" if config["require_mention"] else "all messages"
        members = config["allowed_members"]
        members_str = ", ".join(str(m) for m in members) if members else "all paired users"
        print(f"  {gid}  trigger={mention}  members={members_str}")


def cmd_user_remove(args: argparse.Namespace) -> None:
    if remove_user(args.user_id):
        print(f"User {args.user_id} removed from allowlist.")
    else:
        print(f"User {args.user_id} not found.")
        sys.exit(1)


def cmd_user_list(args: argparse.Namespace) -> None:
    data = load_auth()
    users = data["allowed_users"]
    if not users:
        print("No allowed users.")
        return
    for uid in users:
        print(f"  {uid}")


def _dispatch_access(
    args: argparse.Namespace,
    access_parser: argparse.ArgumentParser,
    user_parser: argparse.ArgumentParser,
    group_parser: argparse.ArgumentParser,
) -> None:
    """Route access subcommands to their handlers."""
    if args.access_command == "user":
        if args.user_command == "remove":
            cmd_user_remove(args)
        elif args.user_command == "list":
            cmd_user_list(args)
        else:
            user_parser.print_help()
    elif args.access_command == "group":
        if args.group_command == "add":
            cmd_group_add(args)
        elif args.group_command == "remove":
            cmd_group_remove(args)
        elif args.group_command == "list":
            cmd_group_list(args)
        else:
            group_parser.print_help()
    else:
        access_parser.print_help()


def run():
    parser = argparse.ArgumentParser(description="Agentic Telegram Bot")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Start the Telegram bot")

    pair_parser = subparsers.add_parser("pair", help="Confirm a pairing code")
    pair_parser.add_argument("code", help="The 6-character pairing code")

    # access group subcommands
    access_parser = subparsers.add_parser("access", help="Manage access control")
    access_sub = access_parser.add_subparsers(dest="access_command")

    # access user subcommands
    user_parser = access_sub.add_parser("user", help="Manage user access")
    user_sub = user_parser.add_subparsers(dest="user_command")

    user_rm = user_sub.add_parser("remove", help="Remove a user from the allowlist")
    user_rm.add_argument("user_id", type=int, help="Telegram user ID")

    user_sub.add_parser("list", help="List allowed users")

    # access group subcommands
    group_parser = access_sub.add_parser("group", help="Manage group access")
    group_sub = group_parser.add_subparsers(dest="group_command")

    group_add = group_sub.add_parser("add", help="Add a group to the allowlist")
    group_add.add_argument("group_id", type=int, help="Telegram group ID (e.g. -1001654782309)")
    group_add.add_argument("--no-mention", action="store_true", help="Respond to all messages, not just @mentions")
    group_add.add_argument("--allow", type=str, help="Comma-separated user IDs that can trigger the bot")

    group_rm = group_sub.add_parser("remove", help="Remove a group from the allowlist")
    group_rm.add_argument("group_id", type=int, help="Telegram group ID")

    group_sub.add_parser("list", help="List allowed groups")

    args = parser.parse_args()

    if args.command == "pair":
        cmd_pair(args)
    elif args.command == "access":
        _dispatch_access(args, access_parser, user_parser, group_parser)
    else:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(start_bot())
