from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys

from bot.agents import OpenAIAgent
from bot.auth import add_group
from bot.auth import allow_user
from bot.auth import confirm_pairing
from bot.auth import get_dm_policy
from bot.auth import remove_group
from bot.auth import remove_user
from bot.auth import set_dm_policy
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
    print(f"Paired successfully! User ID {user_id} has been added.")


def cmd_allow(args: argparse.Namespace) -> None:
    if allow_user(args.user_id):
        print(f"User {args.user_id} allowed.")
    else:
        print(f"User {args.user_id} already allowed.")


def cmd_remove(args: argparse.Namespace) -> None:
    if remove_user(args.user_id):
        print(f"User {args.user_id} removed.")
    else:
        print(f"User {args.user_id} not found.")
        sys.exit(1)


def cmd_policy(args: argparse.Namespace) -> None:
    """Show or set the DM policy."""
    if args.policy is None:
        print(f"Current dmPolicy: {get_dm_policy()}")
        return
    try:
        set_dm_policy(args.policy)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    print(f"dmPolicy set to: {args.policy}")


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


def _dispatch_access(
    args: argparse.Namespace,
    access_parser: argparse.ArgumentParser,
    group_parser: argparse.ArgumentParser,
) -> None:
    """Route access subcommands to their handlers."""
    if args.access_command == "pair":
        cmd_pair(args)
    elif args.access_command == "policy":
        cmd_policy(args)
    elif args.access_command == "allow":
        cmd_allow(args)
    elif args.access_command == "remove":
        cmd_remove(args)
    elif args.access_command == "group":
        if args.group_command == "add":
            cmd_group_add(args)
        elif args.group_command == "remove":
            cmd_group_remove(args)
        else:
            group_parser.print_help()
    else:
        access_parser.print_help()


def run():
    parser = argparse.ArgumentParser(description="Agentic Telegram Bot")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Start the Telegram bot")

    # access subcommands
    access_parser = subparsers.add_parser("access", help="Manage access control")
    access_sub = access_parser.add_subparsers(dest="access_command")

    # access pair subcommand
    pair_parser = access_sub.add_parser("pair", help="Confirm a pairing code")
    pair_parser.add_argument("code", help="The 6-character pairing code")

    # access policy subcommand
    policy_parser = access_sub.add_parser("policy", help="Show or set DM policy")
    policy_parser.add_argument(
        "policy",
        nargs="?",
        choices=["pairing", "allowlist", "disabled"],
        default=None,
        help="Set DM policy (omit to show current)",
    )

    # access allow / remove
    allow_parser = access_sub.add_parser("allow", help="Allow a user")
    allow_parser.add_argument("user_id", type=int, help="Telegram user ID")

    remove_parser = access_sub.add_parser("remove", help="Remove a user")
    remove_parser.add_argument("user_id", type=int, help="Telegram user ID")

    # access group subcommands
    group_parser = access_sub.add_parser("group", help="Manage group access")
    group_sub = group_parser.add_subparsers(dest="group_command")

    group_add = group_sub.add_parser("add", help="Add a group")
    group_add.add_argument("group_id", type=int, help="Telegram group ID (e.g. -1001654782309)")
    group_add.add_argument("--no-mention", action="store_true", help="Respond to all messages, not just @mentions")
    group_add.add_argument("--allow", type=str, help="Comma-separated user IDs that can trigger the bot")

    group_rm = group_sub.add_parser("remove", help="Remove a group")
    group_rm.add_argument("group_id", type=int, help="Telegram group ID")

    args = parser.parse_args()

    if args.command == "access":
        _dispatch_access(args, access_parser, group_parser)
    else:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(start_bot())
