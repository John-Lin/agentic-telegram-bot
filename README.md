# agentic-telegram-bot

A simple Telegram bot that uses the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) to interact with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers.

See also: [agentic-slackbot](https://github.com/John-Lin/agentic-slackbot) — a similar demo bot for Slack.

## Features

- Private chat and group chat support
- Configurable DM policy (pairing / allowlist / disabled)
- Connects to any MCP server via `servers_config.json`
- Supports OpenAI, Azure OpenAI endpoints
- Per-conversation history with automatic truncation
- Per-user rate limiting

## Install Dependencies

```bash
uv sync
```

## Telegram Bot Setup

1. Create a new bot using the [BotFather](https://t.me/botfather) on Telegram.
2. Get the bot token and username.
3. Setting for [privacy mode](https://core.telegram.org/bots/features#privacy-mode):
   - Use the command `/setprivacy` in the BotFather chat.
   - Select your bot.
   - Choose "Disable" to allow the bot to receive all messages in groups.
4. Set the bot token and username in the `.envrc` or `.env` file.

## Environment Variables

Create a `.envrc` or `.env` file in the root directory:

```
# Telegram bot
export BOT_USERNAME="@your_bot_username"
export TELEGRAM_BOT_TOKEN=""

# OpenAI API
export OPENAI_API_KEY=""
export OPENAI_MODEL="gpt-4.1"
```

If you are using Azure OpenAI, set these instead:

```
export AZURE_OPENAI_API_KEY=""
export AZURE_OPENAI_ENDPOINT="https://<myopenai>.azure.com/"
export OPENAI_MODEL="gpt-4.1"
export OPENAI_API_VERSION="2025-03-01-preview"
```

## MCP Server Configuration

Edit `servers_config.json` to add your MCP servers:

```json
{
  "instructions": "Your custom system prompt here.",
  "mcpServers": {
    "my-server": {
      "command": "uvx",
      "args": ["my-mcp-server"]
    }
  }
}
```

For local MCP servers, use `uv --directory`:

```json
{
  "instructions": "Your custom system prompt here.",
  "mcpServers": {
    "my-server": {
      "command": "uv",
      "args": ["--directory", "/path/to/my-server", "run", "my-entrypoint"]
    }
  }
}
```

## Running the Bot

```bash
uv run bot
```

## Access Control

All access is managed via `access.json` (auto-created, gitignored).

### DM Policy

The bot supports three DM policies:

| Policy | Behaviour |
|---|---|
| `pairing` (default) | Unknown users receive a 6-character pairing code |
| `allowlist` | Unknown users are silently ignored |
| `disabled` | All messages dropped, including allowed users and groups |

```bash
# Show current policy
uv run bot access policy

# Set policy
uv run bot access policy <pairing|allowlist|disabled>
```

### Users

```bash
# Directly allow a user by ID
uv run bot access allow <USER_ID>

# Remove a user
uv run bot access remove <USER_ID>
```

When `dmPolicy` is `pairing`, unknown users receive a 6-character code via DM. Confirm in your terminal:

```bash
uv run bot access pair <CODE>
```

### Groups

Groups are blocked by default.

```bash
# Add a group (default: bot responds only to @mentions)
uv run bot access group add <GROUP_ID>

# Respond to all messages, not just @mentions
uv run bot access group add <GROUP_ID> --no-mention

# Restrict to specific members
uv run bot access group add <GROUP_ID> --allow 111,222

# Remove a group
uv run bot access group remove <GROUP_ID>
```

Group members do not need to pair individually — access is controlled at the group level.

## Docker

```bash
docker build -t agentic-telegram-bot .

docker run -d \
  --name telegent \
  -e BOT_USERNAME="@your_bot_username" \
  -e TELEGRAM_BOT_TOKEN="" \
  -e OPENAI_API_KEY="" \
  -e OPENAI_MODEL="gpt-4.1" \
  -v /path/to/servers_config.json:/app/servers_config.json \
  -v /path/to/access.json:/app/access.json \
  agentic-telegram-bot
```
