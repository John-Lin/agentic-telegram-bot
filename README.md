# agentic-telegram-bot

A simple Telegram bot that uses the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) to interact with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers.

See also: [agentic-slackbot](https://github.com/John-Lin/agentic-slackbot) — a similar demo bot for Slack.

## Features

- Private chat and group chat support
- Configurable DM policy (pairing / allowlist / disabled)
- Connects to any MCP server via `servers_config.json`
- Supports OpenAI, Azure OpenAI endpoints
- Per-conversation history with automatic truncation
- Group reply chain — after `@mention`, anyone can continue by replying
- Optional local shell via `ShellTool`, controlled by `SHELL_ENABLED` and `SHELL_SKILLS_DIR`

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
export OPENAI_MODEL="gpt-5.4"

# Local shell (disabled by default)
# export SHELL_ENABLED=1
# export SHELL_SKILLS_DIR="./skills"  # optional; mount skills alongside the shell

# Optional verbose OpenAI Agents SDK logging
# export AGENT_VERBOSE_LOG=1
```

## Agent Instructions

The bot loads its system prompt from `instructions.md` in the project root.
If the file is missing, the bot fails fast at startup.

You can copy `instructions.md.example` as a starting point:

```bash
cp instructions.md.example instructions.md
```

If you are using Azure OpenAI (v1 API), set these instead:

```
export OPENAI_API_KEY=""
export OPENAI_BASE_URL="https://<resource-name>.openai.azure.com/openai/v1/"
export OPENAI_MODEL="gpt-5.4"
```

## MCP Server Configuration (Optional)

Create a `servers_config.json` file to add your MCP servers. If this file is not provided, the bot starts with no MCP servers configured.

```json
{
  "mcpServers": {
    "my-server": {
      "command": "uvx",
      "args": ["my-mcp-server"]
    }
  }
}
```

For HTTP-based MCP servers (Streamable HTTP), use `url`:

```json
{
  "mcpServers": {
    "my-server": {
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

For local MCP servers, use `uv --directory`:

```json
{
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

## Local Shell (Optional)

The bot can expose a local `ShellTool`. This is **disabled by default**. Enable it with:

```
export SHELL_ENABLED=1
```

With just `SHELL_ENABLED=1`, the agent gets bare local shell access with no pre-defined skills.

### Shell Skills (Optional)

You can optionally mount a skills directory alongside the shell. Each immediate subdirectory containing a `SKILL.md` file is registered as a skill and exposed to the agent as a hint (skills are advisory metadata — they do **not** sandbox command execution).

```
export SHELL_ENABLED=1
export SHELL_SKILLS_DIR="./skills"
```

`SHELL_SKILLS_DIR` is ignored unless `SHELL_ENABLED` is set. If the directory is missing or contains no valid skills, the bot falls back to a bare shell and logs a warning.

The `SKILL.md` file should have YAML frontmatter with `name` and `description` fields:

```markdown
---
name: my-skill
description: A brief description of what this skill does
---

Detailed instructions for the agent...
```

## Docker

```bash
docker build -t agentic-telegram-bot .

docker run -d \
  --name telegent \
  -e BOT_USERNAME="@your_bot_username" \
  -e TELEGRAM_BOT_TOKEN="" \
  -e OPENAI_API_KEY="" \
  -e OPENAI_MODEL="gpt-5.4" \
  -v /path/to/instructions.md:/app/instructions.md \
  -v /path/to/access.json:/app/access.json \
  agentic-telegram-bot
```

To use MCP servers, mount your config file:

```bash
docker run -d \
  --name telegent \
  -e BOT_USERNAME="@your_bot_username" \
  -e TELEGRAM_BOT_TOKEN="" \
  -e OPENAI_API_KEY="" \
  -e OPENAI_MODEL="gpt-5.4" \
  -v /path/to/instructions.md:/app/instructions.md \
  -v /path/to/servers_config.json:/app/servers_config.json \
  -v /path/to/access.json:/app/access.json \
  agentic-telegram-bot
```
