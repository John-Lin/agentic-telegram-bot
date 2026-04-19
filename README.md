# agentic-telegram-bot

A Telegram bot powered by [agent-core](https://github.com/John-Lin/agent-core) that interacts with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers. Supports two interchangeable providers:

- **OpenAI** (default) — via [OpenAI Agents SDK](https://github.com/openai/openai-agents-python); works with OpenAI and Azure OpenAI v1.
- **Claude** — via [claude-agent-sdk](https://pypi.org/project/claude-agent-sdk/).

Switch providers by setting `"provider"` in `agent_config.json`.

See also: [agentic-slackbot](https://github.com/John-Lin/agentic-slackbot) and [agentic-discord-bot](https://github.com/John-Lin/agentic-discord-bot) — similar bots for Slack and Discord.

## Features

- Private chat and group chat support
- Configurable DM policy (pairing / allowlist / disabled)
- Per-conversation history with automatic truncation
- Group reply chain — after `@mention`, anyone can continue by replying
- Connects to any MCP server via `agent_config.json`
- Optional local shell: `ShellTool` (OpenAI, via `provider.shell`) or `Bash`/`Write`/`Edit`/… (Claude, via `provider.allowedTools`). Read-only built-ins (`Read`, `Glob`, `Grep`) are always on for Claude.
- Supports OpenAI, Azure OpenAI v1, and Anthropic Claude

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
4. Set the bot token and username in the `.env` file.

## Environment Variables

Create a `.env` file in the root directory. Set the key(s) for the provider you plan to use:

```
# Telegram bot
BOT_USERNAME="@your_bot_username"
TELEGRAM_BOT_TOKEN=""

# OpenAI provider (default)
OPENAI_API_KEY=""

# Claude provider
# ANTHROPIC_API_KEY=""

# Optional: override SQLite path for session storage (in-memory by default)
# SESSION_DB_PATH="./sessions.db"

# Optional: override the path to the instructions file (default ./instructions.md)
# AGENT_INSTRUCTIONS_PATH="./instructions.md"

# Optional verbose OpenAI Agents SDK logging (OpenAI only)
# AGENT_VERBOSE_LOG=1
```

If you are using Azure OpenAI (v1 API):

```
BOT_USERNAME="@your_bot_username"
TELEGRAM_BOT_TOKEN=""
OPENAI_API_KEY=""
OPENAI_BASE_URL="https://<resource-name>.openai.azure.com/openai/v1/"
```

## Agent Instructions

Create an `instructions.md` file in the project root with the agent system prompt:

```markdown
You are a helpful financial assistant. Help users look up stock data,
news, and market information. Always include ticker symbols.
Respond in the user's language. Keep responses concise.
```

An example is provided in `instructions.md.example`. The bot will fail to start if this file is missing.

## Provider & MCP Server Configuration (Optional)

Create an `agent_config.json` to choose a provider and connect MCP servers. If the file is absent, the bot starts with the default OpenAI provider and no tools.

`provider` is a tagged union keyed by `type` (`"openai"` or `"anthropic"`). `mcp` uses an opencode-style schema keyed by server name, with `type: "local" | "remote"`.

### OpenAI provider (default)

```json
{
  "provider": {
    "type": "openai",
    "model": "gpt-5.4",
    "apiType": "responses",
    "historyTurns": 10
  },
  "mcp": {
    "my-server": {
      "type": "local",
      "command": ["uvx", "my-mcp-server"]
    }
  }
}
```

All `provider` fields are optional (`model` defaults to `gpt-5.4`, `apiType` to `"responses"`, `historyTurns` to `10`). Each MCP entry also accepts `timeout` (seconds, default `30.0`) and `enabled` (default `true`).

### Claude provider

```json
{
  "provider": {
    "type": "anthropic",
    "model": "claude-sonnet-4-6",
    "allowedTools": ["WebFetch"]
  },
  "mcp": {
    "my-stdio": {
      "type": "local",
      "command": ["python", "-m", "srv"],
      "environment": {"FOO": "bar"}
    },
    "my-http": {
      "type": "remote",
      "url": "https://example.com/mcp",
      "headers": {"Authorization": "Bearer x"}
    }
  }
}
```

Requires `ANTHROPIC_API_KEY`. Read-only built-ins (`Read`, `Glob`, `Grep`) are always on; `allowedTools` extends that set with any tool that can mutate files or run commands (`Bash`, `Write`, `Edit`, `WebFetch`, …). Tool names are case-sensitive and validated by the SDK — an unrecognized name is silently dropped. Billing/rate-limit/`error_max_turns` errors are surfaced to the chat as a readable message via `AgentError`.

### Remote (HTTP) MCP servers

```json
{
  "mcp": {
    "my-server": {
      "type": "remote",
      "url": "https://mcp.example.com/mcp",
      "headers": {"Accept": "application/json, text/event-stream"}
    }
  }
}
```

### Local MCP servers (via `uv --directory`)

```json
{
  "mcp": {
    "my-server": {
      "type": "local",
      "command": ["uv", "--directory", "/path/to/my-server", "run", "my-entrypoint"]
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

## Conversation History

Each chat maintains its own conversation history. Replying to the bot's message continues the same conversation via the group reply chain.

OpenAI history length is controlled by `provider.historyTurns` in `agent_config.json` (default `10`). Claude history is managed on disk by `claude-agent-sdk` and resumed across restarts via a `chat_id -> session_id` mapping in SQLite (`SESSION_DB_PATH`).

## Local Shell (Optional)

Local shell tools are **disabled by default** and are configured in `agent_config.json` per provider.

### OpenAI — `provider.shell`

```json
{
  "provider": {
    "type": "openai",
    "shell": {
      "enabled": true,
      "skillsDir": "./skills"
    }
  }
}
```

`provider.shell.enabled` must be a bool (strings are rejected). `provider.shell.skillsDir` is optional and mounts a skills directory alongside the `ShellTool`.

### Claude — `provider.allowedTools`

Read-only built-ins (`Read`, `Glob`, `Grep`) are always on. Add mutating or exec-capable tools explicitly:

```json
{
  "provider": {
    "type": "anthropic",
    "allowedTools": ["Bash", "Write", "Edit", "WebFetch"]
  }
}
```

### Shell Skills (OpenAI only)

Each immediate subdirectory of `skillsDir` containing a `SKILL.md` file is registered as a skill and exposed to the agent as a hint (skills are advisory metadata — they do **not** sandbox command execution). If the directory is missing or contains no valid skills, the bot falls back to a bare shell and logs a warning.

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

# OpenAI provider
docker run -d \
  --name agentic-telegram-bot \
  -e BOT_USERNAME="@your_bot_username" \
  -e TELEGRAM_BOT_TOKEN="" \
  -e OPENAI_API_KEY="" \
  -v /path/to/instructions.md:/app/instructions.md \
  -v /path/to/access.json:/app/access.json \
  agentic-telegram-bot

# Claude provider (agent_config.json must set "provider": {"type": "anthropic"})
docker run -d \
  --name agentic-telegram-bot \
  -e BOT_USERNAME="@your_bot_username" \
  -e TELEGRAM_BOT_TOKEN="" \
  -e ANTHROPIC_API_KEY="" \
  -v /path/to/instructions.md:/app/instructions.md \
  -v /path/to/agent_config.json:/app/agent_config.json \
  -v /path/to/access.json:/app/access.json \
  agentic-telegram-bot
```

To use MCP servers with OpenAI, also mount the config:

```bash
docker run -d \
  --name agentic-telegram-bot \
  -e BOT_USERNAME="@your_bot_username" \
  -e TELEGRAM_BOT_TOKEN="" \
  -e OPENAI_API_KEY="" \
  -v /path/to/instructions.md:/app/instructions.md \
  -v /path/to/agent_config.json:/app/agent_config.json \
  -v /path/to/access.json:/app/access.json \
  agentic-telegram-bot
```

### Docker Compose

A `docker-compose.yml` and `run.sh` are provided for convenience. Both mount `instructions.md`, `agent_config.json`, `access.json`, persist sessions to `./data`, and mount `./skills` as the agent skills directory.

```bash
docker compose up -d --build
```

Or run the container directly:

```bash
./run.sh
```
