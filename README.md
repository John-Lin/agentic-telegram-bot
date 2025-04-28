# agentic-telegram-bot
A simple Telegram bot that uses the OpenAI Agents SDK to interact with the Model Context Protocol (MCP) server.

## Install Dependencies

```bash
uv sync
```


## Environment Variables

Create a `.envrc` file in the root directory of the project and add the following environment variables:

```
export OPENAI_API_KEY=""
export TELEGRAM_BOT_TOKEN=""
export OPENAI_MODEL="gpt-4o"
```

## Running the Bot

```bash
uv run main.py
