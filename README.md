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

If you are using Azure OpenAI, you can set the following environment variables instead:

```
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_ENDPOINT="https://<myopenai>.azure.com/"
OPENAI_MODEL="gpt-4o"
OPENAI_API_VERSION="2024-12-01-preview"
```

## Running the Bot

```bash
uv run main.py
