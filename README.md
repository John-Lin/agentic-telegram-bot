# agentic-telegram-bot
A simple Telegram bot that uses the OpenAI Agents SDK to interact with the Model Context Protocol (MCP) server.

## Install Dependencies

```bash
uv sync
```


## Environment Variables

Create a `.envrc` file in the root directory of the project and add the following environment variables:

```
# Telegram bot
export BOT_USERNAME=""
export TELEGRAM_BOT_TOKEN=""

# OpenAI API
export OPENAI_API_KEY=""
export OPENAI_MODEL="gpt-4.1"

# Langfuse API key (Optional)
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_SECRET_KEY=""
LANGFUSE_HOST=""

# Firecrawl API key(Optional)
FIRECRAWL_API_KEY=""
```

If you are using Azure OpenAI, you can set the following environment variables instead:

```
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_ENDPOINT="https://<myopenai>.azure.com/"
OPENAI_MODEL="gpt-4.1"
AZURE_OPENAI_API_VERSION="2025-03-01-preview"
```

## Running the Bot

```bash
uv run main.py
```
