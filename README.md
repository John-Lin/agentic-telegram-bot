# agentic-telegram-bot
A simple Telegram bot that uses the OpenAI Agents SDK to interact with the Model Context Protocol (MCP) server.

## Docker

```
# Build the Docker image
docker build -t agentic-telegram-bot .

# Run the Docker container
docker run -d \
  --name telegent \
  -e BOT_USERNAME="" \
  -e TELEGRAM_BOT_TOKEN="" \
  -e OPENAI_API_KEY="" \
  -e OPENAI_MODEL="gpt-4.1" johnlin/agentic-telegram-bot
```

## Install Dependencies

```bash
uv sync
```

## Telegram Bot setup

1. Create a new bot using the [BotFather](https://t.me/botfather) on Telegram.
2. Get the bot token and username.
3. Setting for [privacy mode](https://core.telegram.org/bots/features#privacy-mode):
   - Use the command `/setprivacy` in the BotFather chat.
   - Select your bot.
   - Choose "Disable" to allow the bot to receive all messages in groups.

4. Set the bot token and username in the `.envrc` or `.env` file.

## Environment Variables

Create a `.envrc` file in the root directory of the project and add the following environment variables:

```
# Telegram bot
export BOT_USERNAME=""
export TELEGRAM_BOT_TOKEN=""

# OpenAI API
export OPENAI_API_KEY=""
export OPENAI_MODEL="gpt-4.1"

# Firecrawl API key for advanced scrape feature(Optional)
FIRECRAWL_API_KEY=""

# Langfuse API key for LLM debug use(Optional)
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_SECRET_KEY=""
LANGFUSE_HOST=""
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
uv run bot
```
