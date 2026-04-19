#!/bin/bash

mkdir -p ./data ./claude-home

docker run -d \
  --env-file .env \
  -e SESSION_DB_PATH=/app/data/sessions.db \
  -v $(pwd)/instructions.md:/app/instructions.md \
  -v $(pwd)/agent_config.json:/app/agent_config.json \
  -v $(pwd)/access.json:/app/access.json \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/claude-home:/app/claude-home \
  -v $(pwd)/skills:/app/.claude/skills \
  --name agentic-telegram-bot \
  agentic-telegram-bot
