#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Run: cp .env.example .env   then edit with your API key"
    exit 1
fi

echo "Starting codex-deepseek on http://127.0.0.1:11435 ..."
uv run python -m src.main
