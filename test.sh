#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Running tests..."
uv run python tests/test_translate.py
