#!/usr/bin/env bash
# Start the Designer Studio web app on http://127.0.0.1:8765 (local only).
cd "$(dirname "$0")"
PORT="${CLIPSMITH_PORT:-8765}"
if curl -fs "http://127.0.0.1:$PORT/api/meta" >/dev/null 2>&1; then
  echo "Designer Studio is already running: http://127.0.0.1:$PORT"
  exit 0
fi
exec uv run python web.py
