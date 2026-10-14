#!/usr/bin/env bash
# Start the clipsmith web app on http://127.0.0.1:8765 (local only).
cd "$(dirname "$0")" && exec uv run python web.py
