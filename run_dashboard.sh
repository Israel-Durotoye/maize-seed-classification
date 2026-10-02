#!/usr/bin/env bash
# Build the dashboard (if needed) and start the live sorter website.
# Usage: ./run_dashboard.sh [--camera 1] [--port 8000] [--host 0.0.0.0] [--roi X Y W H]
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c "import keras, tensorflow, cv2, fastapi, uvicorn" 2>/dev/null; then
  echo "Installing Python dependencies (first run only)…"
  .venv/bin/python -m pip install -r web/backend/requirements.txt
fi
if [ ! -f web/frontend/dist/index.html ] || [ -n "$(find web/frontend/src web/frontend/index.html -newer web/frontend/dist/index.html 2>/dev/null)" ]; then
  echo "Building the website…"
  (cd web/frontend && { [ -d node_modules ] || npm install; } && npm run build)
fi
exec .venv/bin/python web/backend/server.py "$@"
