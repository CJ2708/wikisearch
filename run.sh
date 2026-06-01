#!/usr/bin/env bash
# Launch the Wikipedia search engine.
#   - ensures Ollama is running
#   - starts the FastAPI server on http://localhost:8666
set -e

cd "$(dirname "$0")"

# Start Ollama if it isn't already up.
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama server..."
  ollama serve >/tmp/ollama.log 2>&1 &
  sleep 2
fi

PORT="${PORT:-8666}"
echo "Open http://localhost:$PORT in your browser"
exec ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
