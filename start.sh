#!/bin/bash
# Launch the chat server and open the browser.
set -e
cd "$(dirname "$0")"

# Check model exists
if [ ! -f "weights/model.pt" ]; then
  echo "✗ Model not found. Run ./setup.sh first to train it."
  exit 1
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║       MyChatAI — Starting            ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "→ Starting server on http://localhost:5000"
echo "  Press Ctrl+C to stop."
echo ""

# Open browser after a 2-second delay
(sleep 2 && open "http://localhost:5000") &

python3 server.py
