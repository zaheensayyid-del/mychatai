#!/bin/bash
# One-time setup: install deps and train the model.
set -e
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║        MyChatAI — Setup              ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Python check ──────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "✗ python3 not found. Install Python 3 from https://python.org"
  exit 1
fi
echo "✓ Python: $(python3 --version)"

# ── pip install ────────────────────────────────────────────────────────────
echo ""
echo "→ Installing dependencies (torch, numpy) …"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet torch numpy

echo "✓ Dependencies installed"

# ── Train ──────────────────────────────────────────────────────────────────
echo ""
echo "→ Training the model (this takes ~5-15 minutes on CPU) …"
echo "  You'll see a progress bar and sample outputs every 20 epochs."
echo ""
python3 train.py

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   ✓ Training complete!               ║"
echo "║   Run: ./start.sh to launch chat     ║"
echo "╚══════════════════════════════════════╝"
echo ""
