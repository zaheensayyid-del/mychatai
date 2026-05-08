#!/bin/bash
cd "$(dirname "$0")"
clear

echo "╔═══════════════════════════════════════════╗"
echo "║           MyChatAI — Starting             ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# ── Check Ollama ──────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo "✗ Ollama is not installed."
  echo ""
  echo "  1. Go to https://ollama.com/download"
  echo "  2. Download and install Ollama for Mac"
  echo "  3. Run this file again"
  echo ""
  open "https://ollama.com/download"
  read -p "Press Enter after installing Ollama…"
fi

# ── Start Ollama if not running ───────────────────────────────────────────
if ! curl -s http://localhost:11434 &>/dev/null; then
  echo "→ Starting Ollama…"
  open -a Ollama 2>/dev/null || ollama serve &>/dev/null &
  sleep 3
fi

# ── Pull model if needed ──────────────────────────────────────────────────
if ! ollama list 2>/dev/null | grep -q "llama3.2"; then
  echo "→ Downloading AI model (llama3.2, ~2GB, one-time only)…"
  echo "  This will take a few minutes depending on your internet."
  echo ""
  ollama pull llama3.2
  echo ""
  echo "✓ Model ready!"
  echo ""
else
  echo "✓ AI model ready (llama3.2)"
fi

# ── Install Python deps ───────────────────────────────────────────────────
python3 -m pip install --quiet --upgrade pip 2>/dev/null
echo "✓ Python ready"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────
echo "→ Opening MyChatAI at http://localhost:8080"
echo "  Press Ctrl+C to stop."
echo ""

(sleep 2 && open "http://localhost:8080") &
python3 server.py
