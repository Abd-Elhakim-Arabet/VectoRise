#!/bin/sh
# One-command Claude Code setup for the VectoRise MCP server.
# Usage:  sh packages/mcp-server/install-claude.sh   (from the repo root)
#
# Registers a user-scope server via `claude mcp add` (works in every project).
# Idempotent: skips if `claude mcp list` already shows vectorise.
set -eu

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -x "$REPO/.venv/bin/python" ]; then
  PY="$REPO/.venv/bin/python"
else
  PY="$(command -v python3 || true)"
fi
if [ -z "${PY:-}" ]; then
  echo "error: no python3 found (create $REPO/.venv or put python3 on PATH)" >&2
  exit 1
fi
if ! "$PY" -c "import vector_mcp" 2>/dev/null; then
  echo "installing vector-mcp into $PY ..."
  "$PY" -m pip install -e "$REPO/packages/mcp-server" || {
    echo "error: pip install failed; run it manually first" >&2
    exit 1
  }
fi
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "warning: ffmpeg/ffprobe not on PATH — conversions will fail until installed" >&2
fi

if ! command -v claude >/dev/null; then
  echo "claude CLI not found. Install it first:"
  echo "  npm install -g @anthropic-ai/claude-code"
  echo "Then register the server:"
  echo "  claude mcp add vectorise --scope user -- $PY -m vector_mcp"
  exit 1
fi
if claude mcp list 2>/dev/null | grep -q "vectorise"; then
  echo "already registered (claude mcp list) — nothing to do"
  exit 0
fi
claude mcp add vectorise --scope user -- "$PY" -m vector_mcp
echo "registered — verify with: claude mcp list"
