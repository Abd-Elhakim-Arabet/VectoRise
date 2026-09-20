#!/bin/sh
# One-command Codex setup for the VectoRise MCP server.
# Usage:  sh packages/mcp-server/install-codex.sh   (from the repo root)
#
# Does: picks a python (repo .venv first, else python3 on PATH),
# verifies vector_mcp is importable, then idempotently adds
# [mcp_servers.vectorise] to ~/.codex/config.toml.
# Restart codex afterwards and check with /mcp.
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

CFG="${HOME}/.codex/config.toml"
mkdir -p "$(dirname "$CFG")"
touch "$CFG"
if grep -q '^\[mcp_servers\.vectorise\]' "$CFG" 2>/dev/null; then
  echo "already present in $CFG — nothing to do (restart codex, check /mcp)"
  exit 0
fi
cat >> "$CFG" <<EOF

[mcp_servers.vectorise]
command = "$PY"
args = ["-m", "vector_mcp"]
cwd = "$REPO"
startup_timeout_sec = 120
EOF
echo "added [mcp_servers.vectorise] to $CFG — restart codex, then /mcp"
