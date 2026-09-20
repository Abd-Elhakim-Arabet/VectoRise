#!/bin/sh
# One-command opencode setup for the VectoRise MCP server.
# Usage:  sh packages/mcp-server/install-opencode.sh   (from the repo root)
#
# Merges mcp.vectorise into ~/.config/opencode/opencode.json (global config),
# preserving every existing key. Backs up before writing. Idempotent.
# Restart opencode afterwards; prompt with `use vectorise`.
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

CFG="${HOME}/.config/opencode/opencode.json"
"$PY" - "$REPO" "$PY" "$CFG" <<'EOF'
import json
import os
import shutil
import sys

repo, py, cfg = sys.argv[1], sys.argv[2], sys.argv[3]
entry = {
    "type": "local",
    "command": [py, "-m", "vector_mcp"],
    "cwd": repo,
    "enabled": True,
    "timeout": 120000,
}
os.makedirs(os.path.dirname(cfg), exist_ok=True)
try:
    with open(cfg) as f:
        data = json.load(f)
except FileNotFoundError:
    data = {"$schema": "https://opencode.ai/config.json"}
except json.JSONDecodeError as exc:
    print(f"error: {cfg} is not valid JSON ({exc}); fix it manually", flush=True)
    sys.exit(1)
if not isinstance(data, dict):
    print(f"error: {cfg} root is not an object; fix it manually", flush=True)
    sys.exit(1)
mcps = data.setdefault("mcp", {})
if not isinstance(mcps, dict):
    print(f"error: {cfg} 'mcp' key is not an object; fix it manually", flush=True)
    sys.exit(1)
if mcps.get("vectorise") == entry:
    print(f"already present in {cfg} — nothing to do (restart opencode)")
    sys.exit(0)
mcps["vectorise"] = entry
note = ""
if os.path.exists(cfg):
    shutil.copyfile(cfg, cfg + ".bak")
    note = f" (backup: {cfg}.bak)"
with open(cfg, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print(f"wrote mcp.vectorise to {cfg}{note} — restart opencode")
EOF
