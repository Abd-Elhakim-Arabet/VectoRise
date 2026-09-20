# vector-mcp

MCP server exposing the VectoRise core engine to AI agents over stdio.
Built on the official `mcp` SDK (v2 API).

## Tools

| Tool | Kind | What it does |
|---|---|---|
| `probe_video(path)` | read-only | ffprobe metadata: resolution, fps, duration, codec. Call first to pick sane params. |
| `video_to_lottie(path, …)` | write | Convert to Lottie JSON (+ MP4 preview by default). `main` = exact, `compressed` = small + flow-tracked. Returns the build report with output paths. |
| `summarize_lottie(json_path)` | read-only | Cheap stats for an existing Lottie JSON (canvas, fps, layers, size). |

All paths must be **absolute, local, and inside `VECTORISE_MCP_ROOTS`**
(os.pathsep-separated; default: repo root + system temp). URLs are rejected,
params are clamped to the same bounds as the web UI, and compressed mode
carries a `max_frames` RAM guard (default 300).

## Run

```bash
pip install -e packages/core        # engine + deps (numpy, opencv, …)
pip install "mcp>=2.0"              # or: pip install -e packages/mcp-server
python -m vector_mcp                # stdio server (no install needed from repo root)
```

## One-command client setup

```bash
sh packages/mcp-server/install-codex.sh     # ~/.codex/config.toml
sh packages/mcp-server/install-opencode.sh   # ~/.config/opencode/opencode.json
sh packages/mcp-server/install-claude.sh     # claude mcp add (user scope)
```

All three are idempotent (safe to re-run), resolve this machine's real
paths, and back up nothing they don't change (opencode config is backed
up to `.bak`). Restart the client afterwards.

Requires `ffmpeg`/`ffprobe` on `PATH`.

## Fresh machine → Codex in 2 minutes

```bash
git clone <this-repo> && cd VectoRise
python3 -m venv .venv && .venv/bin/pip install -e packages/core -e packages/mcp-server
sh packages/mcp-server/install-codex.sh   # wires [mcp_servers.vectorise] into ~/.codex/config.toml
```

Restart codex, type `/mcp` to confirm `vectorise` is listed, then try:
`use vectorise to probe /tmp/clip.mp4`.

## Client config (example — Claude Desktop / Cursor / opencode)

```json
{
  "mcpServers": {
    "vectorise": {
      "command": "/abs/VectoRise/.venv/bin/python",
      "args": ["-m", "vector_mcp"],
      "cwd": "/abs/VectoRise",
      "env": { "VECTORISE_MCP_ROOTS": "/abs/VectoRise:/tmp" }
    }
  }
}
```

Requires `ffmpeg`/`ffprobe` on `PATH` (the server inherits the client's
environment, so make sure the GUI app was launched with Homebrew on PATH —
`~/.zprofile` usually covers login shells, or set `PATH` in `env` above).
