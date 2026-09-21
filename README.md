# VectoRise

Video-to-vector (Lottie) conversion pipeline — one engine, three doors:
a **CLI**, a **secure web app**, and an **MCP server** for AI agents.

```
clip.mp4  →  quantize + patch extraction  →  vector trace  →  out.json (+ preview.mp4)
```

## Requirements

* Python ≥ 3.10, `ffmpeg` + `ffprobe` on `PATH`
  (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux)

## Quick start (CLI)

```bash
pip install -e packages/core -e packages/cli
vectorise clip.mp4 --mp4
vectorise clip.mp4 --mode compressed --mp4
```

### Modes (`vectorise --mode`)

* `main` (default): exact per-frame stills, streaming build, flat memory,
  bigger JSON. Best quality, always works.
* `compressed`: scene-split + optical-flow patch tracking, sparse
  keyframes, much smaller JSON, motion-approximated, batch build
  (whole clip in RAM — use `--max-frames` guard for long clips).

```bash
vectorise clip.mp4 -o out/clip.json --num-colors 12
vectorise clip.mp4 --outdir out/ --quality medium --fps low --mp4
vectorise clip.mp4 --compressed --flow farneback --keyframe-step 1 --mp4
```

Resolution presets: `--quality low/medium/max` = longest side 384/720/1080px;
`--fps low/medium/max` = 12/24/30fps. `--max-dim` / `--fps-value` override.

### Python API

```python
from core_engine import video_to_lottie
video_to_lottie("clip.mp4", "out.json", mode="main")
video_to_lottie("clip.mp4", "small.json", mode="compressed",
                flow_method="dis", keyframe_step=2)
```

Legacy aliases (`BraindeadVideoConfig`, `build_braindead_video_lottie`,
`SceneVideoConfig`, `build_scene_video_lottie`) still import but are
deprecated — use `VideoConfig` / `build_video_lottie` and
`CompressedVideoConfig` / `build_compressed_video_lottie`.

## Web app (`apps/web/`)

Bare-bones, stdlib-only upload → sliders → MP4 preview → JSON download.
No npm, no framework. Details in [`apps/web/README.md`](apps/web/README.md).

```bash
python apps/web/server.py --port 8000        # http://127.0.0.1:8000/
python apps/web/server.py --reload           # dev: auto-restart on server.py changes
```

Limits: **10 MB / 10 seconds** per clip (server-enforced via size caps +
`ffprobe` duration gate). Hardened by design: extension allowlist +
magic-byte sniff, uuid job dirs outside the webroot, server-side param
clamping, per-IP rate limit, 2 parallel conversions, 30-min TTL sweeper,
same-origin POST check, strict security headers.

Always-on (macOS launchd, starts at login, restarts on crash) —
see `apps/web/launchd/` and the web README.

## MCP server (`packages/mcp-server/`)

`vectorise` MCP server (official SDK v2, stdio) with three tools —
`probe_video`, `video_to_lottie`, `summarize_lottie` — returning typed
structured output. Paths are jailed to `VECTORISE_MCP_ROOTS`
(default: repo root + tmp). Details in
[`packages/mcp-server/README.md`](packages/mcp-server/README.md).

```bash
pip install -e packages/core -e packages/mcp-server
python -m vector_mcp
```

One-command client setup (idempotent, machine-local paths auto-detected):

```bash
sh packages/mcp-server/install-codex.sh     # → ~/.codex/config.toml
sh packages/mcp-server/install-opencode.sh   # → ~/.config/opencode/opencode.json
sh packages/mcp-server/install-claude.sh     # → claude mcp add (user scope)
```

Or manually for Claude Code:

```bash
claude mcp add vectorise --scope user -- /abs/.venv/bin/python -m vector_mcp
```

## Repo layout

```
packages/core/src/core_engine/   the engine (preprocess → track → vectorize → lottie)
packages/cli/src/vector_cli/     `vectorise` command
packages/mcp-server/             MCP server + per-client install scripts
apps/web/                        secure web app (server.py + static/, launchd plist)
apps/web/static/assets/          local-only showcase clip (git-ignored, see its README)
deploy/                          production bundle (Docker + Caddy + TLS, see its README)
apps/worker/                     (stub)   shared/  (stub)   docs/  (stub)
```

## Developing

```bash
python -m venv .venv && .venv/bin/pip install -e packages/core -e packages/cli -e "packages/mcp-server" "mcp>=2.0"
.venv/bin/pytest packages/core/tests/   # core test-suite (tests/ is git-ignored: local only)
```
