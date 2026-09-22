# VectoRise

Video-to-vector (Lottie) conversion pipeline — one engine, three doors:
a **CLI**, a **secure web app**, and an **MCP server** for AI agents.

Live: **https://vectoriseai.com** — drop a clip, get vectors back.

```
clip.mp4  →  quantize + patch extraction  →  vector trace  →  out.json (+ preview.mp4)
```

The JSON is plain Lottie: open it in After Effects or Figma via the
LottieFiles plugin, or play it anywhere Lottie runs.

## Requirements

* Python 3.10–3.12, `ffmpeg` + `ffprobe` on `PATH`
  (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux).
  The web server needs ≤ 3.12 (it uses stdlib `cgi`, removed in 3.13+).

## Quick start (CLI)

```bash
pip install -e packages/core -e packages/cli
vectorise clip.mp4 --mp4
vectorise clip.mp4 --mode compressed --mp4
```

### Modes

* `main` (default): exact per-frame stills, streaming build with flat
  memory, bigger JSON. Best quality, always works.
* `compressed`: scene-split + optical-flow patch tracking, sparse
  keyframes, much smaller JSON, motion approximated. Batch build holds
  the clip in RAM — cap it with `--max-frames` for long clips.

### Recipes

```bash
vectorise clip.mp4 -o out/clip.json --num-colors 12
vectorise clip.mp4 --outdir out/ --quality medium --fps low --mp4
vectorise clip.mp4 --compressed --flow farneback --keyframe-step 1 --mp4
```

* Resolution: `--quality low/medium/max` caps the longest side at
  384/720/1080px (`--max-dim` overrides with an exact value).
* Frame rate: `--fps low/medium/max` = 12/24/30fps
  (`--fps-value` overrides). Lower fps ≈ smaller JSON.
* Palette: `--num-colors 8..24` (default 16). Smoothing: `--merge-area`
  dissolves patches smaller than N px into neighbours.
* `--mp4` also renders an `<stem>_converted.mp4` preview from the JSON.
* `-q` silences per-frame progress; `--version` prints the version.

### Python API

```python
from core_engine import video_to_lottie
video_to_lottie("clip.mp4", "out.json", mode="main")
video_to_lottie("clip.mp4", "small.json", mode="compressed",
                flow_method="dis", keyframe_step=2)
```

## Web app (`apps/web/`)

Stdlib-only: upload → sliders → live MP4 preview → JSON download.
No npm, no framework. Details in [`apps/web/README.md`](apps/web/README.md).

```bash
python apps/web/server.py --port 8000        # http://127.0.0.1:8000/
python apps/web/server.py --reload           # dev: auto-restart on server.py changes
```

Limits: **10 MB / 10 seconds** per clip (size caps + `ffprobe` duration
gate, both server-enforced). Hardened by design: extension allowlist +
magic-byte sniff, uuid job dirs outside the webroot, server-side param
clamping, per-visitor rate limit, bounded queue, 2 parallel conversions,
enforced convert timeout, 30-min file TTL, same-origin POST check,
strict security headers. Always-on macOS setup: see `apps/web/launchd/`.

## Deploy (`deploy/`)

Site + conversion API are one process — every lane below hosts both.
Full guide in [`deploy/README.md`](deploy/README.md).

| Lane | Cost | Fit |
|---|---|---|
| VPS + Caddy (`compose.yml`) | $0 Oracle / ~€4 Hetzner | Always on, full CPU, auto-TLS |
| Render free (`render.yaml`) | $0, no card | Standby-grade: sleeps when idle, 512 MB RAM |
| Mac + Cloudflare Tunnel | $0 | Fastest conversions, sleeps with the Mac |

Public-facing limits are env-tunable (`VECTORISE_RATE_LIMIT_N`,
`VECTORISE_MAX_QUEUED`, `VECTORISE_CONVERT_TIMEOUT_SEC`).

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

## Repo layout

```
packages/core/src/core_engine/   the engine (preprocess → track → vectorize → lottie)
packages/cli/src/vector_cli/     `vectorise` command
packages/mcp-server/             MCP server + per-client install scripts
apps/web/                        secure web app (server.py + static/, launchd plist)
apps/web/static/assets/          showcase clip (committed so deploys serve it)
deploy/                          production bundle (Docker + Caddy + TLS, Render lane)
render.yaml                      Render free-tier blueprint (standby lane)
apps/worker/                     (stub)   shared/  (stub)   docs/  (stub)
```

## Developing

```bash
python -m venv .venv && .venv/bin/pip install -e packages/core -e packages/cli -e "packages/mcp-server" "mcp>=2.0"
.venv/bin/pytest packages/core/tests/   # core test-suite (tests/ is git-ignored: local only)
```
