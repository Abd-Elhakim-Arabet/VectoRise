# VectoRise web

Bare-bones, stdlib-only secure converter. No npm, no pip packages beyond `vector-core`.

## Run

```bash
pip install -e packages/core   # once (numpy, opencv, pillow, vtracer, lottie, scenedetect)
python apps/web/server.py --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/
```

Dev (auto-restart on `server.py` changes; static files never need a restart):

```bash
python apps/web/server.py --reload
```

## Always on (macOS, no manual launching)

Installed as a launchd service (`com.vectorise.web`): starts at login,
restarts itself on crash. Source plist: `apps/web/launchd/`.

```bash
launchctl list com.vectorise.web          # check it's running
tail -f ~/Library/Logs/vectorise-web.log  # logs (.err for errors)
launchctl stop com.vectorise.web          # pause (stays installed)
launchctl start com.vectorise.web         # resume
launchctl unload -w ~/Library/LaunchAgents/com.vectorise.web.plist  # remove
```

To reinstall after editing the plist (substitute your paths; the committed
copy is a template — never commit the installed one):

```bash
sed -e "s|@REPO@|$PWD|" -e "s|@HOME@|$HOME|" \
  apps/web/launchd/com.vectorise.web.plist \
  > ~/Library/LaunchAgents/com.vectorise.web.plist
launchctl load -w ~/Library/LaunchAgents/com.vectorise.web.plist
```

Note: the service holds port 8000 — stop it before a manual dev run,
or dev on another port (`--port 8001`).

## What it does

Upload video → sliders set params → server runs `core_engine.video_to_lottie`
(main or compressed) + MP4 preview → page shows `<video>` preview, report,
and JSON/MP4 download links. Frontend polls `GET /api/status?id=…`.

Sliders map 1:1 to CLI flags: `num_colors` (8–24), `max_dim` (128–480),
`fps` (5–16), `merge_area` (1–100), plus compressed-only `keyframe_step`,
`scene_threshold`, `scene_min_len`, `flow`. Web defaults: 480px @ 12fps,
`max_frames=300` hard cap on compressed (RAM guard).

## Security model

See `server.py` docstring. Highlights: 10MB / 3s caps (size checked pre-read,
duration via ffprobe pre-convert), extension allowlist +
magic-byte sniff, uuid job dirs under system temp (never webroot, never
client filename), server-side clamping, per-visitor rate limit
(`CF-Connecting-IP` behind Cloudflare, socket IP direct; 5/10min public,
10/10min local — env-tunable), bounded queue (`503` past cap), max 2
parallel conversions, enforced convert timeout, 30-min TTL sweeper,
same-origin POST check, CSP/nosniff/DENY framing headers,
`Content-Disposition: attachment` for JSON, generic error messages
(tracebacks to stderr only), argv-only subprocesses (no shell).
