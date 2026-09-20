# VectoRise web

Bare-bones, stdlib-only secure converter. No npm, no pip packages beyond `vector-core`.

## Run

```bash
pip install -e packages/core   # once (numpy, opencv, pillow, vtracer, lottie, scenedetect)
python apps/web/server.py --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/
```

## What it does

Upload video → sliders set params → server runs `core_engine.video_to_lottie`
(main or compressed) + MP4 preview → page shows `<video>` preview, report,
and JSON/MP4 download links. Frontend polls `GET /api/status?id=…`.

Sliders map 1:1 to CLI flags: `num_colors` (8–24), `max_dim` (128–1080),
`fps` (5–30), `merge_area` (1–100), plus compressed-only `keyframe_step`,
`scene_threshold`, `scene_min_len`, `flow`. Web defaults: 480px @ 12fps,
`max_frames=300` hard cap on compressed (RAM guard).

## Security model

See `server.py` docstring. Highlights: 50MB cap, extension allowlist +
magic-byte sniff, uuid job dirs under system temp (never webroot, never
client filename), server-side clamping, per-IP rate limit (10/10min),
max 2 parallel conversions, 30-min TTL sweeper, same-origin POST check,
CSP/nosniff/DENY framing headers, `Content-Disposition: attachment` for
JSON, generic error messages (tracebacks to stderr only), argv-only
subprocesses (no shell).
