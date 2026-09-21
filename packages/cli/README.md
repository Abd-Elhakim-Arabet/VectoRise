# vector-cli

`vectorise`: video → Lottie JSON (+ MP4 preview) from the command line.
Thin wrapper over `core_engine.video_to_lottie` — same builders and bounds
as the web UI and MCP server.

* `main` (default): exact per-frame stills, bigger JSON, always works.
* `compressed`: scene-split + flow-tracked, smaller JSON, motion
  approximated, capped by `--max-frames`.

```bash
vectorise input.mp4 out.json [--mode main|compressed] [--preview out.mp4]
```

Install: `pip install -e packages/cli` (pulls in `packages/core`).
Entry point: `vector_cli.main` (`project.scripts` in `pyproject.toml`).
