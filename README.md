# VectoRise

Video-to-vector (Lottie) conversion pipeline.

## Quick start

```bash
pip install -e packages/core -e packages/cli
vectorise clip.mp4 --mp4
vectorise clip.mp4 --mode compressed --mp4
```

## Modes (`vectorise --mode`)

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

## Python API

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
