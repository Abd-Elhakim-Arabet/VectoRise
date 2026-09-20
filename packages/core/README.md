# Core Engine

Video-to-vector (Lottie) conversion pipeline.

* `pipeline.preprocessor`: ffprobe probe + ffmpeg frame streaming + quantization.
* `pipeline.vectorizer`: vtracer raster-to-Bezier tracing.
* `pipeline.tracker`: Farneback dense + Lucas-Kanade sparse optical flow.
* `pipeline.stabilizer`: topology-locked path propagation (legacy `vectorize()` path).
* `pipeline.scenes`: PySceneDetect cuts + background-aware refinement.
* `pipeline.layers`: connected-color patches + exact contour tracing.
* `pipeline.lottie_builder`: Lottie assembly (morph / still / drift / tracked).
* `pipeline.patches_lottie`: verified still builds with render-back checks.
* `pipeline.scene_video`: whole-video builders —
  `VideoConfig` + `build_video_lottie` (main, exact, streaming),
  `CompressedVideoConfig` + `build_compressed_video_lottie`
  (scene-split + flow-tracked, small JSON, batch).

Entry points: `core_engine.video_to_lottie(path, out, mode="main"|"compressed")`,
legacy `core_engine.vectorize(cfg)` (stabilizer path) and
`core_engine.frame_to_lottie(frame, out)`.
