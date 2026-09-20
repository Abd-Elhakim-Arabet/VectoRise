"""Main entry-point function(s)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core_engine.config import VectorizeConfig
from core_engine.pipeline.layers import extract_layers, trace_layer
from core_engine.pipeline.lottie_builder import (
    build_layered_lottie_animation,
    build_lottie_animation,
    save_lottie_json,
)
from core_engine.pipeline.preprocessor import VideoPreprocessor
from core_engine.pipeline.stabilizer import PathStabilizer
from core_engine.pipeline.tracker import MotionField, MotionTracker
from core_engine.pipeline.vectorizer import trace_frame


def vectorize(config: VectorizeConfig) -> str:
    """Run the full video-to-Lottie pipeline.

    Preprocess → optical flow → trace keyframes → stabilize topology →
    assemble Lottie JSON, saved to ``config.output_path``.

    Only keyframes are traced (every ``config.keyframe_interval``
    frames, plus error-triggered re-traces); intermediate frames ride
    the stabilized topology, so cost stays flat in clip length.

    Returns:
        The output path written.
    """
    if not config.output_path:
        raise ValueError("config.output_path must be set")

    frames = VideoPreprocessor(config.input_path, config).extract_all_frames()
    if not frames:
        raise ValueError(f"no frames decoded from {config.input_path}")

    motion_fields = [
        MotionField(flow_vectors=f.flow_vectors, frame_index=f.frame_index)
        for f in MotionTracker().analyze_sequence(frames)
    ]
    base = trace_frame(np.ascontiguousarray(frames[0]), config, frame_index=0)

    def _retrace(i: int):
        return trace_frame(np.ascontiguousarray(frames[i]), config, frame_index=i)

    stable = PathStabilizer(config).stabilize_sequence(
        [base], motion_fields, retrace_fn=_retrace
    )
    animation = build_lottie_animation(stable, config)
    save_lottie_json(animation, config.output_path)
    return config.output_path


def frame_to_lottie(
    frame: np.ndarray,
    output_path: str,
    config: VectorizeConfig | None = None,
    num_colors: int = 16,
    min_layer_area: int = 10,
    fps: float | None = None,
) -> str:
    """Convert a single frame into a multi-layer Lottie JSON file.

    Frame → connected-color :func:`layers <extract_layers>` → each layer
    traced individually → one Lottie ``ShapeLayer`` per layer
    (``layer_<id>``). The result is a still (single-frame) animation; it
    is the per-frame basis that multi-frame motion will keyframe later.

    Args:
        frame: ``uint8`` RGB ``[H, W, 3]`` frame.
        output_path: where to write the ``.json`` file.
        config: optional ``VectorizeConfig`` (only ``path_precision`` and
            ``target_fps`` are read); a default is built when omitted.
        num_colors: quantization palette size, must be in [2, 24].
        min_layer_area: regions smaller than this (pixels) are dropped.
        fps: frame rate tag; falls back to ``config.target_fps``, then 30.

    Returns:
        The output path written.
    """
    if not output_path:
        raise ValueError("output_path must be set")
    cfg = config or VectorizeConfig(input_path="", output_path=output_path)
    layers = extract_layers(
        np.ascontiguousarray(frame),
        num_colors=num_colors,
        min_layer_area=min_layer_area,
    )
    if not layers:
        raise ValueError("no layers extracted from frame")
    h, w, _ = np.ascontiguousarray(frame).shape
    shapes_per_layer = [trace_layer(layer, cfg) for layer in layers]
    rate = fps or cfg.target_fps or 30.0
    animation = build_layered_lottie_animation(
        layers, shapes_per_layer, w, h, rate
    )
    save_lottie_json(animation, output_path)
    return output_path


def video_to_lottie(
    input_path: str,
    output_path: str,
    mode: str = "main",
    target_fps: float | None = 24.0,
    max_dimension: int | None = 720,
    num_colors: int = 16,
    merge_min_area: int = 10,
    preview_mp4: str | None = None,
    verbose: bool = False,
    **compressed_kwargs,
) -> dict:
    """Convert a video file to Lottie JSON with the chosen builder.

    Args:
        input_path: source video file.
        output_path: destination ``.json`` path.
        mode: ``"main"`` (exact per-frame stills, streaming) or
            ``"compressed"`` (scene-split + flow-tracked, smaller JSON).
        target_fps: working frame rate (None = keep source).
        max_dimension: longest-side cap in px (None = keep source).
        num_colors: palette size per frame/scene reference (2..24).
        merge_min_area: patches smaller than this dissolve into neighbours.
        preview_mp4: optional preview MP4 path (H264/yuv420p/faststart).
        verbose: print builder progress lines.
        **compressed_kwargs: extra :class:`CompressedVideoConfig` fields
            (``flow_method``, ``keyframe_step``, ``scene_threshold``,
            ``scene_min_len``, ``max_frames``) -- only used in
            compressed mode.

    Returns:
        The builder report dict (includes ``mode``, ``num_frames``,
        ``num_tracks``, ``size_kb``, ``sample_checks``).
    """
    from core_engine.pipeline.scene_video import (
        CompressedVideoConfig,
        VideoConfig,
        build_compressed_video_lottie,
        build_video_lottie,
    )

    if mode not in ("main", "compressed"):
        raise ValueError(f"mode must be 'main'/'compressed', got {mode!r}")
    if not input_path or not Path(input_path).is_file():
        raise ValueError(f"input not found: {input_path}")
    if not output_path:
        raise ValueError("output_path must be set")
    if mode == "main":
        if compressed_kwargs:
            raise ValueError(
                f"compressed options {sorted(compressed_kwargs)} "
                "need mode='compressed'"
            )
        return build_video_lottie(
            VideoConfig(
                input_path=input_path, output_path=output_path,
                target_fps=target_fps, max_dimension=max_dimension,
                num_colors=num_colors, merge_min_area=merge_min_area,
            ),
            verbose=verbose, preview_mp4=preview_mp4,
        )
    allowed = {"flow_method", "keyframe_step", "scene_threshold",
               "scene_min_len", "flow_samples_per_layer", "max_frames"}
    unknown = sorted(set(compressed_kwargs) - allowed)
    if unknown:
        raise ValueError(f"unknown compressed options: {unknown}")
    return build_compressed_video_lottie(
        CompressedVideoConfig(
            input_path=input_path, output_path=output_path,
            target_fps=target_fps if target_fps is not None else 8.0,
            max_dimension=max_dimension if max_dimension is not None else 384,
            num_colors=num_colors, merge_min_area=merge_min_area,
            **compressed_kwargs,
        ),
        verbose=verbose, preview_mp4=preview_mp4,
    )
