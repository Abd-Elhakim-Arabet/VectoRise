"""Main entry-point function(s)."""

from __future__ import annotations

import numpy as np

from core_engine.config import VectorizeConfig
from core_engine.pipeline.lottie_builder import build_lottie_animation, save_lottie_json
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
