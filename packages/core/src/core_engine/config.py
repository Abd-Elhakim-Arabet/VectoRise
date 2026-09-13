"""Data structures (VectorizeConfig, etc.)."""

from dataclasses import dataclass

# Allowed palette size for color quantization.
MIN_COLOR_COUNT = 2
MAX_COLOR_COUNT = 24


@dataclass
class VectorizeConfig:
    """Configuration for a vectorization run."""

    input_path: str
    output_path: str
    # Target output FPS. None = keep source FPS. Downsampling is done
    # in FFmpeg via the `fps=` video filter.
    target_fps: float | None = None
    # Longest-side cap in pixels (e.g. 720, 1080). None = no scaling.
    # Keeps downstream optical flow memory-bounded. Aspect ratio is
    # preserved and output dims are rounded to even numbers (required
    # by most codecs / filters).
    max_dimension: int | None = None
    # Safety cap for batch loading (extract_all_frames). None = no cap.
    max_frames: int | None = None
    # Number of palette colors to reduce each frame to via
    # quantization (2-24). None = skip quantization (keep raw colors).
    color_count: int | None = 16
    # Decimal precision for fitted Bezier control points in traced SVG.
    path_precision: int = 2
    # Drop traced shapes with polygon area below this (square pixels);
    # filters speckle/noise. Also forwarded to vtracer's filter_speckle.
    min_shape_area: int = 10
    # Mean per-anchor flow magnitude (px) above which the propagated
    # topology is deemed untrustworthy and a keyframe re-trace is forced.
    max_tracking_error: float = 5.0
    # Hard cap on propagated frames: every Nth frame becomes a keyframe
    # and re-syncs topology even if tracking error stays low.
    keyframe_interval: int = 15
