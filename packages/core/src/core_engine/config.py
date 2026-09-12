"""Data structures (VectorizeConfig, etc.)."""

from dataclasses import dataclass


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
