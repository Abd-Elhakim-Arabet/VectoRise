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
    # Number of palette colors to reduce each frame to via K-Means
    # quantization. None = skip quantization (keep raw colors).
    color_count: int | None = 8
    # Toggle edge-preserving bilateral filtering (smooths noise/flat
    # areas while keeping object boundaries sharp).
    enable_smoothing: bool = True
    # Bilateral filter diameter (neighborhood size) and sigma
    # (filter strength in color + coordinate space).
    bilateral_d: int = 9
    bilateral_sigma: float = 75.0
    # Decimal precision for fitted Bezier control points in traced SVG.
    path_precision: int = 2
    # Drop traced shapes with polygon area below this (square pixels);
    # filters speckle/noise. Also forwarded to vtracer's filter_speckle.
    min_shape_area: int = 10
