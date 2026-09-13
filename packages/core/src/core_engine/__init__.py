"""Core engine public API."""

from core_engine.api import frame_to_lottie, vectorize
from core_engine.config import VectorizeConfig
from core_engine.pipeline.layers import Layer, composite_layers, extract_layers, layer_to_rgba, merge_small_layers, rasterize_shapes, trace_layer
from core_engine.pipeline.patches_lottie import (
    LottieVerifyError,
    patches_to_lottie,
    patches_to_lottie_verified,
    render_lottie_layers,
    verify_lottie_file,
    verify_lottie_patches,
)
from core_engine.pipeline.scenes import (
    BoundaryReport,
    Scene,
    SceneConfig,
    SceneDetector,
)

__all__ = [
    "vectorize",
    "frame_to_lottie",
    "VectorizeConfig",
    "BoundaryReport",
    "Scene",
    "SceneConfig",
    "SceneDetector",
    "Layer",
    "composite_layers",
    "extract_layers",
    "layer_to_rgba",
    "merge_small_layers",
    "rasterize_shapes",
    "trace_layer",
    "LottieVerifyError",
    "patches_to_lottie",
    "patches_to_lottie_verified",
    "render_lottie_layers",
    "verify_lottie_file",
    "verify_lottie_patches",
]
