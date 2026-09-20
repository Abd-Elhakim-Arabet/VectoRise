"""Core engine public API."""

from core_engine.api import frame_to_lottie, vectorize, video_to_lottie
from core_engine.config import VectorizeConfig
from core_engine.pipeline.layers import Layer, composite_layers, extract_layers, layer_to_rgba, merge_small_layers, rasterize_shapes, trace_layer
from core_engine.pipeline.patches_lottie import (
    LottieVerifyError,
    layers_to_lottie,
    patches_to_lottie,
    patches_to_lottie_verified,
    render_lottie_layers,
    verify_lottie_file,
    verify_lottie_patches,
)
from core_engine.pipeline.scene_video import (
    BraindeadVideoConfig,
    CompressedVideoConfig,
    SceneVideoConfig,
    SceneVideoError,
    VideoConfig,
    VideoError,
    build_braindead_video_lottie,
    build_compressed_video_lottie,
    build_scene_video_lottie,
    build_video_lottie,
    clear_patch_images,
    render_animation_frames,
    render_animation_to_mp4,
)
from core_engine.pipeline.scenes import (
    BoundaryReport,
    Scene,
    SceneConfig,
    SceneDetector,
)

__all__ = [
    "vectorize",
    "video_to_lottie",
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
    "layers_to_lottie",
    "patches_to_lottie",
    "patches_to_lottie_verified",
    "render_lottie_layers",
    "verify_lottie_file",
    "verify_lottie_patches",
    "SceneVideoConfig",
    "CompressedVideoConfig",
    "VideoConfig",
    "SceneVideoError",
    "VideoError",
    "build_scene_video_lottie",
    "build_compressed_video_lottie",
    "BraindeadVideoConfig",
    "build_braindead_video_lottie",
    "build_video_lottie",
    "clear_patch_images",
    "render_animation_frames",
    "render_animation_to_mp4",
]
