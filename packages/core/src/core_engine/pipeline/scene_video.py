"""Whole-video Lottie from per-scene fixed patches + optical-flow tracking.

Pipeline per video::

    scenes (explain_splits) -> per scene: reference frame -> fixed patches
      (extract + merge, never change within the scene) -> DIS/Farneback
      dense flow -> per-layer translation track -> one Lottie JSON for the
      whole video (layers carry scene in/out lifetimes).

Typical usage::

    from core_engine.pipeline.scene_video import (
        SceneVideoConfig, build_scene_video_lottie)

    cfg = SceneVideoConfig(input_path="clip.mp4", output_path="video.json")
    report = build_scene_video_lottie(cfg)

Knobs (all on :class:`SceneVideoConfig`):

* ``target_fps`` / ``max_dimension`` -- resampled working resolution.
* ``num_colors`` / ``merge_min_area`` -- reference patch decomposition.
* ``flow_method`` -- ``"dis"`` (DISOpticalFlow/MEDIUM, strongest flow
  shipped with OpenCV) or ``"farneback"`` (the repo's default).
* ``keyframe_step`` -- emit a path keyframe every N frames (players
  interpolate between them; 1 = every frame).
* ``scene_threshold`` / ``scene_min_len`` -- cut sensitivity for the
  existing scene splitter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core_engine.config import VectorizeConfig
from core_engine.pipeline.layers import (
    extract_layers,
    merge_small_layers,
    trace_layer,
)
from core_engine.pipeline.lottie_builder import (
    build_tracked_layers_animation,
    save_lottie_json,
)
from core_engine.pipeline.patches_lottie import (
    _flatten_loop,
    check_renders_against_layers,
    render_lottie_layers,
)
from core_engine.pipeline.preprocessor import VideoPreprocessor
from core_engine.pipeline.scenes import SceneConfig, explain_splits
from core_engine.pipeline.tracker import compute_dense_flow

try:  # Optional at import time; functions raise a clear error if missing.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


class SceneVideoError(ValueError):
    """Raised when scene video planning, flow, or assembly fails."""


@dataclass
class SceneVideoConfig:
    """Knobs for whole-video patch tracking."""

    input_path: str
    output_path: str
    # Working resolution: resampled FPS and longest-side cap.
    target_fps: float = 8.0
    max_dimension: int = 384
    # Reference patch decomposition (fixed for the whole scene).
    num_colors: int = 8
    merge_min_area: int = 10
    # Dense flow backend: "dis" (strongest in-box) or "farneback".
    flow_method: str = "dis"
    # Emit one path keyframe every N frames (1 = every frame).
    keyframe_step: int = 2
    # Scene splitter sensitivity (higher threshold = fewer cuts).
    scene_threshold: float = 27.0
    scene_min_len: int = 15
    # Max reference-mask pixels sampled per layer per flow step.
    flow_samples_per_layer: int = 2000


def _require_cv2() -> None:
    if cv2 is None:
        raise SceneVideoError(
            "OpenCV (cv2) is required for scene video tracking: "
            "install opencv-python-headless"
        )


def _make_flow_fn(method: str):
    """Build a ``(prev_gray, curr_gray) -> [H, W, 2] float32`` flow closure."""
    _require_cv2()
    if method == "dis":
        try:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        except Exception as exc:
            raise SceneVideoError(f"DIS optical flow unavailable: {exc}") from exc

        def _dis(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
            flow = dis.calc(prev_gray, curr_gray, None)
            return np.ascontiguousarray(flow, dtype=np.float32)

        return _dis
    if method == "farneback":
        def _fb(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
            prev_rgb = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2RGB)
            curr_rgb = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2RGB)
            return compute_dense_flow(prev_rgb, curr_rgb)

        return _fb
    raise SceneVideoError(f"unknown flow_method: {method!r} (use 'dis'/'farneback')")


def _track_scene_translations(
    grays: list[np.ndarray],
    layers,
    flow_fn,
    max_samples: int,
) -> dict[int, list[tuple[float, float]]]:
    """Per-layer ``(dx, dy)`` translation for each frame of a scene.

    The reference is the middle frame (zero motion); translations
    accumulate forward (median flow sampled on the translated mask) and
    backward. Masks never change -- only positions move.
    """
    n = len(grays)
    ref = n // 2
    h, w = grays[0].shape
    # Forward flows between consecutive frames (n-1 of them).
    flows = [flow_fn(grays[t], grays[t + 1]) for t in range(n - 1)]

    positions: dict[int, list] = {}
    for lyr in layers:
        ys, xs = np.nonzero(lyr.mask)
        step = max(1, len(xs) // max(max_samples, 1))
        xs, ys = xs[::step], ys[::step]
        traj: list[tuple[float, float] | None] = [None] * n
        traj[ref] = (0.0, 0.0)
        pos = np.zeros(2, dtype=np.float64)
        for t in range(ref + 1, n):
            sx = np.clip((xs + int(round(pos[0]))), 0, w - 1)
            sy = np.clip((ys + int(round(pos[1]))), 0, h - 1)
            delta = np.median(flows[t - 1][sy, sx].astype(np.float64), axis=0)
            pos = pos + delta
            traj[t] = (float(pos[0]), float(pos[1]))
        pos = np.zeros(2, dtype=np.float64)
        for t in range(ref - 1, -1, -1):
            sx = np.clip((xs + int(round(pos[0]))), 0, w - 1)
            sy = np.clip((ys + int(round(pos[1]))), 0, h - 1)
            delta = np.median(flows[t][sy, sx].astype(np.float64), axis=0)
            pos = pos - delta
            traj[t] = (float(pos[0]), float(pos[1]))
        positions[lyr.id] = [p if p is not None else (0.0, 0.0) for p in traj]
    return positions


def build_scene_video_lottie(config: SceneVideoConfig, verbose: bool = True) -> dict:
    """Build one Lottie JSON for the entire video. Returns a report dict."""
    _require_cv2()
    if not config.input_path or not Path(config.input_path).is_file():
        raise SceneVideoError(f"input not found: {config.input_path}")
    if not config.output_path:
        raise SceneVideoError("output_path must be set")
    if config.keyframe_step < 1:
        raise SceneVideoError(f"keyframe_step must be >= 1: {config.keyframe_step}")
    flow_fn = _make_flow_fn(config.flow_method)

    scenes, _ = explain_splits(
        config.input_path,
        SceneConfig(threshold=config.scene_threshold,
                    min_scene_len=config.scene_min_len),
    )
    vcfg = VectorizeConfig(
        input_path=config.input_path, output_path="",
        target_fps=config.target_fps, max_dimension=config.max_dimension,
        color_count=None,
    )
    pre = VideoPreprocessor(config.input_path, vcfg)
    frames = pre.extract_all_frames()
    if not frames:
        raise SceneVideoError(f"no frames decoded from {config.input_path}")
    n, out_fps = len(frames), float(pre.output_fps)
    H, W, _ = frames[0].shape
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    # Map scene times (source seconds) onto processed frame indices.
    ranges: list[tuple[int, int]] = []
    for s in scenes:
        i0 = max(0, min(n - 1, int(round(s.start_time * out_fps))))
        i1 = max(i0 + 1, min(n, int(round(s.end_time * out_fps))))
        ranges.append((i0, i1))
    ranges[-1] = (ranges[-1][0], n)  # last scene runs to the final frame

    cfg = VectorizeConfig(input_path="", output_path=config.output_path)
    tracks: list[dict] = []
    scene_reports = []
    for si, (i0, i1) in enumerate(ranges):
        ref = (i0 + i1) // 2
        layers = merge_small_layers(
            extract_layers(frames[ref], num_colors=config.num_colors,
                           min_layer_area=1),
            min_area=config.merge_min_area,
        )
        if not layers:
            raise SceneVideoError(f"scene {si}: no layers extracted")
        traced = [trace_layer(lyr, cfg) for lyr in layers]
        if verbose:
            print(f"scene {si}: frames [{i0}:{i1}] ref={ref} "
                  f"{len(layers)} patches, tracking...", flush=True)
        positions = _track_scene_translations(
            grays[i0:i1], layers, flow_fn, config.flow_samples_per_layer)
        times = list(range(i0, i1, config.keyframe_step))
        if times[-1] != i1 - 1:
            times.append(i1 - 1)
        mags = [float(np.hypot(dx, dy))
                for lyr in layers for dx, dy in positions[lyr.id]]
        # Back-to-front: smallest first so the backdrop lands last
        # (Lottie paints layers[0] on top).
        shapes_by_id = {lyr.id: shapes for lyr, shapes in zip(layers, traced)}
        for lyr in sorted(layers, key=lambda l: l.area):
            tracks.append({
                "name": f"s{si}_layer_{lyr.id}",
                "in_point": i0,
                "out_point": i1,
                "fill_color": lyr.color,
                "shapes": shapes_by_id[lyr.id],
                "motions": [(t, *positions[lyr.id][t - i0]) for t in times],
            })
        scene_reports.append({
            "index": si, "start_frame": i0, "end_frame": i1,
            "ref_frame": ref, "num_layers": len(layers),
            "mean_motion_px": float(np.mean(mags)) if mags else 0.0,
            "max_motion_px": float(np.max(mags)) if mags else 0.0,
        })
        if verbose:
            print(f"  -> {len(layers)} tracks, mean motion "
                  f"{scene_reports[-1]['mean_motion_px']:.1f}px, "
                  f"max {scene_reports[-1]['max_motion_px']:.1f}px", flush=True)

    animation = build_tracked_layers_animation(tracks, W, H, n, out_fps)
    save_lottie_json(animation, config.output_path)
    size_kb = Path(config.output_path).stat().st_size / 1024.0
    if verbose:
        print(f"saved {config.output_path} ({size_kb:.0f} KB, "
              f"{len(tracks)} tracks, {n} frames @ {out_fps:.1f}fps)")
    return {
        "ok": True,
        "output_path": config.output_path,
        "num_scenes": len(ranges),
        "num_frames": n,
        "fps": out_fps,
        "width": W,
        "height": H,
        "num_tracks": len(tracks),
        "size_kb": size_kb,
        "scenes": scene_reports,
    }


def _lerp_loops(loops0: list[dict], loops1: list[dict], e: float) -> list[dict]:
    """Linearly interpolate two bezier-loop lists (same structure)."""
    out = []
    for b0, b1 in zip(loops0, loops1):
        loop = {"c": b0.get("c", True), "v": [], "i": [], "o": []}
        for key in ("v", "i", "o"):
            a0, a1 = b0.get(key, []), b1.get(key, [])
            loop[key] = [
                [p0[0] + e * (p1[0] - p0[0]), p0[1] + e * (p1[1] - p0[1])]
                for p0, p1 in zip(a0, a1)
            ]
        out.append(loop)
    return out


def _path_loops_at(item: dict, t: int) -> list[dict]:
    """Bezier loops of a path item at integer time ``t`` (lerped)."""
    ks = item.get("ks", {})
    if ks.get("a", 0) == 0:
        return ks["k"][0]["s"]
    kfs = ks.get("k", [])
    if not kfs:
        raise SceneVideoError("animated path has no keyframes")
    if t <= kfs[0]["t"]:
        return kfs[0]["s"]
    for k0, k1 in zip(kfs, kfs[1:]):
        if t <= k1["t"]:
            span = max(k1["t"] - k0["t"], 1)
            return _lerp_loops(k0["s"], k1["s"], (t - k0["t"]) / span)
    return kfs[-1]["s"]


def render_animation_frames(
    animation: dict | str | Path, frame_indices: list[int] | None = None
) -> list[np.ndarray]:
    """Rasterize a tracked-layers Lottie animation to RGB frames.

    Parses the JSON (not the in-memory shapes), so this is what a player
    would show: back-to-front compositing, scene lifetimes honored,
    keyframes linearly interpolated. Returns ``uint8`` RGB frames.
    """
    _require_cv2()
    if isinstance(animation, (str, Path)):
        with open(str(animation), encoding="utf-8") as fh:
            animation = json.load(fh)
    if not isinstance(animation, dict):
        raise SceneVideoError("animation must be a Lottie dict or JSON path")
    w, h = int(animation.get("w", 0)), int(animation.get("h", 0))
    n = int(animation.get("op", 0))
    if w < 1 or h < 1 or n < 1:
        raise SceneVideoError(f"invalid animation canvas/timeline: {(w, h, n)}")
    todo = list(range(n)) if frame_indices is None else list(frame_indices)

    frames = []
    for t in todo:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        # layers[0] paints on top: composite back-to-front (reversed).
        for layer in reversed(animation.get("layers", [])):
            if not isinstance(layer, dict):
                continue
            if not (int(layer.get("ip", 0)) <= t < int(layer.get("op", 0))):
                continue
            for group in layer.get("shapes", []):
                if not isinstance(group, dict) or group.get("ty") != "gr":
                    continue
                fill = None
                for item in group.get("it", []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("ty") == "fl" and fill is None:
                        c = item.get("c", {}).get("k", [0, 0, 0])[:3]
                        fill = tuple(max(0, min(255, int(round(float(v) * 255.0)))) for v in c)
                    elif item.get("ty") == "sh":
                        if fill is None:
                            raise SceneVideoError("path before fill")
                        for loop in _path_loops_at(item, t):
                            poly = _flatten_loop(
                                loop.get("v", []), loop.get("i", []),
                                loop.get("o", []), bool(loop.get("c", True)))
                            if len(poly) < 3:
                                continue
                            cv2.fillPoly(
                                canvas, [poly.astype(np.int32).reshape(-1, 1, 2)],
                                color=fill)
        frames.append(canvas)
    return frames


def render_animation_to_mp4(
    animation: dict | str | Path,
    output_mp4: str | Path,
    fps: float | None = None,
) -> str:
    """Render a Lottie JSON to an MP4 file (one MP4 copy of the animation)."""
    _require_cv2()
    if isinstance(animation, (str, Path)):
        with open(str(animation), encoding="utf-8") as fh:
            anim = json.load(fh)
    else:
        anim = animation
    rate = float(fps or anim.get("fr", 30.0))
    if rate <= 0:
        raise SceneVideoError(f"invalid fps: {rate}")
    frames = render_animation_frames(anim)
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(
        str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), rate, (w, h))
    if not writer.isOpened():
        raise SceneVideoError(f"could not open VideoWriter for {output_mp4}")
    try:
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return str(output_mp4)


@dataclass
class BraindeadVideoConfig:
    """Knobs for the brain-dead version: every frame gets its own still.

    No scenes, no tracking, no interpolation: frame ``t`` becomes
    ``f{t}_layer_<id>`` ShapeLayers with lifetime exactly ``[t, t+1)``.
    Patch sets differ frame to frame (flicker included, correctness
    guaranteed).
    """

    input_path: str
    output_path: str
    # Working resolution: resampled FPS and longest-side cap.
    target_fps: float = 4.0
    max_dimension: int = 256
    # Per-frame patch decomposition.
    num_colors: int = 8
    merge_min_area: int = 10


def build_braindead_video_lottie(
    config: BraindeadVideoConfig, verbose: bool = True
) -> dict:
    """Quantize -> Lottie each frame independently, patch them together.

    Returns a report with per-frame layer counts plus render-back sample
    checks (first/middle/last frame re-rendered from the saved JSON and
    compared to the original patches).
    """
    _require_cv2()
    if not config.input_path or not Path(config.input_path).is_file():
        raise SceneVideoError(f"input not found: {config.input_path}")
    if not config.output_path:
        raise SceneVideoError("output_path must be set")

    vcfg = VectorizeConfig(
        input_path=config.input_path, output_path="",
        target_fps=config.target_fps, max_dimension=config.max_dimension,
        color_count=None,
    )
    pre = VideoPreprocessor(config.input_path, vcfg)
    frames = pre.extract_all_frames()
    if not frames:
        raise SceneVideoError(f"no frames decoded from {config.input_path}")
    n, out_fps = len(frames), float(pre.output_fps)
    H, W, _ = frames[0].shape
    cfg = VectorizeConfig(input_path="", output_path=config.output_path)

    check_idx = {0, n // 2, n - 1}
    tracks: list[dict] = []
    frame_reports = []
    kept: dict[int, list] = {}
    for t, frame in enumerate(frames):
        layers = merge_small_layers(
            extract_layers(frame, num_colors=config.num_colors,
                           min_layer_area=1),
            min_area=config.merge_min_area,
        )
        if not layers:
            raise SceneVideoError(f"frame {t}: no layers extracted")
        traced = [trace_layer(lyr, cfg) for lyr in layers]
        shapes_by_id = {lyr.id: s for lyr, s in zip(layers, traced)}
        # Back-to-front within the frame: smallest first.
        for lyr in sorted(layers, key=lambda l: l.area):
            tracks.append({
                "name": f"f{t}_layer_{lyr.id}",
                "in_point": t,
                "out_point": t + 1,
                "fill_color": lyr.color,
                "shapes": shapes_by_id[lyr.id],
                "motions": [(t, 0.0, 0.0)],
            })
        frame_reports.append(
            {"frame": t, "num_layers": len(layers)})
        if t in check_idx:
            kept[t] = layers
        if verbose and (t % 25 == 0 or t == n - 1):
            print(f"  frame {t}/{n - 1}: {len(layers)} patches", flush=True)

    animation = build_tracked_layers_animation(tracks, W, H, n, out_fps)
    save_lottie_json(animation, config.output_path)
    size_kb = Path(config.output_path).stat().st_size / 1024.0

    sample_checks = []
    for t in sorted(kept):
        renders = render_lottie_layers(
            {  # single-frame still dicts reuse the still renderer
                "layers": [
                    lyr for lyr in animation["layers"]
                    if lyr.get("nm", "").startswith(f"f{t}_layer_")
                ],
                "w": W, "h": H,
            },
            (H, W),
        )
        rep = check_renders_against_layers(kept[t], renders, (H, W))
        sample_checks.append({
            "frame": t,
            "num_layers": len(kept[t]),
            "missing_pixels": sum(p["missing_pixels"] for p in rep["layers"]),
            "canvas_diff_pixels": rep["canvas_diff_pixels"],
            "ok": rep["ok"],
        })
    if verbose:
        print(f"saved {config.output_path} ({size_kb:.0f} KB, "
              f"{len(tracks)} tracks, {n} frames @ {out_fps:.1f}fps)")
    return {
        "ok": True,
        "output_path": config.output_path,
        "num_frames": n,
        "fps": out_fps,
        "width": W,
        "height": H,
        "num_tracks": len(tracks),
        "size_kb": size_kb,
        "frames": frame_reports,
        "sample_checks": sample_checks,
    }


def clear_patch_images(patches_dir: str | Path) -> int:
    """Delete all image files in a patches folder. Returns the count removed."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
    removed = 0
    for p in Path(patches_dir).iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            p.unlink()
            removed += 1
    return removed


__all__ = [
    "BraindeadVideoConfig",
    "SceneVideoConfig",
    "SceneVideoError",
    "build_braindead_video_lottie",
    "build_scene_video_lottie",
    "render_animation_frames",
    "render_animation_to_mp4",
    "clear_patch_images",
]
