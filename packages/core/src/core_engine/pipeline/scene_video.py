"""Whole-video Lottie: main (exact) + compressed (tracked) builders.

Two builders share the same per-frame patch decomposition
(``extract_layers`` + ``merge_small_layers`` + ``trace_layer``) and the
same :func:`build_tracked_layers_animation` envelope, but differ in how
frames relate to each other:

* **main** (:class:`VideoConfig` / :func:`build_video_lottie`) -- every
  frame gets its own still (``f{t}_layer_<id>`` ShapeLayers with lifetime
  exactly ``[t, t+1)``). No scenes, no tracking, no interpolation.
  Patch sets differ frame to frame (flicker included, correctness
  guaranteed). Streaming build with bounded RAM.
* **compressed** (:class:`CompressedVideoConfig` /
  :func:`build_compressed_video_lottie`) -- the video is split into
  scenes (``explain_splits``); each scene keeps ONE fixed patch set
  (from its middle reference frame) and every other frame only moves
  those patches with dense optical-flow translations. One ShapeLayer per
  patch with scene lifetime + sparse path keyframes. Much smaller JSON,
  motion-approximated.

Typical usage::

    from core_engine.pipeline.scene_video import (
        VideoConfig, build_video_lottie,
        CompressedVideoConfig, build_compressed_video_lottie)

    report = build_video_lottie(VideoConfig(
        input_path="clip.mp4", output_path="video.json"))
    report = build_compressed_video_lottie(CompressedVideoConfig(
        input_path="clip.mp4", output_path="video_small.json"))

Knobs for :class:`CompressedVideoConfig`:

* ``target_fps`` / ``max_dimension`` -- resampled working resolution.
* ``num_colors`` / ``merge_min_area`` -- reference patch decomposition.
* ``flow_method`` -- ``"dis"`` (DISOpticalFlow/MEDIUM, strongest flow
  shipped with OpenCV) or ``"farneback"`` (the repo's default).
* ``keyframe_step`` -- emit a path keyframe every N frames (players
  interpolate between them; 1 = every frame).
* ``scene_threshold`` / ``scene_min_len`` -- cut sensitivity for the
  existing scene splitter.
* ``max_frames`` -- safety cap on decoded working frames (compressed is
  a batch build: the whole clip lives in RAM, unlike the streaming main
  builder -- use main for long clips).
"""

from __future__ import annotations

import json
import shutil
import subprocess
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


class VideoError(ValueError):
    """Raised when video planning, flow, or assembly fails."""


# Back-compat alias: the error was previously called ``SceneVideoError``.
SceneVideoError = VideoError


@dataclass
class CompressedVideoConfig:
    """Knobs for the compressed builder: fixed patches + flow tracking."""

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
    # Safety cap on decoded working frames (batch build holds the whole
    # clip in RAM). None = no cap (like before).
    max_frames: int | None = None


# Back-compat alias: previously ``SceneVideoConfig``.
SceneVideoConfig = CompressedVideoConfig


def _require_cv2() -> None:
    if cv2 is None:
        raise VideoError(
            "OpenCV (cv2) is required for video tracking/rendering: "
            "install opencv-python-headless"
        )


def _make_flow_fn(method: str):
    """Build a ``(prev_gray, curr_gray) -> [H, W, 2] float32`` flow closure."""
    _require_cv2()
    if method == "dis":
        try:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        except Exception as exc:
            raise VideoError(f"DIS optical flow unavailable: {exc}") from exc

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
    raise VideoError(f"unknown flow_method: {method!r} (use 'dis'/'farneback')")


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


def _validate_compressed_config(config: CompressedVideoConfig) -> None:
    """Validate a compressed config early with actionable messages."""
    if not config.input_path or not Path(config.input_path).is_file():
        raise VideoError(f"input not found: {config.input_path}")
    if not config.output_path:
        raise VideoError("output_path must be set")
    if config.target_fps is not None and config.target_fps <= 0:
        raise VideoError(f"target_fps must be > 0: {config.target_fps}")
    if config.max_dimension is not None and config.max_dimension < 16:
        raise VideoError(f"max_dimension must be >= 16: {config.max_dimension}")
    if not 2 <= config.num_colors <= 24:
        raise VideoError(f"num_colors must be 2..24, got {config.num_colors}")
    if config.merge_min_area < 1:
        raise VideoError(f"merge_min_area must be >= 1: {config.merge_min_area}")
    if config.keyframe_step < 1:
        raise VideoError(f"keyframe_step must be >= 1: {config.keyframe_step}")
    if config.flow_method not in ("dis", "farneback"):
        raise VideoError(
            f"unknown flow_method: {config.flow_method!r} (use 'dis'/'farneback')"
        )
    if config.scene_threshold <= 0:
        raise VideoError(f"scene_threshold must be > 0: {config.scene_threshold}")
    if config.scene_min_len < 1:
        raise VideoError(f"scene_min_len must be >= 1: {config.scene_min_len}")
    if config.flow_samples_per_layer < 1:
        raise VideoError(
            f"flow_samples_per_layer must be >= 1: {config.flow_samples_per_layer}"
        )
    if config.max_frames is not None and config.max_frames < 1:
        raise VideoError(f"max_frames must be >= 1: {config.max_frames}")


def _map_scenes_to_frames(scenes, n: int) -> list[tuple[int, int]]:
    """Map scene time ranges onto working frame indices with guarantees.

    Source scenes live in the *source* time domain while working frames
    live in the *resampled* domain (``target_fps``). Mapping by rounded
    time drifts, so this helper clamps, sorts, de-duplicates, drops
    empty ranges, and forces full coverage ``[0, n)`` with the last
    scene extended to ``n``. Never returns an empty list for ``n >= 1``.
    """
    if n < 1:
        raise VideoError("no frames to map scenes onto")
    if not scenes:
        return [(0, n)]
    # Working fps is unknown here; callers pass scenes already in seconds
    # plus out_fps separately -- keep this helper pure on indices by
    # expecting pre-mapped pairs? No: do the time mapping in the caller
    # and only normalize here.
    bounds: list[int] = [0]
    for s in scenes:
        bounds.append(int(s[0]))
        bounds.append(int(s[1]))
    bounds.append(n)
    bounds = sorted(set(max(0, min(n, b)) for b in bounds))
    ranges = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    ranges = [(a, b) for a, b in ranges if b > a]
    if not ranges:
        return [(0, n)]
    # Merge a trailing 1-frame sliver into its predecessor to avoid a
    # degenerate single-frame scene from rounding.
    if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] < 2:
        ranges[-2] = (ranges[-2][0], n)
        ranges.pop()
    else:
        ranges[-1] = (ranges[-1][0], n)
    return ranges


def _write_preview_mp4_streaming(
    animation: dict | str | Path,
    mp4_path: str | Path,
    fps: float,
) -> None:
    """Render a finished animation to MP4 one frame at a time (bounded RAM).

    Uses the same H264/yuv420p/faststart flags as the main builder's
    single-pass preview so compressed previews play everywhere.
    """
    import shutil as _shutil
    import subprocess as _subprocess

    ffmpeg = _shutil.which("ffmpeg")
    if ffmpeg is None:
        raise VideoError("ffmpeg binary not found on PATH")
    if isinstance(animation, (str, Path)):
        with open(str(animation), encoding="utf-8") as fh:
            anim = json.load(fh)
    else:
        anim = animation
    w, h = int(anim.get("w", 0)), int(anim.get("h", 0))
    total = int(anim.get("op", 0))
    if w < 1 or h < 1 or total < 1:
        raise VideoError(f"invalid animation canvas/timeline: {(w, h, total)}")
    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(float(fps)), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(mp4_path)]
    proc = _subprocess.Popen(cmd, stdin=_subprocess.PIPE,
                             stdout=_subprocess.DEVNULL, stderr=_subprocess.PIPE)
    try:
        for t in range(total):
            frame = render_animation_frames(anim, [t])[0]
            try:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except BrokenPipeError as exc:
                raise VideoError(
                    f"ffmpeg preview pipe broke at frame {t}") from exc
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise VideoError(
                f"ffmpeg preview failed for {mp4_path}: "
                f"{(stderr or b'').decode(errors='replace').strip()}")
    except Exception:
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        proc.wait()
        raise


def build_compressed_video_lottie(
    config: CompressedVideoConfig,
    verbose: bool = True,
    preview_mp4: str | Path | None = None,
) -> dict:
    """Build one small Lottie JSON via per-scene fixed patches + flow.

    Each scene keeps ONE patch set (middle reference frame); all other
    frames only translate those patches with dense optical flow. Batch
    build: the whole working clip lives in RAM (see ``max_frames``).
    For long clips or bounded memory use :func:`build_video_lottie`.

    Progress lines include ``frame i/n`` markers so CLI progress bars
    work for both modes. Returns a report dict with the same core keys
    as the main builder (``mode="compressed"``) plus per-scene stats
    and render-back ``sample_checks`` (MAE of the player render vs the
    working frame at each scene reference -- approximation quality,
    not exactness).
    """
    _require_cv2()
    _validate_compressed_config(config)
    flow_fn = _make_flow_fn(config.flow_method)

    scenes, _ = explain_splits(
        config.input_path,
        SceneConfig(threshold=config.scene_threshold,
                    min_scene_len=config.scene_min_len),
    )
    vcfg = VectorizeConfig(
        input_path=config.input_path, output_path="",
        target_fps=config.target_fps, max_dimension=config.max_dimension,
        color_count=None, max_frames=config.max_frames,
    )
    pre = VideoPreprocessor(config.input_path, vcfg)
    frames = pre.extract_all_frames()
    if not frames:
        raise VideoError(f"no frames decoded from {config.input_path}")
    n, out_fps = len(frames), float(pre.output_fps)
    H, W, _ = frames[0].shape
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    # Map scene times (source seconds) onto working frame indices, then
    # normalize to guaranteed full coverage.
    raw_pairs = []
    for s in scenes:
        i0 = max(0, min(n - 1, int(round(s.start_time * out_fps))))
        i1 = max(i0 + 1, min(n, int(round(s.end_time * out_fps))))
        raw_pairs.append((i0, i1))
    if raw_pairs:
        flat = [(a, b) for a, b in raw_pairs]
        bounds = sorted(set([0, n] + [x for p in flat for x in p]))
        bounds = [max(0, min(n, b)) for b in bounds]
        ranges = [(bounds[i], bounds[i + 1])
                  for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    else:
        ranges = [(0, n)]
    ranges = _map_scenes_to_frames(ranges, n)

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
            raise VideoError(f"scene {si}: no layers extracted")
        traced = [trace_layer(lyr, cfg) for lyr in layers]
        if verbose:
            total_s = f"{n - 1}"
            print(f"scene {si + 1}/{len(ranges)}: frames [{i0}:{i1}] ref={ref} "
                  f"{len(layers)} patches, tracking...", flush=True)
            print(f"  frame {i0}/{total_s}: scene {si + 1} start", flush=True)
        try:
            positions = _track_scene_translations(
                grays[i0:i1], layers, flow_fn, config.flow_samples_per_layer)
        except Exception as exc:
            raise VideoError(f"scene {si}: flow tracking failed: {exc}") from exc
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
            print(f"  frame {i1 - 1}/{total_s}: scene {si + 1} done", flush=True)
        # Free per-scene flow fields promptly; frames/grays stay for later
        # scenes (batch build) but intermediate flow arrays are released.
        del positions, traced, layers, shapes_by_id

    animation = build_tracked_layers_animation(tracks, W, H, n, out_fps)
    save_lottie_json(animation, config.output_path)
    size_kb = Path(config.output_path).stat().st_size / 1024.0

    mp4_out = str(Path(preview_mp4)) if preview_mp4 else None
    if mp4_out:
        if verbose:
            print(f"rendering preview MP4 -> {mp4_out} ...", flush=True)
        _write_preview_mp4_streaming(animation, mp4_out, out_fps)

    # Sample checks: render each scene reference from the SAVED json and
    # compare against the working frame (approximation quality).
    sample_checks = []
    for rep in scene_reports:
        t = int(rep["ref_frame"])
        rendered = render_animation_frames(config.output_path, [t])[0]
        orig = frames[t].astype(np.float32)
        mae = float(np.abs(rendered.astype(np.float32) - orig).mean())
        sample_checks.append({
            "frame": t,
            "num_layers": int(rep["num_layers"]),
            "mae": mae,
            "mean_motion_px": float(rep["mean_motion_px"]),
            "ok": True,
        })

    if verbose:
        print(f"saved {config.output_path} ({size_kb:.0f} KB, "
              f"{len(tracks)} tracks, {n} frames @ {out_fps:.1f}fps)")
    # Release heavy buffers before returning.
    del frames, grays
    return {
        "ok": True,
        "mode": "compressed",
        "output_path": config.output_path,
        "num_scenes": len(ranges),
        "num_frames": n,
        "fps": out_fps,
        "width": W,
        "height": H,
        "num_tracks": len(tracks),
        "size_kb": size_kb,
        "scenes": scene_reports,
        "sample_checks": sample_checks,
        "preview_mp4": mp4_out,
    }


# Back-compat alias: previously ``build_scene_video_lottie``.
def build_scene_video_lottie(
    config: CompressedVideoConfig,
    verbose: bool = True,
    preview_mp4: str | Path | None = None,
) -> dict:
    """Deprecated alias of :func:`build_compressed_video_lottie`."""
    return build_compressed_video_lottie(config, verbose=verbose,
                                         preview_mp4=preview_mp4)


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
        raise VideoError("animated path has no keyframes")
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
        raise VideoError("animation must be a Lottie dict or JSON path")
    w, h = int(animation.get("w", 0)), int(animation.get("h", 0))
    n = int(animation.get("op", 0))
    if w < 1 or h < 1 or n < 1:
        raise VideoError(f"invalid animation canvas/timeline: {(w, h, n)}")
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
                            raise VideoError("path before fill")
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
    """Render a Lottie JSON to an MP4 file (one MP4 copy of the animation).

    Prefers ffmpeg H264 + yuv420p + faststart (plays in browsers/Jupyter);
    falls back to cv2's mp4v writer when ffmpeg is unavailable.
    """
    _require_cv2()
    if isinstance(animation, (str, Path)):
        with open(str(animation), encoding="utf-8") as fh:
            anim = json.load(fh)
    else:
        anim = animation
    rate = float(fps or anim.get("fr", 30.0))
    if rate <= 0:
        raise VideoError(f"invalid fps: {rate}")
    import shutil as _shutil

    if _shutil.which("ffmpeg") is not None:
        _write_preview_mp4_streaming(anim, output_mp4, rate)
        return str(output_mp4)
    frames = render_animation_frames(anim)
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(
        str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), rate, (w, h))
    if not writer.isOpened():
        raise VideoError(f"could not open VideoWriter for {output_mp4}")
    try:
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return str(output_mp4)


@dataclass
class VideoConfig:
    """Knobs for the main builder: every frame gets its own still.

    No scenes, no tracking, no interpolation: frame ``t`` becomes
    ``f{t}_layer_<id>`` ShapeLayers with lifetime exactly ``[t, t+1)``.
    Patch sets differ frame to frame (flicker included, correctness
    guaranteed). This is the default ``vectorise`` path.
    """

    input_path: str
    output_path: str
    # Working resolution: resampled FPS and longest-side cap.
    target_fps: float = 4.0
    max_dimension: int = 256
    # Per-frame patch decomposition.
    num_colors: int = 8
    merge_min_area: int = 10


# Back-compat alias: previously ``BraindeadVideoConfig``.
BraindeadVideoConfig = VideoConfig


def build_video_lottie(
    config: VideoConfig,
    verbose: bool = True,
    preview_mp4: str | Path | None = None,
) -> dict:
    """Quantize -> Lottie each frame independently, patch them together.

    Streaming build with bounded RAM: frames are pulled from the FFmpeg
    pipe one at a time (never the whole clip in memory), each frame's
    layers are serialized to a temp JSONL file as soon as they are traced,
    and the final JSON is assembled by streaming that file -- so a 1080p
    clip that would OOM the batch build completes on modest machines.
    The finished JSON is content-identical to the batch build (same layer
    dicts in the same order, same header keys from the same builder).

    Args:
        config: input/output paths and per-frame knobs.
        verbose: print per-frame progress + the final summary.
        preview_mp4: optional MP4 path. When set, each frame is rasterized
            from its just-traced layers and piped to ffmpeg in the same
            pass (same flags as the notebook cell: H264 + yuv420p +
            faststart), so JSON + preview MP4 complete with one bounded-
            memory pass and the giant JSON is never re-parsed.

    Returns a report with per-frame layer counts plus render-back sample
    checks (first/middle/last frame re-rendered from the saved JSON and
    compared to the original patches).
    """
    _require_cv2()
    if not config.input_path or not Path(config.input_path).is_file():
        raise VideoError(f"input not found: {config.input_path}")
    if not config.output_path:
        raise VideoError("output_path must be set")
    if config.target_fps is not None and config.target_fps <= 0:
        raise VideoError(f"target_fps must be > 0: {config.target_fps}")
    if config.max_dimension is not None and config.max_dimension < 16:
        raise VideoError(f"max_dimension must be >= 16: {config.max_dimension}")
    if not 2 <= config.num_colors <= 24:
        raise VideoError(f"num_colors must be 2..24, got {config.num_colors}")
    if config.merge_min_area < 1:
        raise VideoError(f"merge_min_area must be >= 1: {config.merge_min_area}")

    vcfg = VectorizeConfig(
        input_path=config.input_path, output_path="",
        target_fps=config.target_fps, max_dimension=config.max_dimension,
        color_count=None,
    )
    pre = VideoPreprocessor(config.input_path, vcfg)
    out_fps = float(pre.output_fps)
    # Estimated total for progress lines (true n is counted as we stream).
    est = pre.estimated_frames
    if not est:
        meta = pre.metadata
        total = meta.frame_count
        if total is None and meta.duration is not None:
            total = int(round(meta.duration * out_fps))
        est = total if total else None
    want_middle = est // 2 if est else None
    cfg = VectorizeConfig(input_path="", output_path=config.output_path)

    out_path = Path(config.output_path)
    tmp_path = out_path.parent / f"{out_path.name}.layers.jsonl.tmp"
    mp4_path = Path(preview_mp4) if preview_mp4 else None

    frame_reports = []
    kept_raw: dict[int, np.ndarray] = {}
    kept_json: dict[int, list[str]] = {}
    last_t, last_raw, last_json = -1, None, None
    n, W, H = 0, 0, 0
    num_tracks = 0
    proc = None
    try:
        with open(tmp_path, "w", encoding="utf-8") as tmp_fh:
            for t, frame in enumerate(pre.extract_frames_generator()):
                frame = np.ascontiguousarray(frame)
                if t == 0:
                    H, W, _ = frame.shape
                    if mp4_path is not None:
                        proc = _open_preview_pipe(mp4_path, W, H, out_fps)
                        if proc is None and verbose:
                            print(f"warning: ffmpeg not found on PATH -- "
                                  f"skipping preview MP4, JSON continues",
                                  flush=True)
                layers = merge_small_layers(
                    extract_layers(frame, num_colors=config.num_colors,
                                   min_layer_area=1),
                    min_area=config.merge_min_area,
                )
                if not layers:
                    raise VideoError(f"frame {t}: no layers extracted")
                traced = [trace_layer(lyr, cfg) for lyr in layers]
                shapes_by_id = {lyr.id: s for lyr, s in zip(layers, traced)}
                # Back-to-front within the frame: smallest first.
                tracks = [{
                    "name": f"f{t}_layer_{lyr.id}",
                    "in_point": t,
                    "out_point": t + 1,
                    "fill_color": lyr.color,
                    "shapes": shapes_by_id[lyr.id],
                    "motions": [(t, 0.0, 0.0)],
                } for lyr in sorted(layers, key=lambda l: l.area)]
                # Layer dicts depend only on tracks (+ canvas/fps), never on
                # the animation length, so per-frame assembly matches the
                # batch build exactly; header (with the true n) is written
                # at the end from the same builder.
                layer_dicts = build_tracked_layers_animation(
                    tracks, W, H, t + 1, out_fps)["layers"]
                layer_json = [json.dumps(ld, separators=(",", ":"))
                              for ld in layer_dicts]
                for s in layer_json:
                    tmp_fh.write(s + "\n")
                num_tracks += len(layer_json)
                if t == 0 or (want_middle is not None and t == want_middle):
                    kept_raw[t] = frame
                    kept_json[t] = layer_json
                last_t, last_raw, last_json = t, frame, layer_json
                frame_reports.append(
                    {"frame": t, "num_layers": len(layers)})
                if proc is not None:
                    try:
                        proc.stdin.write(
                            _render_static_frame(layer_dicts, W, H).tobytes())
                    except BrokenPipeError as exc:
                        raise VideoError(
                            f"ffmpeg preview pipe broke at frame {t}") from exc
                if verbose:
                    total_s = f"{est - 1}" if est else "?"
                    if t % 25 == 0:
                        print(f"  frame {t}/{total_s}: {len(layers)} patches",
                              flush=True)
                n = t + 1
                del layers, traced, shapes_by_id, tracks, layer_dicts, layer_json
        if n == 0:
            raise VideoError(f"no frames decoded from {config.input_path}")
        if proc is not None:
            _close_preview_pipe(proc, mp4_path)
            proc = None

        # Header keys from the real builder (zero-track animation), so the
        # streamed file carries the identical envelope; layers stream after.
        header = build_tracked_layers_animation([], W, H, n, out_fps)
        with open(out_path, "w", encoding="utf-8") as out_fh:
            with open(tmp_path, encoding="utf-8") as tmp_fh:
                out_fh.write("{")
                first = True
                for k, v in header.items():
                    if k == "layers":
                        continue
                    if not first:
                        out_fh.write(",")
                    out_fh.write(json.dumps(k) + ":"
                                 + json.dumps(v, separators=(",", ":")))
                    first = False
                out_fh.write(',"layers":[')
                first_layer = True
                for line in tmp_fh:
                    line = line.strip()
                    if not line:
                        continue
                    if not first_layer:
                        out_fh.write(",")
                    out_fh.write(line)
                    first_layer = False
                out_fh.write("]}")
        size_kb = Path(config.output_path).stat().st_size / 1024.0

        # Sample checks re-extract from the kept raw frames (the pipeline
        # is deterministic, so these equal the streamed layers) and render
        # the kept layer dicts back through the still renderer.
        kept_raw[last_t] = last_raw
        kept_json[last_t] = last_json
        sample_checks = []
        for t in sorted(kept_raw):
            layers_again = merge_small_layers(
                extract_layers(kept_raw[t], num_colors=config.num_colors,
                               min_layer_area=1),
                min_area=config.merge_min_area,
            )
            renders = render_lottie_layers(
                {"layers": [json.loads(s) for s in kept_json[t]],
                 "w": W, "h": H},
                (H, W),
            )
            rep = check_renders_against_layers(layers_again, renders, (H, W))
            sample_checks.append({
                "frame": t,
                "num_layers": len(layers_again),
                "missing_pixels": sum(p["missing_pixels"] for p in rep["layers"]),
                "canvas_diff_pixels": rep["canvas_diff_pixels"],
                "ok": rep["ok"],
            })
        if verbose:
            print(f"saved {config.output_path} ({size_kb:.0f} KB, "
                  f"{num_tracks} tracks, {n} frames @ {out_fps:.1f}fps)")
        return {
            "ok": True,
            "mode": "main",
            "output_path": config.output_path,
            "num_frames": n,
            "fps": out_fps,
            "width": W,
            "height": H,
            "num_tracks": num_tracks,
            "size_kb": size_kb,
            "frames": frame_reports,
            "sample_checks": sample_checks,
            "preview_mp4": str(mp4_path) if mp4_path is not None else None,
        }
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except OSError:
                pass
            proc.wait()


# Back-compat alias: previously ``build_braindead_video_lottie``.
def build_braindead_video_lottie(
    config: VideoConfig,
    verbose: bool = True,
    preview_mp4: str | Path | None = None,
) -> dict:
    """Deprecated alias of :func:`build_video_lottie`."""
    return build_video_lottie(config, verbose=verbose, preview_mp4=preview_mp4)


def _open_preview_pipe(mp4_path: Path, w: int, h: int, fr: float):
    """Open the ffmpeg preview pipe (notebook cell-2 flags); None if missing."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fr), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(mp4_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _close_preview_pipe(proc, mp4_path: Path) -> None:
    """Drain a preview pipe; raise if ffmpeg failed (JSON is complete)."""
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise VideoError(
            f"ffmpeg preview failed for {mp4_path}: "
            f"{(stderr or b'').decode(errors='replace').strip()}")


def _render_static_frame(layer_dicts: list[dict], width: int, height: int) -> np.ndarray:
    """Rasterize one main-builder frame's layer dicts to RGB.

    Applies exactly the compositing ``render_animation_frames`` gives these
    layers (back-to-front, ``layers[0]`` on top, one keyframe per path), so
    the single-pass preview MP4 is pixel-identical to rendering the
    finished JSON -- without ever holding the whole animation in RAM.
    """
    _require_cv2()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for layer in reversed(layer_dicts):
        if not isinstance(layer, dict):
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
                    fill = tuple(max(0, min(255, int(round(float(v) * 255.0))))
                                 for v in c)
                elif item.get("ty") == "sh":
                    if fill is None:
                        raise VideoError("path before fill")
                    for loop in _path_loops_at(item, 0):
                        poly = _flatten_loop(
                            loop.get("v", []), loop.get("i", []),
                            loop.get("o", []), bool(loop.get("c", True)))
                        if len(poly) < 3:
                            continue
                        cv2.fillPoly(
                            canvas, [poly.astype(np.int32).reshape(-1, 1, 2)],
                            color=fill)
    return canvas


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
    "VideoConfig",
    "CompressedVideoConfig",
    "VideoError",
    "BraindeadVideoConfig",
    "SceneVideoConfig",
    "SceneVideoError",
    "build_video_lottie",
    "build_compressed_video_lottie",
    "build_braindead_video_lottie",
    "build_scene_video_lottie",
    "render_animation_frames",
    "render_animation_to_mp4",
    "clear_patch_images",
]
