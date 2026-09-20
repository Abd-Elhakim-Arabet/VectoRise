"""MCP tools wrapping the VectoRise core engine.

Tools:
  * ``probe_video`` — ffprobe metadata for a local video file.
  * ``video_to_lottie`` — convert a local video to Lottie JSON (+ MP4 preview).
  * ``summarize_lottie`` — cheap stats for an existing Lottie JSON file.

Security model (local server, strict anyway):
  * Every path must resolve inside ``VECTORISE_MCP_ROOTS`` (os.pathsep
    separated; default: repo root + system temp). Symlink escapes rejected.
  * Only local files — no URLs, no fetching, no shell (core uses argv-only
    subprocesses for ffmpeg/ffprobe).
  * All numeric/enum params are clamped/validated server-side with the same
    bounds as the web UI; compressed mode carries a ``max_frames`` RAM guard.
  * Outputs default next to the input (``<stem>.json``); nothing is written
    outside the roots.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Literal, TypedDict

_HERE = Path(__file__).resolve()
_PKG_DIR = _HERE.parent.parent.parent  # packages/mcp-server/
_REPO_ROOT = _PKG_DIR.parent.parent
_CORE_SRC = _REPO_ROOT / "packages" / "core" / "src"
if str(_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(_CORE_SRC))

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "The 'mcp' package (v2+) is required: pip install 'mcp>=2.0'"
    ) from exc

mcp = MCPServer("vectorise")

# --- limits (same bounds as the web UI) --------------------------------------
BOUNDS = {
    "num_colors": (8, 24),
    "max_dimension": (128, 1080),
    "target_fps": (5.0, 30.0),
    "merge_min_area": (1, 100),
    "keyframe_step": (1, 10),
    "scene_threshold": (1.0, 100.0),
    "scene_min_len": (1, 60),
    "max_frames": (1, 600),
}
ALLOWED_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def _roots() -> list[Path]:
    raw = os.environ.get("VECTORISE_MCP_ROOTS", "")
    if raw.strip():
        cands = [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]
    else:
        cands = [_REPO_ROOT, Path(tempfile.gettempdir())]
    roots = []
    for c in cands:
        try:
            roots.append(c.resolve())
        except OSError:
            continue
    if not roots:
        raise ValueError("no usable VECTORISE_MCP_ROOTS")
    return roots


def _resolve_inside_roots(user_path: str, *, must_exist: bool) -> Path:
    """Resolve a user-supplied path, rejecting escapes outside the roots."""
    if not user_path or not user_path.strip():
        raise ValueError("path must be set")
    lowered = user_path.strip().lower()
    if lowered.startswith(("http://", "https://", "file://", "ftp://")):
        raise ValueError("only local file paths are accepted, not URLs")
    p = Path(user_path.strip()).expanduser()
    if not p.is_absolute():
        raise ValueError(f"path must be absolute: {user_path!r}")
    try:
        resolved = p.resolve()
    except OSError as exc:
        raise ValueError(f"cannot resolve path: {user_path!r}") from exc
    if not any(resolved == r or r in resolved.parents for r in _roots()):
        raise ValueError(
            f"path outside allowed roots {[str(r) for r in _roots()]}: {user_path!r}"
        )
    if must_exist:
        if not resolved.is_file():
            raise ValueError(f"file not found: {user_path!r}")
    else:
        if resolved.suffix.lower() not in (".json", ".mp4"):
            raise ValueError("output must end in .json or .mp4")
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"cannot create output dir: {exc}") from exc
    return resolved


def _check_video_ext(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_EXTS:
        raise ValueError(
            f"extension {path.suffix or '?'} not allowed "
            "(mp4/mov/webm/mkv/avi)"
        )


def _clamp_int(name: str, value: int) -> int:
    lo, hi = BOUNDS[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be {lo}..{hi}, got {value}")
    return value


def _clamp_float(name: str, value: float) -> float:
    lo, hi = BOUNDS[name]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be {lo}..{hi}, got {value}")
    return value


_READONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
_WRITE = ToolAnnotations(readOnlyHint=False, openWorldHint=False)


class ProbeResult(TypedDict):
    path: str
    width: int
    height: int
    fps: float | None
    duration_s: float | None
    frame_count: int | None
    codec: str | None
    pix_fmt: str | None


class ConvertResult(TypedDict):
    mode: str | None
    json_path: str
    preview_mp4: str | None
    num_frames: int | None
    fps: float | None
    num_tracks: int | None
    size_kb: float


class SummaryResult(TypedDict):
    path: str
    width: int | None
    height: int | None
    fps: int | None
    in_point: int | None
    out_point: int | None
    num_layers: int | None
    size_kb: float


@mcp.tool(
    description="Probe a local video file (resolution, fps, duration, codec). "
    "Call this before converting to pick sane parameters."
    " Rejects paths outside VECTORISE_MCP_ROOTS.",
    annotations=_READONLY,
    structured_output=True,
)
def probe_video(path: str) -> ProbeResult:
    """Return ffprobe metadata for a local video file."""
    from core_engine.pipeline.preprocessor import probe_video as _probe

    resolved = _resolve_inside_roots(path, must_exist=True)
    _check_video_ext(resolved)
    try:
        meta = _probe(str(resolved))
    except ValueError as exc:
        raise ValueError(f"cannot probe video: {exc}") from exc
    return {
        "path": str(resolved),
        "width": meta.width,
        "height": meta.height,
        "fps": round(meta.fps, 3) if meta.fps else None,
        "duration_s": round(meta.duration, 3) if meta.duration is not None else None,
        "frame_count": meta.frame_count,
        "codec": meta.codec,
        "pix_fmt": meta.pix_fmt,
    }


@mcp.tool(
    description="Convert a local video to Lottie JSON (+ optional MP4 preview). "
    "mode 'main' = exact per-frame stills (bigger JSON, always works); "
    "'compressed' = scene-split + flow-tracked (smaller JSON, motion "
    "approximated, capped by max_frames). Returns the builder report."
    " Rejects paths outside VECTORISE_MCP_ROOTS.",
    annotations=_WRITE,
    structured_output=True,
)
def video_to_lottie(
    path: Annotated[str, "Absolute local path to the input video."],
    output_path: Annotated[
        str | None,
        "Absolute .json destination. Defaults to <input-stem>.json next to the input.",
    ] = None,
    mode: Annotated[Literal["main", "compressed"], "Builder to use."] = "main",
    num_colors: Annotated[int, "Palette size per frame (8..24)."] = 16,
    max_dimension: Annotated[int, "Longest-side working resolution in px (128..1080)."] = 480,
    target_fps: Annotated[float, "Working frame rate (5..30)."] = 12.0,
    merge_min_area: Annotated[int, "Patches smaller than this dissolve (1..100 px)."] = 10,
    preview: Annotated[bool, "Also render an MP4 preview next to the JSON."] = True,
    flow_method: Annotated[
        Literal["dis", "farneback"], "Compressed-mode dense flow backend."
    ] = "dis",
    keyframe_step: Annotated[int, "Compressed: path keyframe every N frames (1..10)."] = 2,
    scene_threshold: Annotated[float, "Compressed: cut sensitivity (1..100)."] = 27.0,
    scene_min_len: Annotated[int, "Compressed: min scene length in frames (1..60)."] = 15,
    max_frames: Annotated[int, "Compressed: RAM guard on decoded frames (1..600)."] = 300,
) -> ConvertResult:
    """Convert a local video file to Lottie JSON, returning the build report."""
    from core_engine import video_to_lottie as _convert

    if mode not in ("main", "compressed"):
        raise ValueError("mode must be 'main' or 'compressed'")
    if flow_method not in ("dis", "farneback"):
        raise ValueError("flow_method must be 'dis' or 'farneback'")
    src = _resolve_inside_roots(path, must_exist=True)
    _check_video_ext(src)
    num_colors = _clamp_int("num_colors", num_colors)
    max_dimension = _clamp_int("max_dimension", max_dimension)
    target_fps = _clamp_float("target_fps", target_fps)
    merge_min_area = _clamp_int("merge_min_area", merge_min_area)
    keyframe_step = _clamp_int("keyframe_step", keyframe_step)
    scene_threshold = _clamp_float("scene_threshold", scene_threshold)
    scene_min_len = _clamp_int("scene_min_len", scene_min_len)
    max_frames = _clamp_int("max_frames", max_frames)

    if output_path:
        out = _resolve_inside_roots(output_path, must_exist=False)
        if out.suffix.lower() != ".json":
            raise ValueError("output_path must end in .json")
    else:
        out = src.with_suffix(".json")
        # Default sits next to the input, which is inside the roots already.
    mp4 = str(out.with_name(out.stem + "_converted.mp4")) if preview else None

    kwargs: dict = {}
    if mode == "compressed":
        kwargs = {
            "flow_method": flow_method,
            "keyframe_step": keyframe_step,
            "scene_threshold": scene_threshold,
            "scene_min_len": scene_min_len,
            "max_frames": max_frames,
        }
    try:
        report = _convert(
            str(src), str(out),
            mode=mode,
            target_fps=target_fps,
            max_dimension=max_dimension,
            num_colors=num_colors,
            merge_min_area=merge_min_area,
            preview_mp4=mp4,
            verbose=False,
            **kwargs,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise ValueError(f"conversion failed: {exc}") from exc
    return {
        "mode": report.get("mode"),
        "json_path": str(out),
        "preview_mp4": mp4 if mp4 and Path(mp4).is_file() else None,
        "num_frames": report.get("num_frames"),
        "fps": report.get("fps"),
        "num_tracks": report.get("num_tracks"),
        "size_kb": round(float(report.get("size_kb", 0)), 1),
    }


@mcp.tool(
    description="Cheap stats for an existing Lottie JSON file (canvas size, "
    "frame rate, layer count, file size). No rendering involved."
    " Rejects paths outside VECTORISE_MCP_ROOTS.",
    annotations=_READONLY,
    structured_output=True,
)
def summarize_lottie(json_path: str) -> SummaryResult:
    """Return canvas/timing/size stats for a Lottie JSON file."""
    resolved = _resolve_inside_roots(json_path, must_exist=True)
    if resolved.suffix.lower() != ".json":
        raise ValueError("not a .json file")
    try:
        data = json.loads(resolved.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read lottie json: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("not a lottie object")
    layers = data.get("layers", [])
    return {
        "path": str(resolved),
        "width": data.get("w"),
        "height": data.get("h"),
        "fps": data.get("fr"),
        "in_point": data.get("ip"),
        "out_point": data.get("op"),
        "num_layers": len(layers) if isinstance(layers, list) else None,
        "size_kb": round(resolved.stat().st_size / 1024, 1),
    }
