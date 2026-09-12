"""Raster-to-Bezier vectorization.

Converts preprocessed, color-quantized RGB frames into structured Bezier
path layers (``VectorFrame``). This stage builds the initial static vector
shapes for keyframes; downstream motion data morphs these paths between
keyframes instead of re-tracing every frame.

Typical usage::

    from core_engine.config import VectorizeConfig
    from core_engine.pipeline.preprocessor import VideoPreprocessor
    from core_engine.pipeline.vectorizer import trace_frame

    cfg = VectorizeConfig(input_path="clip.mp4", output_path="out.json")
    frame = VideoPreprocessor(cfg.input_path, cfg).extract_all_frames()[0]
    vf = trace_frame(frame, cfg)
    print(len(vf.shapes), vf.shapes[0].fill_color)

Tracing backend is ``vtracer`` (Rust, via ``convert_pixels_to_svg``); the
SVG it emits is parsed into native dataclasses. Handles are absolute
pixel coordinates; straight anchors carry zero-length handles
(``handle_in == handle_out == (x, y)``).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core_engine.config import VectorizeConfig

try:  # Optional at import time; trace_frame raises a clear error if missing.
    import vtracer  # type: ignore
except ImportError:  # pragma: no cover
    vtracer = None  # type: ignore[assignment]


class VectorizationError(ValueError):
    """Raised when a frame is invalid, vtracer is missing, or SVG parsing fails."""


@dataclass(frozen=True)
class PathPoint:
    """Single anchor with cubic-Bezier handles (absolute pixel coords)."""

    x: float
    y: float
    handle_in: tuple[float, float]
    handle_out: tuple[float, float]


@dataclass(frozen=True)
class VectorShape:
    """One filled path: color + ordered anchors."""

    fill_color: tuple[int, int, int]
    points: list[PathPoint] = field(default_factory=list)
    is_closed: bool = True


@dataclass(frozen=True)
class VectorFrame:
    """All traced shapes for one frame."""

    frame_index: int
    shapes: list[VectorShape] = field(default_factory=list)
    width: int = 0
    height: int = 0


def _require_vtracer() -> None:
    if vtracer is None:
        raise VectorizationError(
            "vtracer is required for vectorization: install vtracer>=0.6"
        )


def _validate_frame(frame: np.ndarray) -> tuple[int, int]:
    if not isinstance(frame, np.ndarray):
        raise VectorizationError("frame must be a numpy ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise VectorizationError(
            f"frame must have shape (H, W, 3), got {frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise VectorizationError(f"frame must be uint8 RGB, got {frame.dtype}")
    h, w, _ = frame.shape
    if h < 1 or w < 1:
        raise VectorizationError(f"frame has invalid dims: {frame.shape}")
    return h, w


def _parse_fill(raw: str | None) -> tuple[int, int, int] | None:
    """Parse an SVG fill into ``(r, g, b)``; ``None`` for missing/``none``."""
    if not raw:
        return None
    s = raw.strip().lower()
    if s in ("none", "transparent"):
        return None
    m = re.fullmatch(r"#([0-9a-f]{6})", s)
    if m:
        v = m.group(1)
        return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
    m = re.fullmatch(r"#([0-9a-f]{3})", s)
    if m:
        v = m.group(1)
        return (int(v[0] * 2, 16), int(v[1] * 2, 16), int(v[2] * 2, 16))
    m = re.fullmatch(r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _parse_translate(raw: str | None) -> tuple[float, float]:
    """Extract ``(tx, ty)`` from ``transform="translate(x[, y])"``."""
    if not raw:
        return (0.0, 0.0)
    m = re.search(
        r"translate\(\s*(-?\d+(?:\.\d+)?)\s*(?:[,\s]\s*(-?\d+(?:\.\d+)?))?\s*\)",
        raw,
    )
    if not m:
        return (0.0, 0.0)
    return (float(m.group(1)), float(m.group(2) or 0.0))


_TOKEN = re.compile(r"[MmLlCcQqZz]|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _parse_subpaths(d: str) -> list[tuple[list[tuple[float, float, tuple[float, float], tuple[float, float]]], bool]]:
    """Parse an SVG path ``d`` into subpaths.

    Returns a list of ``(anchors, closed)`` where each anchor is
    ``(x, y, handle_in, handle_out)``. Supports M/L/C/Q/Z (absolute and
    relative); other commands (H/V/S/T/A) raise ``VectorizationError``.
    """
    tokens = _TOKEN.findall(d or "")
    # Each M starts a new subpath: (points, closed, cursor, start).
    subpaths: list[list] = []
    pts: list = []
    closed = False
    cx = cy = 0.0
    sx = sy = 0.0
    i = 0

    def flush() -> None:
        nonlocal pts, closed
        if pts:
            subpaths.append((pts, closed))
        pts = []
        closed = False

    def anchor(x: float, y: float) -> None:
        pts.append([x, y, (x, y), (x, y)])

    while i < len(tokens):
        t = tokens[i]
        i += 1
        if t in ("M", "m"):
            rel = t == "m"
            first = True
            while i + 1 < len(tokens) and tokens[i] not in "MmLlCcQqZz":
                x = float(tokens[i])
                y = float(tokens[i + 1])
                i += 2
                if rel:
                    x += cx
                    y += cy
                if first:
                    flush()
                    first = False
                cx, cy = x, y
                sx, sy = x, y
                anchor(x, y)
        elif t in ("L", "l"):
            rel = t == "l"
            while i + 1 < len(tokens) and tokens[i] not in "MmLlCcQqZz":
                x = float(tokens[i])
                y = float(tokens[i + 1])
                i += 2
                if rel:
                    x += cx
                    y += cy
                cx, cy = x, y
                anchor(x, y)
        elif t in ("C", "c"):
            rel = t == "c"
            while i + 5 < len(tokens) and tokens[i] not in "MmLlCcQqZz":
                x1, y1, x2, y2, x, y = (float(tokens[i + k]) for k in range(6))
                i += 6
                if rel:
                    x1 += cx
                    y1 += cy
                    x2 += cx
                    y2 += cy
                    x += cx
                    y += cy
                if pts:
                    pts[-1][3] = (x1, y1)
                anchor(x, y)
                pts[-1][2] = (x2, y2)
                cx, cy = x, y
        elif t in ("Q", "q"):
            rel = t == "q"
            while i + 3 < len(tokens) and tokens[i] not in "MmLlCcQqZz":
                qx, qy, x, y = (float(tokens[i + k]) for k in range(4))
                i += 4
                if rel:
                    qx += cx
                    qy += cy
                    x += cx
                    y += cy
                if pts:
                    px, py = pts[-1][0], pts[-1][1]
                    pts[-1][3] = (
                        px + 2.0 / 3.0 * (qx - px),
                        py + 2.0 / 3.0 * (qy - py),
                    )
                anchor(x, y)
                pts[-1][2] = (
                    x + 2.0 / 3.0 * (qx - x),
                    y + 2.0 / 3.0 * (qy - y),
                )
                cx, cy = x, y
        elif t in ("Z", "z"):
            closed = True
            cx, cy = sx, sy
        else:
            raise VectorizationError(f"Unsupported SVG path command: {t!r}")
    flush()
    return [
        ([(x, y, hi, ho) for x, y, hi, ho in p], c) for p, c in subpaths
    ]


def _polygon_area(points: Sequence[PathPoint]) -> float:
    """Shoelace area of anchor polygon (0.0 for < 3 points)."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    n = len(points)
    for k in range(n):
        x0, y0 = points[k].x, points[k].y
        x1, y1 = points[(k + 1) % n].x, points[(k + 1) % n].y
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def trace_frame(
    frame: np.ndarray,
    config: VectorizeConfig,
    frame_index: int = 0,
) -> VectorFrame:
    """Trace a quantized RGB frame into Bezier path layers.

    Args:
        frame: ``uint8`` RGB ``[H, W, 3]`` (ideally color-quantized so
            each flat region becomes one path).
        config: ``path_precision`` controls Bezier decimal precision;
            ``min_shape_area`` drops speckle shapes (also forwarded to
            vtracer's own ``filter_speckle``).
        frame_index: identifier stored on the returned ``VectorFrame``.

    Returns:
        ``VectorFrame`` with shapes sorted largest-area first and all
        coordinates clamped to frame bounds.
    """
    _require_vtracer()
    h, w = _validate_frame(frame)
    if config.path_precision is not None and config.path_precision < 0:
        raise VectorizationError(
            f"path_precision must be >= 0: {config.path_precision}"
        )
    if config.min_shape_area is not None and config.min_shape_area < 0:
        raise VectorizationError(
            f"min_shape_area must be >= 0: {config.min_shape_area}"
        )
    min_area = float(config.min_shape_area or 0)

    # vtracer wants RGBA tuples; frames are opaque RGB.
    flat = frame.reshape(-1, 3)
    alpha = np.full((flat.shape[0], 1), 255, dtype=np.uint8)
    rgba = np.concatenate([flat, alpha], axis=1)
    pixels = [tuple(int(v) for v in px) for px in rgba.tolist()]

    try:
        svg = vtracer.convert_pixels_to_svg(
            pixels,
            (w, h),
            colormode="color",
            # 'cutout' tiles regions seam-free with shared boundaries, so
            # small regions survive instead of being swallowed by stacking.
            hierarchical="cutout",
            mode="spline",
            filter_speckle=int(config.min_shape_area or 0),
            # Full channel precision: input is pre-quantized, keep its exact
            # palette instead of re-merging close colors.
            color_precision=8,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            splice_threshold=45,
            path_precision=int(config.path_precision or 0),
        )
    except Exception as exc:
        raise VectorizationError(f"vtracer tracing failed: {exc}") from exc

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise VectorizationError(f"Could not parse vtracer SVG: {exc}") from exc

    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    shapes: list[VectorShape] = []
    for el in root.iter():
        name = _local(el.tag)
        if name == "path":
            fill = _parse_fill(el.get("fill"))
            if fill is None:
                continue
            tx, ty = _parse_translate(el.get("transform"))
            try:
                subpaths = _parse_subpaths(el.get("d") or "")
            except VectorizationError:
                continue
            for anchors, closed in subpaths:
                pts = [
                    PathPoint(
                        x=min(max(x + tx, 0.0), float(w)),
                        y=min(max(y + ty, 0.0), float(h)),
                        handle_in=(
                            min(max(hi[0] + tx, 0.0), float(w)),
                            min(max(hi[1] + ty, 0.0), float(h)),
                        ),
                        handle_out=(
                            min(max(ho[0] + tx, 0.0), float(w)),
                            min(max(ho[1] + ty, 0.0), float(h)),
                        ),
                    )
                    for x, y, hi, ho in anchors
                ]
                if len(pts) < 2:
                    continue
                if _polygon_area(pts) < min_area:
                    continue
                shapes.append(
                    VectorShape(fill_color=fill, points=pts, is_closed=closed)
                )
        elif name == "rect":
            # Fallback: some tracers emit background rects.
            fill = _parse_fill(el.get("fill"))
            if fill is None:
                continue
            try:
                rx = float(el.get("x", 0) or 0)
                ry = float(el.get("y", 0) or 0)
                rw = float(el.get("width", w) or w)
                rh = float(el.get("height", h) or h)
            except (TypeError, ValueError):
                continue
            x0, y0 = max(rx, 0.0), max(ry, 0.0)
            x1, y1 = min(rx + rw, float(w)), min(ry + rh, float(h))
            pts = [
                PathPoint(x0, y0, (x0, y0), (x0, y0)),
                PathPoint(x1, y0, (x1, y0), (x1, y0)),
                PathPoint(x1, y1, (x1, y1), (x1, y1)),
                PathPoint(x0, y1, (x0, y1), (x0, y1)),
            ]
            if _polygon_area(pts) < min_area:
                continue
            shapes.append(VectorShape(fill_color=fill, points=pts, is_closed=True))

    shapes.sort(key=_polygon_area_for_sort, reverse=True)
    return VectorFrame(
        frame_index=frame_index, shapes=shapes, width=w, height=h
    )


def _polygon_area_for_sort(shape: VectorShape) -> float:
    return _polygon_area(shape.points)


__all__ = [
    "VectorizationError",
    "PathPoint",
    "VectorShape",
    "VectorFrame",
    "trace_frame",
]
