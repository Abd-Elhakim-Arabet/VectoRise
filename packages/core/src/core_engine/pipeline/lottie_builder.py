"""Lottie schema assembly.

Converts stabilized ``VectorFrame`` sequences into a valid Lottie
animation dict. Shapes keep one ``ShapeLayer`` each across the whole
clip — per-frame topology becomes path keyframes (``t`` = frame number)
instead of duplicated layers, so players interpolate the morph.

Typical usage::

    from core_engine.pipeline.lottie_builder import (
        build_lottie_animation,
        save_lottie_json,
    )

    anim = build_lottie_animation(stable_frames, config)
    save_lottie_json(anim, "out.json")

Backend is the ``lottie`` package (python-lottie objects API); the final
artifact is a plain ``dict`` straight from ``Animation.to_dict()``.
"""

from __future__ import annotations

import json

from core_engine.config import VectorizeConfig
from core_engine.pipeline.layers import Layer
from core_engine.pipeline.vectorizer import PathPoint, VectorFrame, VectorShape

try:  # Optional at import time; builder raises a clear error if missing.
    from lottie import NVector
    from lottie.objects import Animation, Bezier, Color, Fill, Group, Path, ShapeLayer
    from lottie.objects.properties import ShapePropKeyframe
except ImportError:  # pragma: no cover
    NVector = None  # type: ignore[assignment]
    Animation = None  # type: ignore[assignment]
    Bezier = None  # type: ignore[assignment]
    Color = None  # type: ignore[assignment]
    Fill = None  # type: ignore[assignment]
    Group = None  # type: ignore[assignment]
    Path = None  # type: ignore[assignment]
    ShapeLayer = None  # type: ignore[assignment]
    ShapePropKeyframe = None  # type: ignore[assignment]


class LottieBuildError(ValueError):
    """Raised when frames are invalid or the lottie package is missing."""


def _require_lottie() -> None:
    if Animation is None:
        raise LottieBuildError(
            "The 'lottie' package is required to build animations: "
            "install lottie>=0.7"
        )


def _to_bezier(shape: VectorShape) -> "Bezier":
    """Convert a ``VectorShape`` to a lottie ``Bezier`` (relative tangents)."""
    bezier = Bezier()
    bezier.closed = bool(shape.is_closed)
    bezier.vertices = [NVector(p.x, p.y) for p in shape.points]
    bezier.in_tangents = [
        NVector(p.handle_in[0] - p.x, p.handle_in[1] - p.y)
        for p in shape.points
    ]
    bezier.out_tangents = [
        NVector(p.handle_out[0] - p.x, p.handle_out[1] - p.y)
        for p in shape.points
    ]
    return bezier


def _to_fill(shape: VectorShape) -> "Fill":
    r, g, b = shape.fill_color
    for c in (r, g, b):
        if not isinstance(c, int) or not 0 <= c <= 255:
            raise LottieBuildError(f"invalid fill channel: {c!r}")
    return Fill(Color(r / 255.0, g / 255.0, b / 255.0))


def shifted(shape: VectorShape, dx: float, dy: float) -> VectorShape:
    """Translate a shape by ``(dx, dy)`` (vertices and absolute handles)."""
    return VectorShape(
        fill_color=shape.fill_color,
        points=[
            PathPoint(
                x=p.x + dx,
                y=p.y + dy,
                handle_in=(p.handle_in[0] + dx, p.handle_in[1] + dy),
                handle_out=(p.handle_out[0] + dx, p.handle_out[1] + dy),
            )
            for p in shape.points
        ],
        is_closed=shape.is_closed,
    )


def build_lottie_animation(
    frames: list[VectorFrame], config: VectorizeConfig
) -> dict:
    """Assemble stabilized frames into a Lottie animation dict.

    Args:
        frames: stabilized sequence (locked topology preferred). Shape
            index ``i`` is treated as one tracked object across frames.
        config: ``target_fps`` sets the frame rate (``None`` falls back
            to 30); canvas size comes from the first frame.

    Returns:
        Plain-dict Lottie animation with standard keys (``v``, ``fr``,
        ``ip``, ``op``, ``w``, ``h``, ``layers``); one layer per shape
        track, one path keyframe per frame where that track exists.
        Layers are emitted back-to-front (largest track last) because
        Lottie paints ``layers[0]`` on top -- emitting the backdrop first
        would cover every other shape with opaque fill.
    """
    _require_lottie()
    fps = config.target_fps if config.target_fps else 30.0
    if fps <= 0:
        raise LottieBuildError(f"frame rate must be > 0: {fps}")
    n = len(frames)
    width = frames[0].width if n else 0
    height = frames[0].height if n else 0
    for i, vf in enumerate(frames):
        if not isinstance(vf, VectorFrame):
            raise LottieBuildError(f"frames[{i}] is not a VectorFrame")
        if (vf.width, vf.height) != (width, height):
            raise LottieBuildError(
                f"frames[0] is {(width, height)} but frames[{i}] is "
                f"{(vf.width, vf.height)}: canvas must match"
            )

    animation = Animation(n, fps)
    animation.width = width
    animation.height = height
    animation.in_point = 0
    animation.out_point = n

    n_tracks = max((len(vf.shapes) for vf in frames), default=0)
    # Back-to-front: track 0 (largest shape) is the backdrop, so it is
    # emitted last; Lottie paints layers[0] on top.
    for track in reversed(range(n_tracks)):
        layer = ShapeLayer()
        layer.name = f"shape_{track}"
        group = Group()
        group.name = f"shape_{track}"
        track_shapes = [
            (t, vf.shapes[track])
            for t, vf in enumerate(frames)
            if track < len(vf.shapes)
        ]
        if not track_shapes:
            continue
        # A track may be born at a later retrace.  Without this range Lottie
        # displays its first path from frame zero, creating the "future shape"
        # artifacts seen in the original output.
        layer.in_point = track_shapes[0][0]
        layer.out_point = track_shapes[-1][0] + 1
        # add_shape inserts before the trailing TransformShape, keeping the
        # After Effects child order (fills/paths first, "tr" last) that
        # Group.transform (shapes[-1]) and third-party players rely on.
        group.add_shape(_to_fill(track_shapes[0][1]))
        path = Path()
        path.name = f"path_{track}"
        keyframes = []
        for t, shape in track_shapes:
            if not shape.points:
                continue
            keyframes.append(ShapePropKeyframe(t, _to_bezier(shape)))
        path.shape.keyframes = keyframes
        path.shape.animated = len(keyframes) > 1
        group.add_shape(path)
        layer.shapes.append(group)
        animation.layers.append(layer)

    return animation.to_dict()


def build_layered_lottie_animation(
    layers: list[Layer],
    shapes_per_layer: list[list[VectorShape]],
    width: int,
    height: int,
    fps: float = 30.0,
) -> dict:
    """Assemble one still frame's layers into a multi-layer Lottie dict.

    Each :class:`Layer` becomes exactly one ``ShapeLayer`` (named
    ``layer_<id>``) holding that layer's traced shapes under a single
    fill, so the Lottie layer structure mirrors the frame decomposition
    1:1. The animation is a single still frame (``ip=0, op=1``); motion
    across frames is a later step that keyframes these same layers.

    Args:
        layers: layers from :func:`extract_layers` (ids set the order).
        shapes_per_layer: traced shapes parallel to ``layers``; a layer
            with no shapes is skipped (keeps the JSON valid).
        width: canvas width in pixels.
        height: canvas height in pixels.
        fps: frame rate tag (a still has no motion; ``None``/invalid
            falls back to 30).

    Returns:
        Plain-dict Lottie animation with one layer per traced ``Layer``,
        emitted back-to-front (``layers[0]`` paints on top, so the
        backdrop -- largest area -- comes last).
    """
    _require_lottie()
    if len(layers) != len(shapes_per_layer):
        raise LottieBuildError(
            f"layers ({len(layers)}) and shapes_per_layer "
            f"({len(shapes_per_layer)}) must align"
        )
    if width < 1 or height < 1:
        raise LottieBuildError(f"invalid canvas: {(width, height)}")
    if not fps or fps <= 0:
        fps = 30.0

    animation = Animation(1, fps)
    animation.width = width
    animation.height = height
    animation.in_point = 0
    animation.out_point = 1

    for layer, shapes in zip(reversed(layers), reversed(shapes_per_layer)):
        if not shapes:
            continue
        shape_layer = ShapeLayer()
        shape_layer.name = f"layer_{layer.id}"
        shape_layer.in_point = 0
        shape_layer.out_point = 1
        r, g, b = layer.color
        fill = Fill(Color(r / 255.0, g / 255.0, b / 255.0))
        for n, shape in enumerate(shapes):
            if not shape.points:
                continue
            group = Group()
            group.name = f"layer_{layer.id}_path_{n}"
            group.add_shape(fill)
            path = Path()
            path.name = f"layer_{layer.id}_path_{n}"
            path.shape.keyframes = [ShapePropKeyframe(0, _to_bezier(shape))]
            path.shape.animated = False
            group.add_shape(path)
            shape_layer.shapes.append(group)
        if not shape_layer.shapes:
            continue
        animation.layers.append(shape_layer)

    return animation.to_dict()


def build_layer_drift_animation(
    layers: list[Layer],
    shapes_per_layer: list[list[VectorShape]],
    width: int,
    height: int,
    fps: float = 10.0,
    n_frames: int = 20,
    amplitude: float = 12.0,
) -> dict:
    """Animate still layers drifting apart and back (seamless loop).

    Each layer slides along its own direction -- angle ``2*pi*id/N`` over
    the layers -- with offset ``amplitude * sin(2*pi*t/(n_frames-1))``:
    frame 0 and the last frame are both at rest, every patch glides out
    and returns, and the clip loops seamlessly. Layers keep one
    ``ShapeLayer`` per layer (back-to-front, like
    :func:`build_layered_lottie_animation`); translation shifts vertices
    only, relative tangents are untouched.

    Args:
        layers: layers from :func:`extract_layers`.
        shapes_per_layer: traced shapes parallel to ``layers``.
        width: canvas width in pixels.
        height: canvas height in pixels.
        fps: frame rate tag.
        n_frames: frame count (a full sine period, hence loopable).
        amplitude: peak drift distance in pixels.

    Returns:
        Plain-dict Lottie animation with ``n_frames`` path keyframes per
        layer path.
    """
    import math

    _require_lottie()
    if len(layers) != len(shapes_per_layer):
        raise LottieBuildError(
            f"layers ({len(layers)}) and shapes_per_layer "
            f"({len(shapes_per_layer)}) must align"
        )
    if width < 1 or height < 1:
        raise LottieBuildError(f"invalid canvas: {(width, height)}")
    if not fps or fps <= 0:
        fps = 10.0
    if n_frames < 2:
        raise LottieBuildError(f"n_frames must be >= 2, got {n_frames}")
    if amplitude < 0:
        raise LottieBuildError(f"amplitude must be >= 0, got {amplitude}")

    n = len(layers)
    animation = Animation(n_frames, fps)
    animation.width = width
    animation.height = height
    animation.in_point = 0
    animation.out_point = n_frames

    for layer, shapes in zip(reversed(layers), reversed(shapes_per_layer)):
        if not shapes:
            continue
        angle = 2.0 * math.pi * layer.id / max(n, 1)
        dx_unit, dy_unit = math.cos(angle), math.sin(angle)
        shape_layer = ShapeLayer()
        shape_layer.name = f"layer_{layer.id}"
        shape_layer.in_point = 0
        shape_layer.out_point = n_frames
        fill = Fill(
            Color(
                layer.color[0] / 255.0,
                layer.color[1] / 255.0,
                layer.color[2] / 255.0,
            )
        )
        for m, shape in enumerate(shapes):
            if not shape.points:
                continue
            group = Group()
            group.name = f"layer_{layer.id}_path_{m}"
            group.add_shape(fill)
            path = Path()
            path.name = f"layer_{layer.id}_path_{m}"
            keyframes = []
            for t in range(n_frames):
                drift = amplitude * math.sin(
                    2.0 * math.pi * t / (n_frames - 1)
                )
                moved = shifted(shape, drift * dx_unit, drift * dy_unit)
                keyframes.append(ShapePropKeyframe(t, _to_bezier(moved)))
            path.shape.keyframes = keyframes
            path.shape.animated = True
            group.add_shape(path)
            shape_layer.shapes.append(group)
        if not shape_layer.shapes:
            continue
        animation.layers.append(shape_layer)

    return animation.to_dict()


def save_lottie_json(animation_dict: dict, output_path: str) -> None:
    """Write a Lottie dict to ``output_path`` as compact JSON."""
    if not isinstance(animation_dict, dict):
        raise LottieBuildError("animation_dict must be a dict")
    if not output_path:
        raise LottieBuildError("empty output path")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(animation_dict, fh, separators=(",", ":"))


__all__ = [
    "LottieBuildError",
    "build_lottie_animation",
    "build_layered_lottie_animation",
    "build_layer_drift_animation",
    "save_lottie_json",
    "shifted",
]
