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
from core_engine.pipeline.vectorizer import VectorFrame, VectorShape

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
    for track in range(n_tracks):
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
        group.shapes.append(_to_fill(track_shapes[0][1]))
        path = Path()
        path.name = f"path_{track}"
        keyframes = []
        for t, shape in track_shapes:
            if not shape.points:
                continue
            keyframes.append(ShapePropKeyframe(t, _to_bezier(shape)))
        path.shape.keyframes = keyframes
        path.shape.animated = len(keyframes) > 1
        group.shapes.append(path)
        layer.shapes.append(group)
        animation.layers.append(layer)

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
    "save_lottie_json",
]
