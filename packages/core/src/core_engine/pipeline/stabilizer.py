"""Temporal path stabilization & topology lock.

Bridges ``vectorizer`` keyframes and ``tracker`` motion fields: instead of
re-tracing every frame (which makes anchor counts jitter and shapes
flicker), the keyframe topology — same shapes, same anchors, same order —
is carried forward by sampling optical flow at each anchor.

Typical usage::

    from core_engine.pipeline.stabilizer import PathStabilizer

    stab = PathStabilizer(config)
    stable = stab.stabilize_sequence(
        [trace_frame(keyframe, config)], motion_fields,
        retrace_fn=lambda i: trace_frame(frames[i], config),
    )

A fresh topology is adopted only at keyframes: every
``config.keyframe_interval`` frames, or when the mean per-anchor flow
magnitude exceeds ``config.max_tracking_error`` (fast motion /
occlusion makes propagation untrustworthy). Without a ``retrace_fn`` the
base topology is propagated for the whole sequence.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from core_engine.config import VectorizeConfig
from core_engine.pipeline.tracker import MotionField
from core_engine.pipeline.vectorizer import PathPoint, VectorFrame, VectorShape


class StabilizationError(ValueError):
    """Raised when inputs are invalid or fields/shapes are incompatible."""


def _sample_flow(flow: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Bilinear-sample a ``[H, W, 2]`` flow field at subpixel ``(x, y)``.

    Coordinates outside the field are clamped to the border (anchors never
    escape the frame, so topology survives edge motion).
    """
    h, w, _ = flow.shape
    x = min(max(x, 0.0), float(w - 1))
    y = min(max(y, 0.0), float(h - 1))
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = flow[y0, x0] * (1.0 - fx) + flow[y0, x1] * fx
    bottom = flow[y1, x0] * (1.0 - fx) + flow[y1, x1] * fx
    vec = top * (1.0 - fy) + bottom * fy
    return (float(vec[0]), float(vec[1]))


def _validate_field(field: MotionField, width: int, height: int) -> np.ndarray:
    flow = field.flow_vectors
    if not isinstance(flow, np.ndarray) or flow.ndim != 3 or flow.shape[2] != 2:
        raise StabilizationError(
            f"flow_vectors must have shape (H, W, 2), got {getattr(flow, 'shape', None)}"
        )
    if (flow.shape[1], flow.shape[0]) != (width, height):
        raise StabilizationError(
            f"flow field {flow.shape[:2][::-1]} does not match "
            f"frame dims {(width, height)}"
        )
    return flow


def mean_tracking_error(shape: VectorShape, motion_field: MotionField) -> float:
    """Mean per-anchor flow magnitude — the keyframe trigger metric.

    Large values mean anchors would jump far in one step (fast motion,
    occlusion, or bad flow), so the propagated topology can't be trusted.
    """
    if not shape.points:
        return 0.0
    flow = motion_field.flow_vectors
    total = 0.0
    for p in shape.points:
        dx, dy = _sample_flow(flow, p.x, p.y)
        total += (dx * dx + dy * dy) ** 0.5
    return total / len(shape.points)


def propagate_shape(shape: VectorShape, motion_field: MotionField) -> VectorShape:
    """Shift a shape by its local flow, preserving point topology.

    Each anchor and both of its handles are translated by the bilinearly
    sampled ``(dx, dy)`` at the anchor position, so curve segments keep
    their form while riding the motion. Point count, order, fill, and
    closed-ness are unchanged; coordinates are clamped to field bounds.
    """
    flow = _validate_field(
        motion_field,
        width=motion_field.flow_vectors.shape[1],
        height=motion_field.flow_vectors.shape[0],
    )
    h, w = flow.shape[:2]
    moved: list[PathPoint] = []
    for p in shape.points:
        dx, dy = _sample_flow(flow, p.x, p.y)
        moved.append(
            PathPoint(
                x=min(max(p.x + dx, 0.0), float(w)),
                y=min(max(p.y + dy, 0.0), float(h)),
                handle_in=(
                    min(max(p.handle_in[0] + dx, 0.0), float(w)),
                    min(max(p.handle_in[1] + dy, 0.0), float(h)),
                ),
                handle_out=(
                    min(max(p.handle_out[0] + dx, 0.0), float(w)),
                    min(max(p.handle_out[1] + dy, 0.0), float(h)),
                ),
            )
        )
    return VectorShape(
        fill_color=shape.fill_color, points=moved, is_closed=shape.is_closed
    )


class PathStabilizer:
    """Propagate a keyframe topology across a clip with locked topology."""

    def __init__(self, config: VectorizeConfig) -> None:
        if config.keyframe_interval < 1:
            raise StabilizationError(
                f"keyframe_interval must be >= 1: {config.keyframe_interval}"
            )
        if config.max_tracking_error < 0:
            raise StabilizationError(
                f"max_tracking_error must be >= 0: {config.max_tracking_error}"
            )
        self.config = config
        self.keyframe_indices: list[int] = []

    def _is_keyframe(self, frame_idx: int, error: float) -> bool:
        if frame_idx > 0 and frame_idx % self.config.keyframe_interval == 0:
            return True
        return error > self.config.max_tracking_error

    def stabilize_sequence(
        self,
        initial_vector_frames: list[VectorFrame],
        motion_fields: list[MotionField],
        retrace_fn: Callable[[int], VectorFrame] | None = None,
    ) -> list[VectorFrame]:
        """Stabilize a frame sequence under one locked topology.

        Args:
            initial_vector_frames: traced keyframes; element 0 supplies
                the base topology (frame 0 output).
            motion_fields: dense fields where ``motion_fields[i]`` bridges
                output frame ``i`` to ``i + 1`` (``MotionTracker`` order).
            retrace_fn: optional ``frame_index -> VectorFrame`` used to
                adopt a fresh topology at keyframes. Without it, the base
                topology is propagated through the whole sequence.

        Returns:
            ``len(motion_fields) + 1`` frames; non-keyframe outputs share
            the exact shape/point counts of their keyframe.
        """
        if not initial_vector_frames:
            raise StabilizationError("need at least one initial VectorFrame")
        base = initial_vector_frames[0]
        if base.width < 1 or base.height < 1:
            raise StabilizationError(
                f"base frame has invalid dims: {(base.width, base.height)}"
            )
        width, height = base.width, base.height
        for f in motion_fields:
            _validate_field(f, width, height)

        self.keyframe_indices = [0]
        current = [
            VectorShape(s.fill_color, list(s.points), s.is_closed)
            for s in base.shapes
        ]
        out = [
            VectorFrame(
                frame_index=0, shapes=current, width=width, height=height
            )
        ]
        for i, field in enumerate(motion_fields, start=1):
            error = (
                max(mean_tracking_error(s, field) for s in current)
                if current
                else 0.0
            )
            if self._is_keyframe(i, error) and retrace_fn is not None:
                fresh = retrace_fn(i)
                if fresh.width >= 1 and fresh.height >= 1:
                    width, height = fresh.width, fresh.height
                current = [
                    VectorShape(s.fill_color, list(s.points), s.is_closed)
                    for s in fresh.shapes
                ]
                self.keyframe_indices.append(i)
            else:
                current = [propagate_shape(s, field) for s in current]
            out.append(
                VectorFrame(
                    frame_index=i, shapes=current, width=width, height=height
                )
            )
        return out


__all__ = [
    "StabilizationError",
    "PathStabilizer",
    "mean_tracking_error",
    "propagate_shape",
]
