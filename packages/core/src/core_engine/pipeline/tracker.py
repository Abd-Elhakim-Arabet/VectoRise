"""Optical flow & motion tracking.

Consumes RGB frames produced by the preprocessor (``VideoPreprocessor``)
and computes per-pixel motion vectors. Downstream stages use this data to
morph Bezier paths between keyframes instead of redrawing paths per frame.

Typical usage::

    from core_engine.pipeline.tracker import MotionTracker

    tracker = MotionTracker()
    fields = tracker.analyze_sequence(frames)  # list[MotionField]
    flow = fields[0].flow_vectors  # [H, W, 2] float32 (dx, dy)

Conventions:
    * Frames are ``uint8`` RGB arrays of shape ``[H, W, 3]`` (preprocessor
      output). Grayscale conversion uses ``COLOR_RGB2GRAY``.
    * Dense flow is ``float32`` ``[H, W, 2]`` with ``[..., 0] = dx`` and
      ``[..., 1] = dy`` in pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:  # Optional at import time; functions raise a clear error if missing.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


class MotionTrackingError(ValueError):
    """Raised when frames/points are invalid or OpenCV is unavailable."""


@dataclass(frozen=True)
class MotionField:
    """Dense motion vectors between frame ``frame_index - 1`` and ``frame_index``."""

    flow_vectors: np.ndarray  # [H, W, 2] float32 (dx, dy)
    frame_index: int  # 1-based index into the analyzed sequence


def _require_cv2() -> None:
    if cv2 is None:
        raise MotionTrackingError(
            "OpenCV (cv2) is required for motion tracking: "
            "install opencv-python-headless"
        )


def _validate_frame(frame: np.ndarray, name: str = "frame") -> tuple[int, int]:
    if not isinstance(frame, np.ndarray):
        raise MotionTrackingError(f"{name} must be a numpy ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise MotionTrackingError(
            f"{name} must have shape (H, W, 3), got {frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise MotionTrackingError(
            f"{name} must be uint8 RGB, got {frame.dtype}"
        )
    h, w, _ = frame.shape
    if h < 1 or w < 1:
        raise MotionTrackingError(f"{name} has invalid dims: {frame.shape}")
    return h, w


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB ``uint8`` frame to single-channel grayscale."""
    _require_cv2()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def compute_dense_flow(prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
    """Compute dense optical flow between two consecutive RGB frames.

    Uses OpenCV's Farneback algorithm on grayscale conversions.

    Returns:
        ``float32`` array of shape ``[H, W, 2]`` with per-pixel
        displacements ``(dx, dy)`` in pixels.
    """
    _require_cv2()
    h0, w0 = _validate_frame(prev_frame, "prev_frame")
    h1, w1 = _validate_frame(curr_frame, "curr_frame")
    if (h0, w0) != (h1, w1):
        raise MotionTrackingError(
            f"frame size mismatch: {prev_frame.shape} vs {curr_frame.shape}"
        )
    prev_gray = _to_grayscale(prev_frame)
    curr_gray = _to_grayscale(curr_frame)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return np.ascontiguousarray(flow, dtype=np.float32)


def track_sparse_points(
    prev_frame: np.ndarray, curr_frame: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Track 2D points across consecutive frames with Lucas-Kanade.

    Args:
        prev_frame: ``uint8`` RGB ``[H, W, 3]`` reference frame.
        curr_frame: ``uint8`` RGB ``[H, W, 3]`` next frame (same size).
        points: ``[N, 2]`` array of ``(x, y)`` pixel coordinates.

    Returns:
        ``(next_points, status)`` where ``next_points`` is ``float32``
        ``[N, 2]`` and ``status`` is ``uint8`` ``[N]`` (1 = tracked,
        0 = dropped).
    """
    _require_cv2()
    h0, w0 = _validate_frame(prev_frame, "prev_frame")
    h1, w1 = _validate_frame(curr_frame, "curr_frame")
    if (h0, w0) != (h1, w1):
        raise MotionTrackingError(
            f"frame size mismatch: {prev_frame.shape} vs {curr_frame.shape}"
        )
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        raise MotionTrackingError(
            f"points must have shape (N, 2), got {getattr(points, 'shape', None)}"
        )
    if points.shape[0] == 0:
        raise MotionTrackingError("points must contain at least one point")

    prev_gray = _to_grayscale(prev_frame)
    curr_gray = _to_grayscale(curr_frame)
    pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 1, 2)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=criteria,
    )
    if next_pts is None or status is None:
        raise MotionTrackingError("Lucas-Kanade tracking failed")
    next_pts = np.ascontiguousarray(next_pts.reshape(-1, 2), dtype=np.float32)
    status = np.ascontiguousarray(status.reshape(-1), dtype=np.uint8)
    return next_pts, status


class MotionTracker:
    """Compute sequential dense motion fields for a clip."""

    def analyze_sequence(
        self, frames: Iterable[np.ndarray]
    ) -> list[MotionField]:
        """Process frames and return motion fields between consecutive pairs.

        Args:
            frames: iterable of ``uint8`` RGB ``[H, W, 3]`` frames (list
                from ``extract_all_frames`` or any generator of processed
                frames). All frames must share shape/dtype.

        Returns:
            ``len(frames) - 1`` fields; field ``i`` holds flow from frame
            ``i`` to ``i + 1`` with ``frame_index = i + 1``. Empty input or
            a single frame yields ``[]``.
        """
        _require_cv2()
        seq = list(frames)
        if len(seq) < 2:
            return []
        # Validate eagerly so shape errors surface before expensive flow.
        for i, f in enumerate(seq):
            _validate_frame(f, f"frames[{i}]")
        h, w = seq[0].shape[:2]
        for i, f in enumerate(seq[1:], start=1):
            if f.shape[:2] != (h, w):
                raise MotionTrackingError(
                    f"frame size mismatch: frames[0] is {(h, w)} "
                    f"but frames[{i}] is {tuple(f.shape[:2])}"
                )
        fields: list[MotionField] = []
        for i in range(1, len(seq)):
            flow = compute_dense_flow(seq[i - 1], seq[i])
            fields.append(MotionField(flow_vectors=flow, frame_index=i))
        return fields


__all__ = [
    "MotionTrackingError",
    "MotionField",
    "MotionTracker",
    "compute_dense_flow",
    "track_sparse_points",
]
