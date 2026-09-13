"""Frame-to-layers decomposition: the basis for Lottie generation.

A layer is one **connected same-colored pixel region** of a quantized
frame: the frame is reduced to ``num_colors`` (2-24) flat colors, then each
8-connected (or 4-connected) monochrome region becomes an independent
:class:`Layer` with its own color, mask, bbox, and centroid.

Why connected color regions -- and not something fancier?
---------------------------------------------------------
* A Lottie ``ShapeLayer`` is fundamentally a fill + path. A fill *is* a
  connected same-color region, so these layers map 1:1 onto the vector
  shapes the tracer already produces (vtracer emits one path per connected
  region). They are the atomic unit -- anything coarser must be built from
  them, not instead of them.
* It is deterministic, millisecond-fast (``cv2.connectedComponentsWithStats``,
  no model weights), and works at full resolution.
* The research alternatives solve a different (harder, heavier) problem:
  *semantic* layers (SAM/SAM2 object masks, MG-Gen's OCR + YOLO + SAM +
  LaMa-inpainting pipeline, LayerDiffuse/Qwen-Image-Layered RGBA diffusion).
  Those group pixels into designer-meaningful objects (text, character,
  background) but need GB-sized models, occlusion inpainting, and still
  bottom out at per-color fills for vector output. The LayerAnimate work
  even notes conventional segmentation of animation yields "over-segmented
  color patches" -- i.e. roughly what we produce on purpose.
* The natural upgrade path stays in this codebase: the motion tracker
  already computes optical flow, so a later step can *merge* these atomic
  layers by shared motion (LayerAnimate-style motion-based merging) to
  recover object-level layers without any new dependency.

Known limitation: one semantic object with internal color variation splits
into several layers, and identical colors in disconnected places become
separate layers (arguably a feature -- they can animate independently).

Typical usage::

    from core_engine.pipeline.layers import extract_layers, composite_layers

    layers = extract_layers(frame, num_colors=16)  # largest area first
    for layer in layers:
        print(layer.id, layer.color, layer.area, layer.bbox)
    rebuilt = composite_layers(layers, frame.shape[:2])  # == quantized frame
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from core_engine.config import VectorizeConfig
from core_engine.pipeline.preprocessor import (
    VideoValidationError,
    validate_color_count,
)
from core_engine.pipeline.vectorizer import PathPoint, VectorShape

try:  # Optional at import time; functions raise a clear error if missing.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Layer:
    """One connected same-colored region of a quantized frame."""

    id: int  # 0-based, sorted by descending area (background first)
    color: tuple[int, int, int]  # RGB fill color
    area: int  # pixel count
    bbox: tuple[int, int, int, int]  # (x, y, w, h) tight bounding box
    centroid: tuple[float, float]  # (x, y) center of mass
    mask: np.ndarray  # bool [H, W], True exactly on this layer's pixels


def _require_deps() -> None:
    if cv2 is None:
        raise VideoValidationError(
            "OpenCV (cv2) is required for layer extraction: "
            "install opencv-python-headless"
        )
    if Image is None:
        raise VideoValidationError(
            "Pillow is required for layer extraction: install pillow"
        )


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise VideoValidationError("frame must be a numpy ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise VideoValidationError(
            f"frame must have shape (H, W, 3), got {frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise VideoValidationError(f"frame must be uint8 RGB, got {frame.dtype}")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise VideoValidationError(f"frame has invalid dims: {frame.shape}")


def _quantize(frame: np.ndarray, num_colors: int) -> np.ndarray:
    """Median-cut quantization (flat regions, no dithering)."""
    quantized = (
        Image.fromarray(frame)
        .quantize(
            colors=num_colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        .convert("RGB")
    )
    return np.ascontiguousarray(quantized, dtype=np.uint8)


def extract_layers(
    frame: np.ndarray,
    num_colors: int = 16,
    min_layer_area: int = 10,
    connectivity: int = 8,
) -> list[Layer]:
    """Chop a frame into patches: connected pixels of the same color.

    Three steps, nothing else:

    1. Quantize the frame to ``num_colors`` flat colors.
    2. For each color, take the mask of same-colored pixels and split it
       into connected blobs (8-connectivity joins diagonals, 4 splits).
    3. Each blob bigger than ``min_layer_area`` is one patch (``Layer``).

    Args:
        frame: ``uint8`` RGB ``[H, W, 3]`` frame (raw or pre-quantized;
            it is always (re-)quantized to ``num_colors`` first so layer
            colors are exact palette entries).
        num_colors: palette size, must be in [2, 24] (same rule as the
            preprocessing quantization step).
        min_layer_area: blobs smaller than this (pixels) are dropped as
            speckle/noise.
        connectivity: 8 (diagonals join) or 4 (diagonals split).

    Returns:
        Patches sorted by descending area (backdrop tends to come first).
    """
    _require_deps()
    _validate_frame(frame)
    validate_color_count(num_colors)
    if min_layer_area < 1:
        raise VideoValidationError(
            f"min_layer_area must be >= 1, got {min_layer_area}"
        )
    if connectivity not in (4, 8):
        raise VideoValidationError(
            f"connectivity must be 4 or 8, got {connectivity}"
        )

    quantized = _quantize(frame, num_colors)

    patches: list[Layer] = []
    for color in np.unique(quantized.reshape(-1, 3), axis=0):
        same_color = np.all(quantized == color, axis=2).astype(np.uint8)
        n_blobs, labels, stats, centroids = cv2.connectedComponentsWithStats(
            same_color, connectivity=connectivity
        )
        for blob in range(1, n_blobs):  # 0 is the background of the mask
            area = int(stats[blob, cv2.CC_STAT_AREA])
            if area < min_layer_area:
                continue
            patches.append(
                Layer(
                    id=-1,  # assigned after area sorting below
                    color=(int(color[0]), int(color[1]), int(color[2])),
                    area=area,
                    bbox=(
                        int(stats[blob, cv2.CC_STAT_LEFT]),
                        int(stats[blob, cv2.CC_STAT_TOP]),
                        int(stats[blob, cv2.CC_STAT_WIDTH]),
                        int(stats[blob, cv2.CC_STAT_HEIGHT]),
                    ),
                    centroid=(
                        float(centroids[blob][0]),
                        float(centroids[blob][1]),
                    ),
                    mask=(labels == blob),
                )
            )

    patches.sort(key=lambda patch: patch.area, reverse=True)
    return [dataclasses.replace(patch, id=i) for i, patch in enumerate(patches)]


def composite_layers(
    layers: list[Layer], shape: tuple[int, int]
) -> np.ndarray:
    """Repaint layers (ascending id order) onto a blank canvas.

    Inverse of :func:`extract_layers`: painting layers back in id order
    reproduces the quantized frame exactly, since layer masks partition
    the frame (disjoint, fully covering). Useful to verify an extraction.
    """
    h, w = shape
    if h < 1 or w < 1:
        raise VideoValidationError(f"invalid canvas shape: {shape}")
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for layer in sorted(layers, key=lambda lyr: lyr.id):
        if layer.mask.shape != (h, w):
            raise VideoValidationError(
                f"layer {layer.id} mask shape {layer.mask.shape} "
                f"does not match canvas {(h, w)}"
            )
        canvas[layer.mask] = layer.color
    return canvas


def merge_small_layers(
    layers: list[Layer], min_area: int = 10
) -> list[Layer]:
    """Absorb tiny patches into neighboring patches.

    Every layer with ``area < min_area`` is dissolved: its pixels are
    recolored to a neighboring layer and the patch disappears (its PNG
    goes away with it). The absorb-target metric is the neighbor sharing
    the longest border (8-neighborhood ring pixel count); ties break
    toward the larger neighbor, then the smaller id, so the result is
    deterministic. Small layers are processed smallest-first, and a
    small layer may absorb into another small layer -- areas accumulate,
    so chains collapse toward the surrounding big patch.

    The surviving masks still partition the frame (disjoint, fully
    covering); only colors change, on exactly the absorbed pixels.

    Args:
        layers: layers from :func:`extract_layers` (ids need not be
            ordered; masks must all share one ``(H, W)`` shape).
        min_area: patches smaller than this (pixels) are absorbed.
            ``1`` keeps everything.

    Returns:
        Surviving layers sorted by descending area with fresh ``0..N-1``
        ids (backdrop first, like :func:`extract_layers`).
    """
    if not layers:
        return []
    if min_area < 1:
        raise VideoValidationError(
            f"min_area must be >= 1, got {min_area}"
        )
    h, w = layers[0].mask.shape
    for lyr in layers:
        if lyr.mask.shape != (h, w):
            raise VideoValidationError(
                f"layer {lyr.id} mask shape {lyr.mask.shape} "
                f"does not match {(h, w)}"
            )
    by_id = {lyr.id: lyr for lyr in layers}
    if len(by_id) != len(layers):
        raise VideoValidationError("layer ids must be unique")
    if min_area <= 1:
        return list(layers)
    if all(lyr.area >= min_area for lyr in layers):
        return list(layers)

    label = np.full((h, w), -1, dtype=np.int32)
    for lyr in layers:
        label[lyr.mask] = lyr.id
    areas = {lyr.id: lyr.area for lyr in layers}
    alive = set(by_id)
    kernel = np.ones((3, 3), dtype=np.uint8)

    for lyr in sorted(layers, key=lambda l: (l.area, l.id)):
        if lyr.area >= min_area or lyr.id not in alive:
            continue
        x, y, bw, bh = lyr.bbox
        x0, x1 = max(x - 1, 0), min(x + bw + 1, w)
        y0, y1 = max(y - 1, 0), min(y + bh + 1, h)
        sub = label[y0:y1, x0:x1]
        m = sub == lyr.id
        if not m.any():
            alive.discard(lyr.id)
            continue
        ring = (cv2.dilate(m.astype(np.uint8), kernel).astype(bool)) & ~m
        if not ring.any():
            continue  # lone layer covering the whole frame: keep it
        uniq, counts = np.unique(sub[ring], return_counts=True)
        cands = [
            (lid, c) for lid, c in zip(uniq.tolist(), counts.tolist())
            if lid != lyr.id and lid in alive
        ]
        if not cands:
            continue
        # Longest shared border wins; ties -> larger area, then smaller id.
        target = min(cands, key=lambda t: (-t[1], -areas[t[0]], t[0]))[0]
        label[label == lyr.id] = target
        areas[target] += areas[lyr.id]
        alive.discard(lyr.id)

    rebuilt: list[Layer] = []
    for lid in alive:
        m = label == lid
        area = int(m.sum())
        if area == 0:
            continue
        ys, xs = np.nonzero(m)
        rebuilt.append(
            Layer(
                id=-1,
                color=by_id[lid].color,
                area=area,
                bbox=(int(xs.min()), int(ys.min()),
                      int(xs.max() - xs.min() + 1),
                      int(ys.max() - ys.min() + 1)),
                centroid=(float(xs.mean()), float(ys.mean())),
                mask=m,
            )
        )
    rebuilt.sort(key=lambda lyr: lyr.area, reverse=True)
    return [dataclasses.replace(lyr, id=i) for i, lyr in enumerate(rebuilt)]


def trace_layer(layer: Layer, config: VectorizeConfig) -> list[VectorShape]:
    """Trace one layer into Bezier shapes in full-frame coordinates.

    The trace is an **exact replica** of the quantized patch -- not an
    approximation. Each connected region's pixel contour is extracted
    with ``cv2.findContours`` (collinear runs compressed losslessly) and
    emitted as straight anchors (zero-length handles): no spline
    smoothing, no corner rounding, no color merging. The fill is the
    layer's exact palette color.

    Holes need no special handling: layers are painted back-to-front and
    their masks partition the frame, so a layer's outer contour -- even
    where it spans pixels owned by layers in front of it -- is invisible
    wherever another layer covers it. What you see is exactly the
    quantized frame.

    Thin regions need one extra step: a single pixel row/column has no 2D
    contour (``findContours`` follows its centerline and returns < 3
    points), so when no contour survives, the patch is emitted as one
    axis-aligned rect per horizontal pixel run instead -- still an exact
    replica when rasterized, so no patch is ever silently lost.

    Args:
        layer: the :class:`Layer` to trace.
        config: ``VectorizeConfig`` (only ``path_precision`` is read, to
            round anchor coordinates).

    Returns:
        Traced shapes (one per contour; a single connected layer almost
        always yields exactly one).
    """
    _require_deps()
    x, y, bw, bh = layer.bbox
    if bw < 1 or bh < 1:
        raise VideoValidationError(f"layer {layer.id} has empty bbox")
    precision = int(config.path_precision if config.path_precision else 0)
    if precision < 0:
        raise VideoValidationError(
            f"path_precision must be >= 0: {config.path_precision}"
        )

    crop = layer.mask[y:y + bh, x:x + bw].astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    shapes = []
    for contour in contours:
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            continue
        anchors = [
            PathPoint(
                x=round(float(cx) + x, precision),
                y=round(float(cy) + y, precision),
                handle_in=(round(float(cx) + x, precision),
                           round(float(cy) + y, precision)),
                handle_out=(round(float(cx) + x, precision),
                            round(float(cy) + y, precision)),
            )
            for cx, cy in pts
        ]
        shapes.append(
            VectorShape(
                fill_color=layer.color, points=anchors, is_closed=True
            )
        )
    if not shapes:
        # Degenerate thin region: emit one rect per horizontal pixel run.
        m = layer.mask[y:y + bh, x:x + bw]
        for yy in range(m.shape[0]):
            row = np.flatnonzero(m[yy])
            if len(row) == 0:
                continue
            cuts = np.flatnonzero(np.diff(row) > 1)
            starts = np.concatenate(([row[0]], row[cuts + 1]))
            ends = np.concatenate((row[cuts], [row[-1]]))
            for x0, x1 in zip(starts.tolist(), ends.tolist()):
                x0f, y0f = round(float(x0) + x, precision), round(float(yy) + y, precision)
                x1f, y1f = round(float(x1) + x + 1, precision), round(float(yy) + y + 1, precision)
                shapes.append(
                    VectorShape(
                        fill_color=layer.color,
                        points=[
                            PathPoint(x0f, y0f, (x0f, y0f), (x0f, y0f)),
                            PathPoint(x1f, y0f, (x1f, y0f), (x1f, y0f)),
                            PathPoint(x1f, y1f, (x1f, y1f), (x1f, y1f)),
                            PathPoint(x0f, y1f, (x0f, y1f), (x0f, y1f)),
                        ],
                        is_closed=True,
                    )
                )
    return shapes


def rasterize_shapes(
    shapes: list[VectorShape], shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize shapes onto a blank canvas (for replica verification).

    Fills each polygon with its fill color in order; later shapes cover
    earlier ones -- the same back-to-front compositing Lottie players do.
    """
    _require_deps()
    h, w = shape
    if h < 1 or w < 1:
        raise VideoValidationError(f"invalid canvas shape: {shape}")
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for s in shapes:
        if len(s.points) < 3:
            continue
        poly = np.array(
            [[p.x, p.y] for p in s.points], dtype=np.float32
        ).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [poly.astype(np.int32)], color=tuple(s.fill_color))
    return canvas


def layer_to_rgba(layer: Layer) -> np.ndarray:
    """Render one layer as an RGBA image (transparent outside the patch).

    Returns:
        ``uint8`` ``[H, W, 4]`` with the layer's RGB color where its mask
        is set and alpha 0 elsewhere -- i.e. the isolated patch,
        ready to save as PNG.
    """
    h, w = layer.mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[layer.mask] = (*layer.color, 255)
    return rgba


__all__ = ["Layer", "extract_layers", "composite_layers", "merge_small_layers", "trace_layer", "rasterize_shapes", "layer_to_rgba"]
