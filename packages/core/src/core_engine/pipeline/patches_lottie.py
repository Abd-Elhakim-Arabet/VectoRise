"""Frame patches -> Lottie with render-back verification.

Pipeline::

    frame -> extract_layers -> trace_layer -> build_layered_lottie_animation
          -> save JSON -> reload JSON -> render each Lottie layer back to PNG
          -> compare every rendered patch against its original patch PNG

The last step is the point of this module: the JSON on disk is parsed
(without touching the in-memory shapes) and each ``layer_<id>`` ShapeLayer
is rasterized with the same back-to-front compositing a player would do.
Every rendered patch must cover its original pixels exactly
(:func:`layer_to_rgba` mask, color and alpha). A traced contour may also
span pixels owned by layers in front of it (a holed backdrop's outer
contour covers its holes -- see :func:`trace_layer`); that overfill is
reported as ``extra_pixels`` and is only acceptable because the stacked
canvas check proves it is invisible. If any patch misses pixels -- or the
full stacked canvas differs -- the output is **refused**: the JSON file
is deleted and :class:`LottieVerifyError` is raised.

Typical usage::

    from core_engine.pipeline.patches_lottie import patches_to_lottie_verified

    report = patches_to_lottie_verified(frame, "still.json", num_colors=8)
    print(report["ok"], report["canvas_diff_pixels"])
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from core_engine.config import VectorizeConfig
from core_engine.pipeline.layers import (
    Layer,
    composite_layers,
    extract_layers,
    layer_to_rgba,
    trace_layer,
)
from core_engine.pipeline.lottie_builder import (
    LottieBuildError,
    build_layered_lottie_animation,
    save_lottie_json,
)

try:  # Optional at import time; functions raise a clear error if missing.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


class LottieVerifyError(ValueError):
    """Raised when a Lottie render-back does not match the original patches."""


def _require_cv2() -> None:
    if cv2 is None:
        raise LottieVerifyError(
            "OpenCV (cv2) is required to render Lottie layers: "
            "install opencv-python-headless"
        )


# ---------------------------------------------------------------------------
# Bezier flattening (Lottie stores cubic segments; rasterization needs polys)
# ---------------------------------------------------------------------------

def _flatten_cubic(
    p0: tuple[float, float],
    c1: tuple[float, float],
    c2: tuple[float, float],
    p1: tuple[float, float],
    tol: float,
    _depth: int = 0,
) -> list[tuple[float, float]]:
    """Flatten one cubic Bezier to a polyline (endpoints inclusive)."""
    if _depth > 12:
        return [p0, p1]
    # Straight segment (what trace_layer always emits): exact, no recursion.
    if (
        math.hypot(c1[0] - p0[0], c1[1] - p0[1]) < 1e-9
        and math.hypot(c2[0] - p1[0], c2[1] - p1[1]) < 1e-9
    ):
        return [p0, p1]
    # Flatness: control-point distance from the chord.
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    chord = math.hypot(dx, dy) or 1e-12
    dist = max(
        abs((c1[0] - p0[0]) * dy - (c1[1] - p0[1]) * dx) / chord,
        abs((c2[0] - p0[0]) * dy - (c2[1] - p0[1]) * dx) / chord,
    )
    if dist <= tol:
        return [p0, p1]
    # Subdivide at t=0.5 (de Casteljau).
    mx = lambda a, b: ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    p01, p12, p23 = mx(p0, c1), mx(c1, c2), mx(c2, p1)
    p012, p123 = mx(p01, p12), mx(p12, p23)
    mid = mx(p012, p123)
    left = _flatten_cubic(p0, p01, p012, mid, tol, _depth + 1)
    right = _flatten_cubic(mid, p123, p23, p1, tol, _depth + 1)
    return left[:-1] + right


def _flatten_loop(
    v: list, i: list, o: list, closed: bool, tol: float = 0.25
) -> np.ndarray:
    """Flatten one Lottie bezier loop to an ``(N, 2)`` float polygon."""
    n = len(v)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts: list[tuple[float, float]] = []
    segs = n if closed else max(n - 1, 0)
    for j in range(segs):
        p0 = (float(v[j][0]), float(v[j][1]))
        c1 = (p0[0] + float(o[j][0]), p0[1] + float(o[j][1]))
        nxt = (j + 1) % n
        p1 = (float(v[nxt][0]), float(v[nxt][1]))
        c2 = (p1[0] + float(i[nxt][0]), p1[1] + float(i[nxt][1]))
        seg = _flatten_cubic(p0, c1, c2, p1, tol)
        pts.extend(seg[:-1] if j < segs - 1 else seg)
    if not closed and n:
        pts.append((float(v[-1][0]), float(v[-1][1])))
    return np.asarray(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Lottie JSON -> per-layer PNG renders
# ---------------------------------------------------------------------------

def _parse_layer_id(name: object) -> int | None:
    # Accepts "layer_<id>" as well as prefixed forms ("s0_layer_<id>",
    # "f12_layer_<id>"): the id is whatever follows the last "layer_".
    if isinstance(name, str) and "layer_" in name:
        try:
            return int(name.rsplit("layer_", 1)[1])
        except ValueError:
            return None
    return None


def _static_fill_color(item: dict) -> tuple[int, int, int]:
    c = item.get("c", {})
    if not isinstance(c, dict) or c.get("a", 0) != 0:
        raise LottieVerifyError(f"animated fill not supported: {c!r}")
    k = c.get("k", [])
    if len(k) < 3:
        raise LottieVerifyError(f"fill has no RGB channels: {c!r}")
    rgb = []
    for ch in k[:3]:
        v = int(round(float(ch) * 255.0))
        rgb.append(max(0, min(255, v)))
    return (rgb[0], rgb[1], rgb[2])


def _static_bezier_loops(item: dict) -> list[dict]:
    ks = item.get("ks", {})
    if not isinstance(ks, dict) or ks.get("a", 0) != 0:
        raise LottieVerifyError("animated path not supported in still renders")
    keyframes = ks.get("k", [])
    if not keyframes:
        raise LottieVerifyError("path has no keyframes")
    loops = keyframes[0].get("s", [])
    if not loops:
        raise LottieVerifyError("path keyframe holds no beziers")
    return loops


def render_lottie_layers(
    animation: dict | str | Path, shape: tuple[int, int]
) -> dict[int, np.ndarray]:
    """Render every ``layer_<id>`` ShapeLayer of a still animation to RGBA.

    Args:
        animation: Lottie dict (as built by
            :func:`build_layered_lottie_animation`) or a path to its JSON
            file -- the file is what the verifier reads, so the check
            covers serialization too.
        shape: ``(h, w)`` canvas size.

    Returns:
        Mapping of layer id to ``uint8`` ``[H, W, 4]`` RGBA: the layer's
        fill color where its paths cover, transparent elsewhere.
    """
    _require_cv2()
    if isinstance(animation, (str, Path)):
        with open(str(animation), encoding="utf-8") as fh:
            animation = json.load(fh)
    if not isinstance(animation, dict):
        raise LottieVerifyError("animation must be a Lottie dict or JSON path")
    h, w = shape
    if h < 1 or w < 1:
        raise LottieVerifyError(f"invalid canvas shape: {shape}")

    renders: dict[int, np.ndarray] = {}
    for layer in animation.get("layers", []):
        if not isinstance(layer, dict):
            continue
        lid = _parse_layer_id(layer.get("nm"))
        if lid is None:
            continue
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        alpha = np.zeros((h, w), dtype=np.uint8)
        for group in layer.get("shapes", []):
            if not isinstance(group, dict) or group.get("ty") != "gr":
                continue  # e.g. the trailing TransformShape
            fill = None
            for item in group.get("it", []):
                if not isinstance(item, dict):
                    continue
                if item.get("ty") == "fl" and fill is None:
                    fill = _static_fill_color(item)
                elif item.get("ty") == "sh":
                    if fill is None:
                        raise LottieVerifyError(
                            f"layer_{lid}: path before fill"
                        )
                    for loop in _static_bezier_loops(item):
                        poly = _flatten_loop(
                            loop.get("v", []),
                            loop.get("i", []),
                            loop.get("o", []),
                            bool(loop.get("c", True)),
                        )
                        if len(poly) < 3:
                            continue
                        # Same int32 truncation as rasterize_shapes: what a
                        # player-aligned fill covers for these exact traces.
                        pts = poly.astype(np.int32).reshape(-1, 1, 2)
                        cv2.fillPoly(rgb, [pts], color=fill)
                        cv2.fillPoly(alpha, [pts], color=255)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = alpha
        renders[lid] = rgba
    if not renders:
        raise LottieVerifyError("no layer_<id> ShapeLayers found to render")
    return renders


# ---------------------------------------------------------------------------
# Verification (render-back vs original patches)
# ---------------------------------------------------------------------------

def check_renders_against_layers(
    layers: list[Layer],
    renders: dict[int, np.ndarray],
    shape: tuple[int, int],
) -> dict:
    """Compare renders to originals without raising (see :func:`verify_*`).

    Per patch, two counts are reported:

    * ``missing_pixels`` -- original mask pixels the render does not
      reproduce (wrong color or transparent). Must be 0.
    * ``extra_pixels`` -- rendered pixels outside the original mask. These
      come from traced contours spanning pixels owned by layers in front
      (e.g. a backdrop's outer contour covering its holes) and are
      allowed as long as the stacked canvas matches exactly.
    """
    h, w = shape
    per_layer = []
    for lyr in layers:
        ref = layer_to_rgba(lyr)
        got = renders.get(lyr.id)
        if got is None:
            per_layer.append(
                {"id": lyr.id, "color": lyr.color, "area": lyr.area,
                 "missing_pixels": h * w, "extra_pixels": 0,
                 "ok": False, "error": "missing"}
            )
            continue
        if got.shape != ref.shape:
            per_layer.append(
                {"id": lyr.id, "color": lyr.color, "area": lyr.area,
                 "missing_pixels": h * w, "extra_pixels": 0,
                 "ok": False, "error": "shape"}
            )
            continue
        mask = lyr.mask
        mask_ok = (
            (got[..., 3] == 255)
            & (got[..., 0] == lyr.color[0])
            & (got[..., 1] == lyr.color[1])
            & (got[..., 2] == lyr.color[2])
        )
        missing = int((mask & ~mask_ok).sum())
        extra = int(((~mask) & (got[..., 3] > 0)).sum())
        per_layer.append(
            {"id": lyr.id, "color": lyr.color, "area": lyr.area,
             "missing_pixels": missing, "extra_pixels": extra,
             "ok": missing == 0}
        )

    # Full canvas: paint renders back-to-front (ascending id, same order as
    # composite_layers) and compare against the stacked original patches.
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for lyr in sorted(layers, key=lambda l: l.id):
        got = renders.get(lyr.id)
        if got is None:
            continue
        mask = got[..., 3] > 0
        canvas[mask] = got[..., :3][mask]
    expected = composite_layers(layers, (h, w))
    canvas_diff = int((canvas != expected).any(axis=2).sum())

    ok = all(p["ok"] for p in per_layer) and canvas_diff == 0
    return {
        "ok": ok,
        "num_layers": len(layers),
        "layers": per_layer,
        "canvas_diff_pixels": canvas_diff,
        "canvas_pixels": h * w,
    }


def _mismatch_message(report: dict) -> str:
    bad = [p for p in report["layers"] if not p["ok"]][:10]
    if bad:
        detail = ", ".join(
            f"layer_{p['id']} (missing {p['missing_pixels']}px"
            + (f", {p['error']}" if p.get("error") else "")
            + ")"
            for p in bad
        )
        msg = f"render-back mismatch: {detail}"
    else:
        msg = "render-back mismatch: all patches cover their pixels"
    if report["canvas_diff_pixels"]:
        msg += (f"; canvas {report['canvas_diff_pixels']}/"
                f"{report['canvas_pixels']}px differ")
    return msg


def verify_lottie_patches(
    layers: list[Layer],
    animation: dict | str | Path,
    shape: tuple[int, int],
) -> dict:
    """Verify renders match originals; raise :class:`LottieVerifyError` if off.

    Returns the check report dict on success.
    """
    renders = render_lottie_layers(animation, shape)
    report = check_renders_against_layers(layers, renders, shape)
    if not report["ok"]:
        raise LottieVerifyError(_mismatch_message(report))
    return report


def verify_lottie_file(
    json_path: str | Path, layers: list[Layer], shape: tuple[int, int]
) -> dict:
    """Verify a Lottie JSON file on disk against its original layers."""
    return verify_lottie_patches(layers, Path(json_path), shape)


# ---------------------------------------------------------------------------
# Build (+ optional verified build that refuses bad output)
# ---------------------------------------------------------------------------

def layers_to_lottie(
    layers: list[Layer],
    shape: tuple[int, int],
    output_path: str,
    config: VectorizeConfig | None = None,
    fps: float = 30.0,
) -> dict:
    """Build a layered Lottie JSON from already-extracted layers.

    Same as :func:`patches_to_lottie` but takes the layers directly, so
    callers can pass post-processed layers (e.g. after
    :func:`merge_small_layers`). Returns the animation dict.
    """
    if not output_path:
        raise LottieBuildError("output_path must be set")
    if not layers:
        raise LottieBuildError("no layers to build from")
    h, w = shape
    cfg = config or VectorizeConfig(input_path="", output_path=output_path)
    traced = [trace_layer(lyr, cfg) for lyr in layers]
    animation = build_layered_lottie_animation(layers, traced, w, h, fps)
    save_lottie_json(animation, output_path)
    return animation


def patches_to_lottie(
    frame: np.ndarray,
    output_path: str,
    num_colors: int = 8,
    min_layer_area: int = 10,
    config: VectorizeConfig | None = None,
    fps: float = 30.0,
) -> list[Layer]:
    """Translate all patches of a frame into a layered Lottie JSON file.

    Returns the extracted layers (needed to verify the file afterwards).
    """
    layers = extract_layers(
        np.ascontiguousarray(frame),
        num_colors=num_colors,
        min_layer_area=min_layer_area,
    )
    if not layers:
        raise LottieBuildError("no layers extracted from frame")
    cfg = config or VectorizeConfig(input_path="", output_path=output_path)
    layers_to_lottie(layers, np.ascontiguousarray(frame).shape[:2],
                     output_path, config=cfg, fps=fps)
    return layers


def patches_to_lottie_verified(
    frame: np.ndarray,
    output_path: str,
    num_colors: int = 8,
    min_layer_area: int = 10,
    config: VectorizeConfig | None = None,
    fps: float = 30.0,
    save_renders_dir: str | Path | None = None,
) -> dict:
    """Build the Lottie, render it back to PNGs, and verify every patch.

    The JSON is reloaded from disk and each ``layer_<id>`` is rasterized
    and compared to its original patch PNG (:func:`layer_to_rgba`). If any
    patch -- or the stacked canvas -- differs, the output is refused: the
    JSON file is deleted and :class:`LottieVerifyError` is raised.

    Returns the verification report (``report["ok"]`` is True) on success.
    """
    frame = np.ascontiguousarray(frame)
    h, w, _ = frame.shape
    layers = patches_to_lottie(
        frame, output_path,
        num_colors=num_colors, min_layer_area=min_layer_area,
        config=config, fps=fps,
    )
    try:
        renders = render_lottie_layers(Path(output_path), (h, w))
        report = check_renders_against_layers(layers, renders, (h, w))
        if not report["ok"]:
            raise LottieVerifyError(_mismatch_message(report))
    except LottieVerifyError:
        try:
            os.unlink(output_path)
        except OSError:
            pass
        raise

    if save_renders_dir is not None:
        _require_cv2()
        out_dir = Path(save_renders_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for lid, rgba in sorted(renders.items()):
            cv2.imwrite(
                str(out_dir / f"render_{lid:04d}.png"),
                cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
            )
        stacked = np.zeros((h, w, 3), dtype=np.uint8)
        for lyr in sorted(layers, key=lambda l: l.id):
            m = renders[lyr.id][..., 3] > 0
            stacked[m] = renders[lyr.id][..., :3][m]
        cv2.imwrite(
            str(out_dir / "render_stacked.png"),
            cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR),
        )
    return report


__all__ = [
    "LottieVerifyError",
    "layers_to_lottie",
    "patches_to_lottie",
    "patches_to_lottie_verified",
    "render_lottie_layers",
    "check_renders_against_layers",
    "verify_lottie_patches",
    "verify_lottie_file",
]
