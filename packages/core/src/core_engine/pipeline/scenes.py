"""Scene splitting via PySceneDetect on the original video file.

Given a video that contains different scenes (different shots/components),
this module finds where to cut it so continuous similar frames stay grouped
as one scene. Detection runs on the **original file at full resolution** --
no downsampling, no proxy decodes.

Two stages:

1. **Global cuts** -- delegated to the external ``PySceneDetect`` library
   (``ContentDetector``). Catches any large frame-to-frame change.
2. **Background-aware refinement** (on by default) -- decomposes every
   boundary into *background change* vs *foreground change*:
   each segment's background is its temporal median frame; a pixel is
   foreground when it differs from that background. This fixes the two
   ways a static background distorts slicing:

   * *missed cuts*: only the subject changes while the backdrop stays
     identical (same room, new arrangement) -- the global score barely
     moves, so the cut is recovered from foreground-content change that
     persists across frames (transient single-frame flashes are ignored);
   * *false cuts*: a global flicker with (almost) identical background
     *and* foreground on both sides is suppressed.

Typical usage::

    from core_engine.pipeline.scenes import split_video_into_scenes, explain_splits

    scenes = split_video_into_scenes("clip.mp4")
    scenes, reports = explain_splits("clip.mp4")  # + per-cut diagnostics
    for r in reports:
        print(r.boundary_frame, r.bg_change, r.fg_change, r.source, r.verdict)

Conventions:
    * ``Scene.start_frame`` is inclusive, ``Scene.end_frame`` exclusive
      (Python slice semantics over the source frame sequence).
    * Times are in seconds.
    * ``bg_change`` is mean absolute background difference (0..255);
      ``fg_change`` is foreground change across the cut (0..1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class SceneDetectionError(ValueError):
    """Raised when the video can't be opened or detection fails."""


@dataclass(frozen=True)
class Scene:
    """One continuous scene of the source video."""

    index: int  # 0-based scene number
    start_frame: int  # inclusive
    end_frame: int  # exclusive
    start_time: float  # seconds
    end_time: float  # seconds

    @property
    def frame_count(self) -> int:
        """Number of frames in this scene."""
        return self.end_frame - self.start_frame

    @property
    def duration(self) -> float:
        """Scene duration in seconds."""
        return self.end_time - self.start_time


@dataclass(frozen=True)
class SceneConfig:
    """Tuning knobs for cut detection."""

    # Cut sensitivity for PySceneDetect's ContentDetector (higher = fewer
    # cuts). Library default is 27.0.
    threshold: float = 27.0
    # Minimum scene length in frames; shorter segments merge into neighbours.
    min_scene_len: int = 15
    # Master switch for stage 2 (background-aware refinement).
    use_background: bool = True
    # Temporal-median background is fit from every Nth frame (sampling only
    # affects the background estimate, never the cut positions).
    bg_sample_step: int = 8
    # Max-channel color distance from the background above which a pixel
    # counts as foreground (0..255).
    fg_bin_thresh: float = 25.0
    # Foreground-change score (0..1) above which a persistent change opens a
    # new cut.
    fg_threshold: float = 0.25
    # A global cut is suppressed as background flicker only when BOTH its
    # background change is below this (0..255) AND its foreground change is
    # below ``suppress_fg`` (0..1). Conservative on purpose.
    suppress_bg: float = 5.0
    suppress_fg: float = 0.05


@dataclass(frozen=True)
class BoundaryReport:
    """Per-boundary decomposition of background vs foreground influence."""

    boundary_frame: int  # first frame of the new scene
    time: float  # seconds
    bg_change: float  # mean abs background difference (0..255)
    fg_change: float  # foreground change across the cut (0..1)
    source: str  # 'pyscenedetect' or 'foreground'
    verdict: str  # 'kept' or 'suppressed'


def _require_scenedetect():
    try:
        from scenedetect import ContentDetector, detect
    except ImportError as exc:
        raise SceneDetectionError(
            "PySceneDetect is required for scene detection: "
            "install it with `pip install scenedetect`"
        ) from exc
    return ContentDetector, detect


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SceneDetectionError(
            "OpenCV (cv2) is required for background-aware scene detection: "
            "install opencv-python-headless"
        ) from exc
    return cv2


def _base_bounds(
    path: Path, cfg: SceneConfig
) -> tuple[list[tuple[int, int]], float]:
    """Global cuts from PySceneDetect; returns ([(start, end)], fps)."""
    ContentDetector, detect = _require_scenedetect()
    try:
        raw = detect(
            str(path),
            ContentDetector(
                threshold=cfg.threshold, min_scene_len=cfg.min_scene_len
            ),
        )
    except Exception as exc:
        raise SceneDetectionError(f"Scene detection failed for {path}: {exc}") from exc

    if not raw:
        # No cuts found: the whole clip is one scene.
        from core_engine.pipeline.preprocessor import probe_video

        try:
            meta = probe_video(path)
        except Exception as exc:
            raise SceneDetectionError(
                f"Could not probe video for single-scene fallback: {exc}"
            ) from exc
        total = meta.frame_count
        if total is None and meta.duration is not None:
            total = int(round(meta.duration * meta.fps))
        if not total:
            return [], meta.fps if "meta" in dir() else 0.0
        return [(0, total)], meta.fps
    fps = raw[0][0].framerate
    return [(int(s.frame_num), int(e.frame_num)) for s, e in raw], float(fps)


def _read_color_at(cv2, cap, idx: int):
    """Seek-grab one full-resolution color (BGR) frame; None on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _fg_masks_and_score(frame_prev, frame_curr, bg_prev, bg_curr, bin_thresh):
    """Foreground masks vs each frame's own segment background + change score.

    All inputs are full-resolution color (BGR) frames; a pixel is foreground
    when its max-channel distance from the background exceeds ``bin_thresh``.
    Color (not grayscale) is used throughout so isoluminant changes
    (e.g. red -> green, nearly identical in gray) still register.

    Returns ``(mask_prev, mask_curr, score)`` where score is
    ``max(shape_change, content_change)`` in 0..1: shape = fraction of
    pixels whose foreground label flipped; content = mean frame difference
    restricted to the foreground union (catches same-silhouette recoloring
    that shape alone misses).
    """
    import numpy as np

    m_prev = (
        np.abs(frame_prev.astype(np.float32) - bg_prev).max(axis=2).reshape(-1)
        > bin_thresh
    )
    m_curr = (
        np.abs(frame_curr.astype(np.float32) - bg_curr).max(axis=2).reshape(-1)
        > bin_thresh
    )
    shape = float((m_prev ^ m_curr).mean())
    union = m_prev | m_curr
    if union.any():
        d = np.abs(
            frame_curr.astype(np.float32) - frame_prev.astype(np.float32)
        ).mean(axis=2).reshape(-1)
        content = float(d[union].mean()) / 255.0
    else:
        content = 0.0
    return m_prev, m_curr, max(shape, content)


def _refine(
    path: Path, cfg: SceneConfig
) -> tuple[list[Scene], list[BoundaryReport], float]:
    """Background-aware refinement; returns (scenes, reports, fps)."""
    import numpy as np

    cv2 = _require_cv2()
    if cfg.min_scene_len < 1:
        raise SceneDetectionError(
            f"min_scene_len must be >= 1, got {cfg.min_scene_len}"
        )
    base, fps = _base_bounds(path, cfg)
    if not base:
        return [], [], fps or 0.0

    cap = cv2.VideoCapture(str(path))
    try:
        # Pass A: sample every Nth full-resolution color frame for
        # background medians.
        samples: dict[int, np.ndarray] = {}
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % max(1, cfg.bg_sample_step) == 0:
                samples[idx] = frame
            idx += 1
        total = idx
        if total == 0:
            return [], [], fps or 0.0
        base = [(s, min(e, total)) for s, e in base if s < total]

        def seg_of(frame_idx: int, bounds: list[tuple[int, int]]) -> int:
            for k, (s, e) in enumerate(bounds):
                if s <= frame_idx < e:
                    return k
            return len(bounds) - 1

        def bg_median(s: int, e: int) -> np.ndarray:
            picked = [f for i, f in samples.items() if s <= i < e]
            if not picked:  # segment shorter than the sampling step
                f = _read_color_at(cv2, cap, min(s, total - 1))
                if f is None:
                    raise SceneDetectionError(f"Could not read frame {s}")
                return f.astype(np.float32)
            return np.median(np.stack(picked).astype(np.float32), axis=0)

        base_bgs = [bg_median(s, e) for s, e in base]
        bounds = [s for s, _ in base] + [base[-1][1]]

        # Pass B: full sequential pass, foreground-change scores within
        # each base segment (never across a known cut).
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_frame = None
        prev_seg = -1
        candidates: list[tuple[int, float]] = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            seg = seg_of(i, base)
            if prev_frame is not None and seg == prev_seg:
                _, _, score = _fg_masks_and_score(
                    prev_frame, frame,
                    base_bgs[prev_seg], base_bgs[seg],
                    cfg.fg_bin_thresh,
                )
                if score >= cfg.fg_threshold:
                    candidates.append((i, score))
            prev_frame, prev_seg = frame, seg
            i += 1

        # Keep a candidate only if the change persists (content before vs
        # after differs -> not a single-frame flash) and both resulting
        # scenes respect min_scene_len. Strongest spikes win: candidates are
        # judged in descending score order so a sharp cut suppresses weaker
        # motion-induced neighbours within min_scene_len.
        new_cuts: list[int] = []
        for c, _ in sorted(candidates, key=lambda t: -t[1]):
            if c - 1 < 0 or c + 1 >= total:
                continue
            f_before = _read_color_at(cv2, cap, c - 1)
            f_after = _read_color_at(cv2, cap, c + 1)
            if f_before is None or f_after is None:
                continue
            sb, sa = seg_of(c - 1, base), seg_of(c + 1, base)
            _, _, persist = _fg_masks_and_score(
                f_before, f_after,
                base_bgs[sb], base_bgs[sa],
                cfg.fg_bin_thresh,
            )
            if persist < cfg.fg_threshold:
                continue
            provisional = sorted(set(bounds + new_cuts + [c]))
            pos = provisional.index(c)
            if c - provisional[pos - 1] < cfg.min_scene_len:
                continue
            if provisional[pos + 1] - c < cfg.min_scene_len:
                continue
            new_cuts.append(c)

        provisional = sorted(set(bounds + new_cuts))
        final_segs = list(zip(provisional[:-1], provisional[1:]))
        final_bgs = [bg_median(s, e) for s, e in final_segs]

        # Reports (+ flicker suppression) for every provisional boundary.
        scenes: list[Scene] = []
        reports: list[BoundaryReport] = []
        kept_bounds = [provisional[0]]
        for k in range(1, len(provisional) - 1):  # skip clip start/end
            b = provisional[k]
            # NOTE: k indexes final_segs; boundary k separates segs k-1, k.
            bg_change = float(np.abs(final_bgs[k - 1] - final_bgs[k]).mean())
            # Foreground change is measured against ONE shared reference
            # (median over both sides): with per-side backgrounds the masks
            # right at the cut would be trivially empty on both sides.
            s_left, e_right = final_segs[k - 1][0], final_segs[k][1]
            ref_picked = [f for i, f in samples.items() if s_left <= i < e_right]
            ref_bg = (
                np.median(np.stack(ref_picked).astype(np.float32), axis=0)
                if ref_picked
                else final_bgs[k - 1]
            )
            f_before = _read_color_at(cv2, cap, max(0, b - 1))
            f_now = _read_color_at(cv2, cap, min(b, total - 1))
            if f_before is None or f_now is None:
                fg_change = 0.0
            else:
                _, _, fg_change = _fg_masks_and_score(
                    f_before, f_now, ref_bg, ref_bg, cfg.fg_bin_thresh
                )
            source = "foreground" if b in new_cuts else "pyscenedetect"
            flicker = (
                b not in new_cuts
                and bg_change < cfg.suppress_bg
                and fg_change < cfg.suppress_fg
            )
            verdict = "suppressed" if flicker else "kept"
            reports.append(
                BoundaryReport(
                    boundary_frame=b,
                    time=b / fps if fps else 0.0,
                    bg_change=bg_change,
                    fg_change=fg_change,
                    source=source,
                    verdict=verdict,
                )
            )
            if not flicker:
                kept_bounds.append(b)
        kept_bounds.append(provisional[-1])  # clip end is not a cut

        kept_bounds = sorted(set(kept_bounds))
        for n, (s, e) in enumerate(zip(kept_bounds[:-1], kept_bounds[1:])):
            scenes.append(
                Scene(n, s, e, s / fps if fps else 0.0, e / fps if fps else 0.0)
            )
        return scenes, reports, fps or 0.0
    finally:
        cap.release()


def split_video_into_scenes(
    input_path: str | Path, config: SceneConfig | None = None
) -> list[Scene]:
    """Split the original video file into scenes.

    Stage 1 finds global cuts with PySceneDetect; stage 2 (unless
    ``config.use_background`` is False) refines them with
    background/foreground analysis: same-background subject changes are
    recovered, background-flicker false cuts are suppressed.

    Args:
        input_path: video file (opened at full resolution; the file is
            never re-encoded or downscaled for detection).
        config: cut sensitivity / minimum scene length / refinement knobs.

    Returns:
        One :class:`Scene` per detected scene, in order. A video with no
        cuts yields a single scene spanning the whole clip.
    """
    cfg = config or SceneConfig()
    path = Path(input_path)
    if not path.is_file():
        raise SceneDetectionError(f"Video file not found: {path}")
    if not cfg.use_background:
        base, fps = _base_bounds(path, cfg)
        return [
            Scene(i, s, e, s / fps if fps else 0.0, e / fps if fps else 0.0)
            for i, (s, e) in enumerate(base)
        ]
    scenes, _, _ = _refine(path, cfg)
    return scenes


def explain_splits(
    input_path: str | Path, config: SceneConfig | None = None
) -> tuple[list[Scene], list[BoundaryReport]]:
    """Split a video and explain each boundary's background influence.

    Returns:
        ``(scenes, reports)`` where each report carries ``bg_change``,
        ``fg_change``, ``source`` (``'pyscenedetect'`` or ``'foreground'``)
        and ``verdict`` (``'kept'`` or ``'suppressed'``). With
        ``use_background=False`` reports still describe the global cuts
        (background fields set from whole-segment medians is skipped, so
        use the default config for meaningful diagnostics).
    """
    cfg = config or SceneConfig()
    path = Path(input_path)
    if not path.is_file():
        raise SceneDetectionError(f"Video file not found: {path}")
    if not cfg.use_background:
        scenes = split_video_into_scenes(path, cfg)
        reports = [
            BoundaryReport(
                boundary_frame=s.start_frame,
                time=s.start_time,
                bg_change=float("nan"),
                fg_change=float("nan"),
                source="pyscenedetect",
                verdict="kept",
            )
            for s in scenes[1:]
        ]
        return scenes, reports
    scenes, reports, _ = _refine(path, cfg)
    return scenes, reports


class SceneDetector:
    """Object-oriented wrapper around the scene-splitting functions."""

    def __init__(self, config: SceneConfig | None = None) -> None:
        self.config = config or SceneConfig()

    def detect(self, input_path: str | Path) -> list[Scene]:
        """Split a video file into scenes (full resolution, no downscale)."""
        return split_video_into_scenes(input_path, self.config)

    def explain(
        self, input_path: str | Path
    ) -> tuple[list[Scene], list[BoundaryReport]]:
        """Split a video and report background influence per boundary."""
        return explain_splits(input_path, self.config)


__all__ = [
    "SceneDetectionError",
    "Scene",
    "SceneConfig",
    "SceneDetector",
    "BoundaryReport",
    "split_video_into_scenes",
    "explain_splits",
]
