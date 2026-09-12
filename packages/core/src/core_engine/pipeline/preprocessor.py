"""FFmpeg & frame handling.

Streams raw RGB frames from arbitrary video containers (MP4, WebM, MOV,
AVI, MKV, ...) via an FFmpeg stdout pipe -- no temporary image files.

Typical usage::

    from core_engine.config import VectorizeConfig
    from core_engine.pipeline.preprocessor import VideoPreprocessor

    cfg = VectorizeConfig(
        input_path="clip.mp4", output_path="out.json",
        target_fps=12.0, max_dimension=720,
    )
    pre = VideoPreprocessor(cfg.input_path, cfg)
    print(pre.metadata)
    for frame in pre.extract_frames_generator():  # HxWx3 uint8 RGB
        ...
    frames = pre.extract_all_frames()  # batch version for short clips

Per-frame enhancement order (after FFmpeg spatial resizing):

    [Raw Streamed Frame] -> [Bilateral Edge Smoothing] -> [Color Quantization]

Both steps are bypassed when the corresponding ``VectorizeConfig``
options are disabled (``enable_smoothing=False`` / ``color_count=None``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from core_engine.config import VectorizeConfig

try:  # Optional at import time; methods raise a clear error if missing.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
 

class VideoValidationError(ValueError):
    """Raised when a video file is missing, invalid, or has no video stream."""


@dataclass(frozen=True)
class VideoMetadata:
    """Probed source-stream metadata (pre-scaling, pre-FPS-conversion)."""

    path: str
    width: int
    height: int
    fps: float
    duration: float | None
    frame_count: int | None
    pix_fmt: str | None
    codec: str | None


def _parse_frame_rate(rate: str | None) -> float | None:
    """Parse an ffprobe frame-rate string like ``"30000/1001"``."""
    if not rate or rate in ("0/0", "0"):
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            num_f, den_f = float(num), float(den)
            if den_f == 0 or num_f <= 0:
                return None
            return num_f / den_f
        value = float(rate)
        return value if value > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(input_path: str | Path) -> VideoMetadata:
    """Inspect a video file with ``ffprobe``.

    Raises:
        VideoValidationError: if the file is missing, unreadable, or has
            no decodable video stream.
    """
    path = Path(input_path)
    if not path.exists():
        raise VideoValidationError(f"Video file not found: {path}")
    if not path.is_file():
        raise VideoValidationError(f"Not a file: {path}")

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise VideoValidationError("ffprobe binary not found on PATH")

    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames,pix_fmt,codec_name",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise VideoValidationError(f"Failed to run ffprobe: {exc}") from exc

    if proc.returncode != 0:
        raise VideoValidationError(
            f"ffprobe failed for {path}: {(proc.stderr or '').strip()}"
        )
    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VideoValidationError(f"Could not parse ffprobe output: {exc}") from exc

    streams = info.get("streams") or []
    if not streams:
        raise VideoValidationError(f"No readable video stream in: {path}")
    stream = streams[0]

    try:
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except (TypeError, ValueError) as exc:
        raise VideoValidationError(f"Invalid dimensions in: {path}") from exc
    if width <= 0 or height <= 0:
        raise VideoValidationError(f"Invalid dimensions in: {path}")

    fps = _parse_frame_rate(stream.get("avg_frame_rate")) or _parse_frame_rate(
        stream.get("r_frame_rate")
    )

    duration: float | None = None
    for candidate in (stream.get("duration"), (info.get("format") or {}).get("duration")):
        try:
            if candidate is not None:
                duration = float(candidate)
                if duration < 0:
                    duration = None
                break
        except (TypeError, ValueError):
            continue

    frame_count: int | None = None
    nb_frames = stream.get("nb_frames")
    try:
        if nb_frames is not None and str(nb_frames).isdigit():
            frame_count = int(nb_frames)
    except (TypeError, ValueError):
        frame_count = None
    if frame_count is None and duration is not None and fps:
        frame_count = int(round(duration * fps))

    if fps is None:
        if frame_count and duration:
            fps = frame_count / duration
        else:
            raise VideoValidationError(
                f"Could not determine FPS for video stream in: {path}"
            )

    return VideoMetadata(
        path=str(path),
        width=width,
        height=height,
        fps=float(fps),
        duration=duration,
        frame_count=frame_count,
        pix_fmt=stream.get("pix_fmt"),
        codec=stream.get("codec_name"),
    )


def _even(value: int) -> int:
    """Round to nearest even int >= 2 (codec/filter friendly)."""
    return max(2, int(round(value / 2.0)) * 2)


def resolve_output_geometry(
    metadata: VideoMetadata, config: VectorizeConfig
) -> tuple[int, int, float]:
    """Compute (out_width, out_height, out_fps) after scaling + FPS filter."""
    out_w, out_h = metadata.width, metadata.height

    if config.max_dimension is not None:
        if config.max_dimension < 16:
            raise VideoValidationError(
                f"max_dimension too small: {config.max_dimension}"
            )
        longest = max(out_w, out_h)
        if longest > config.max_dimension:
            scale = config.max_dimension / longest
            if out_w >= out_h:
                out_w = _even(out_w * scale)
                out_h = _even(out_h * scale)
            else:
                out_h = _even(out_h * scale)
                out_w = _even(out_w * scale)

    out_fps = metadata.fps
    if config.target_fps is not None:
        if config.target_fps <= 0:
            raise VideoValidationError(f"target_fps must be > 0: {config.target_fps}")
        out_fps = float(config.target_fps)

    return out_w, out_h, out_fps


class VideoPreprocessor:
    """Validate, probe, and stream RGB frames from a video file."""

    def __init__(
        self,
        input_path: str | Path | None = None,
        config: VectorizeConfig | None = None,
    ) -> None:
        if config is None and input_path is None:
            raise VideoValidationError("Provide input_path and/or a VectorizeConfig")
        resolved = str(input_path) if input_path is not None else str(config.input_path)  # type: ignore[union-attr]
        if not resolved:
            raise VideoValidationError("Empty input video path")
        self.input_path = resolved
        self.config = config or VectorizeConfig(
            input_path=resolved, output_path=""
        )
        self._validate_enhancement_config()
        self._metadata = probe_video(self.input_path)
        self._out_w, self._out_h, self._out_fps = resolve_output_geometry(
            self._metadata, self.config
        )

    def _validate_enhancement_config(self) -> None:
        """Validate smoothing / quantization options early (fail fast)."""
        cfg = self.config
        if cfg.color_count is not None:
            if not isinstance(cfg.color_count, int) or cfg.color_count < 1:
                raise VideoValidationError(
                    f"color_count must be a positive int or None: {cfg.color_count}"
                )
        if cfg.bilateral_d < 1:
            raise VideoValidationError(
                f"bilateral_d must be >= 1: {cfg.bilateral_d}"
            )
        if cfg.bilateral_sigma <= 0:
            raise VideoValidationError(
                f"bilateral_sigma must be > 0: {cfg.bilateral_sigma}"
            )

    @property
    def metadata(self) -> VideoMetadata:
        """Source stream metadata (before scaling / FPS conversion)."""
        return self._metadata

    # Alias -- some callers expect `.info`.
    @property
    def info(self) -> VideoMetadata:
        return self._metadata

    @property
    def output_width(self) -> int:
        return self._out_w

    @property
    def output_height(self) -> int:
        return self._out_h

    @property
    def output_fps(self) -> float:
        return self._out_fps

    @property
    def estimated_frames(self) -> int | None:
        """Expected frame count after FPS conversion (None if unknown)."""
        if self._metadata.duration is None:
            return self._metadata.frame_count
        return int(round(self._metadata.duration * self._out_fps))

    @staticmethod
    def _require_cv2() -> None:
        if cv2 is None:
            raise VideoValidationError(
                "OpenCV (cv2) is required for frame enhancement: "
                "install opencv-python-headless"
            )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise VideoValidationError("frame must be a numpy ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise VideoValidationError(
                f"frame must have shape (H, W, 3), got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise VideoValidationError(
                f"frame must be uint8 RGB, got {frame.dtype}"
            )
        if frame.shape[0] < 1 or frame.shape[1] < 1:
            raise VideoValidationError(f"frame has invalid dims: {frame.shape}")

    def apply_edge_preserving_filter(self, frame: np.ndarray) -> np.ndarray:
        """Smooth noise/flat areas while keeping object boundaries sharp.

        Uses OpenCV's bilateral filter (permutationally symmetric, so RGB
        order needs no BGR conversion). Output shape/dtype match input.
        """
        self._validate_frame(frame)
        self._require_cv2()
        d = int(self.config.bilateral_d)
        sigma = float(self.config.bilateral_sigma)
        if d < 1:
            raise VideoValidationError(f"bilateral_d must be >= 1: {d}")
        if sigma <= 0:
            raise VideoValidationError(f"bilateral_sigma must be > 0: {sigma}")
        # sigmaColor == sigmaSpace == sigma: single strength knob.
        filtered = cv2.bilateralFilter(frame, d, sigma, sigma)
        return np.ascontiguousarray(filtered, dtype=np.uint8)

    def _kmeans_centers(self, frame: np.ndarray, num_colors: int) -> np.ndarray:
        """Return deterministic RGB K-Means centers for one frame.

        Runs ``cv2.kmeans`` (KMEANS_PP_CENTERS) on a capped pixel sample
        for speed.
        """
        self._validate_frame(frame)
        self._require_cv2()
        if not isinstance(num_colors, int) or num_colors < 1:
            raise VideoValidationError(
                f"num_colors must be a positive int, got {num_colors}"
            )
        h, w, _ = frame.shape
        n = h * w
        k = min(num_colors, n)
        if k == n:
            # Fewer pixels than requested colors: nothing to cluster.
            return np.ascontiguousarray(frame.reshape(-1, 3).copy(), dtype=np.uint8)

        pixels = frame.reshape(-1, 3)
        # Cap clustering sample for speed on large frames; stride sample
        # is deterministic (no RNG dependence across runs).
        max_sample = 50_000
        if n > max_sample:
            step = max(1, n // max_sample)
            sample = pixels[::step].astype(np.float32)
        else:
            sample = pixels.astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0,
        )
        try:
            cv2.setRNGSeed(0)
        except Exception:
            pass
        _compact, _labels, centers = cv2.kmeans(
            sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        return np.clip(np.rint(centers), 0, 255).astype(np.uint8)

    @staticmethod
    def _assign_palette(frame: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Map every RGB pixel to its nearest palette center."""
        h, w, _ = frame.shape
        n = h * w
        pixels = frame.reshape(-1, 3)

        # Nearest-center assignment in chunks to bound memory.
        # NOTE: int32 (not int16) -- squared channel diffs sum to ~195k,
        # which overflows int16 and scrambles the palette mapping.
        flat = pixels.astype(np.int32)
        c = centers.astype(np.int32)
        chunk = 100_000
        out_idx = np.empty(n, dtype=np.int64)
        for start in range(0, n, chunk):
            block = flat[start:start + chunk]  # (m, 3)
            # Squared Euclidean distance to each center: (m, k).
            dists = ((block[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
            out_idx[start:start + chunk] = np.argmin(dists, axis=1)
        quantized = centers[out_idx].reshape(h, w, 3)
        return np.ascontiguousarray(quantized, dtype=np.uint8)

    def quantize_colors(self, frame: np.ndarray, num_colors: int) -> np.ndarray:
        """Reduce a frame to ``num_colors`` discrete RGB values via K-Means."""
        self._validate_frame(frame)
        centers = self._kmeans_centers(frame, num_colors)
        return self._assign_palette(frame, centers)

    def fit_temporal_palette(self, frames: list[np.ndarray], num_colors: int) -> np.ndarray:
        """Fit one deterministic palette over representative clip pixels.

        Reusing these centers for every frame is essential for temporal
        vectorization: otherwise each frame's independent K-Means labels can
        change both fill colors and region boundaries.
        """
        if not frames:
            raise VideoValidationError("cannot fit a palette to zero frames")
        for frame in frames:
            self._validate_frame(frame)
        if not isinstance(num_colors, int) or num_colors < 1:
            raise VideoValidationError("num_colors must be a positive int")
        # Evenly sample frames across the whole clip (not just the head,
        # or scene changes later in the video get no palette entries) and
        # pixels within them, retaining a bounded but representative
        # training set for long clips.
        n_pick = min(len(frames), 32)
        picked = [frames[i] for i in np.linspace(0, len(frames) - 1, n_pick, dtype=int)]
        per_frame = max(1, 50_000 // n_pick)
        samples = []
        for frame in picked:
            pixels = frame.reshape(-1, 3)
            step = max(1, len(pixels) // per_frame)
            samples.append(pixels[::step][:per_frame])
        training = np.concatenate(samples, axis=0)
        h = len(training)
        proxy = training.reshape(h, 1, 3)
        return self._kmeans_centers(proxy, min(num_colors, h))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply enhancement chain: smoothing then quantization (if enabled)."""
        if self.config.enable_smoothing:
            frame = self.apply_edge_preserving_filter(frame)
        if self.config.color_count is not None:
            frame = self.quantize_colors(frame, self.config.color_count)
        return frame

    def _video_filters(self) -> list[str]:
        filters: list[str] = []
        if (
            self.config.target_fps is not None
            and abs(self.config.target_fps - self._metadata.fps) > 1e-6
        ):
            filters.append(f"fps={self._out_fps}")
        if (self._out_w, self._out_h) != (
            self._metadata.width,
            self._metadata.height,
        ):
            filters.append(f"scale={self._out_w}:{self._out_h}:flags=lanczos")
        return filters

    def _ffmpeg_cmd(self) -> list[str]:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise VideoValidationError("ffmpeg binary not found on PATH")
        cmd = [ffmpeg, "-v", "error", "-i", self.input_path]
        filters = self._video_filters()
        if filters:
            cmd += ["-vf", ",".join(filters)]
        # Raw RGB bytes on stdout: easy np.frombuffer parsing, no temp files.
        cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        return cmd

    def _read_exact(
        self, stream, nbytes: int, proc: subprocess.Popen
    ) -> bytes | None:
        """Read exactly nbytes or return None on clean EOF."""
        chunks: list[bytes] = []
        remaining = nbytes
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if not chunks:
            return None  # clean EOF
        data = b"".join(chunks)
        if len(data) < nbytes:
            stderr = b""
            try:
                _, stderr = proc.communicate(timeout=5)
            except Exception:
                pass
            raise VideoValidationError(
                f"Truncated FFmpeg frame ({len(data)}/{nbytes} bytes): "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return data

    def _extract_frames(self, *, quantize: bool) -> Iterator[np.ndarray]:
        """Yield frames one-by-one as ``uint8`` RGB arrays ``[H, W, 3]``.

        FFmpeg handles spatial resizing; each decoded frame then passes
        through :meth:`process_frame` (smoothing -> quantization) unless
        bypassed via config.
        """
        cmd = self._ffmpeg_cmd()
        frame_size = self._out_w * self._out_h * 3
        max_frames = self.config.max_frames
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except OSError as exc:
            raise VideoValidationError(f"Failed to launch ffmpeg: {exc}") from exc

        assert proc.stdout is not None
        count = 0
        try:
            while True:
                if max_frames is not None and count >= max_frames:
                    break
                raw = self._read_exact(proc.stdout, frame_size, proc)
                if raw is None:
                    break  # EOF
                frame = (
                    np.frombuffer(raw, dtype=np.uint8)
                    .reshape((self._out_h, self._out_w, 3))
                    .copy()
                )
                if self.config.enable_smoothing:
                    frame = self.apply_edge_preserving_filter(frame)
                if quantize and self.config.color_count is not None:
                    frame = self.quantize_colors(frame, self.config.color_count)
                count += 1
                yield frame
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                _, stderr = proc.communicate(timeout=10)
            except Exception:
                proc.kill()
                _, stderr = proc.communicate()
            if proc.returncode not in (0, None) and count == 0:
                msg = (
                    stderr.decode(errors="replace").strip()
                    if isinstance(stderr, bytes)
                    else str(stderr or "").strip()
                )
                raise VideoValidationError(f"FFmpeg decode failed: {msg}")

    def extract_frames_generator(self) -> Iterator[np.ndarray]:
        """Yield independently enhanced frames for streaming consumers.

        Use :meth:`extract_all_frames` for video-to-Lottie work: it can fit a
        palette across the whole clip and therefore avoids temporal palette
        flicker.
        """
        yield from self._extract_frames(quantize=True)

    def extract_all_frames(self) -> list[np.ndarray]:
        """Load all frames into memory. Only for short clips."""
        frames = list(self._extract_frames(quantize=not (
            self.config.temporal_palette and self.config.color_count is not None
        )))
        if frames and self.config.temporal_palette and self.config.color_count is not None:
            centers = self.fit_temporal_palette(frames, self.config.color_count)
            frames = [self._assign_palette(frame, centers) for frame in frames]
        return frames


__all__ = [
    "VideoValidationError",
    "VideoMetadata",
    "VideoPreprocessor",
    "probe_video",
    "resolve_output_geometry",
]
