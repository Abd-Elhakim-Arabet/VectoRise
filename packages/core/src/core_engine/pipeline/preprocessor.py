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
        self._metadata = probe_video(self.input_path)
        self._out_w, self._out_h, self._out_fps = resolve_output_geometry(
            self._metadata, self.config
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

    def extract_frames_generator(self) -> Iterator[np.ndarray]:
        """Yield frames one-by-one as ``uint8`` RGB arrays ``[H, W, 3]``."""
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

    def extract_all_frames(self) -> list[np.ndarray]:
        """Load all frames into memory. Only for short clips."""
        return list(self.extract_frames_generator())


__all__ = [
    "VideoValidationError",
    "VideoMetadata",
    "VideoPreprocessor",
    "probe_video",
    "resolve_output_geometry",
]
