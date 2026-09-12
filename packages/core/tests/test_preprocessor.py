"""Unit tests for the video preprocessor.

Prefers a real clip at ``tests/samples/sample1.mp4`` (or the
``packages/core/samples/`` alt; both git-ignored). Falls back to a
synthetic testsrc clip when no sample video is present.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from core_engine.config import VectorizeConfig
from core_engine.pipeline.preprocessor import (
    VideoPreprocessor,
    VideoValidationError,
    probe_video,
)

# Sample clip candidates (local, git-ignored -- drop your own clip in either).
# tests/samples/ sits next to this file; packages/core/samples/ is the alt.
_HERE = Path(__file__).resolve().parent
SAMPLE_CANDIDATES = [
    _HERE / "samples" / "sample1.mp4",
    _HERE.parent / "samples" / "sample1.mp4",
]


def _find_sample() -> Path | None:
    for cand in SAMPLE_CANDIDATES:
        if cand.is_file() and cand.stat().st_size > 0:
            return cand
    return None

SYNTH_WIDTH, SYNTH_HEIGHT, SYNTH_FPS, SYNTH_DURATION = 64, 48, 30, 1


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH")


def _make_clip(path: Path, w: int = SYNTH_WIDTH, h: int = SYNTH_HEIGHT) -> Path:
    _require_ffmpeg()
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"testsrc=size={w}x{h}:rate={SYNTH_FPS}:duration={SYNTH_DURATION}",
        "-pix_fmt", "yuv420p", "-c:v", "libx264",
        str(path),
    ]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    """Real sample if present, else a synthetic clip."""
    sample = _find_sample()
    if sample is not None:
        return sample
    return _make_clip(tmp_path / "clip.mp4")


@pytest.fixture()
def is_real_sample(clip: Path) -> bool:
    return clip.name == "sample1.mp4"


def test_probe_valid(clip: Path, is_real_sample: bool) -> None:
    meta = probe_video(clip)
    if is_real_sample:
        # Arbitrary user clip: only sanity-check.
        assert meta.width > 0 and meta.height > 0
        assert meta.fps > 0
        assert meta.duration is None or meta.duration > 0
    else:
        assert (meta.width, meta.height) == (SYNTH_WIDTH, SYNTH_HEIGHT)
        assert meta.fps == pytest.approx(SYNTH_FPS)
        assert meta.duration == pytest.approx(SYNTH_DURATION, abs=0.1)
        assert meta.frame_count == pytest.approx(SYNTH_FPS * SYNTH_DURATION, abs=2)


def test_probe_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoValidationError):
        probe_video(tmp_path / "nope.mp4")


def test_probe_corrupt_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video" * 100)
    with pytest.raises(VideoValidationError):
        probe_video(bad)


def test_probe_audio_only_raises(tmp_path: Path) -> None:
    _require_ffmpeg()
    out = tmp_path / "audio.m4a"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "anullsrc=r=44100:cl=mono:d=1",
             "-c:a", "aac", str(out)],
            check=True,
        )
    except subprocess.CalledProcessError:
        pytest.skip("could not synthesize audio-only file")
    with pytest.raises(VideoValidationError):
        probe_video(out)


def test_generator_shape_dtype_and_fps(clip: Path) -> None:
    meta = probe_video(clip)
    cfg = VectorizeConfig(
        input_path=str(clip), output_path="", target_fps=10.0
    )
    pre = VideoPreprocessor(str(clip), cfg)
    frames = list(pre.extract_frames_generator())
    assert len(frames) > 0
    if meta.duration is not None:
        expected = meta.duration * 10.0
        assert len(frames) == pytest.approx(expected, abs=max(2, expected * 0.2))
    for f in frames:
        assert isinstance(f, np.ndarray)
        assert f.dtype == np.uint8
        assert f.shape == (meta.height, meta.width, 3)


def test_spatial_downscale(clip: Path) -> None:
    cfg = VectorizeConfig(
        input_path=str(clip), output_path="", max_dimension=32
    )
    pre = VideoPreprocessor(str(clip), cfg)
    assert max(pre.output_width, pre.output_height) <= 32
    assert pre.output_width % 2 == 0 and pre.output_height % 2 == 0
    frame = next(iter(pre.extract_frames_generator()))
    assert frame.shape == (pre.output_height, pre.output_width, 3)


def test_batch_matches_generator(clip: Path) -> None:
    # max_frames bounds memory on long real clips; both sides share the
    # config so batch-vs-generator equality is still exact.
    cfg = VectorizeConfig(
        input_path=str(clip), output_path="", target_fps=10.0, max_frames=10
    )
    pre = VideoPreprocessor(str(clip), cfg)
    gen_frames = list(pre.extract_frames_generator())
    batch = pre.extract_all_frames()
    assert len(batch) == len(gen_frames)
    assert all(np.array_equal(a, b) for a, b in zip(batch, gen_frames))


def test_max_frames_cap(clip: Path) -> None:
    cfg = VectorizeConfig(
        input_path=str(clip), output_path="", target_fps=10.0, max_frames=3
    )
    pre = VideoPreprocessor(str(clip), cfg)
    assert len(pre.extract_all_frames()) == 3
