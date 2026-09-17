#!/usr/bin/env python3
"""``vectorise``: video -> Lottie JSON (+ MP4 preview) from the command line.

Self-contained CLI over the exact notebook path
(``packages/core/tests/preprocessing_demo.ipynb``):

* cell 1 -- :func:`build_braindead_video_lottie`: preprocess at the working
  resolution, quantize each frame to ``--num-colors`` colors, trace one
  Lottie ``ShapeLayer`` per patch, patch all frames into one JSON, streaming
  each frame to disk so memory stays flat.
* cell 2 -- with ``--mp4`` the finished JSON is additionally converted to
  ``<video-stem>_converted.mp4`` in the same streaming pass (H264, yuv420p,
  faststart: plays in browsers/Jupyter -- cv2's mp4v does not).

Typical usage::

    vectorise clip.mp4
    vectorise clip.mp4 -o out/clip.json --num-colors 12
    vectorise clip.mp4 --outdir out/ --quality medium --fps low --mp4

Two knobs control the working resolution (both default to ``medium``,
the notebook's 720p @ 24fps sweet spot):

* ``--quality`` -- longest-side cap in px: low=384, medium=720, max=1080.
* ``--fps`` -- working frame rate: low=12, medium=24, max=30.

``--max-dim`` / ``--fps-value`` override the preset with an exact number.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import shutil
import sys
from pathlib import Path

from vector_cli import __version__

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback: plain prints, no bar
    tqdm = None  # type: ignore[assignment]

# Presets: --quality picks the longest-side cap, --fps the working rate.
# medium matches the notebook (720px @ 24fps).
QUALITY_DIMS = {"low": 384, "medium": 720, "max": 1080}
FPS_RATES = {"low": 12.0, "medium": 24.0, "max": 30.0}

MIN_COLORS, MAX_COLORS = 8, 24

# Runtime pieces the pipeline needs, with the pip package that provides each
# (checked up front so a broken env fails fast with a fix, not mid-run).
_REQUIRED_MODULES = (
    ("cv2", "opencv-python-headless"),
    ("PIL", "pillow"),
    ("lottie", "lottie"),
)
_REQUIRED_BINARIES = ("ffmpeg", "ffprobe")


def _check_environment() -> str | None:
    """Return an error message if a runtime dependency is missing, else None."""
    for module, package in _REQUIRED_MODULES:
        try:
            __import__(module)
        except ImportError:
            return (f"missing required package '{package}' "
                    f"(import {module} failed). Fix: pip install -e packages/core")
    for binary in _REQUIRED_BINARIES:
        if shutil.which(binary) is None:
            return (f"missing required binary '{binary}' on PATH. Fix: "
                    f"brew install ffmpeg  (macOS)  /  sudo apt install ffmpeg  (Linux)")
    return None

# Matches the core builder's progress lines: "  frame 25/609: 4 patches".
_FRAME_RE = re.compile(r"^frame\s+(\d+)\s*/\s*(\d+)\s*:")


class _ProgressWriter:
    """Stdout proxy that folds the core's ``frame t/n`` lines into a bar.

    Lets the CLI show progress without touching working core code: the
    builder's per-frame prints are consumed as bar updates, every other
    line passes through to the real stdout untouched.
    """

    def __init__(self, stream, bar) -> None:
        self._stream = stream
        self._bar = bar
        self._buf = ""
        self._last = -1

    def write(self, s: str):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            m = _FRAME_RE.match(line.strip())
            if m and self._bar is not None:
                t, total = int(m.group(1)), int(m.group(2)) + 1
                if self._bar.total != total:
                    self._bar.total = total
                self._bar.update(t - self._last)
                self._last = t
            else:
                self._stream.write(line + "\n")
        return len(s)

    def flush(self):
        if self._buf:
            self._stream.write(self._buf)
            self._buf = ""
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vectorise",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="input video file (mp4/mov/webm/avi/mkv/...)")
    ap.add_argument(
        "-o", "--output", default=None,
        help="output .json path (default: same name as the input with a "
             ".json extension, next to the input; a directory places the "
             "default name inside it; mutually exclusive with --outdir)",
    )
    ap.add_argument(
        "--outdir", default=None, metavar="DIR",
        help="write <video-stem>.json (+ .mp4 preview) into DIR "
             "(mutually exclusive with -o/--output)",
    )
    ap.add_argument(
        "--num-colors", type=int, default=16, metavar="N",
        help=f"palette size per frame, {MIN_COLORS}..{MAX_COLORS} "
             f"(default: 16)",
    )
    ap.add_argument(
        "--quality", choices=("low", "medium", "max"), default="medium",
        help="resolution preset (longest side capped at 384/720/1080px; "
             "default: medium). Overridden by --max-dim.",
    )
    ap.add_argument(
        "--fps", choices=("low", "medium", "max"), default="medium",
        help="frame-rate preset: 12/24/30 fps (default: medium). "
             "Overridden by --fps-value.",
    )
    ap.add_argument(
        "--max-dim", type=int, default=None, metavar="PX",
        help="exact longest-side cap in px (overrides --quality)",
    )
    ap.add_argument(
        "--fps-value", type=float, default=None, metavar="FPS",
        help="exact working frame rate (overrides --fps)",
    )
    ap.add_argument(
        "--merge-area", type=int, default=10, metavar="PX",
        help="patches smaller than this (px) dissolve into neighbours "
             "(default: 10)",
    )
    ap.add_argument(
        "--mp4", action="store_true",
        help="also convert the finished JSON to <video-stem>_converted.mp4 "
             "next to it via ffmpeg (default: JSON only)",
    )
    ap.add_argument(
        "-q", "--quiet", action="store_true",
        help="only print the final summary (hide per-frame progress)",
    )
    ap.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    src = Path(args.input)
    if not src.is_file():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 2
    if not (MIN_COLORS <= args.num_colors <= MAX_COLORS):
        print(f"error: --num-colors must be {MIN_COLORS}..{MAX_COLORS}, "
              f"got {args.num_colors}", file=sys.stderr)
        return 2
    max_dim = args.max_dim if args.max_dim is not None else QUALITY_DIMS[args.quality]
    if max_dim < 16:
        print(f"error: --max-dim must be >= 16, got {max_dim}", file=sys.stderr)
        return 2
    fps = args.fps_value if args.fps_value is not None else FPS_RATES[args.fps]
    if fps <= 0:
        print(f"error: fps must be > 0, got {fps}", file=sys.stderr)
        return 2
    if args.merge_area < 1:
        print(f"error: --merge-area must be >= 1, got {args.merge_area}",
              file=sys.stderr)
        return 2

    env_error = _check_environment()
    if env_error is not None:
        print(f"error: {env_error}", file=sys.stderr)
        return 1

    if args.output is not None and args.outdir is not None:
        print("error: --outdir and -o/--output are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.outdir is not None:
        out = Path(args.outdir) / f"{src.stem}.json"
    elif args.output is None:
        out = src.with_suffix(".json")
    else:
        out = Path(args.output)
        if out.is_dir() or args.output.endswith(("/", "\\")):
            out = out / f"{src.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    from core_engine.pipeline.scene_video import build_braindead_video_lottie
    from core_engine.pipeline.scene_video import BraindeadVideoConfig

    show_bar = tqdm is not None and not args.quiet
    mp4_path = out.parent / f"{src.stem}_converted.mp4" if args.mp4 else None
    try:
        if show_bar:
            bar = tqdm(desc="vectorising", unit="frames")
            try:
                with contextlib.redirect_stdout(_ProgressWriter(sys.stdout, bar)):
                    report = build_braindead_video_lottie(BraindeadVideoConfig(
                        input_path=str(src), output_path=str(out),
                        target_fps=fps, max_dimension=max_dim,
                        num_colors=args.num_colors,
                        merge_min_area=args.merge_area,
                    ), verbose=True, preview_mp4=mp4_path)
            finally:
                bar.close()
        else:
            report = build_braindead_video_lottie(BraindeadVideoConfig(
                input_path=str(src), output_path=str(out),
                target_fps=fps, max_dimension=max_dim,
                num_colors=args.num_colors, merge_min_area=args.merge_area,
            ), verbose=not args.quiet, preview_mp4=mp4_path)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\n{report['num_frames']} frames @ {report['fps']:.1f}fps -> "
          f"{report['num_tracks']} tracks, {report['size_kb']:.0f} KB -> {out}")
    for c in report["sample_checks"]:
        print(f"  sample frame {c['frame']:{3}d}: {c['num_layers']:{3}d} patches, "
              f"missing={c['missing_pixels']}px canvas={c['canvas_diff_pixels']}px")
    if mp4_path is not None:
        print(f"{out.name} -> wrote {mp4_path.name} "
              f"({mp4_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
