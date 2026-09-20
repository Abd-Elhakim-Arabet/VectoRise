#!/usr/bin/env python3
"""``vectorise``: video -> Lottie JSON (+ MP4 preview) from the command line.

Two builders share the same patch decomposition, pick via ``--mode``:

* ``main`` (default) -- :func:`build_video_lottie`: every frame gets its
  own still (``f{t}_layer_<id>``, lifetime ``[t, t+1)``). Exact,
  streaming with flat memory, bigger JSON.
* ``compressed`` -- :func:`build_compressed_video_lottie`: the video is
  split into scenes, each scene keeps ONE fixed patch set (middle
  reference frame) and all other frames only translate those patches
  with dense optical flow + sparse keyframes. Much smaller JSON,
  motion-approximated, batch build (whole clip in RAM).

* main path: preprocess at the working resolution, quantize each frame
  to ``--num-colors`` colors, trace one Lottie ``ShapeLayer`` per patch,
  patch all frames into one JSON, streaming each frame to disk.
* with ``--mp4`` the finished JSON is additionally rendered to
  ``<video-stem>_converted.mp4`` (H264, yuv420p, faststart: plays in
  browsers/Jupyter -- cv2's mp4v does not). Main does it in the same
  streaming pass; compressed renders streaming from the saved JSON.

Typical usage::

    vectorise clip.mp4
    vectorise clip.mp4 -o out/clip.json --num-colors 12
    vectorise clip.mp4 --outdir out/ --quality medium --fps low --mp4
    vectorise clip.mp4 --mode compressed --mp4
    vectorise clip.mp4 --compressed --flow farneback --keyframe-step 1

Two knobs control the working resolution (both default to ``medium``,
the notebook's 720p @ 24fps sweet spot):

* ``--quality`` -- longest-side cap in px: low=384, medium=720, max=1080.
* ``--fps`` -- working frame rate: low=12, medium=24, max=30.

``--max-dim`` / ``--fps-value`` override the preset with an exact number.
Compressed-only knobs (``--flow``, ``--keyframe-step``,
``--scene-threshold``, ``--scene-min-len``, ``--max-frames``) are ignored
in main mode.
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

# Matches both builders' progress lines: "  frame 25/609: 4 patches"
# (main) and "  frame 12/120: scene 1 done" (compressed).
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
        "--mode", choices=("main", "compressed"), default=None,
        help="builder: 'main' = exact per-frame stills (default), "
             "'compressed' = scene-split + flow-tracked (smaller JSON). "
             "Shortcut: --compressed means --mode compressed.",
    )
    ap.add_argument(
        "--compressed", action="store_true",
        help="shortcut for --mode compressed",
    )
    ap.add_argument(
        "--flow", choices=("dis", "farneback"), default="dis",
        help="compressed-mode dense flow backend (default: dis). "
             "Ignored in main mode.",
    )
    ap.add_argument(
        "--keyframe-step", type=int, default=2, metavar="N",
        help="compressed-mode: emit one path keyframe every N frames "
             "(default: 2; 1 = every frame). Ignored in main mode.",
    )
    ap.add_argument(
        "--scene-threshold", type=float, default=27.0, metavar="F",
        help="compressed-mode scene cut sensitivity, higher = fewer cuts "
             "(default: 27.0). Ignored in main mode.",
    )
    ap.add_argument(
        "--scene-min-len", type=int, default=15, metavar="N",
        help="compressed-mode minimum scene length in frames (default: 15). "
             "Ignored in main mode.",
    )
    ap.add_argument(
        "--max-frames", type=int, default=None, metavar="N",
        help="compressed-mode safety cap on decoded working frames "
             "(batch build holds the clip in RAM; default: none). "
             "Ignored in main mode.",
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
    mode = args.mode or "main"
    if args.compressed:
        if args.mode is not None and args.mode != "compressed":
            # Explicit --mode main + --compressed is contradictory; the flag
            # wins but warn so scripts don't silently get the wrong builder.
            print("warning: --compressed overrides --mode "
                  f"{args.mode!r} -> 'compressed'", file=sys.stderr)
        mode = "compressed"
    if args.keyframe_step < 1:
        print(f"error: --keyframe-step must be >= 1, got {args.keyframe_step}",
              file=sys.stderr)
        return 2
    if args.scene_threshold <= 0:
        print(f"error: --scene-threshold must be > 0, got {args.scene_threshold}",
              file=sys.stderr)
        return 2
    if args.scene_min_len < 1:
        print(f"error: --scene-min-len must be >= 1, got {args.scene_min_len}",
              file=sys.stderr)
        return 2
    if args.max_frames is not None and args.max_frames < 1:
        print(f"error: --max-frames must be >= 1, got {args.max_frames}",
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

    from core_engine.pipeline.scene_video import (
        CompressedVideoConfig,
        VideoConfig,
        build_compressed_video_lottie,
        build_video_lottie,
    )

    mp4_path = out.parent / f"{src.stem}_converted.mp4" if args.mp4 else None

    def _run_main(verbose: bool) -> dict:
        return build_video_lottie(VideoConfig(
            input_path=str(src), output_path=str(out),
            target_fps=fps, max_dimension=max_dim,
            num_colors=args.num_colors,
            merge_min_area=args.merge_area,
        ), verbose=verbose, preview_mp4=mp4_path)

    def _run_compressed(verbose: bool) -> dict:
        return build_compressed_video_lottie(CompressedVideoConfig(
            input_path=str(src), output_path=str(out),
            target_fps=fps, max_dimension=max_dim,
            num_colors=args.num_colors,
            merge_min_area=args.merge_area,
            flow_method=args.flow,
            keyframe_step=args.keyframe_step,
            scene_threshold=args.scene_threshold,
            scene_min_len=args.scene_min_len,
            max_frames=args.max_frames,
        ), verbose=verbose, preview_mp4=mp4_path)

    run = _run_compressed if mode == "compressed" else _run_main

    show_bar = tqdm is not None and not args.quiet
    try:
        if show_bar:
            bar = tqdm(desc=f"vectorising ({mode})", unit="frames")
            try:
                with contextlib.redirect_stdout(_ProgressWriter(sys.stdout, bar)):
                    report = run(verbose=True)
            finally:
                bar.close()
        else:
            report = run(verbose=not args.quiet)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\n[{mode}] {report['num_frames']} frames @ {report['fps']:.1f}fps -> "
          f"{report['num_tracks']} tracks, {report['size_kb']:.0f} KB -> {out}")
    if mode == "compressed":
        for s in report.get("scenes", []):
            print(f"  scene {s['index']}: frames [{s['start_frame']}:{s['end_frame']}] "
                  f"ref={s['ref_frame']} {s['num_layers']} patches, "
                  f"motion mean={s['mean_motion_px']:.1f}px max={s['max_motion_px']:.1f}px")
    for c in report.get("sample_checks", []):
        if "missing_pixels" in c:
            print(f"  sample frame {c['frame']:{3}d}: {c['num_layers']:{3}d} patches, "
                  f"missing={c['missing_pixels']}px canvas={c['canvas_diff_pixels']}px")
        else:
            print(f"  sample frame {c['frame']:{3}d}: {c['num_layers']:{3}d} patches, "
                  f"mae={c.get('mae', 0.0):.2f} motion={c.get('mean_motion_px', 0.0):.1f}px")
    if mp4_path is not None:
        print(f"{out.name} -> wrote {mp4_path.name} "
              f"({mp4_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
