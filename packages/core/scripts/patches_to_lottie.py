#!/usr/bin/env python3
"""Translate all patches of a given frame into a verified Lottie file.

Usage::

    python patches_to_lottie.py still.png -o still.json
    python patches_to_lottie.py clip.mp4 --frame-number 200 -o still.json --save-renders

The script builds one Lottie ``ShapeLayer`` per patch (connected
same-color region), then renders the saved JSON back to PNGs and compares
every rendered patch against its original patch PNG pixel-for-pixel. If
any patch is off, the JSON is deleted (refused) and the script exits
non-zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core_engine.pipeline.patches_lottie import (  # noqa: E402
    LottieVerifyError,
    patches_to_lottie_verified,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_frame(source: Path, frame_number: int) -> np.ndarray:
    """Load an ``uint8`` RGB frame from an image file or a video file."""
    if source.suffix.lower() in IMAGE_EXTS:
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not read image: {source}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        raise SystemExit(f"no frames decoded from: {source}")
    frame_number = max(0, min(frame_number, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame_number} from: {source}")
    print(f"frame {frame_number}/{total - 1} of {source.name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="frame image or video file")
    ap.add_argument("-o", "--output", default=None,
                    help="output .json path (default: <source-stem>_patches.json)")
    ap.add_argument("--frame-number", type=int, default=0,
                    help="frame to use when source is a video (default: 0)")
    ap.add_argument("--num-colors", type=int, default=16,
                    help="quantization palette size 2..24 (default: 16)")
    ap.add_argument("--min-layer-area", type=int, default=10,
                    help="drop patches smaller than this (px, default: 10)")
    ap.add_argument("--save-renders", action="store_true",
                    help="save per-patch render PNGs + stacked composite next "
                         "to the output for eyeballing")
    args = ap.parse_args(argv)

    src = Path(args.source)
    if not src.is_file():
        print(f"error: not found: {src}", file=sys.stderr)
        return 2
    out = Path(args.output) if args.output else Path(f"{src.stem}_patches.json")

    frame = load_frame(src, args.frame_number)
    print(f"frame {frame.shape[1]}x{frame.shape[0]}, "
          f"num_colors={args.num_colors}, min_layer_area={args.min_layer_area}")

    renders_dir = out.parent / f"{out.stem}_renders" if args.save_renders else None
    try:
        report = patches_to_lottie_verified(
            frame, str(out),
            num_colors=args.num_colors,
            min_layer_area=args.min_layer_area,
            save_renders_dir=renders_dir,
        )
    except LottieVerifyError as exc:
        print(f"REFUSED: {out} did not round-trip -- deleted. {exc}",
              file=sys.stderr)
        return 1

    for p in report["layers"]:
        print(f"  layer_{p['id']:4d} color={str(p['color']):18s} "
              f"area={p['area']:7d} missing={p['missing_pixels']:6d} "
              f"extra={p['extra_pixels']:6d} "
              f"{'PASS' if p['ok'] else 'FAIL'}")
    print(f"{report['num_layers']} patches, canvas "
          f"{report['canvas_diff_pixels']}/{report['canvas_pixels']}px differ "
          f"-> {out} ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
