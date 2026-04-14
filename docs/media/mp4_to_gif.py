"""Convert the recorded MP4 demos to optimized GIFs for README embedding.

Usage:
    uv run --extra media python docs/media/mp4_to_gif.py docs/media/baseline.mp4
    uv run --extra media python docs/media/mp4_to_gif.py docs/media/perturbed.mp4

Produces <name>.gif next to the input MP4. Uses a single shared palette quantized
from all frames to keep file size small enough for README embedding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


def convert(mp4_path: Path, fps: int, max_width: int, colors: int, frame_stride: int) -> Path:
    """Read an MP4 and write an optimized GIF with a shared global palette."""
    gif_path = mp4_path.with_suffix(".gif")
    frames = iio.imread(mp4_path, plugin="pyav")[::frame_stride]

    h, w = frames.shape[1], frames.shape[2]
    target_size: tuple[int, int] | None = None
    if w > max_width:
        target_size = (max_width, int(h * max_width / w))

    # Build a shared palette from a sampled subset of frames
    sample = frames[:: max(1, len(frames) // 20)]
    sample_pil = [Image.fromarray(f) for f in sample]
    if target_size is not None:
        sample_pil = [img.resize(target_size, Image.LANCZOS) for img in sample_pil]
    mosaic = Image.fromarray(np.concatenate([np.array(img) for img in sample_pil], axis=0))
    palette_img = mosaic.convert("P", palette=Image.ADAPTIVE, colors=colors)

    resized: list[Image.Image] = []
    for f in frames:
        img = Image.fromarray(f)
        if target_size is not None:
            img = img.resize(target_size, Image.LANCZOS)
        resized.append(img.quantize(palette=palette_img, dither=Image.FLOYDSTEINBERG))

    resized[0].save(
        gif_path,
        save_all=True,
        append_images=resized[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mp4", type=Path, help="Input MP4 file")
    parser.add_argument("--fps", type=int, default=15, help="Output GIF fps (default: 15)")
    parser.add_argument(
        "--max-width", type=int, default=360, help="Max output width in pixels (default: 360)"
    )
    parser.add_argument("--colors", type=int, default=64, help="Palette size 2-256 (default: 64)")
    parser.add_argument(
        "--frame-stride", type=int, default=2, help="Keep every Nth frame (default: 2)"
    )
    args = parser.parse_args()

    gif_path = convert(args.mp4, args.fps, args.max_width, args.colors, args.frame_stride)
    print(f"Saved: {gif_path} ({gif_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
