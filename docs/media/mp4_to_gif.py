"""Convert the recorded MP4 demos to optimized GIFs for README embedding.

Usage:
    uv run --extra media python docs/media/mp4_to_gif.py docs/media/baseline.mp4
    uv run --extra media python docs/media/mp4_to_gif.py docs/media/perturbed.mp4

Produces <name>.gif next to the input MP4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
from PIL import Image


def convert(mp4_path: Path, fps: int, max_width: int) -> Path:
    """Read an MP4 and write an optimized GIF with the same basename."""
    gif_path = mp4_path.with_suffix(".gif")
    frames = iio.imread(mp4_path, plugin="pyav")

    h, w = frames.shape[1], frames.shape[2]
    target_size: tuple[int, int] | None = None
    if w > max_width:
        target_size = (max_width, int(h * max_width / w))

    resized: list[Image.Image] = []
    for f in frames:
        img = Image.fromarray(f)
        if target_size is not None:
            img = img.resize(target_size, Image.LANCZOS)
        resized.append(img.convert("P", palette=Image.ADAPTIVE))

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
    parser.add_argument("--fps", type=int, default=20, help="Output GIF fps (default: 20)")
    parser.add_argument(
        "--max-width", type=int, default=480, help="Max output width in pixels (default: 480)"
    )
    args = parser.parse_args()

    gif_path = convert(args.mp4, args.fps, args.max_width)
    print(f"Saved: {gif_path}")


if __name__ == "__main__":
    main()
