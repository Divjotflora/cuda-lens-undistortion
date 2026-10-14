#!/usr/bin/env python3
"""Builds before/after PNGs for the proof-of-execution artifacts.

For each sampled frame it writes  <out>/<name>_before_after.png  with
  [ distorted input | GPU-corrected output | ground truth (if present) ]
and a combined <out>/contact_sheet.png with one row per sampled frame.

Requires numpy and Pillow.
"""

import argparse
import os
import sys

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    print("Pillow not installed; skipping contact sheet "
          "(pip install pillow)")
    sys.exit(0)


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    tokens, pos = [], 0
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        tokens.append(data[pos:end])
        pos = end
    pos += 1
    w, h = int(tokens[1]), int(tokens[2])
    return np.frombuffer(data[pos:pos + w * h * 3], np.uint8).reshape(h, w, 3)


def labeled(img, text, width):
    im = Image.fromarray(img)
    im = im.resize((width, int(im.height * width / im.width)), Image.BILINEAR)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, 8 * len(text) + 10, 18], fill=(0, 0, 0))
    draw.text((5, 3), text, fill=(255, 255, 255))
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True,
                        help="directory with corrected images")
    parser.add_argument("--gt_dir", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--tile_width", type=int, default=400)
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"no corrected images in {args.output_dir}; skipping")
        return 0
    names = sorted(n for n in os.listdir(args.output_dir)
                   if n.endswith(".ppm"))
    if not names:
        print(f"no corrected images in {args.output_dir}")
        return 1
    step = max(1, len(names) // args.samples)
    picks = names[::step][:args.samples]
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for name in picks:
        tiles = [
            labeled(read_ppm(os.path.join(args.input_dir, name)),
                    f"input (distorted) {name}", args.tile_width),
            labeled(read_ppm(os.path.join(args.output_dir, name)),
                    "GPU corrected", args.tile_width),
        ]
        gt_path = os.path.join(args.gt_dir, name) if args.gt_dir else ""
        if gt_path and os.path.exists(gt_path):
            tiles.append(labeled(read_ppm(gt_path), "ground truth",
                                 args.tile_width))
        row = Image.new("RGB", (sum(t.width for t in tiles) + 4 *
                                (len(tiles) - 1), tiles[0].height),
                        (255, 255, 255))
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width + 4
        stem = os.path.splitext(name)[0]
        row.save(os.path.join(args.out, f"{stem}_before_after.png"))
        rows.append(row)

    sheet = Image.new("RGB", (max(r.width for r in rows),
                              sum(r.height for r in rows) + 4 *
                              (len(rows) - 1)), (255, 255, 255))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + 4
    sheet.save(os.path.join(args.out, "contact_sheet.png"))
    print(f"wrote {len(rows)} before/after images + contact_sheet.png "
          f"to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
